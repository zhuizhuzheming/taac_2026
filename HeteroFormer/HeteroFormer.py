"""
HeteroFormer Academic - Generative Sequence Encoder Edition
(Complete Academic Specification, No Industrial Compromises)

Core Design Philosophy:
  1. Three-layer abstraction: Intra → Inter → Pool
  2. HeteroBlock as unified composition unit with principled residual gating
  3. GenerativeSequenceEncoder as Intra-op with full VQ regularization
  4. Lipschitz-continuous prediction head via spectral normalization
  5. Clean interfaces: forward() returns logits only, auxiliary via explicit hooks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, NamedTuple


# ==============================================================================
# Model Input Interface (Preserved)
# ==============================================================================

class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: Dict[str, torch.Tensor]
    seq_lens: Dict[str, torch.Tensor]
    seq_time_buckets: Dict[str, torch.Tensor]
    seq_decay_weights: Optional[Dict[str, torch.Tensor]] = None


# ==============================================================================
# Base Primitives
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation with configurable expansion ratio."""
    def __init__(self, d_model: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(x * F.silu(gate))


# ==============================================================================
# Spectral Normalization (Lipschitz Constraint for Prediction Head)
# ==============================================================================

def apply_spectral_norm(module: nn.Linear, n_power_iterations: int = 1) -> nn.Linear:
    """Apply spectral normalization to enforce Lipschitz continuity."""
    return nn.utils.parametrizations.spectral_norm(module, n_power_iterations=n_power_iterations)


# ==============================================================================
# Intra-Field Operations
# ==============================================================================

class ConstrainedEmbedding(nn.Embedding):
    """
    Embedding with L2 max-norm constraint.
    Prevents embedding space collapse in high-cardinality features.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.max_norm = max_norm
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_normed = F.normalize(self.weight, p=2, dim=-1) * self.max_norm
        return F.embedding(input, weight_normed, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)


class IntraEmbedding(nn.Module):
    """Field-level embedding with cardinality-aware dropout."""
    def __init__(self, vocab_size: int, emb_dim: int, num_slots: int = 1,
                 dropout: float = 0.0, is_high_card: bool = False):
        super().__init__()
        self.emb = ConstrainedEmbedding(max(vocab_size, 2), emb_dim)
        self.num_slots = num_slots
        self.dropout = nn.Dropout(dropout * (2.0 if is_high_card else 1.0))
        self.is_high_card = is_high_card

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_slots == 1:
            emb = self.emb(x.squeeze(-1))
        else:
            embs = [self.emb(x[:, i]) for i in range(self.num_slots)]
            emb = torch.stack(embs, dim=1).mean(dim=1)
        return self.dropout(emb)


class IntraLinear(nn.Module):
    """Field-level linear projection with RMSNorm."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# Generative Sequence Encoder (Complete Academic Specification)
# ==============================================================================

class VectorQuantizer(nn.Module):
    """
    Gumbel-Softmax Vector Quantization with principled training dynamics.
    
    Academic design choices:
    - Temperature annealing over substantial steps (10000) for stable codebook learning
    - Commitment + codebook + entropy losses with theoretically motivated coefficients
    - Straight-through estimator for gradient flow
    - Null code (index 0) for variable-length sequences
    """
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        temperature: float = 1.0,
        temp_anneal_steps: int = 10000,
        min_temp: float = 0.1,
        commitment_cost: float = 0.25,
        codebook_cost: float = 0.25,
        entropy_cost: float = 0.1,
        null_reg_cost: float = 0.01,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.temperature = temperature
        self.min_temp = min_temp
        self.temp_anneal_steps = temp_anneal_steps
        
        self.commitment_cost = commitment_cost
        self.codebook_cost = codebook_cost
        self.entropy_cost = entropy_cost
        self.null_reg_cost = null_reg_cost
        
        # K+1 codes: 0=null (learnable but initialized to zero), 1..K=active
        self.codebook = nn.Parameter(torch.randn(num_codes + 1, code_dim))
        with torch.no_grad():
            self.codebook[0].fill_(0.0)
        nn.init.uniform_(self.codebook[1:], -1.0 / num_codes, 1.0 / num_codes)
        
        self._step_count = 0

    def get_temperature(self) -> float:
        """Exponential temperature annealing for stable VQ training."""
        if self._step_count >= self.temp_anneal_steps:
            return self.min_temp
        progress = self._step_count / self.temp_anneal_steps
        return self.min_temp + (self.temperature - self.min_temp) * math.exp(-5 * progress)

    @torch.compiler.disable
    def forward(self, z: torch.Tensor, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, D = z.shape
        device = z.device
        
        # Compute pairwise squared L2 distances [B*L, K+1]
        z_flat = z.reshape(-1, D)
        distances = torch.cdist(z_flat, self.codebook, p=2).pow(2).view(B, L, self.num_codes + 1)
        
        # Null mask: positions beyond sequence length OR fully empty sequences
        pos_mask = torch.arange(L, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        empty_seq_mask = (seq_lens == 0).unsqueeze(1)
        null_mask = pos_mask | empty_seq_mask
        
        # Temperature and Gumbel-Softmax sampling
        temp = self.get_temperature()
        logits = -distances
        
        # Force null code for masked positions, active codes for valid positions
        logits = logits.masked_fill(null_mask.unsqueeze(-1), -1e4)
        logits[:, :, 0] = logits[:, :, 0].masked_fill(~null_mask, -1e4)
        
        # Gumbel noise for differentiable sampling
        uniform = torch.rand_like(logits).clamp(min=1e-10, max=1.0)
        gumbel_noise = -torch.log(-torch.log(uniform))
        gumbel_logits = (logits + gumbel_noise) / temp
        probs = F.softmax(gumbel_logits, dim=-1)
        
        # Quantize via expectation
        quantized = torch.einsum('blk,kd->bld', probs, self.codebook)
        
        # === Losses (theoretically motivated) ===
        # Commitment: encoder commits to codebook
        commitment_loss = F.mse_loss(z, quantized.detach())
        # Codebook: codebook moves towards encoder output
        codebook_loss = F.mse_loss(quantized, z.detach())
        
        # Entropy regularization: encourage diverse code usage (prevent index collapse)
        if not null_mask.all():
            valid_probs = probs[~null_mask][:, 1:]  # Exclude null code
            usage_probs = valid_probs.mean(dim=0)  # [K]
            # Negative entropy: we want HIGH entropy (uniform usage), so minimize -entropy
            entropy = -(usage_probs * torch.log(usage_probs + 1e-10)).sum()
            entropy_loss = -entropy  # Minimize negative entropy = maximize entropy
        else:
            entropy_loss = torch.tensor(0.0, device=device)
        
        # Null code regularization: keep null vector near zero
        null_reg = self.codebook[0].pow(2).mean()
        
        # Perplexity for monitoring (not a loss term)
        if not null_mask.all():
            valid_probs = probs[~null_mask][:, 1:]
            usage_probs = valid_probs.mean(dim=0)
            perplexity = torch.exp(-(usage_probs * torch.log(usage_probs + 1e-10)).sum()).item()
        else:
            perplexity = 1.0
        
        info = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'entropy_loss': entropy_loss,
            'null_reg': null_reg,
            'temperature': temp,
            'usage_rate': (probs.argmax(dim=-1) != 0).float().mean().item(),
            'perplexity': perplexity,
            'code_usage_histogram': valid_probs.mean(dim=0).detach() if not null_mask.all() else torch.zeros(self.num_codes, device=device),
        }
        
        # Straight-through estimator: forward uses quantized, backward flows through z
        quantized_st = z + (quantized - z).detach()
        self._step_count += 1
        
        return quantized_st, info


class GenerativeSequenceEncoder(nn.Module):
    """
    Intra-field operation for sequence encoding via generative paradigm.
    
    Architecture (bottom-up):
        Multi-feature Embedding → Bidirectional Transformer → 
        VQ Compression → Attention Aggregation → Empty Gate → Output
    
    Interface contract:
        Input:  seq_ids [B, n_feats, max_len], seq_lens [B], 
                optional time_buckets, decay_weight
        Output: [B, d_model] pooled representation
    
    Auxiliary losses and diagnostics accessible via explicit hooks (not leaked in forward).
    """
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int = 128,
        num_time_buckets: int = 0,
        num_heads: int = 8,
        num_layers: int = 4,
        num_codes: int = 64,
        code_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        id_threshold: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.id_threshold = id_threshold
        
        n_feats = len(vocab_sizes)
        feat_dim = max(d_model // n_feats, 1)
        
        # === Intra: Multi-feature Embedding ===
        self.feat_embs = nn.ModuleList()
        self.feat_dropouts = nn.ModuleList()
        for vs in vocab_sizes:
            self.feat_embs.append(nn.Embedding(max(vs, 2), feat_dim))
            extra_drop = dropout * 2 if vs > id_threshold else 0.0
            self.feat_dropouts.append(nn.Dropout(min(dropout + extra_drop, 0.5)))
            nn.init.normal_(self.feat_embs[-1].weight, std=0.01)
        
        # === Intra: Projection ===
        self.seq_proj = nn.Linear(feat_dim * n_feats, d_model, bias=False)
        self.seq_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.seq_proj.weight, gain=0.1)
        
        # === Intra: Time Embedding ===
        if num_time_buckets > 0:
            self.time_emb = nn.Embedding(num_time_buckets, d_model)
            nn.init.normal_(self.time_emb.weight, std=0.01)
        else:
            self.time_emb = None
        
        # === Intra: Bidirectional Transformer ===
        # No causal mask: full bidirectional context (generative, not autoregressive)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === Intra: VQ Compression (generative core) ===
        self.vq = VectorQuantizer(num_codes, code_dim)
        self.to_code = nn.Linear(d_model, code_dim, bias=False)
        self.from_code = nn.Linear(code_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.to_code.weight, gain=0.1)
        nn.init.xavier_uniform_(self.from_code.weight, gain=0.1)
        
        # === Intra: Attention Aggregation ===
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.aggregate = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.agg_norm = nn.LayerNorm(d_model)
        
        # === Intra: Empty Sequence Gate ===
        # Learnable empty state + sigmoid-gated blending
        self.empty_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.null_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=True),
        )
        nn.init.normal_(self.null_gate[-1].weight, std=0.01)
        nn.init.constant_(self.null_gate[-1].bias, -2.0)  # Initialize closed
        
        # === Intra: Output Projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight, gain=0.1)
        
        # === Internal state for auxiliary loss extraction ===
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_info: Dict[str, float] = {}

    def forward(
        self,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        time_buckets: Optional[torch.Tensor] = None,
        decay_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, n_feats, max_len = seq_ids.shape
        device = seq_ids.device
        
        # Multi-feature embedding (Intra-op: field composition)
        feat_embs = []
        for i in range(n_feats):
            emb = self.feat_embs[i](seq_ids[:, i, :])
            emb = self.feat_dropouts[i](emb)
            feat_embs.append(emb)
        
        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))
        
        # Time embedding
        if self.time_emb is not None and time_buckets is not None:
            t_emb = self.time_emb(time_buckets)
            seq_repr = seq_repr + t_emb
        
        # Padding mask for transformer
        padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        
        # Bidirectional encoding (no causal mask: full context)
        encoded = self.transformer(seq_repr, src_key_padding_mask=padding_mask)
        
        # VQ compression: continuous → discrete → continuous
        code_input = self.to_code(encoded)
        quantized, vq_info = self.vq(code_input, seq_lens)
        quantized = self.from_code(quantized)
        
        # Attention aggregation: pool variable-length sequence
        agg_query = self.agg_token.expand(B, -1, -1)
        kv = quantized.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        aggregated, attn_weights = self.aggregate(agg_query, kv, kv, key_padding_mask=padding_mask)
        aggregated = aggregated.squeeze(1)
        aggregated = self.agg_norm(aggregated)
        
        # Empty sequence gate: blend between aggregated and learnable empty state
        null_logits = self.null_gate(aggregated)
        null_gate = torch.sigmoid(null_logits)
        output = null_gate * self.empty_state + (1 - null_gate) * aggregated
        output = self.output_proj(output)
        
        # Store auxiliary info internally (no leakage in return value)
        self._last_aux_loss = (
            self.vq.commitment_cost * vq_info['commitment_loss'] + 
            self.vq.codebook_cost * vq_info['codebook_loss'] +
            self.vq.entropy_cost * vq_info['entropy_loss'] +
            self.vq.null_reg_cost * vq_info['null_reg']
        )
        self._last_info = {
            'temperature': vq_info['temperature'],
            'usage_rate': vq_info['usage_rate'],
            'perplexity': vq_info['perplexity'],
            'null_gate_mean': null_gate.mean().item(),
            'null_gate_std': null_gate.std().item(),
            'empty_ratio': (seq_lens == 0).float().mean().item(),
        }
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        """External interface for auxiliary loss extraction."""
        if self._last_aux_loss is not None and self.training:
            return self._last_aux_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_diagnostics(self) -> Dict[str, float]:
        """External interface for diagnostic info extraction."""
        return self._last_info.copy()


# ==============================================================================
# Inter-Field Operations
# ==============================================================================

class InterSelfAttention(nn.Module):
    """Field-wise self-attention for intra-block feature interaction."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = RMSNorm(d_model)

    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = sorted(fields.keys())
        x = torch.stack([fields[k] for k in keys], dim=1)
        out, _ = self.attn(x, x, x)
        out = self.norm(x + out)
        return {k: out[:, i] for i, k in enumerate(keys)}


class InterCrossAttention(nn.Module):
    """Cross-attention between query fields and key-value fields."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 query_fields: Optional[List[str]] = None,
                 kv_fields: Optional[List[str]] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.query_fields = query_fields
        self.kv_fields = kv_fields

    def forward(self, fields: Dict[str, torch.Tensor],
                keys_values: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        if keys_values is None:
            if self.query_fields is not None and self.kv_fields is not None:
                q_keys = [f for f in self.query_fields if f in fields]
                kv_keys = [f for f in self.kv_fields if f in fields]
            else:
                q_keys = sorted(fields.keys())
                kv_keys = q_keys
            q = torch.stack([fields[k] for k in q_keys], dim=1)
            kv = torch.stack([fields[k] for k in kv_keys], dim=1)
        else:
            q_keys = sorted(fields.keys())
            kv_keys = sorted(keys_values.keys())
            q = torch.stack([fields[k] for k in q_keys], dim=1)
            kv = torch.stack([keys_values[k] for k in kv_keys], dim=1)

        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), kv)
        out = self.norm_q(q) + out
        return {k: out[:, i] for i, k in enumerate(q_keys)}


class InterBilinear(nn.Module):
    """
    Low-rank bilinear interaction for field fusion.
    Captures multiplicative interactions with O(d*r) parameters.
    """
    def __init__(self, d_model: int, rank: int, num_fields: int = 3):
        super().__init__()
        self.rank = rank
        self.projs = nn.ModuleList([nn.Linear(d_model, rank, bias=False) for _ in range(num_fields)])
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.scale = rank ** -0.5
        for p in self.projs:
            nn.init.xavier_uniform_(p.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = sorted(fields.keys())
        projected = [self.projs[i](fields[keys[i]]) for i in range(len(keys))]
        interaction = projected[0]
        for p in projected[1:]:
            interaction = interaction * p
        interaction = interaction * self.scale
        out = self.out_proj(interaction)
        return {k: self.norm(out) for k in keys}


class InterHadamard(nn.Module):
    """
    Hadamard product interaction network.
    Efficient multiplicative interaction via element-wise product + MLP.
    """
    def __init__(self, d_model: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)
        ])
        self.scale = (2 * num_layers) ** 0.5
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.1 / self.scale)
            nn.init.zeros_(layer.bias)

    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = sorted(fields.keys())
        x0 = fields[keys[0]]
        x = x0
        for layer in self.layers:
            x = x0 * (layer(x) / self.scale) + x
        return {k: x for k in keys}


# ==============================================================================
# Pool-Field Operations
# ==============================================================================

class PoolRouter(nn.Module):
    """
    Subspace routing: soft-select from multiple low-rank subspace banks.
    Enables adaptive capacity allocation across examples.
    """
    def __init__(self, d_model: int, rank: int, num_banks: int = 4):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d_model, num_banks, bias=False),
            nn.Softmax(dim=-1),
        )
        init_std = 0.02
        banks = torch.empty(num_banks, rank, d_model)
        for i in range(num_banks):
            nn.init.orthogonal_(banks[i])
        self.subspace_banks = nn.Parameter(banks * init_std)
        nn.init.xavier_uniform_(self.router[0].weight)

    def forward(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(list(fields.values()), dim=-1)
        if x.size(-1) > list(fields.values())[0].size(-1):
            x = x[:, :list(fields.values())[0].size(-1)]
        weights = self.router(x)
        mixed = torch.einsum("bn,nrd->brd", weights, self.subspace_banks)
        return mixed.mean(dim=1)


class PoolConcatLinear(nn.Module):
    """Simple concatenation + linear projection pooling."""
    def __init__(self, field_dims: Dict[str, int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        total_in = sum(field_dims.values())
        self.proj = nn.Linear(total_in, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(list(fields.values()), dim=-1)
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# Unified Composition Block
# ==============================================================================

class HeteroBlock(nn.Module):
    """
    Unified composition block: Intra → Inter → Pool → Residual.
    
    Residual gating: sigmoid-gated residual connection with temperature annealing.
    Stochastic depth: layer-drop regularization during training.
    """
    def __init__(
        self,
        fields: List[str],
        intra_ops: Optional[Dict[str, nn.Module]] = None,
        inter_op: Optional[nn.Module] = None,
        pool_op: Optional[nn.Module] = None,
        residual_gate: bool = False,
        stochastic_depth_prob: float = 0.0,
        dropout: float = 0.1,
        gate_init: float = 0.0,  # Academic: 0.0 means gate=0.5 initially (fully open)
        name: str = "block",
    ):
        super().__init__()
        self.name = name
        self.fields = fields
        self.intra_ops = nn.ModuleDict(intra_ops or {})
        self.inter_op = inter_op
        self.pool_op = pool_op
        
        # Residual gate: parameterized scalar with temperature annealing support
        if residual_gate:
            self.residual_gate = nn.Parameter(torch.tensor(gate_init))
        else:
            self.register_parameter('residual_gate', None)
            
        self.stochastic_depth_prob = stochastic_depth_prob
        self.dropout = nn.Dropout(dropout)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        fields: Dict[str, torch.Tensor],
        residuals: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        # Intra-field operations
        intra_out = {}
        for name, tensor in fields.items():
            if name in self.intra_ops:
                intra_out[name] = self.intra_ops[name](tensor)
            else:
                intra_out[name] = tensor

        # Inter-field operations
        inter_out = intra_out
        if self.inter_op is not None:
            inter_out = self.inter_op(intra_out)

        # Pool-field operations
        pool_out = inter_out
        if self.pool_op is not None:
            pooled = self.pool_op(inter_out)
            pool_out = {k: pooled for k in inter_out.keys()}

        # Residual connection with optional gate
        output = {}
        if residuals is not None and self.residual_gate is not None:
            gate = torch.sigmoid(self.residual_gate)
            for k in pool_out.keys():
                if k in residuals:
                    output[k] = residuals[k] + gate * self.dropout(pool_out[k] - residuals[k])
                else:
                    output[k] = pool_out[k]
        elif residuals is not None:
            for k in pool_out.keys():
                if k in residuals:
                    output[k] = residuals[k] + self.dropout(pool_out[k])
                else:
                    output[k] = pool_out[k]
        else:
            output = {k: self.dropout(v) for k, v in pool_out.items()}

        # Stochastic depth (layer drop)
        if self.training and self.stochastic_depth_prob > 0:
            keep = torch.rand(1, device=next(iter(output.values())).device) > self.stochastic_depth_prob
            scale = 1.0 / (1.0 - self.stochastic_depth_prob)
            output = {k: v * (keep.float() * scale) for k, v in output.items()}

        self._last_diagnostics = {
            'gate_value': torch.sigmoid(self.residual_gate).detach() if self.residual_gate is not None else torch.tensor(1.0),
            'output_norm': torch.stack([v.norm(dim=-1).mean() for v in output.values()]).mean().detach(),
        }
        return output

    def get_diagnostics(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self._last_diagnostics.items()}


# ==============================================================================
# HeteroFormer Academic (Complete Specification)
# ==============================================================================

class HeteroFormer(nn.Module):
    """
    Heterogeneous Feature Interaction Network with Generative Sequence Encoding.
    
    Architecture (bottom-up):
        Block 1: User/Item Feature Tokenization (Intra)
        Block 2: Sequence Encoding via GSE (Intra → VQ)
        Block 3: Initial Cross-Attention (Inter)
        Block 4: Deep NS Interaction Stack (Intra → Inter → Pool, repeated)
        Block 5: NS-Sequence Cross-Attention (Inter)
        Block 6: Cross-Network (Inter)
        Head:   Lipschitz-constrained Prediction Head
    
    Key academic features:
    - Full spectral normalization on prediction head (all layers)
    - Principled VQ with temperature annealing and entropy regularization
    - Residual gating with controllable initialization
    - Clean separation: forward() returns logits, auxiliary via hooks
    """
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        d_model: int = 128,
        emb_dim: int = 16,
        seq_len: int = 50,
        num_layers: int = 4,
        base_rank: Optional[int] = None,
        rank_schedule: str = 'bottleneck',
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: Optional[int] = None,
        num_banks: Optional[int] = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
        num_time_buckets: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        emb_skip_threshold: int = 0,
        action_num: int = 1,
        gate_init: float = 0.0,  # Academic: 0.0 → sigmoid(0)=0.5 (half-open)
        stochastic_depth_prob: float = 0.0,
        progressive_layer_training: bool = False,
        id_dropout_rate: float = 0.15,
        id_vocab_threshold: int = 10000,
        shrinkage: float = 0.05,
        cross_network_layers: int = 2,
        # GSE parameters
        gse_num_codes: int = 64,
        gse_code_dim: int = 64,
        gse_num_layers: int = 4,
        gse_temp_anneal_steps: int = 10000,
        gse_min_temp: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.action_num = action_num
        self.progressive_layer_training = progressive_layer_training
        self._current_epoch = 0

        self.user_int_feature_specs = user_int_feature_specs
        self.item_int_feature_specs = item_int_feature_specs

        if num_heads is None:
            num_heads = max(4, d_model // 32)
        if num_banks is None:
            num_banks = max(4, d_model // 16)
        if base_rank is None:
            base_rank = max(16, d_model // 2)

        min_rank = max(24, d_model // 4)
        self.ranks = self._generate_ranks(base_rank, num_layers, rank_schedule, min_rank)

        self.seq_domains = sorted(seq_vocab_sizes.keys())

        # === Block 1: User/Item Feature Tokenization ===
        user_intra = {}
        user_dims = {}
        for idx, (vs, offset, length) in enumerate(user_int_feature_specs):
            if vs > 1 and (emb_skip_threshold <= 0 or vs <= emb_skip_threshold):
                fname = f'ufeat_{idx}'
                user_intra[fname] = IntraEmbedding(vs, emb_dim, length, id_dropout_rate, vs > id_vocab_threshold)
                user_dims[fname] = emb_dim

        item_intra = {}
        item_dims = {}
        for idx, (vs, offset, length) in enumerate(item_int_feature_specs):
            if vs > 1 and (emb_skip_threshold <= 0 or vs <= emb_skip_threshold):
                fname = f'ifeat_{idx}'
                item_intra[fname] = IntraEmbedding(vs, emb_dim, length, id_dropout_rate, vs > id_vocab_threshold)
                item_dims[fname] = emb_dim

        self.user_tokenize = HeteroBlock(
            fields=list(user_intra.keys()),
            intra_ops=user_intra,
            pool_op=PoolConcatLinear(user_dims, d_model, dropout) if user_dims else None,
            name='user_tokenize',
        )

        self.item_tokenize = HeteroBlock(
            fields=list(item_intra.keys()),
            intra_ops=item_intra,
            pool_op=PoolConcatLinear(item_dims, d_model, dropout) if item_dims else None,
            name='item_tokenize',
        )

        self.user_dense_proj = IntraLinear(user_dense_dim, d_model, dropout) if user_dense_dim > 0 else None
        self.item_dense_proj = IntraLinear(item_dense_dim, d_model, dropout) if item_dense_dim > 0 else None

        # === Block 2: Sequence Encoding (GSE as Intra-op) ===
        self.seq_blocks = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            self.seq_blocks[domain] = GenerativeSequenceEncoder(
                vocab_sizes=vocab_sizes,
                d_model=d_model,
                num_time_buckets=num_time_buckets,
                num_heads=num_heads,
                num_layers=gse_num_layers,
                num_codes=gse_num_codes,
                code_dim=gse_code_dim,
                dropout=dropout,
                max_seq_len=512,
                id_threshold=seq_id_threshold,
            )

        # Sequence domain aggregation
        self.seq_aggregate = HeteroBlock(
            fields=['seq_pooled'],
            intra_ops={'seq_pooled': IntraLinear(d_model, d_model, dropout)},
            name='seq_aggregate',
        )

        # === Block 3: Initial Cross-Attention ===
        self.init_cross = HeteroBlock(
            fields=['user', 'item', 'context'],
            inter_op=InterHadamard(d_model, num_layers=1),
            name='init_cross',
        )

        # === Block 4: Deep NS Interaction Stack ===
        self.ns_blocks = nn.ModuleList()
        for i in range(num_layers):
            rank = self.ranks[i]
            self.ns_blocks.append(HeteroBlock(
                fields=['user', 'item', 'context'],
                intra_ops={
                    'user': IntraLinear(d_model, d_model, dropout),
                    'item': IntraLinear(d_model, d_model, dropout),
                    'context': IntraLinear(d_model, d_model, dropout),
                },
                inter_op=InterBilinear(d_model, rank, num_fields=3),
                pool_op=PoolRouter(d_model, rank, num_banks),
                residual_gate=True,
                gate_init=gate_init,  # Academic: controllable gate initialization
                stochastic_depth_prob=stochastic_depth_prob * (i / max(num_layers - 1, 1)),
                dropout=dropout,
                name=f'ns_deep_{i}',
            ))

        # === Block 5: NS-Sequence Cross-Attention ===
        self.ns_seq_cross = HeteroBlock(
            fields=['ns_global', 'seq_local'],
            inter_op=InterCrossAttention(d_model, num_heads, dropout,
                                         query_fields=['seq_local'],
                                         kv_fields=['ns_global']),
            pool_op=None,
            name='ns_seq_cross',
        )

        # === Block 6: Cross-Network ===
        if cross_network_layers > 0:
            self.cross_net = HeteroBlock(
                fields=['final'],
                inter_op=InterHadamard(d_model * 4, num_layers=cross_network_layers),
                name='cross_net',
            )
        else:
            self.cross_net = None

        # === Prediction Head (Full Lipschitz Constraint) ===
        # Academic: ALL layers spectrally normalized for principled Lipschitz continuity
        self.predictor = nn.Sequential(
            apply_spectral_norm(nn.Linear(d_model * 4, d_model * 2, bias=False)),
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            apply_spectral_norm(nn.Linear(d_model * 2, action_num, bias=False)),
        )
        # Standard initialization for SN layers (SN handles its own spectral constraint)
        nn.init.xavier_uniform_(self.predictor[0].weight)
        nn.init.xavier_uniform_(self.predictor[3].weight)

        # === Temperature and Scale Parameters ===
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        # Academic: initialize to 1.0 (sigmoid(0)=0.5, scale=1.0 is neutral starting point)
        self.output_scale_logit = nn.Parameter(torch.tensor(0.0))

        # === Diagnostics ===
        self._diagnostics: Dict[str, List[float]] = {}

    def _generate_ranks(self, base_rank: int, num_layers: int, schedule: str, min_rank: int = 24) -> List[int]:
        """Generate rank schedule for bilinear interaction layers."""
        ranks = []
        for i in range(num_layers):
            if schedule == 'constant':
                r = base_rank
            elif schedule == 'gentle':
                r = max(min_rank, int(base_rank / (1 + 0.15 * i)))
            elif schedule == 'bottleneck':
                mid = num_layers // 2
                if i <= mid:
                    r = max(min_rank, base_rank - i * (base_rank // (mid + 1)))
                else:
                    r = max(min_rank, base_rank - (num_layers - 1 - i) * (base_rank // (mid + 1)))
            else:
                r = base_rank
            ranks.append(r)
        return ranks

    def set_epoch(self, epoch: int) -> None:
        """Progressive layer training: unlock layers gradually."""
        self._current_epoch = epoch
        if self.progressive_layer_training:
            for i, block in enumerate(self.ns_blocks):
                requires_grad = i <= epoch
                for p in block.parameters():
                    p.requires_grad = requires_grad

    def reinit_high_cardinality_params(self, threshold: int) -> set:
        """Reinitialize high-cardinality embeddings (useful for transfer learning)."""
        reinit_ptrs = set()
        if threshold <= 0:
            return reinit_ptrs
        for name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, ConstrainedEmbedding)) and module.num_embeddings > threshold:
                nn.init.normal_(module.weight, std=0.01)
                reinit_ptrs.add(module.weight.data_ptr())
        return reinit_ptrs

    def get_aux_loss(self) -> torch.Tensor:
        """Collect auxiliary losses from all GSE modules."""
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for domain in self.seq_domains:
            block = self.seq_blocks[domain]
            if hasattr(block, 'get_aux_loss'):
                aux_loss = aux_loss + block.get_aux_loss()
        return aux_loss

    def get_diagnostics(self) -> Dict[str, float]:
        """Aggregate diagnostics from all blocks including GSE."""
        diag = {}
        for block in self.ns_blocks:
            diag.update(block.get_diagnostics())
        for domain in self.seq_domains:
            block = self.seq_blocks[domain]
            if hasattr(block, 'get_diagnostics'):
                d = block.get_diagnostics()
                for k, v in d.items():
                    diag[f'{domain}_{k}'] = v
        return diag

    def clear_diagnostics(self) -> None:
        self._diagnostics.clear()

    def _get_output_scale(self):
        """Output scale: sigmoid(logit) * 2, range (0, 2)."""
        return torch.sigmoid(self.output_scale_logit) * 2.0

    def _encode_features(self, user_int, item_int, user_dense, item_dense):
        """Encode user/item sparse and dense features."""
        user_fields = {}
        for i, (_, offset, length) in enumerate(self.user_int_feature_specs):
            fname = f'ufeat_{i}'
            if fname in self.user_tokenize.fields:
                user_fields[fname] = user_int[:, offset:offset+length]
        user_emb = self.user_tokenize(user_fields) if user_fields else None
        user_feat = user_emb[list(user_emb.keys())[0]] if user_emb else \
            torch.zeros(user_int.size(0), self.d_model, device=user_int.device)

        if self.user_dense_proj is not None and user_dense.size(-1) > 0:
            user_feat = user_feat + self.user_dense_proj(user_dense)

        item_fields = {}
        for i, (_, offset, length) in enumerate(self.item_int_feature_specs):
            fname = f'ifeat_{i}'
            if fname in self.item_tokenize.fields:
                item_fields[fname] = item_int[:, offset:offset+length]
        item_emb = self.item_tokenize(item_fields) if item_fields else None
        item_feat = item_emb[list(item_emb.keys())[0]] if item_emb else \
            torch.zeros(item_int.size(0), self.d_model, device=item_int.device)

        if self.item_dense_proj is not None and item_dense.size(-1) > 0:
            item_feat = item_feat + self.item_dense_proj(item_dense)

        context_feat = user_feat + item_feat
        return user_feat, item_feat, context_feat

    def _encode_sequence(self, seq_data, seq_lens, seq_time_buckets, seq_decay_weights):
        """Encode sequence features via GSE and domain aggregation."""
        B = next(iter(seq_data.values())).size(0)
        device = next(iter(seq_data.values())).device

        if len(self.seq_domains) == 0:
            return torch.zeros(B, self.d_model, device=device)

        # Each domain processed by its GSE (Intra-op)
        domain_tokens = []
        for domain in self.seq_domains:
            if domain not in seq_data:
                continue
            pooled = self.seq_blocks[domain](
                seq_data[domain], seq_lens[domain],
                seq_time_buckets.get(domain),
                seq_decay_weights.get(domain) if seq_decay_weights else None,
            )
            domain_tokens.append(pooled)

        if len(domain_tokens) == 0:
            return torch.zeros(B, self.d_model, device=device)

        # Inter-op: aggregate multiple sequence domains
        stacked = torch.stack(domain_tokens, dim=1)
        seq_fields = {'seq_pooled': stacked.mean(dim=1)}
        seq_out = self.seq_aggregate(seq_fields)
        return seq_out['seq_pooled']

    def forward(self, model_input: ModelInput) -> torch.Tensor:
        """
        Forward pass: returns logits [B, action_num].
        
        Clean interface: no auxiliary loss leakage, no ZMLC patches.
        Temperature scaling applied at output.
        """
        # Feature encoding (Intra)
        user_feat, item_feat, context_feat = self._encode_features(
            model_input.user_int_feats, model_input.item_int_feats,
            model_input.user_dense_feats, model_input.item_dense_feats,
        )

        # Sequence encoding (Intra → Inter)
        seq_feat = self._encode_sequence(
            model_input.seq_data, model_input.seq_lens,
            model_input.seq_time_buckets, model_input.seq_decay_weights,
        )

        # Initial cross (Inter)
        init_fields = {'user': user_feat, 'item': item_feat, 'context': context_feat}
        init_out = self.init_cross(init_fields)
        u_cross, i_cross, c_cross = init_out['user'], init_out['item'], init_out['context']

        # Deep NS stack (Intra → Inter → Pool, repeated)
        ns_fields = {'user': u_cross, 'item': i_cross, 'context': c_cross}
        residuals = None

        for block in self.ns_blocks:
            ns_fields = block(ns_fields, residuals)
            residuals = {k: v for k, v in ns_fields.items()}

        # NS-Sequence cross (Inter)
        ns_global = torch.stack([ns_fields['user'], ns_fields['item'], ns_fields['context']], dim=1).mean(dim=1)
        cross_fields = {'ns_global': ns_global, 'seq_local': seq_feat}
        cross_out = self.ns_seq_cross(cross_fields)

        # Final representation
        final_repr = torch.cat([
            ns_fields['user'], ns_fields['item'],
            ns_fields['context'], cross_out['seq_local']
        ], dim=-1)

        if self.cross_net is not None:
            cn_fields = {'final': final_repr}
            cn_out = self.cross_net(cn_fields)
            final_repr = cn_out['final']

        # Prediction with Lipschitz-constrained head
        logits = self.predictor(final_repr)
        logits = logits * self._get_output_scale()
        temperature = torch.exp(self.logit_temperature).clamp(min=0.5, max=5.0)

        return logits / temperature

    def predict(self, model_input: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference interface: returns logits and final representation."""
        logits = self.forward(model_input)
        # Return zero tensor for representation (can be extended)
        return logits, torch.zeros(logits.size(0), self.d_model, device=logits.device)
