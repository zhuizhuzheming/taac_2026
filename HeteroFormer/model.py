"""
PCVRHeteroFormer v7.3 - Collaborative Multi-Objective Edition
torch.compile optimized: minimal changes, original class structure preserved
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, NamedTuple


# ==============================================================================
# Spectral Normalization (preserved)
# ==============================================================================

def apply_spectral_norm(module, n_power_iterations=1):
    """Apply spectral normalization to a linear layer."""
    return nn.utils.parametrizations.spectral_norm(module, n_power_iterations=n_power_iterations)

# ==============================================================================
# Model Input (preserved)
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
# Base Primitives (preserved)
# ==============================================================================

class SwiGLU(nn.Module):
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
# Intra-Field Operations (preserved + GSE)
# ==============================================================================

class ConstrainedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.max_norm = max_norm
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_normed = F.normalize(self.weight, p=2, dim=-1) * self.max_norm
        return F.embedding(input, weight_normed, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)


class IntraEmbedding(nn.Module):
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
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# GenerativeSequenceEncoder (GSE, preserved with compile-friendly forward)
# ==============================================================================

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        temperature: float = 1.0,
        temp_anneal_steps: int = 500,
        min_temp: float = 0.2,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.init_temp = temperature
        self.min_temp = min_temp
        self.temp_anneal_steps = temp_anneal_steps

        codebook = torch.empty(num_codes + 1, code_dim)
        nn.init.uniform_(codebook, -1.0 / num_codes, 1.0 / num_codes)
        codebook[0] = 0.0
        self.codebook = nn.Parameter(codebook)

        self.register_buffer('_step_count', torch.tensor(0, dtype=torch.long))

    def get_temperature(self) -> torch.Tensor:
        progress = torch.clamp(self._step_count.float() / self.temp_anneal_steps, 0.0, 1.0)
        return self.min_temp + (self.init_temp - self.min_temp) * torch.exp(-5.0 * progress)

    def forward(
        self, z: torch.Tensor, seq_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns tuple of tensors instead of dict to avoid graph break.
        (quantized_st, commitment_loss, codebook_loss, entropy_loss, null_reg, temperature, usage_rate)
        """
        B, L, D = z.shape
        device = z.device

        z_safe = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        z_flat = z_safe.reshape(-1, D)
        distances = torch.cdist(z_flat, self.codebook, p=2).pow(2).view(B, L, self.num_codes + 1)
        distances = torch.clamp(distances, min=0.0, max=1e8)

        # Build null mask without data-dependent control flow
        len_mask = torch.arange(L, device=device).unsqueeze(0) >= seq_lens.unsqueeze(1)
        zero_len_mask = (seq_lens == 0).unsqueeze(1)
        null_mask = len_mask | zero_len_mask

        temp = self.get_temperature()

        logits = -distances
        logits = logits.masked_fill(null_mask.unsqueeze(-1), -1e4)
        non_null_mask = ~null_mask
        logits[:, :, 0] = logits[:, :, 0].masked_fill(non_null_mask, -1e4)

        noise = torch.rand_like(logits).clamp(min=1e-10)
        noise = -torch.log(-torch.log(noise) + 1e-10)
        gumbel_logits = (logits + noise) / temp.clamp(min=1e-5)
        gumbel_logits = torch.clamp(gumbel_logits, min=-50, max=50)
        soft_probs = F.softmax(gumbel_logits, dim=-1)
        soft_quantized = torch.einsum('blk,kd->bld', soft_probs, self.codebook)

        hard_indices = logits.argmax(dim=-1)
        hard_quantized = self.codebook[hard_indices]
        hard_probs = F.one_hot(hard_indices, self.num_codes + 1).float()

        soft_weight = (temp > 0.2).float()
        quantized = soft_weight * soft_quantized + (1 - soft_weight) * hard_quantized
        probs = soft_weight * soft_probs + (1 - soft_weight) * hard_probs

        commitment_loss = F.mse_loss(z_safe, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z_safe.detach())

        probs_active = probs[:, :, 1:]
        log_probs = torch.log(probs_active + 1e-10)
        entropy_per_pos = -(probs_active * log_probs).sum(dim=-1)

        valid_count = non_null_mask.sum()
        safe_count = torch.clamp(valid_count, min=1)
        entropy_loss = torch.where(
            valid_count > 0,
            (entropy_per_pos * non_null_mask).sum() / safe_count,
            torch.tensor(0.0, device=device)
        )

        null_reg = self.codebook[0].pow(2).mean()

        quantized_st = z_safe + (quantized - z_safe).detach()
        quantized_st = torch.nan_to_num(quantized_st, nan=0.0)

        usage_rate = (probs.argmax(dim=-1) != 0).float().mean()

        self._step_count += 1
        return quantized_st, commitment_loss, codebook_loss, entropy_loss, null_reg, temp, usage_rate


class GenerativeSequenceEncoder(nn.Module):
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
        seq_id_dropout_rate: float = 0.10,
        init_const_bias: float = -0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.id_threshold = id_threshold
        self.seq_id_dropout_rate = seq_id_dropout_rate
        
        n_feats = len(vocab_sizes)
        feat_dim = max(d_model // n_feats, 1)
        
        self.feat_embs = nn.ModuleList()
        self.feat_dropouts = nn.ModuleList()
        for vs in vocab_sizes:
            self.feat_embs.append(nn.Embedding(max(vs, 2), feat_dim))
            extra_drop = dropout * 2 if vs > id_threshold else 0.0
            self.feat_dropouts.append(nn.Dropout(min(dropout + extra_drop, 0.5)))
            nn.init.normal_(self.feat_embs[-1].weight, std=0.01)
        
        self.seq_proj = nn.Linear(feat_dim * n_feats, d_model, bias=False)
        self.seq_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.seq_proj.weight, gain=0.1)
        
        if num_time_buckets > 0:
            self.time_emb = nn.Embedding(num_time_buckets, d_model)
            nn.init.normal_(self.time_emb.weight, std=0.01)
        else:
            self.time_emb = None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.vq = VectorQuantizer(num_codes, code_dim)
        self.to_code = nn.Linear(d_model, code_dim, bias=False)
        self.from_code = nn.Linear(code_dim, d_model, bias=False)
        nn.init.xavier_uniform_(self.to_code.weight, gain=0.1)
        nn.init.xavier_uniform_(self.from_code.weight, gain=0.1)
        
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.aggregate = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.agg_norm = nn.LayerNorm(d_model)
        
        self.empty_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.empty_state = nn.Parameter(torch.randn(d_model) * 0.1)
        self.null_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1, bias=True),
        )
        nn.init.normal_(self.null_gate[-1].weight, std=0.01)
        nn.init.constant_(self.null_gate[-1].bias, init_const_bias)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight, gain=0.1)
        
        # Buffers for diagnostics (accessed outside forward)
        self.register_buffer('_last_aux_loss', torch.tensor(0.0))
        self.register_buffer('_last_info_temp', torch.tensor(0.0))
        self.register_buffer('_last_info_usage', torch.tensor(0.0))
        self.register_buffer('_last_info_null_mean', torch.tensor(0.0))
        self.register_buffer('_last_info_null_std', torch.tensor(0.0))
        self.register_buffer('_last_info_empty_ratio', torch.tensor(0.0))

    def forward(
        self,
        seq_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        time_buckets: Optional[torch.Tensor] = None,
        decay_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, n_feats, max_len = seq_ids.shape
        device = seq_ids.device
        
        # Training-aware dropout using torch.where
        if self.seq_id_dropout_rate > 0:
            valid_mask = seq_ids > 0
            drop_mask = torch.rand_like(seq_ids.float()) > self.seq_id_dropout_rate
            dropped = seq_ids * (valid_mask & drop_mask).long()
            # Use training flag as tensor for torch.where
            training_flag = torch.tensor(self.training, device=device)
            seq_ids = torch.where(training_flag.bool(), dropped, seq_ids)
        
        feat_embs = []
        for i in range(n_feats):
            emb = self.feat_embs[i](seq_ids[:, i, :])
            emb = self.feat_dropouts[i](emb)
            feat_embs.append(emb)
        
        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))
        
        # Empty sequence handling: pure torch.where
        empty_mask = (seq_lens == 0)
        empty_mask_3d = empty_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, max_len, self.d_model)
        empty_token_broadcast = self.empty_token.expand(B, max_len, -1)
        pos0_mask = torch.zeros(B, max_len, 1, device=device, dtype=torch.bool)
        pos0_mask[:, 0, :] = True
        replace_mask = empty_mask_3d & pos0_mask.expand(-1, -1, self.d_model)
        seq_repr = torch.where(replace_mask, empty_token_broadcast, seq_repr)
        
        effective_lens = torch.where(empty_mask, torch.ones_like(seq_lens), seq_lens)
        
        if self.time_emb is not None and time_buckets is not None:
            t_emb = self.time_emb(time_buckets)
            seq_repr = seq_repr + t_emb
        
        padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= effective_lens.unsqueeze(1)
        
        encoded = self.transformer(seq_repr, src_key_padding_mask=padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1e4, neginf=-1e4)
        
        code_input = self.to_code(encoded)
        # VQ returns tuple
        quantized, commitment_loss, codebook_loss, entropy_loss, null_reg, temp, usage_rate = self.vq(code_input, effective_lens)
        quantized = self.from_code(quantized) + encoded
        
        kv = quantized.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        aggregated, _ = self.aggregate(self.agg_token.expand(B, -1, -1), kv, kv, key_padding_mask=padding_mask)
        aggregated = aggregated.squeeze(1)
        aggregated = torch.nan_to_num(aggregated, nan=0.0, posinf=1e4, neginf=-1e4)
        aggregated = self.agg_norm(aggregated)

        null_logits = self.null_gate(aggregated)
        null_gate = torch.sigmoid(null_logits)
        empty_state_broadcast = self.empty_state.unsqueeze(0).expand(B, -1)
        output = null_gate * empty_state_broadcast + (1 - null_gate) * aggregated
        output = self.output_proj(output)

        # Store diagnostics in buffers (no Python side effects in forward)
        aux_loss = (
            0.1 * commitment_loss +
            0.1 * codebook_loss +
            0.05 * entropy_loss +
            0.01 * null_reg
        )
        self._last_aux_loss = aux_loss.detach()
        self._last_info_temp = temp.detach()
        self._last_info_usage = usage_rate.detach()
        self._last_info_null_mean = null_gate.mean().detach()
        self._last_info_null_std = null_gate.std().detach()
        self._last_info_empty_ratio = empty_mask.float().mean().detach()
        
        return output
    
    def get_aux_loss(self) -> torch.Tensor:
        if self.training:
            return self._last_aux_loss
        return torch.tensor(0.0, device=self._last_aux_loss.device)
    
    def get_diagnostics(self) -> Dict[str, float]:
        return {
            'temperature': self._last_info_temp.item(),
            'usage_rate': self._last_info_usage.item(),
            'null_gate_mean': self._last_info_null_mean.item(),
            'null_gate_std': self._last_info_null_std.item(),
            'empty_ratio': self._last_info_empty_ratio.item(),
        }


# ==============================================================================
# Inter-Field & Pool Operations (preserved, dict interface for init only)
# ==============================================================================

class InterSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = sorted(fields.keys())
        x = torch.stack([fields[k] for k in keys], dim=1)
        out, _ = self.attn(x, x, x)
        out = self.norm(x + out)
        return {k: out[:, i] for i, k in enumerate(keys)}


class InterCrossAttention(nn.Module):
    """
    Cross-attention with compile-friendly fixed-field mode.
    Falls back to dynamic mode only if fields not specified at init.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 query_fields: Optional[List[str]] = None,
                 kv_fields: Optional[List[str]] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_kv = nn.LayerNorm(d_model, eps=1e-5)
        
        # 冻结为 tuple，视为编译期常量
        self._q_fields = tuple(query_fields) if query_fields is not None else None
        self._kv_fields = tuple(kv_fields) if kv_fields is not None else None
        
        # 标记模式：固定字段 vs 动态字段
        self._fixed_mode = self._q_fields is not None and self._kv_fields is not None

    def _stack_fixed(self, fields: Dict[str, torch.Tensor], 
                    keys_values: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]:
        """固定字段模式：无动态分支，Dynamo 可完全 trace"""
        q = torch.stack([fields[f] for f in self._q_fields], dim=1)
        if keys_values is None:
            kv = torch.stack([fields[f] for f in self._kv_fields], dim=1)
        else:
            kv = torch.stack([keys_values[f] for f in self._kv_fields], dim=1)
        return q, kv, self._q_fields

    def _stack_dynamic(self, fields: Dict[str, torch.Tensor],
                       keys_values: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """动态字段模式：保留原始行为，但会 graph break"""
        if keys_values is None:
            q_keys = sorted(fields.keys())
            kv_keys = q_keys
        else:
            q_keys = sorted(fields.keys())
            kv_keys = sorted(keys_values.keys())
        q = torch.stack([fields[k] for k in q_keys], dim=1)
        kv = torch.stack([keys_values[k] for k in kv_keys], dim=1) if keys_values is not None else q
        return q, kv, q_keys

    def forward(self, fields: Dict[str, torch.Tensor],
                keys_values: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        # 用 torch.where 无法选择不同代码路径，所以用 Python 条件
        # 但 _fixed_mode 是 bool 常量（init 后不变），Dynamo 会将其视为静态 guard
        if self._fixed_mode:
            q, kv, q_keys = self._stack_fixed(fields, keys_values)
        else:
            q, kv, q_keys = self._stack_dynamic(fields, keys_values)

        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), kv)
        out = self.norm_q(q) + out
        
        # 返回：固定模式用 tuple 迭代（常量），动态模式用 list 迭代
        return {k: out[:, i] for i, k in enumerate(q_keys)}


class InterBilinear(nn.Module):
    def __init__(self, d_model: int, rank: int, num_fields: int = 3):
        super().__init__()
        self.rank = rank
        self.projs = nn.ModuleList([nn.Linear(d_model, rank, bias=False) for _ in range(num_fields)])
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
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


class PoolRouter(nn.Module):
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
    def __init__(self, field_dims: Dict[str, int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        total_in = sum(field_dims.values())
        self.proj = nn.Linear(total_in, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(list(fields.values()), dim=-1)
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# HeteroBlock (preserved, dict interface for init, but forward uses internal tensor path)
# ==============================================================================

class HeteroBlock(nn.Module):
    def __init__(
        self,
        fields: List[str],
        intra_ops: Optional[Dict[str, nn.Module]] = None,
        inter_op: Optional[nn.Module] = None,
        pool_op: Optional[nn.Module] = None,
        residual_gate: bool = False,
        stochastic_depth_prob: float = 0.0,
        dropout: float = 0.1,
        name: str = "block",
    ):
        super().__init__()
        self.name = name
        self.fields = fields
        self.intra_ops = nn.ModuleDict(intra_ops or {})
        self.inter_op = inter_op
        self.pool_op = pool_op
        if residual_gate:
            self.residual_gate = nn.Parameter(torch.tensor(-1.0))
        else:
            self.register_parameter('residual_gate', None)
        self.stochastic_depth_prob = stochastic_depth_prob
        self.dropout = nn.Dropout(dropout)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def _fields_to_tensor(self, fields: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, field_dim] dict -> [B, num_fields, d_model] tensor"""
        return torch.stack([fields[k] for k in self.fields], dim=1)

    def _tensor_to_fields(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """[B, num_fields, d_model] tensor -> [B, field_dim] dict"""
        return {self.fields[i]: x[:, i] for i in range(len(self.fields))}

    def forward(self, fields: Dict[str, torch.Tensor],
                residuals: Optional[Dict[str, torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        # 1. 先应用 intra_ops，统一所有字段维度
        if self.intra_ops:
            processed = {}
            for name in self.fields:
                if name in self.intra_ops:
                    processed[name] = self.intra_ops[name](fields[name])
                else:
                    processed[name] = fields[name]
        else:
            processed = fields
    
        # 2. 转换为 [B, N, D]
        x = self._fields_to_tensor(processed)
    
        # 3. Inter-field 操作（可能只返回部分字段）
        if self.inter_op is not None:
            fields_dict = self._tensor_to_fields(x)
            inter_dict = self.inter_op(fields_dict)
            # 补全缺失字段，保证所有 self.fields 都存在
            for name in self.fields:
                if name not in inter_dict:
                    inter_dict[name] = fields_dict[name]
            x = self._fields_to_tensor(inter_dict)
    
        # 4. Pool 操作
        if self.pool_op is not None:
            fields_dict = self._tensor_to_fields(x)
            pooled = self.pool_op(fields_dict)
            x = pooled.unsqueeze(1).expand(-1, len(self.fields), -1)
    
        # 5. 残差连接（保持原始逻辑）
        if residuals is not None:
            res = self._fields_to_tensor(residuals)
            if self.residual_gate is not None:
                gate = torch.sigmoid(self.residual_gate)
                x = res + gate * self.dropout(x - res)
            else:
                x = res + self.dropout(x - res)
        else:
            x = self.dropout(x)
    
        # 6. 随机深度
        if self.stochastic_depth_prob > 0:
            training_flag = torch.tensor(self.training, device=x.device)
            keep = torch.rand(1, device=x.device) > self.stochastic_depth_prob
            scale = 1.0 / (1.0 - self.stochastic_depth_prob)
            sd_scale = torch.where(
                training_flag & keep,
                torch.tensor(scale, device=x.device),
                torch.tensor(1.0, device=x.device)
            )
            x = x * sd_scale
    
        # 7. 存储诊断
        self._last_diagnostics = {
            'gate_value': torch.sigmoid(self.residual_gate).detach()
                          if self.residual_gate is not None else torch.tensor(1.0),
            'output_norm': x.norm(dim=-1).mean().detach(),
        }
        return self._tensor_to_fields(x)

    def get_diagnostics(self) -> Dict[str, float]:
        return {k: v.item() for k, v in self._last_diagnostics.items()}


# ==============================================================================
# PCVRHeteroFormer v7.3 - Collaborative Multi-Objective (compile-friendly)
# ==============================================================================

class PCVRHeteroFormer(nn.Module):
    """
    v7.3: Three-stream architecture for collaborative optimization
    - Main stream: Focal + Prior + ECE (calibrated probability)
    - Rank stream: LambdaRank (independent head, shared early features)
    - Calib stream: Residual calibration with ZMLC constraint
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
        gate_anneal_steps: int = 2000,
        stochastic_depth_prob: float = 0.0,
        progressive_layer_training: bool = False,
        id_dropout_rate: float = 0.15,
        seq_id_dropout_rate: float = 0.10,
        id_vocab_threshold: int = 10000,
        shrinkage: float = 0.05,
        cross_network_layers: int = 2,
        use_zmlc: bool = True,
        zmlc_lambda: float = 0.05,
        gse_num_codes: int = 64,
        gse_code_dim: int = 64,
        gse_num_layers: int = 4,
        rank_head_hidden_dim: Optional[int] = None,
        calib_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.action_num = action_num
        self.progressive_layer_training = progressive_layer_training
        self._current_epoch = 0
        self.use_zmlc = use_zmlc
        self.calib_residual_scale = calib_residual_scale

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
        logging.info(f"Rank schedule: {self.ranks} (min_rank={min_rank})")

        self.seq_domains = sorted(seq_vocab_sizes.keys())

        # === Block 1: User/Item Feature Tokenization (preserved) ===
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

        # === Block 2: Sequence Encoding (GSE) ===
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
                seq_id_dropout_rate=seq_id_dropout_rate,
            )

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

        # === THREE-STREAM ARCHITECTURE ===
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2, bias=False),
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            apply_spectral_norm(nn.Linear(d_model * 2, action_num, bias=False)),
        )
        nn.init.xavier_uniform_(self.predictor[0].weight, gain=0.5)

        rank_hidden = rank_head_hidden_dim or d_model
        self.rank_feature_proj = nn.Sequential(
            nn.Linear(d_model * 4, rank_hidden, bias=False),
            nn.LayerNorm(rank_hidden),
            nn.SiLU(),
        )
        self.rank_predictor = nn.Linear(rank_hidden, action_num, bias=False)
        nn.init.xavier_uniform_(self.rank_feature_proj[0].weight, gain=0.05)
        nn.init.zeros_(self.rank_predictor.weight)

        self.calibrator = nn.Sequential(
            nn.Linear(action_num, max(action_num * 4, d_model // 4), bias=False),
            nn.SiLU(),
            nn.Linear(max(action_num * 4, d_model // 4), action_num, bias=False)
        )
        nn.init.normal_(self.calibrator[0].weight, std=0.01)
        nn.init.zeros_(self.calibrator[-1].weight)

        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.output_scale_logit = nn.Parameter(torch.tensor(0.0))

        self._sparse_params: List[nn.Parameter] = []
        self._dense_params: List[nn.Parameter] = []
        self._gate_params: List[nn.Parameter] = []
        self._scale_params: List[nn.Parameter] = []
        self._rank_params: List[nn.Parameter] = []
        self._calib_params: List[nn.Parameter] = []
        self._register_params()

        self._diagnostics: Dict[str, List[float]] = {}
        self._alignment_penalty = torch.tensor(0.0)

    def _generate_ranks(self, base_rank: int, num_layers: int, schedule: str, min_rank: int = 24) -> List[int]:
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

    def _register_params(self):
        for name, param in self.named_parameters():
            if 'output_scale' in name:
                self._scale_params.append(param)
            elif any(k in name for k in ('residual_gate', 'high_order_scale', 'attn_temperature')):
                self._gate_params.append(param)
            elif 'rank_' in name or 'rank_predictor' in name:
                self._rank_params.append(param)
            elif 'calibrator' in name:
                self._calib_params.append(param)
            elif 'embedding' in name or 'emb' in name.lower():
                self._sparse_params.append(param)
            else:
                self._dense_params.append(param)

    def get_sparse_params(self) -> List[nn.Parameter]:
        return self._sparse_params

    def get_dense_params(self) -> List[nn.Parameter]:
        return self._dense_params

    def get_gate_params(self) -> List[nn.Parameter]:
        return self._gate_params

    def get_scale_params(self) -> List[nn.Parameter]:
        return self._scale_params

    def get_rank_params(self) -> List[nn.Parameter]:
        return self._rank_params

    def get_calib_params(self) -> List[nn.Parameter]:
        return self._calib_params

    def get_aux_loss(self) -> torch.Tensor:
        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for domain in self.seq_domains:
            block = self.seq_blocks[domain]
            if hasattr(block, 'get_aux_loss'):
                aux_loss = aux_loss + block.get_aux_loss()
        return aux_loss

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
        if self.progressive_layer_training:
            for i, block in enumerate(self.ns_blocks):
                requires_grad = i <= epoch
                for p in block.parameters():
                    p.requires_grad = requires_grad
            logging.info(f"Progressive training: epoch {epoch}, unlocked layers 0-{epoch}")

    def reinit_high_cardinality_params(self, threshold: int) -> set:
        reinit_ptrs = set()
        if threshold <= 0:
            return reinit_ptrs
        for name, module in self.named_modules():
            if isinstance(module, (nn.Embedding, ConstrainedEmbedding)) and module.num_embeddings > threshold:
                nn.init.normal_(module.weight, std=0.01)
                reinit_ptrs.add(module.weight.data_ptr())
        return reinit_ptrs

    def get_diagnostics(self) -> Dict[str, float]:
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
        return torch.sigmoid(self.output_scale_logit) * 2.0

    def _encode_features(self, user_int, item_int, user_dense, item_dense):
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
        B = next(iter(seq_data.values())).size(0)
        device = next(iter(seq_data.values())).device

        if len(self.seq_domains) == 0:
            return torch.zeros(B, self.d_model, device=device)

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

        stacked = torch.stack(domain_tokens, dim=1)
        seq_fields = {'seq_pooled': stacked.mean(dim=1)}
        seq_out = self.seq_aggregate(seq_fields)
        return seq_out['seq_pooled']

    def forward(self, model_input: ModelInput, return_components: bool = False):
        """
        Dynamo  sees single return path, no graph break.
        """
        # Feature encoding
        user_feat, item_feat, context_feat = self._encode_features(
            model_input.user_int_feats, model_input.item_int_feats,
            model_input.user_dense_feats, model_input.item_dense_feats,
        )

        # Sequence encoding
        seq_feat = self._encode_sequence(
            model_input.seq_data, model_input.seq_lens,
            model_input.seq_time_buckets, model_input.seq_decay_weights,
        )

        # Initial cross
        init_fields = {'user': user_feat, 'item': item_feat, 'context': context_feat}
        init_out = self.init_cross(init_fields)
        u_cross, i_cross, c_cross = init_out['user'], init_out['item'], init_out['context']

        # Deep NS stack
        ns_fields = {'user': u_cross, 'item': i_cross, 'context': c_cross}
        residuals = None

        for block in self.ns_blocks:
            ns_fields = block(ns_fields, residuals)
            residuals = {k: v for k, v in ns_fields.items()}

        # NS-Sequence cross
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

        # === THREE-STREAM OUTPUT (always computed) ===
        logits_main = self.predictor(final_repr)
        rank_feat = self.rank_feature_proj(final_repr)
        logits_rank = self.rank_predictor(rank_feat)
        calib_residual = self.calibrator(logits_main)
        
        logits = logits_main + self.calib_residual_scale * calib_residual
        
        logits = logits * self._get_output_scale()
        temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
        logits = logits / temperature

        # Always return 4-tuple, regardless of return_components flag.
        # This eliminates the if-branch graph break while maintaining API compatibility.
        return logits, logits_main, logits_rank, calib_residual

    def predict(self, model_input: ModelInput):
        logits, _, _, _ = self.forward(model_input)
        return logits, torch.zeros(logits.size(0), self.d_model, device=logits.device)