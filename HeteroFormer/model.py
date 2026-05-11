"""
PCVRHeteroFormer v8.1-patch2 - Zero-Hessian Symplectic Prototype Learning
==========================================================================
修复 (v8.1-patch1 → v8.1-patch2):
1. vMF Prototype: 给 kappa 添加独立正则损失，绕过 Sinkhorn 梯度衰减
   - kappa_var: 鼓励不同原型有不同集中度
   - entropy_mismatch: 对齐 kappa 与分配锐度
2. forward 返回 kappa_loss，供 trainer 直接优化

Author: v8.1-patch2
Date: 2026-05-09
"""

import math
import logging
from typing import Tuple, Optional, List, Dict, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ==============================================================================
# Model Input (preserved interface)
# ==============================================================================

class ModelInput(NamedTuple):
    user_int_feats: Tensor
    item_int_feats: Tensor
    user_dense_feats: Tensor
    item_dense_feats: Tensor
    seq_data: Dict[str, Tensor]
    seq_lens: Dict[str, Tensor]
    seq_time_buckets: Dict[str, Tensor]
    seq_decay_weights: Optional[Dict[str, Tensor]] = None


# ==============================================================================
# Base Primitives
# ==============================================================================

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.fc1(x)
        x_part, gate = x_proj.chunk(2, dim=-1)
        return self.fc2(x_part * F.silu(gate))


# ==============================================================================
# Intra-Field Operations (preserved)
# ==============================================================================

class ConstrainedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.max_norm = max_norm
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input: Tensor) -> Tensor:
        weight_normed = F.normalize(self.weight, p=2, dim=-1) * self.max_norm
        return F.embedding(
            input, weight_normed, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


class IntraEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_slots: int = 1,
                 dropout: float = 0.0, is_high_card: bool = False):
        super().__init__()
        self.emb = ConstrainedEmbedding(max(vocab_size, 2), emb_dim)
        self.num_slots = num_slots
        self.dropout = nn.Dropout(dropout * (2.0 if is_high_card else 1.0))
        self.is_high_card = is_high_card

    def forward(self, x: Tensor) -> Tensor:
        if self.num_slots == 1:
            emb = self.emb(x.squeeze(-1))
        else:
            embs = torch.stack([self.emb(x[:, i]) for i in range(self.num_slots)], dim=1)
            emb = embs.mean(dim=1)
        return self.dropout(emb)


class IntraLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# Sequence Encoder (preserved, Linear Attention O(n))
# ==============================================================================

class LinearAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * feature_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L, D = x.shape
        H, fd = self.num_heads, self.feature_dim

        Q = self.q_proj(x).view(B, L, H, fd)
        K = self.k_proj(x).view(B, L, H, fd)
        V = self.v_proj(x).view(B, L, H, fd)

        phi_Q = F.elu(Q) + 1.0
        phi_K = F.elu(K) + 1.0

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
            phi_K = phi_K.masked_fill(mask_expanded.expand(B, H, L, fd).permute(0, 2, 1, 3), 0.0)
            V = V.masked_fill(mask_expanded.expand(B, H, L, fd).permute(0, 2, 1, 3), 0.0)

        KV_state = torch.einsum('blhf,blhg->bhfg', phi_K, V)
        Z_state = torch.einsum('blhf->bhf', phi_K)

        numerator = torch.einsum('blhf,bhfg->blhg', phi_Q, KV_state)
        denominator = torch.einsum('blhf,bhf->blh', phi_Q, Z_state).unsqueeze(-1) + 1e-8

        out = numerator / denominator
        out = out.reshape(B, L, -1)

        return self.dropout(self.o_proj(out))


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int = 128,
        num_time_buckets: int = 0,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        id_threshold: int = 10000,
        seq_id_dropout_rate: float = 0.10,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_id_dropout_rate = seq_id_dropout_rate
        self.max_seq_len = max_seq_len

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

        self.layers = nn.ModuleList([
            LinearAttentionLayer(d_model, num_heads, d_model // num_heads * 2, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.empty_state = nn.Parameter(torch.randn(d_model) * 0.1)

        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.aggregate = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.agg_norm = nn.LayerNorm(d_model)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        nn.init.xavier_uniform_(self.output_proj[0].weight, gain=0.1)

    def forward(
        self,
        seq_ids: Tensor,
        seq_lens: Tensor,
        time_buckets: Optional[Tensor] = None,
        decay_weight: Optional[Tensor] = None,
    ) -> Tensor:
        B, n_feats, max_len = seq_ids.shape
        device = seq_ids.device

        if self.training and self.seq_id_dropout_rate > 0:
            valid_mask = seq_ids > 0
            drop_mask = torch.rand_like(seq_ids.float()) > self.seq_id_dropout_rate
            dropped = seq_ids * (valid_mask & drop_mask).long()
            training_flag = torch.tensor(self.training, device=device)
            seq_ids = torch.where(training_flag, dropped, seq_ids)

        feat_embs = []
        for i in range(n_feats):
            emb = self.feat_embs[i](seq_ids[:, i, :])
            emb = self.feat_dropouts[i](emb)
            feat_embs.append(emb)

        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))

        empty_mask = (seq_lens == 0)
        empty_broadcast = self.empty_state.view(1, 1, -1).expand(B, max_len, -1)
        pos0_mask = torch.arange(max_len, device=device).unsqueeze(0) == 0
        replace_mask = empty_mask.unsqueeze(-1).unsqueeze(-1) & pos0_mask.unsqueeze(-1)
        seq_repr = torch.where(replace_mask.expand(-1, -1, self.d_model), empty_broadcast, seq_repr)

        effective_lens = torch.where(empty_mask, torch.ones_like(seq_lens), seq_lens)

        if self.time_emb is not None and time_buckets is not None:
            t_emb = self.time_emb(time_buckets)
            seq_repr = seq_repr + t_emb

        if decay_weight is not None:
            seq_repr = seq_repr * decay_weight.unsqueeze(-1)

        padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= effective_lens.unsqueeze(1)

        hidden = seq_repr
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = hidden
            hidden = layer(hidden, mask=padding_mask)
            hidden = norm(hidden + residual)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        hidden_masked = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        aggregated, _ = self.aggregate(
            self.agg_token.expand(B, -1, -1),
            hidden_masked,
            hidden_masked,
            key_padding_mask=padding_mask,
        )
        aggregated = aggregated.squeeze(1)
        aggregated = torch.nan_to_num(aggregated, nan=0.0, posinf=1e4, neginf=-1e4)
        aggregated = self.agg_norm(aggregated)

        aggregated = torch.where(
            empty_mask.unsqueeze(-1),
            self.empty_state.unsqueeze(0).expand(B, -1),
            aggregated
        )

        return self.output_proj(aggregated)


# ==============================================================================
# vMF Prototype Layer (patch2: kappa 独立正则)
# ==============================================================================

class VonMisesFisherPrototype(nn.Module):
    """
    vMF分布原型 v8.1-patch2: 方向与集中度解耦，kappa 独立正则。

    修正:
    - 方向 mu = F.normalize(eta, dim=-1)
    - 集中度 kappa = softplus(kappa_log) + 0.5，独立参数
    - kappa_loss: 绕过 Sinkhorn 的直接梯度源
        * kappa_var: 鼓励集中度多样性
        * entropy_mismatch: 对齐 kappa 与分配锐度
    """
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iter: int = 20,
        min_mass_ratio: float = 0.016,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.min_mass_ratio = min_mass_ratio

        self.eta = nn.Parameter(torch.randn(num_codes, code_dim))
        self.kappa_log = nn.Parameter(torch.ones(num_codes) * 0.1)
        self.empty_prior = nn.Parameter(torch.zeros(num_codes))

        self.match_proj = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.xavier_uniform_(self.match_proj.weight, gain=0.1)

    def get_prototypes(self) -> Tuple[Tensor, Tensor]:
        mu = F.normalize(self.eta, p=2, dim=-1)
        kappa = F.softplus(self.kappa_log) + 0.5
        return mu, kappa

    def sinkhorn(self, log_probs: Tensor) -> Tensor:
        B, K = log_probs.shape
        device = log_probs.device
        eps = self.sinkhorn_epsilon

        C = -log_probs
        K_exp = torch.exp(-C / eps)

        u = torch.zeros(B, device=device)
        v = torch.zeros(K, device=device)

        uniform_mass = B / K
        min_mass = uniform_mass * self.min_mass_ratio

        for _ in range(self.sinkhorn_iter):
            u = -torch.logsumexp(torch.log(K_exp + 1e-10) + v.unsqueeze(0), dim=1)
            v = -torch.logsumexp(torch.log(K_exp + 1e-10) + u.unsqueeze(1), dim=0)
            col_sums = torch.sum(torch.exp(u.unsqueeze(1)) * K_exp * torch.exp(v.unsqueeze(0)), dim=0)
            deficit = torch.relu(min_mass - col_sums)
            v = v + torch.log(1 + deficit / (col_sums + 1e-8))

        Pi = torch.exp(u.unsqueeze(1)) * K_exp * torch.exp(v.unsqueeze(0))
        Pi = Pi / (Pi.sum(dim=1, keepdim=True) + 1e-8)
        return Pi

    def forward(self, z_seq: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            proto_weights: [B, K]
            proto_repr: [B, D]
            kappa_mean: scalar
            kappa_loss: scalar (patch2 新增，独立正则)
        """
        B = z_seq.size(0)
        device = z_seq.device

        mu, kappa = self.get_prototypes()

        z_norm = F.normalize(z_seq, p=2, dim=-1)
        log_probs = kappa.unsqueeze(0) * (z_norm @ mu.T)

        proto_weights = self.sinkhorn(log_probs)

        empty_mask = (seq_lens == 0).unsqueeze(-1).float()
        empty_prior_weights = F.softmax(self.empty_prior, dim=-1).unsqueeze(0).expand(B, -1)
        proto_weights = empty_mask * empty_prior_weights + (1.0 - empty_mask) * proto_weights

        proto_repr_raw = torch.einsum('bk,kd->bd', proto_weights, mu)
        proto_repr = F.normalize(proto_repr_raw, p=2, dim=-1)

        # ===== patch2: kappa 独立正则（绕过 Sinkhorn 梯度衰减）=====
        # 1. 多样性：鼓励不同原型有不同集中度
        kappa_var = kappa.var()

        # 2. 锐度对齐：分配熵应与 kappa 匹配
        # 高 kappa → 低熵（尖锐），低 kappa → 高熵（平缓）
        proto_entropy = -(proto_weights * torch.log(proto_weights + 1e-10)).sum(dim=-1).mean()
        # 理论目标熵代理
        target_entropy = math.log(self.num_codes) - torch.logsumexp(kappa, dim=0) + kappa.mean()
        entropy_mismatch = (proto_entropy - target_entropy).pow(2)

        # 组合：多样性奖励 + 锐度对齐惩罚
        kappa_loss = -0.1 * kappa_var + 0.5 * entropy_mismatch

        return proto_weights, proto_repr, kappa.mean(), kappa_loss


# ==============================================================================
# Grassmannian Packing (零SVD谱正则)
# ==============================================================================

class GrassmannianPacking(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, coherence_threshold: float = 0.1):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.coherence_threshold = coherence_threshold

    def forward(self, prototypes: Tensor) -> Tensor:
        K = prototypes.size(0)
        gram = torch.mm(prototypes, prototypes.t())
        mask = 1.0 - torch.eye(K, device=prototypes.device)
        off_diag = gram * mask
        excess = F.relu(torch.abs(off_diag) - self.coherence_threshold)
        loss = (excess ** 2).sum()
        return loss


# ==============================================================================
# Prototype-Item Interaction (preserved)
# ==============================================================================

class PrototypeInteraction(nn.Module):
    def __init__(self, d_model: int, num_codes: int):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        self.item_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.item_proj.weight, gain=0.1)

        self.match_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        for layer in self.match_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)

    def forward(self, proto_weights: Tensor, item_repr: Tensor, prototypes: Tensor) -> Tensor:
        B, K = proto_weights.shape
        z_i = self.item_proj(item_repr)

        z_i_exp = z_i.unsqueeze(1).expand(B, K, -1)
        proto_exp = prototypes.unsqueeze(0).expand(B, -1, -1)
        match_input = torch.cat([z_i_exp, proto_exp], dim=-1)
        match_scores = self.match_mlp(match_input).squeeze(-1)

        ctr_logits = (proto_weights * match_scores).sum(dim=-1, keepdim=True)
        return ctr_logits


# ==============================================================================
# ForceNet Hamiltonian (零autograd.grad)
# ==============================================================================

class ForceNetHamiltonian(nn.Module):
    def __init__(self, d_model: int, num_fields: int = 3, num_steps: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.num_steps = num_steps

        self.force_net = nn.Sequential(
            nn.Linear(d_model * num_fields, d_model * num_fields),
            nn.LayerNorm(d_model * num_fields),
            nn.SiLU(),
            nn.Linear(d_model * num_fields, d_model * num_fields),
            nn.Tanh(),
        )

        self.mass_log = nn.Parameter(torch.zeros(num_fields, d_model))
        self.momentum_init = nn.Linear(d_model, d_model)
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        keys = sorted(fields.keys())
        q = torch.stack([fields[k] for k in keys], dim=1)
        B, F, D = q.shape

        p = self.momentum_init(q.mean(dim=1)).unsqueeze(1).expand(-1, F, -1)
        M = torch.exp(self.mass_log).unsqueeze(0)
        dt = torch.sigmoid(self.dt) * 0.2

        for _ in range(self.num_steps):
            q, p = self._verlet_step(q, p, M, dt)

        return {k: q[:, i] for i, k in enumerate(keys)}

    def _verlet_step(self, q: Tensor, p: Tensor, M: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        B, F, D = q.shape
        q_half = q + 0.5 * dt * p / M
        q_flat = q_half.reshape(B, -1)
        force_flat = self.force_net(q_flat)
        force = force_flat.reshape(B, F, D)
        p_new = p + dt * force
        q_new = q_half + 0.5 * dt * p_new / M
        return q_new, p_new


# ==============================================================================
# Inter-Field Operations (preserved)
# ==============================================================================

class InterSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        keys = sorted(fields.keys())
        x = torch.stack([fields[k] for k in keys], dim=1)
        out, _ = self.attn(x, x, x)
        out = self.norm(x + out)
        return {k: out[:, i] for i, k in enumerate(keys)}


class InterCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1,
                 query_fields: Optional[List[str]] = None,
                 kv_fields: Optional[List[str]] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_kv = nn.LayerNorm(d_model, eps=1e-5)
        self._q_fields = tuple(query_fields) if query_fields is not None else None
        self._kv_fields = tuple(kv_fields) if kv_fields is not None else None
        self._fixed_mode = self._q_fields is not None and self._kv_fields is not None

    def forward(self, fields: Dict[str, Tensor],
                keys_values: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        if self._fixed_mode:
            q = torch.stack([fields[f] for f in self._q_fields], dim=1)
            if keys_values is None:
                kv = torch.stack([fields[f] for f in self._kv_fields], dim=1)
            else:
                kv = torch.stack([keys_values[f] for f in self._kv_fields], dim=1)
            q_keys = self._q_fields
        else:
            q_keys = sorted(fields.keys())
            kv_keys = sorted(keys_values.keys()) if keys_values is not None else q_keys
            q = torch.stack([fields[k] for k in q_keys], dim=1)
            kv = torch.stack([keys_values[k] for k in kv_keys], dim=1) if keys_values is not None else q

        out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), kv)
        out = self.norm_q(q) + out
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

    def forward(self, fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        keys = sorted(fields.keys())
        projected = [self.projs[i](fields[keys[i]]) for i in range(len(keys))]
        interaction = projected[0]
        for p in projected[1:]:
            interaction = interaction * p
        interaction = interaction * self.scale
        out = self.out_proj(interaction)
        return {k: self.norm(out) for k in keys}


class PoolConcatLinear(nn.Module):
    def __init__(self, field_dims: Dict[str, int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        total_in = sum(field_dims.values())
        self.proj = nn.Linear(total_in, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, fields: Dict[str, Tensor]) -> Tensor:
        x = torch.cat(list(fields.values()), dim=-1)
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# HeteroBlock (preserved)
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

    def _fields_to_tensor(self, fields: Dict[str, Tensor]) -> Tensor:
        return torch.stack([fields[k] for k in self.fields], dim=1)

    def _tensor_to_fields(self, x: Tensor) -> Dict[str, Tensor]:
        return {self.fields[i]: x[:, i] for i in range(len(self.fields))}

    def forward(self, fields: Dict[str, Tensor],
                residuals: Optional[Dict[str, Tensor]] = None
               ) -> Dict[str, Tensor]:
        if self.intra_ops:
            processed = {}
            for name in self.fields:
                processed[name] = self.intra_ops[name](fields[name]) if name in self.intra_ops else fields[name]
        else:
            processed = fields

        x = self._fields_to_tensor(processed)

        if self.inter_op is not None:
            fields_dict = self._tensor_to_fields(x)
            inter_dict = self.inter_op(fields_dict)
            for name in self.fields:
                if name not in inter_dict:
                    inter_dict[name] = fields_dict[name]
            x = self._fields_to_tensor(inter_dict)

        if self.pool_op is not None:
            fields_dict = self._tensor_to_fields(x)
            pooled = self.pool_op(fields_dict)
            x = pooled.unsqueeze(1).expand(-1, len(self.fields), -1)

        if residuals is not None:
            res = self._fields_to_tensor(residuals)
            if self.residual_gate is not None:
                gate = torch.sigmoid(self.residual_gate)
                x = res + gate * self.dropout(x - res)
            else:
                x = res + self.dropout(x - res)
        else:
            x = self.dropout(x)

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

        return self._tensor_to_fields(x)


# ==============================================================================
# Fusion Gate v8.1-patch2 (添加bias)
# ==============================================================================

class FusionGate(nn.Module):
    def __init__(self, d_model: int, num_codes: int):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        self.static_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2, bias=False),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        self.proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + num_codes, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion_head = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.fusion_head.weight, gain=0.1)
        nn.init.zeros_(self.fusion_head.bias)

    def forward(self, final_repr: Tensor, proto_repr: Tensor,
                user_feat: Tensor, item_feat: Tensor, proto_weights: Tensor) -> Tensor:
        static_repr = self.static_proj(final_repr)
        proto_aligned = self.proto_proj(proto_repr)

        context = torch.cat([user_feat, item_feat], dim=-1)
        gate_input = torch.cat([context, proto_weights], dim=-1)
        weights = self.gate(gate_input)

        fused_repr = weights[:, 0:1] * static_repr + weights[:, 1:2] * proto_aligned
        fused_logits = self.fusion_head(fused_repr)
        return fused_logits


# ==============================================================================
# PCVRHeteroFormer v8.1-patch2
# ==============================================================================

class PCVRHeteroFormer(nn.Module):
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
        num_codes: int = 64,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iter: int = 20,
        min_mass_ratio: float = 0.016,
        coherence_threshold: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.action_num = action_num
        self.progressive_layer_training = progressive_layer_training
        self._current_epoch = 0
        self.num_codes = num_codes

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

        # Block 1: Feature Tokenization
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

        # Block 2: Sequence Encoding
        self.seq_blocks = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            self.seq_blocks[domain] = SequenceEncoder(
                vocab_sizes=vocab_sizes,
                d_model=d_model,
                num_time_buckets=num_time_buckets,
                num_heads=num_heads,
                num_layers=4,
                dropout=dropout,
                max_seq_len=512,
                id_threshold=seq_id_threshold,
                seq_id_dropout_rate=seq_id_dropout_rate,
            )

        # Block 3: vMF Prototype Quantization
        self.prototype_vqs = nn.ModuleDict()
        for domain in self.seq_domains:
            self.prototype_vqs[domain] = VonMisesFisherPrototype(
                num_codes=num_codes,
                code_dim=d_model,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_iter=sinkhorn_iter,
                min_mass_ratio=min_mass_ratio,
            )

        # Grassmannian Packing
        self.grassmannian_packing = GrassmannianPacking(
            num_codes=num_codes,
            code_dim=d_model,
            coherence_threshold=coherence_threshold,
        )

        # Block 4: Prototype-Item Interaction
        self.proto_interaction = PrototypeInteraction(d_model, num_codes)

        # Block 5: ForceNet Hamiltonian Initial Cross
        self.init_cross = HeteroBlock(
            fields=['user', 'item', 'context'],
            inter_op=ForceNetHamiltonian(d_model, num_fields=3, num_steps=2),
            name='init_cross',
        )

        # Block 6: Deep NS Stack
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
                pool_op=None,
                residual_gate=True,
                stochastic_depth_prob=stochastic_depth_prob * (i / max(num_layers - 1, 1)),
                dropout=dropout,
                name=f'ns_deep_{i}',
            ))

        # Block 7: NS-Sequence Cross
        self.ns_seq_cross = HeteroBlock(
            fields=['ns_global', 'seq_local'],
            inter_op=InterCrossAttention(d_model, num_heads, dropout,
                                         query_fields=['seq_local'],
                                         kv_fields=['ns_global']),
            name='ns_seq_cross',
        )

        # Block 8: Fusion Gate
        self.fusion_gate = FusionGate(d_model, num_codes)

        # Static Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2, bias=False),
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, action_num, bias=False),
        )
        nn.init.xavier_uniform_(self.predictor[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.1)

        # Logit Temperature & Scale
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.output_scale_logit = nn.Parameter(torch.tensor(0.0))

        # 全局 logit_bias，抑制均值持续下漂
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

        # Parameter Registration
        self._sparse_params: List[nn.Parameter] = []
        self._dense_params: List[nn.Parameter] = []
        self._gate_params: List[nn.Parameter] = []
        self._scale_params: List[nn.Parameter] = []
        self._proto_params: List[nn.Parameter] = []
        self._register_params()

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
            elif 'prototype' in name or 'proto_' in name or 'empty_prior' in name or 'match_vectors' in name or 'eta' in name or 'kappa_log' in name:
                self._proto_params.append(param)
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

    def get_proto_params(self) -> List[nn.Parameter]:
        return self._proto_params

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

    def _get_output_scale(self):
        return torch.sigmoid(self.output_scale_logit) * 2.0

    def _encode_features(self, user_int, item_int, user_dense, item_dense):
        user_fields = {}
        for i, (_, offset, length) in enumerate(self.user_int_feature_specs):
            fname = f'ufeat_{i}'
            if fname in self.user_tokenize.fields:
                user_fields[fname] = user_int[:, offset:offset+length]
        user_emb = self.user_tokenize(user_fields) if user_fields else None
        user_feat = user_emb[list(user_emb.keys())[0]] if user_emb else             torch.zeros(user_int.size(0), self.d_model, device=user_int.device)

        if self.user_dense_proj is not None and user_dense.size(-1) > 0:
            user_feat = user_feat + self.user_dense_proj(user_dense)

        item_fields = {}
        for i, (_, offset, length) in enumerate(self.item_int_feature_specs):
            fname = f'ifeat_{i}'
            if fname in self.item_tokenize.fields:
                item_fields[fname] = item_int[:, offset:offset+length]
        item_emb = self.item_tokenize(item_fields) if item_fields else None
        item_feat = item_emb[list(item_emb.keys())[0]] if item_emb else             torch.zeros(item_int.size(0), self.d_model, device=item_int.device)

        if self.item_dense_proj is not None and item_dense.size(-1) > 0:
            item_feat = item_feat + self.item_dense_proj(item_dense)

        context_feat = user_feat + item_feat
        return user_feat, item_feat, context_feat

    def _encode_sequence(self, seq_data, seq_lens, seq_time_buckets, seq_decay_weights):
        B = next(iter(seq_data.values())).size(0)
        device = next(iter(seq_data.values())).device

        if len(self.seq_domains) == 0:
            return torch.zeros(B, self.d_model, device=device), {}

        domain_outputs = {}
        for domain in self.seq_domains:
            if domain not in seq_data:
                continue
            pooled = self.seq_blocks[domain](
                seq_data[domain], seq_lens[domain],
                seq_time_buckets.get(domain),
                seq_decay_weights.get(domain) if seq_decay_weights else None,
            )
            domain_outputs[domain] = pooled

        if len(domain_outputs) == 0:
            return torch.zeros(B, self.d_model, device=device), {}

        stacked = torch.stack(list(domain_outputs.values()), dim=1)
        seq_repr = stacked.mean(dim=1)

        return seq_repr, domain_outputs

    def forward(self, model_input: ModelInput):
        # 1. Feature Encoding
        user_feat, item_feat, context_feat = self._encode_features(
            model_input.user_int_feats, model_input.item_int_feats,
            model_input.user_dense_feats, model_input.item_dense_feats,
        )

        # 2. Sequence Encoding
        seq_feat, domain_outputs = self._encode_sequence(
            model_input.seq_data, model_input.seq_lens,
            model_input.seq_time_buckets, model_input.seq_decay_weights,
        )

        # 3. vMF Prototype Quantization
        first_domain = self.seq_domains[0] if len(self.seq_domains) > 0 else None
        if first_domain is not None and first_domain in model_input.seq_lens:
            ref_lens = model_input.seq_lens[first_domain]
            proto_vq = self.prototype_vqs[first_domain]
            proto_weights, proto_repr, kappa_mean, kappa_loss = proto_vq(seq_feat, ref_lens)
        else:
            B = user_feat.size(0)
            device = user_feat.device
            proto_weights = torch.ones(B, self.num_codes, device=device) / self.num_codes
            proto_repr = torch.zeros(B, self.d_model, device=device)
            kappa_mean = torch.tensor(0.0, device=device)
            kappa_loss = torch.tensor(0.0, device=device)

        # 4. ForceNet Hamiltonian Initial Cross
        init_fields = {'user': user_feat, 'item': item_feat, 'context': context_feat}
        init_out = self.init_cross(init_fields)
        u_cross, i_cross, c_cross = init_out['user'], init_out['item'], init_out['context']

        # 5. Deep NS Stack
        ns_fields = {'user': u_cross, 'item': i_cross, 'context': c_cross}
        residuals = None
        for block in self.ns_blocks:
            ns_fields = block(ns_fields, residuals)
            residuals = {k: v for k, v in ns_fields.items()}

        # 6. NS-Sequence Cross
        ns_global = torch.stack([ns_fields['user'], ns_fields['item'], ns_fields['context']], dim=1).mean(dim=1)
        cross_fields = {'ns_global': ns_global, 'seq_local': seq_feat}
        cross_out = self.ns_seq_cross(cross_fields)

        # 7. Final Representation
        final_repr = torch.cat([
            ns_fields['user'], ns_fields['item'],
            ns_fields['context'], cross_out['seq_local']
        ], dim=-1)

        # 8. Static Prediction
        static_logits = self.predictor(final_repr)

        # 9. Prototype Prediction + Packing Loss (统一 mu 作用域)
        first_proto_vq = self.prototype_vqs[list(self.prototype_vqs.keys())[0]] if len(self.prototype_vqs) > 0 else None
        if first_proto_vq is not None:
            mu, _ = first_proto_vq.get_prototypes()
            proto_ctr = self.proto_interaction(proto_weights, item_feat, mu)
            packing_loss = self.grassmannian_packing(mu)
        else:
            proto_ctr = torch.zeros_like(static_logits)
            packing_loss = torch.tensor(0.0, device=seq_feat.device)

        # 10. Fusion Gate
        fused_logits = self.fusion_gate(final_repr, proto_repr, user_feat, item_feat, proto_weights)

        # 11. Output scaling + bias
        logits = fused_logits * self._get_output_scale() + self.logit_bias
        temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
        logits = logits / temperature

        # 返回 8 元组（patch2 新增 kappa_loss）
        return logits, seq_feat, proto_weights, proto_repr, kappa_mean, packing_loss, proto_ctr, kappa_loss

    def predict(self, model_input: ModelInput):
        with torch.no_grad():
            logits, *_ = self.forward(model_input)
        return logits, None