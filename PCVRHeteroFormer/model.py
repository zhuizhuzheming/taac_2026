"""PCVRHeteroFormer v11.2-MoS - Hybrid (Explicit Interaction + RoPE + SwiGLU + Top-K + Theme Cross-Domain)
================================================================================
嫁接 HyFormer 组件：
1. RoPE 旋转位置编码（替代可学习 pos_embed）
2. SwiGLU 激活（替代标准 FFN）
3. Top-K 序列压缩（LongerEncoder 思想，可选）
4. ThemeCrossDomainLayer（MoS arXiv:2604.20858 深层跨 Domain 嫁接）

保留 v11.1 显式交互：
- InterestExtractor（候选 Target Attention）
- CrossFieldLayer（User×Item/Seq 显式二阶交叉）
- Type-aware Q/K Bias（打破对称注意力）
"""

import math
import logging
from typing import Tuple, Optional, List, Dict, NamedTuple, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class ModelInput(NamedTuple):
    user_int_feats: Tensor
    item_int_feats: Tensor
    user_dense_feats: Tensor
    item_dense_feats: Tensor
    seq_data: Dict[str, Tensor]
    seq_lens: Dict[str, Tensor]
    seq_time_buckets: Dict[str, Tensor]
    seq_decay_weights: Optional[Dict[str, Tensor]] = None
    seq_timestamps_raw: Optional[Dict[str, Tensor]] = None


def safe_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def check_nan_tensor(tensor: Tensor, name: str = "tensor") -> bool:
    has_nan = torch.isnan(tensor).any() or torch.isinf(tensor).any()
    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logging.warning(f"【NaN检测】{name}: NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
    return has_nan


# ==============================================================================
# HyFormer: RoPE Components
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values (torch.compile friendly)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ==============================================================================
# HyFormer: SwiGLU
# ==============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2). Replaces standard FFN."""
    def __init__(self, d_model: int, hidden_mult: int = 2) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


# ==============================================================================
# HyFormer: RoPE Multi-head Attention (with Gating)
# ==============================================================================

class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with RoPE and output gating."""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
        q_rope_cos: Optional[Tensor] = None,
        q_rope_sin: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)
            if self.rope_on_q:
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        sdpa_attn_mask = None
        if key_padding_mask is not None:
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            bool_attn = (attn_mask == 0)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_attn_mask, dropout_p=dropout_p)
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


# ==============================================================================
# MoS: ThemeCrossDomainLayer (Deep cross-domain interaction via theme routing)
# ==============================================================================

class ThemeCrossDomainLayer(nn.Module):
    """MoS-inspired deep cross-domain interaction.

    For each latent theme k:
    1. Soft-assign every seq token (across all domains) to K themes.
    2. Aggregate per-domain theme centers (each domain contributes one vector per theme).
    3. Apply cross-domain self-attention *within* the same theme.
    4. Diffuse the enhanced theme centers back to individual tokens via gating.

    This realizes "same-theme cross-domain deep interaction" at the Backbone level.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_themes: int,
        num_domains: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_themes = num_themes
        self.num_domains = num_domains
        self.d_model = d_model

        # Learnable theme prototypes
        self.theme_protos = nn.Parameter(torch.randn(num_themes, d_model) * 0.02)
        self.scale = d_model ** -0.5

        # Per-theme cross-domain attention: M domains attend to each other
        self.cross_domain_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_themes)
        ])
        self.domain_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_themes)])

        # Gated residual fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, type_ids: Tensor, mask: Optional[Tensor]) -> Tensor:
        B, N, D = x.shape
        device = x.device

        # Only seq tokens participate (type_ids >= 4)
        seq_type_mask = type_ids >= 4  # [B, N]

        # 1. Theme affinity scores for all tokens [B, N, K]
        scores = torch.einsum('bnd,kd->bnk', x, self.theme_protos) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        scores = scores.masked_fill(~seq_type_mask.unsqueeze(-1), float('-inf'))

        theme_w = F.softmax(scores, dim=1)  # [B, N, K]
        theme_w = torch.nan_to_num(theme_w, nan=0.0)

        # 2. Per-domain theme centers [B, K, M, D]
        theme_centers = torch.zeros(B, self.num_themes, self.num_domains, D, device=device)
        for k in range(self.num_themes):
            for m in range(self.num_domains):
                type_id = 4 + m
                domain_mask = (type_ids == type_id)
                if mask is not None:
                    domain_mask = domain_mask & (~mask)
                w = theme_w[..., k] * domain_mask.float()  # [B, N]
                center = (x * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, D]
                theme_centers[:, k, m, :] = center

        # 3. Cross-domain attention within each theme
        enhanced = torch.zeros_like(theme_centers)
        for k in range(self.num_themes):
            centers = theme_centers[:, k, :, :]  # [B, M, D]
            out, _ = self.cross_domain_attn[k](centers, centers, centers)
            enhanced[:, k, :, :] = self.domain_norms[k](centers + self.dropout(out))

        # 4. Diffuse back to tokens
        domain_idx = torch.clamp(type_ids - 4, min=0, max=self.num_domains - 1)  # [B, N]

        # Gather enhanced center for each token's domain: [B, N, K, D]
        m_idx = domain_idx.unsqueeze(2).expand(B, N, self.num_themes)            # [B, N, K]
        k_idx = torch.arange(self.num_themes, device=device).view(1, 1, -1).expand(B, N, -1)
        b_idx = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, N, self.num_themes)

        gathered = enhanced[b_idx, k_idx, m_idx, :]  # [B, N, K, D]
        token_enhanced = (gathered * theme_w.unsqueeze(-1)).sum(dim=2)  # [B, N, D]
        token_enhanced = token_enhanced * seq_type_mask.unsqueeze(-1).float()

        # Gated residual
        gate = self.gate_proj(torch.cat([x, token_enhanced], dim=-1))
        out = x + self.dropout(gate * token_enhanced)
        out = self.out_norm(out)
        return out


# ==============================================================================
# Base Primitives
# ==============================================================================

class ConstrainedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.max_norm = max_norm
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, input: Tensor) -> Tensor:
        weight_normed = F.normalize(self.weight, p=2, dim=-1) * self.max_norm
        return F.embedding(input, weight_normed, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class IntraEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_slots: int = 1):
        super().__init__()
        self.emb = ConstrainedEmbedding(max(vocab_size, 2), emb_dim)
        self.num_slots = num_slots

    def forward(self, x: Tensor) -> Tensor:
        if self.num_slots == 1:
            emb = self.emb(x.squeeze(-1))
        else:
            embs = torch.stack([self.emb(x[:, i]) for i in range(self.num_slots)], dim=1)
            emb = embs.mean(dim=1)
        return emb


# ==============================================================================
# FeatureTokenizer
# ==============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        dense_dim: int,
        d_model: int,
        emb_dim: int = 16,
        id_vocab_threshold: int = 10000,
        emb_skip_threshold: int = 0,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.field_embs = nn.ModuleDict()
        self.field_projs = nn.ModuleDict()
        self.field_names = []
        self.field_meta = []

        for idx, (vocab_size, offset, length) in enumerate(feature_specs):
            if vocab_size > 1 and (emb_skip_threshold <= 0 or vocab_size <= emb_skip_threshold):
                fname = f'f_{idx}'
                self.field_embs[fname] = IntraEmbedding(vocab_size, emb_dim, length)
                if emb_dim != d_model:
                    self.field_projs[fname] = nn.Sequential(
                        nn.Linear(emb_dim, d_model, bias=False),
                        nn.LayerNorm(d_model),
                    )
                self.field_names.append(fname)
                self.field_meta.append((fname, offset, length, False))

        if dense_dim > 0:
            self.dense_proj = nn.Sequential(
                nn.Linear(dense_dim, d_model, bias=False),
                nn.LayerNorm(d_model),
            )
            self.field_names.append('dense')
            self.field_meta.append(('dense', 0, dense_dim, True))
        else:
            self.dense_proj = None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, int_feats: Tensor, dense_feats: Optional[Tensor] = None) -> Tensor:
        B = int_feats.size(0)
        tokens = []
        for fname, offset, length, is_dense in self.field_meta:
            if is_dense:
                if dense_feats is not None and dense_feats.size(-1) > 0:
                    t = self.dense_proj(dense_feats)
                else:
                    t = torch.zeros(B, self.d_model, device=int_feats.device)
            else:
                x = int_feats[:, offset:offset + length]
                t = self.field_embs[fname](x)
                if fname in self.field_projs:
                    t = self.field_projs[fname](t)
            tokens.append(t)

        out = torch.stack(tokens, dim=1)
        return self.dropout(out)


# ==============================================================================
# CondTimeAttention (保留时间偏置 + 候选门控)
# ==============================================================================

class CondTimeAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.cond_gate_q = nn.Linear(d_model, d_model)
        self.cond_gate_k = nn.Linear(d_model, d_model)

        self.time_bias_proj = nn.Linear(1, num_heads, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.cond_gate_q.weight, 0.0)
        nn.init.constant_(self.cond_gate_q.bias, 0.0)
        nn.init.constant_(self.cond_gate_k.weight, 0.0)
        nn.init.constant_(self.cond_gate_k.bias, 0.0)

    def forward(self, x, cond, time_deltas, mask=None):
        B, L, D = x.shape

        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        gate_q = torch.sigmoid(self.cond_gate_q(cond)).view(B, 1, self.num_heads, self.d_head)
        gate_k = torch.sigmoid(self.cond_gate_k(cond)).view(B, 1, self.num_heads, self.d_head)
        q = q * gate_q
        k = k * gate_k

        scores = torch.einsum('blhd,bmhd->blhm', q, k) * self.scale

        if time_deltas is not None and time_deltas.numel() > 0:
            time_compressed = torch.log1p(time_deltas.clamp(min=0).float())
            t_bias = self.time_bias_proj(time_compressed.unsqueeze(-1))
            t_bias = torch.tanh(t_bias) * 2.0
            scores = scores + t_bias.unsqueeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_max = torch.where(torch.isinf(scores_max), torch.zeros_like(scores_max), scores_max)
        scores = scores - scores_max.detach()
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.einsum('blhm,bmhd->blhd', attn, v).reshape(B, L, D)
        out = self.out_proj(out)
        return out


# ==============================================================================
# CondTimeEncoderLayer (FFN -> SwiGLU)
# ==============================================================================

class CondTimeEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = CondTimeAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult=2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, time_deltas, mask=None):
        attn_out = self.attn(x, cond, time_deltas, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.swiglu(x))
        return x


# ==============================================================================
# InterestExtractor (显式 Target Attention)
# ==============================================================================

class InterestExtractor(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_hidden: Tensor, seq_mask: Optional[Tensor], candidate: Tensor) -> Tuple[Tensor, Tensor]:
        cq = candidate.unsqueeze(1)
        interest, _ = self.cross_attn(
            cq, seq_hidden, seq_hidden,
            key_padding_mask=seq_mask if seq_mask is not None else None,
            need_weights=False,
        )
        interest = interest.squeeze(1)

        interest_expanded = interest.unsqueeze(1).expand(-1, seq_hidden.size(1), -1)
        gate = self.gate(torch.cat([seq_hidden, interest_expanded], dim=-1))
        enhanced = seq_hidden + self.dropout(gate * interest_expanded)
        return self.norm(enhanced), interest


# ==============================================================================
# Top-K Sequence Compressor (LongerEncoder 思想嫁接)
# ==============================================================================

class _SeqCompressor(nn.Module):
    """当序列长度 > top_k 时，用 Cross-Attention 压缩到 top_k 个最近 token。"""
    def __init__(self, d_model: int, num_heads: int, top_k: int, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def _gather_top_k(self, x: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        B, L, D = x.shape
        device = x.device
        valid_len = (~mask).sum(dim=1) if mask is not None else torch.full((B,), L, device=device, dtype=torch.long)
        actual_k = torch.clamp(valid_len, max=self.top_k)
        start_pos = valid_len - actual_k
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        indices = start_pos.unsqueeze(1) + offsets
        indices = torch.clamp(indices, max=L - 1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        top_k_x = torch.gather(x, dim=1, index=indices_expanded)

        pos = torch.arange(self.top_k, device=device).unsqueeze(0)
        pad_count = self.top_k - actual_k
        new_mask = pos < pad_count.unsqueeze(1)
        return top_k_x, new_mask, indices

    def forward(self, x: Tensor, mask: Optional[Tensor], cond: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        B, L, D = x.shape
        if self.top_k <= 0 or L <= self.top_k:
            return x, mask
        q, new_mask, _ = self._gather_top_k(x, mask)
        # cond 增强 query
        q = q + cond.unsqueeze(1)
        out, _ = self.cross_attn(q, x, x, key_padding_mask=mask)
        out = self.norm(q + out)
        return out, new_mask


# ==============================================================================
# CondAwareSequenceEncoder (+ Compressor + SwiGLU)
# ==============================================================================

class CondAwareSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: List[int],
        d_model: int = 128,
        num_layers: int = 2,
        max_seq_len: int = 512,
        dropout_rate: float = 0.15,
        top_k: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.top_k = top_k
        n_feats = len(vocab_sizes)
        feat_dim = max(d_model // n_feats, 1)

        self.feat_embs = nn.ModuleList()
        for vs in vocab_sizes:
            self.feat_embs.append(nn.Embedding(max(vs, 2), feat_dim))
            nn.init.normal_(self.feat_embs[-1].weight, std=0.01)

        self.seq_proj = nn.Linear(feat_dim * n_feats, d_model, bias=False)
        self.seq_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.seq_proj.weight, gain=0.1)

        # Optional top-k compressor (HyFormer LongerEncoder idea)
        self.compressor = _SeqCompressor(
            d_model, num_heads=max(4, d_model // 32), top_k=top_k, dropout=dropout_rate
        ) if top_k > 0 else None

        self.layers = nn.ModuleList([
            CondTimeEncoderLayer(d_model, num_heads=max(4, d_model // 32), dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.interest_extractor = InterestExtractor(
            d_model, num_heads=max(4, d_model // 32), dropout=dropout_rate
        )

        self.empty_mu = nn.Parameter(torch.randn(d_model) * 0.1)
        self.empty_log_sigma = nn.Parameter(torch.randn(d_model) * 0.01 - 2.0)

        self.short_proj = nn.Linear(d_model, d_model)
        self.long_proj = nn.Linear(d_model, d_model)
        self.static_proj = nn.Linear(d_model * 3, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(
        self,
        seq_ids: Tensor,
        seq_lens: Tensor,
        cond: Tensor,
        timestamps_raw: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, n_feats, L = seq_ids.shape
        device = seq_ids.device

        feat_embs = [self.feat_embs[i](seq_ids[:, i, :]) for i in range(n_feats)]
        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))
        seq_repr = self.dropout(seq_repr)

        if timestamps_raw is not None:
            time_deltas = timestamps_raw
        else:
            time_deltas = torch.zeros(B, L, device=device)

        positions = torch.arange(L, device=device).unsqueeze(0)
        mask = positions >= seq_lens.unsqueeze(1)

        # === HyFormer: Top-K compression before deep processing ===
        hidden = seq_repr
        if self.compressor is not None:
            hidden, mask = self.compressor(hidden, mask, cond)
            L = hidden.size(1)

        for layer, norm in zip(self.layers, self.layer_norms):
            residual = hidden
            hidden = layer(hidden, cond, time_deltas, mask)
            hidden = norm(hidden + residual)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
            hidden = self.dropout(hidden)

        hidden, interest_vec = self.interest_extractor(hidden, mask, cond)

        valid_mask = (~mask).float()

        short_weights = torch.exp(-torch.arange(L, device=device).float() / 10.0).unsqueeze(0) * valid_mask
        short_weights = short_weights / (short_weights.sum(dim=1, keepdim=True) + 1e-8)
        s_short = torch.sum(hidden * short_weights.unsqueeze(-1), dim=1)
        s_short = self.short_proj(s_short)

        long_weights = valid_mask / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        s_long = torch.sum(hidden * long_weights.unsqueeze(-1), dim=1)
        s_long = self.long_proj(s_long)

        seq_mean = (hidden * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sq_mean = ((hidden ** 2) * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        seq_std = torch.sqrt(torch.clamp(sq_mean - seq_mean ** 2, min=0.0) + 1e-8)
        last_idx = (seq_lens - 1).clamp(min=0).long()
        # Adjust last_idx for compressed length
        if self.compressor is not None:
            valid_len = (~mask).sum(dim=1)
            last_idx = (valid_len - 1).clamp(min=0)
        seq_last = hidden[torch.arange(B, device=device), last_idx]
        s_static = self.static_proj(torch.cat([seq_mean, seq_std, seq_last], dim=-1))

        empty_sigma = torch.exp(self.empty_log_sigma)
        empty_noise = torch.randn_like(self.empty_mu) * empty_sigma if self.training else 0
        empty_state = self.empty_mu + empty_noise
        empty_mask = (seq_lens == 0).unsqueeze(-1).float()
        for s in [s_short, s_long, s_static]:
            s = torch.where(empty_mask.bool(), empty_state.unsqueeze(0), s)

        return {
            'short': self.output_norm(s_short),
            'long': self.output_norm(s_long),
            'static': self.output_norm(s_static),
            'full_seq': hidden,
            'interest_vec': interest_vec,
        }


# ==============================================================================
# CrossFieldLayer (显式二阶交叉)
# ==============================================================================

class CrossFieldLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.cross_gate = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.cross_gate, std=0.01)

        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, user_vec: Tensor, item_vec: Tensor, seq_vec: Tensor) -> Tensor:
        ui = user_vec * item_vec
        us = user_vec * seq_vec
        is_ = item_vec * seq_vec

        crosses = torch.stack([ui, us, is_], dim=1)
        gated_crosses = (crosses * torch.sigmoid(self.cross_gate.unsqueeze(0))).sum(dim=1)

        fused = self.fusion(torch.cat([user_vec, ui, us, is_], dim=-1))
        return self.norm(fused + gated_crosses)


# ==============================================================================
# UnifiedTransformerLayer (RoPE + Type-aware + SwiGLU)
# ==============================================================================

class UnifiedTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        num_types: int = 8,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Type-aware Q/K modulation
        self.type_q_embed = nn.Embedding(num_types, d_model)
        self.type_k_embed = nn.Embedding(num_types, d_model)
        nn.init.normal_(self.type_q_embed.weight, std=0.02)
        nn.init.normal_(self.type_k_embed.weight, std=0.02)

        # SwiGLU FFN
        self.swiglu = SwiGLU(d_model, hidden_mult=2)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = drop_path

    def forward(self, x, mask=None, type_ids=None, rope_cos=None, rope_sin=None):
        if type_ids is not None:
            q = x + self.type_q_embed(type_ids)
            k = x + self.type_k_embed(type_ids)
            v = x
        else:
            q = k = v = x

        attn_out, _ = self.attn(
            q, k, v,
            key_padding_mask=mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        if self.drop_path > 0 and self.training and torch.rand(1).item() < self.drop_path:
            pass
        else:
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.swiglu(x))
        return x


# ==============================================================================
# UnifiedBackbone (RoPE + Type-aware + CrossField + ThemeCrossDomain)
# ==============================================================================

class UnifiedBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
        num_types: int = 8,
        cross_interval: int = 2,
        max_tokens: int = 1024,
        rope_base: float = 10000.0,
        num_domains: int = 4,
        num_themes: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.cross_interval = cross_interval
        self.num_domains = num_domains

        self.layers = nn.ModuleList([
            UnifiedTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                drop_path=(i / max(num_layers - 1, 1)) * stochastic_depth_prob,
                num_types=num_types,
                rope_base=rope_base,
            )
            for i in range(num_layers)
        ])

        # v11.2-MoS: Theme-aware cross-domain layers (deep interaction)
        self.theme_cross_layers = nn.ModuleDict({
            str(i): ThemeCrossDomainLayer(
                d_model=d_model,
                num_heads=max(1, num_heads // 2),
                num_themes=num_themes,
                num_domains=num_domains,
                dropout=dropout,
            )
            for i in range(cross_interval - 1, num_layers, cross_interval)
        })

        self.cross_layers = nn.ModuleDict({
            str(i): CrossFieldLayer(d_model, dropout)
            for i in range(cross_interval - 1, num_layers, cross_interval)
        })

        # RoPE cache (precomputed for torch.compile compatibility)
        head_dim = d_model // num_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_tokens,
            base=rope_base,
        )

    def forward(self, x, mask=None, type_ids=None, return_all_layers: bool = False):
        B, N, D = x.shape
        device = x.device

        # Precompute RoPE for current sequence length
        rope_cos, rope_sin = self.rotary_emb(N, device)

        all_layers = [x] if return_all_layers else None

        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, type_ids=type_ids, rope_cos=rope_cos, rope_sin=rope_sin)

            # v11.2-MoS: Theme-aware cross-domain interaction BEFORE CrossField
            if str(i) in self.theme_cross_layers and type_ids is not None:
                x = self.theme_cross_layers[str(i)](x, type_ids, mask)

            if str(i) in self.cross_layers and type_ids is not None:
                x = self._apply_cross_field(x, type_ids, mask, self.cross_layers[str(i)])

            if return_all_layers:
                all_layers.append(x)

        if return_all_layers:
            return x, all_layers
        return x

    def _apply_cross_field(self, x, type_ids, mask, cross_layer):
        user_vec = self._masked_mean(x, (type_ids == 1), mask)
        item_vec = self._masked_mean(x, (type_ids == 2), mask)
        seq_vec = self._masked_mean(x, (type_ids >= 4), mask)
        cross_out = cross_layer(user_vec, item_vec, seq_vec)
        x[:, 0] = x[:, 0] + cross_out
        return x

    def _masked_mean(self, x, type_mask, pad_mask):
        if pad_mask is not None:
            valid = type_mask & (~pad_mask)
        else:
            valid = type_mask
        if not valid.any():
            return torch.zeros(x.size(0), x.size(-1), device=x.device)
        valid_float = valid.float().unsqueeze(-1)
        return (x * valid_float).sum(dim=1) / valid_float.sum(dim=1).clamp(min=1.0)


# ==============================================================================
# EarlyExitCTRHead + UncertaintyHead
# ==============================================================================

class EarlyExitCTRHead(nn.Module):
    def __init__(self, d_model: int, num_layers: int, exit_interval: int = 2):
        super().__init__()
        self.exit_indices = list(range(exit_interval - 1, num_layers, exit_interval))
        if (num_layers - 1) not in self.exit_indices:
            self.exit_indices.append(num_layers - 1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1),
            ) for _ in self.exit_indices
        ])
        for head in self.heads:
            nn.init.constant_(head[-1].bias, -2.0)
            nn.init.xavier_uniform_(head[-1].weight, gain=0.1)

    def forward(self, layer_outputs: List[Tensor]) -> List[Tensor]:
        logits = []
        for idx, head in zip(self.exit_indices, self.heads):
            if idx < len(layer_outputs):
                cls = layer_outputs[idx][:, 0]
                logits.append(head(cls).squeeze(-1))
        return logits


class UncertaintyHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: Tensor) -> Tensor:
        return self.net(cls_token).squeeze(-1)


# ==============================================================================
# PCVRHeteroFormer (Hybrid v11.2-MoS)
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
        num_layers: int = 4,
        num_heads: Optional[int] = None,
        dropout: float = 0.15,
        stochastic_depth_prob: float = 0.1,
        id_vocab_threshold: int = 10000,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        num_codes: int = 128,
        seq_top_k: int = 0,  # HyFormer: top-k compression (0=disabled)
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        num_heads = num_heads or max(4, d_model // 32)

        self.user_tokenizer = FeatureTokenizer(
            user_int_feature_specs, user_dense_dim, d_model, emb_dim,
            id_vocab_threshold, emb_skip_threshold, dropout,
        )
        self.item_tokenizer = FeatureTokenizer(
            item_int_feature_specs, item_dense_dim, d_model, emb_dim,
            id_vocab_threshold, emb_skip_threshold, dropout,
        )

        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_encoders = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            self.seq_encoders[domain] = CondAwareSequenceEncoder(
                vocab_sizes=vocab_sizes, d_model=d_model, num_layers=2,
                max_seq_len=512, dropout_rate=dropout,
                top_k=seq_top_k,
            )

        self.num_types = 4 + len(self.seq_domains)
        self.type_embed = nn.Embedding(self.num_types, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)

        # Removed: learnable pos_embed (replaced by RoPE in backbone)
        self.max_tokens = 1024

        # v11.2-MoS: pass num_domains / num_themes to backbone
        num_themes = kwargs.get('num_themes', 2)

        self.backbone = UnifiedBackbone(
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            dropout=dropout, stochastic_depth_prob=stochastic_depth_prob,
            num_types=self.num_types,
            cross_interval=2,
            max_tokens=self.max_tokens,
            rope_base=rope_base,
            num_domains=len(self.seq_domains),
            num_themes=num_themes,
        )

        self.early_exit_head = EarlyExitCTRHead(d_model, num_layers, exit_interval=2)
        self.uncertainty_head = UncertaintyHead(d_model)

        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

        self._register_param_groups()

    def _register_param_groups(self):
        self._sparse_params = []
        self._backbone_params = []
        self._seq_params = []
        self._head_params = []
        self._other_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if 'emb' in name.lower() and 'weight' in name and p.numel() > 1000:
                self._sparse_params.append(p)
            elif 'seq_encoders' in name:
                self._seq_params.append(p)
            elif 'backbone' in name:
                self._backbone_params.append(p)
            elif 'head' in name or 'temperature' in name or 'bias' in name:
                self._head_params.append(p)
            else:
                self._other_params.append(p)

        total = sum(1 for _ in self.parameters() if _.requires_grad)
        assigned = sum(len(g) for g in [self._sparse_params, self._backbone_params,
                                        self._seq_params, self._head_params, self._other_params])
        logging.info(f"【v11.2-MoS参数分组】{assigned}/{total} 已分配 | "
                     f"sparse={len(self._sparse_params)}, backbone={len(self._backbone_params)}, "
                     f"seq={len(self._seq_params)}, head={len(self._head_params)}")

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            'sparse': self._sparse_params,
            'backbone': self._backbone_params,
            'seq_encoder': self._seq_params,
            'head': self._head_params,
            'other': self._other_params,
        }

    def _build_unified_tokens(
        self,
        model_input: ModelInput,
        item_summary: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Returns: unified_tokens, pad_mask, type_ids"""
        B = model_input.user_int_feats.size(0)
        device = model_input.user_int_feats.device

        user_tokens = self.user_tokenizer(model_input.user_int_feats, model_input.user_dense_feats)
        item_tokens = self.item_tokenizer(model_input.item_int_feats, model_input.item_dense_feats)

        seq_token_list = []
        seq_masks = []
        for domain in self.seq_domains:
            if domain not in model_input.seq_data:
                continue
            timestamps = (model_input.seq_timestamps_raw or {}).get(domain)
            seq_out = self.seq_encoders[domain](
                model_input.seq_data[domain],
                model_input.seq_lens[domain],
                cond=item_summary,
                timestamps_raw=timestamps,
            )
            full_seq = seq_out['full_seq']
            seq_token_list.append(full_seq)

            L = full_seq.size(1)
            positions = torch.arange(L, device=device).unsqueeze(0)
            mask = positions >= model_input.seq_lens[domain].unsqueeze(1)
            seq_masks.append(mask)

        type_ids = []
        tokens = []

        # CLS: type 0
        cls = torch.zeros(B, 1, self.d_model, device=device)
        tokens.append(cls)
        type_ids.append(torch.full((B, 1), 0, dtype=torch.long, device=device))

        # User: type 1
        type_ids.append(torch.full((B, user_tokens.size(1)), 1, dtype=torch.long, device=device))
        tokens.append(user_tokens)

        # Item: type 2
        type_ids.append(torch.full((B, item_tokens.size(1)), 2, dtype=torch.long, device=device))
        tokens.append(item_tokens)

        # Context: type 3
        ctx = (user_tokens.mean(dim=1, keepdim=True) + item_tokens.mean(dim=1, keepdim=True)) * 0.5
        type_ids.append(torch.full((B, 1), 3, dtype=torch.long, device=device))
        tokens.append(ctx)

        # Sequences: type 4+
        all_seq_mask = []
        for i, (seq_tok, mask) in enumerate(zip(seq_token_list, seq_masks)):
            type_id = 4 + i
            type_ids.append(torch.full((B, seq_tok.size(1)), type_id, dtype=torch.long, device=device))
            tokens.append(seq_tok)
            all_seq_mask.append(mask)

        unified = torch.cat(tokens, dim=1)
        type_ids_cat = torch.cat(type_ids, dim=1)

        # Only type embedding (RoPE handles position)
        type_emb = self.type_embed(type_ids_cat)
        unified = unified + type_emb

        pad_mask = torch.zeros(B, unified.size(1), dtype=torch.bool, device=device)
        offset = 1 + user_tokens.size(1) + item_tokens.size(1) + 1
        for mask in all_seq_mask:
            L = mask.size(1)
            pad_mask[:, offset:offset + L] = mask
            offset += L

        return unified, pad_mask, type_ids_cat

    def forward(self, model_input: ModelInput, task_id: str = 'ctr', return_all_layers: bool = True):
        B = model_input.user_int_feats.size(0)
        device = model_input.user_int_feats.device

        item_tokens = self.item_tokenizer(model_input.item_int_feats, model_input.item_dense_feats)
        item_summary = item_tokens.mean(dim=1)

        unified, pad_mask, type_ids = self._build_unified_tokens(model_input, item_summary)

        # 根据 return_all_layers 决定 backbone 输出
        if return_all_layers:
            hidden, all_layers = self.backbone(unified, mask=pad_mask, type_ids=type_ids, return_all_layers=True)
        else:
            hidden = self.backbone(unified, mask=pad_mask, type_ids=type_ids, return_all_layers=False)
            all_layers = [hidden]  # 兼容 early_exit_head 的索引访问

        # 只在训练或需要时计算 early exit
        if return_all_layers:
            exit_logits = self.early_exit_head(all_layers[1:])
        else:
            exit_logits = []

        cls_final = hidden[:, 0]
        main_logits = exit_logits[-1] if exit_logits else torch.zeros(B, device=device)

        uncertainty = self.uncertainty_head(cls_final)

        temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
        main_logits = main_logits / temperature + self.logit_bias

        if not self.training:
            main_logits = main_logits / (1.0 + uncertainty * 0.5)

        main_logits = torch.clamp(main_logits.unsqueeze(-1), min=-20.0, max=20.0)

        return (
            main_logits,
            torch.zeros(B, 1, device=device),
            cls_final,
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
            torch.zeros(B, self.d_model * 4, device=device),
            uncertainty,
            torch.tensor(0.0, device=device),
            uncertainty,
            torch.tensor(0.0, device=device),
            main_logits.squeeze(-1),
            cls_final,
        )

    def predict(self, model_input: ModelInput):
        out = self.forward(model_input, task_id='ctr', return_all_layers=False)
        return out[0], None


# ==============================================================================
# Preserved HeteroBlock (兼容保留)
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
        dropout: float = 0.0,
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

    def _fields_to_tensor(self, fields: Dict[str, Tensor]) -> Tensor:
        return torch.stack([fields[k] for k in self.fields], dim=1)

    def _tensor_to_fields(self, x: Tensor) -> Dict[str, Tensor]:
        return {self.fields[i]: x[:, i] for i in range(len(self.fields))}

    def forward(self, fields: Dict[str, Tensor],
                residuals: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
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
                x = res + gate * (x - res)
            else:
                x = res + (x - res)

        return self._tensor_to_fields(x)