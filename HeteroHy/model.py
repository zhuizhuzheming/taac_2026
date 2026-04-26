"""PCVRHyFormer-Hetero: 融合 HeteroFormer 高阶交互的 PCVRHyFormer

针对 TAAC2026 序列建模 × 特征交互难题的融合方案：
- 底盘：PCVRHyFormer（多域长序列、LongerEncoder、RoPE、工程完整性）
- 注入：HeteroFormer（ProgressiveLowRankNSInteraction、Highway Gate、显式 einsum 交互）
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}


# ═══════════════════════════════════════════════════════════════════════════════
# HeteroFormer 注入模块
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """1D 因果卷积，保持时序因果性。输入输出均为 (B, L, D)。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel,
            padding=self.padding, dilation=dilation, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out.transpose(1, 2)  # (B, L, D)


class ProgressiveLowRankNSInteraction(nn.Module):
    """HeteroFormer 核心：三元低秩非序列交互。

    支持子空间银行路由、Pre-Norm、层深自适应初始化。
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        num_banks: int = 4,
        dropout: float = 0.1,
        layer_idx: int = 0,
        pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.pre_norm = pre_norm

        self.norm_user = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.norm_item = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.norm_context = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        init_std = 0.02 / (layer_idx + 1)
        self.subspace_banks = nn.Parameter(
            torch.randn(num_banks, rank, d_model) * init_std
        )

        self.router = nn.Sequential(
            nn.Linear(d_model * 3, num_banks),
            nn.Softmax(dim=-1)
        )

        self.W_Q = nn.Linear(d_model, rank, bias=False)
        self.W_K = nn.Linear(d_model, rank, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.high_order_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )

        self.gate_user = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_item = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_context = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5

    def forward(
        self,
        user_cross: torch.Tensor,
        item_cross: torch.Tensor,
        context_cross: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u_norm = self.norm_user(user_cross)
        i_norm = self.norm_item(item_cross)
        c_norm = self.norm_context(context_cross)

        joint_state = torch.cat([u_norm, i_norm, c_norm], dim=-1)
        router_weights = self.router(joint_state)
        mixed_basis = torch.einsum('bn,nrd->brd', router_weights, self.subspace_banks)

        Q_lr = self.W_Q(u_norm)
        K_lr = self.W_K(i_norm)
        V_hr = self.W_V(c_norm)

        attn_scores = torch.einsum('br,br->b', Q_lr, K_lr) * self.scale
        attn_weights = torch.sigmoid(attn_scores).unsqueeze(-1)
        fused = attn_weights * V_hr

        high_order = self.high_order_proj(fused)

        gate_u = self.gate_user(user_cross)
        gate_i = self.gate_item(item_cross)
        gate_c = self.gate_context(context_cross)

        user_b = gate_u * user_cross + (1 - gate_u) * self.dropout(high_order)
        item_b = gate_i * item_cross + (1 - gate_i) * self.dropout(high_order)
        context_b = gate_c * context_cross + (1 - gate_c) * high_order

        ns_matrix = torch.stack([user_b, item_b, context_b], dim=1)
        return user_b, item_b, context_b, ns_matrix


class NSTripletEnhancer(nn.Module):
    """NS Token 三元交互增强器。

    将 PCVRHyFormer 的统一 NS Token 按语义切分为 User/Item/Context，
    通过 ProgressiveLowRankNSInteraction 做高阶耦合，再用 Highway Gate 保护原始特征。
    """

    def __init__(
        self,
        d_model: int,
        num_user_ns: int,
        num_item_ns: int,
        num_context_ns: int,
        rank: int = 32,
        num_banks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_user_ns = num_user_ns
        self.num_item_ns = num_item_ns
        self.num_context_ns = num_context_ns
        self.total_ns = num_user_ns + num_item_ns + num_context_ns

        self.interaction = ProgressiveLowRankNSInteraction(
            d_model=d_model,
            rank=rank,
            num_banks=num_banks,
            dropout=dropout,
            layer_idx=0,
            pre_norm=True,
        )

        self.merge = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
        )

        # Highway Gate：保护原始 NS 全局信息
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ns_tokens: (B, total_ns, D)
        Returns:
            enhanced: (B, total_ns, D)
        """
        B, N, D = ns_tokens.shape

        # 1. 按语义分组并 Mean Pool
        u = ns_tokens[:, :self.num_user_ns].mean(dim=1)           # (B, D)
        i = ns_tokens[:, self.num_user_ns:self.num_user_ns + self.num_item_ns].mean(dim=1)
        c = ns_tokens[:, self.num_user_ns + self.num_item_ns:].mean(dim=1)

        # 2. HeteroFormer 三元低秩交互
        u_out, i_out, c_out, _ = self.interaction(u, i, c)

        # 3. 合并为统一增强向量
        fused = torch.cat([u_out, i_out, c_out], dim=-1)  # (B, 3D)
        enhanced = self.merge(fused)                       # (B, D)

        # 4. Highway Gate：惰性残差，防止高阶交互淹没原始特征
        ns_global = ns_tokens.mean(dim=1)                  # (B, D)
        gate = self.gate(ns_global)
        mixed = gate * ns_global + (1 - gate) * enhanced   # (B, D)

        # 5. 广播残差回原始 NS shape
        return ns_tokens + mixed.unsqueeze(1)              # (B, N, D)


class HeteroCrossAttention(nn.Module):
    """融合 PCVRHyFormer RoPE CrossAttn + HeteroFormer 显式 einsum 交互。

    在标准 Cross Attention 基础上，增加 NS-Sequence 显式点积交互分支和
    CausalConv1d 时序增强，强化非序列 token 与序列 token 的耦合。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_len: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        ln_mode: str = 'pre',
        use_hetero_branch: bool = True,
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode
        self.use_hetero_branch = use_hetero_branch

        # 标准 RoPE Cross Attention（PCVRHyFormer）
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ('pre', 'post'):
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

        # HeteroFormer 显式交互分支
        if use_hetero_branch:
            self.ns_proj = nn.Linear(d_model, d_model)
            self.seq_proj = nn.Linear(d_model, d_model)
            self.time_cnn = CausalConv1d(d_model, d_model, kernel_size, dilation=1)
            self.cnn_norm = nn.LayerNorm(d_model)
            self.hetero_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,        # (B, Nq, D)
        key_value: torch.Tensor,    # (B, L, D)
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = query

        if self.ln_mode == 'pre':
            q = self.norm_q(query)
            kv = self.norm_kv(key_value)
        else:
            q, kv = query, key_value

        # === 分支 A：标准 RoPE Cross Attention ===
        attn_out, _ = self.attn(
            query=q, key=kv, value=kv,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )

        # === 分支 B：HeteroFormer 显式 NS-Seq 交互 ===
        if self.use_hetero_branch:
            B, Nq, D = q.shape
            L = kv.shape[1]

            # NS 广播到序列长度
            q_broadcast = q.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, Nq, D)
            q_p = self.ns_proj(q_broadcast)                     # (B, L, Nq, D)
            s_p = self.seq_proj(kv).unsqueeze(2)                # (B, L, 1, D)

            # einsum 显式点积交互（HeteroFormer 原汁原味）
            interaction = torch.einsum('blnd,bld->bln', q_p, s_p.squeeze(2))
            interaction = F.silu(interaction)                   # (B, L, Nq)

            # 用交互权重聚合 NS 信息到序列侧
            weighted_ns = torch.einsum('bln,blnd->bld', interaction, q_p)  # (B, L, D)

            # CausalConv1d 时序增强
            cnn_out = self.time_cnn(weighted_ns)
            cnn_out = self.cnn_norm(cnn_out + weighted_ns)

            # 将序列侧增强信息按交互权重聚合回 Query 维度
            hetero_out = torch.einsum('bln,bld->bnd', interaction, cnn_out)  # (B, Nq, D)

            attn_out = attn_out + self.hetero_dropout(hetero_out)

        out = residual + attn_out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class HighwayRankMixerBlock(nn.Module):
    """RankMixerBlock + HeteroFormer Highway Gate。

    在 Token Mixing + FFN 后，用 Sigmoid 门控做惰性残差融合，
    保护原始 Query token 不被 FFN 变换完全覆盖。
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full',
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total}"
                )
            self.d_sub = d_model // n_total

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

        # HeteroFormer Highway Gate
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        B, T, D = Q.shape
        Q_split = Q.view(B, T, self.T, self.d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()
        return Q_rewired.view(B, T, D)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return Q

        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:
            Q_hat = Q

        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Highway Gate：保护原始 Q
        gate = self.gate(Q)
        Q_boost = gate * Q + (1 - gate) * Q_e

        return self.post_norm(Q_boost)


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer 基础设施（保留核心设计）
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
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

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(x, cos, sin):
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        return self.fc_out(x)


class RoPEMultiheadAttention(nn.Module):
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
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
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=sdpa_attn_mask, dropout_p=dropout_p,
        )
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)
        return out, None


class CrossAttention(nn.Module):
    """标准 Cross Attention（保留作为 fallback）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode
        self.attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, rope_on_q=False,
        )
        if ln_mode in ('pre', 'post'):
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = query
        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)
        out, _ = self.attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )
        out = residual + out
        if self.ln_mode == 'post':
            out = self.norm_q(out)
        return out


class RankMixerBlock(nn.Module):
    """原始 RankMixerBlock（保留）。"""

    def __init__(
        self,
        d_model: int,
        n_total: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode
        if mode == 'none':
            return
        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(f"d_model={d_model} must be divisible by T={n_total}")
            self.d_sub = d_model // n_total
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        B, T, D = Q.shape
        Q_split = Q.view(B, T, self.T, self.d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()
        return Q_rewired.view(B, T, D)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return Q
        Q_hat = self.token_mixing(Q) if self.mode == 'full' else Q
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)
        Q_boost = Q + Q_e
        return self.post_norm(Q_boost)


class MultiSeqQueryGenerator(nn.Module):
    """多序列 Query 生成器（完全保留）。"""

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        global_info_dim = (num_ns + 1) * d_model
        self.global_info_norm = nn.LayerNorm(global_info_dim)
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                ) for _ in range(num_queries)
            ]) for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)
        q_tokens_list = []
        for i in range(self.num_sequences):
            valid_mask = ~seq_padding_masks[i]
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)
            seq_pooled = seq_sum / seq_count
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)
            q_tokens_list.append(q_tokens)
        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders（完全保留）
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLUEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, **kwargs):
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, rope_on_q=True,
        )
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
        )
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x, key_padding_mask


class LongerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, rope_on_q=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout)
        )

    def _gather_top_k(self, x, key_padding_mask):
        B, L, D = x.shape
        device = x.device
        valid_len = (~key_padding_mask).sum(dim=1)
        actual_k = torch.clamp(valid_len, max=self.top_k)
        start_pos = valid_len - actual_k
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        indices = start_pos.unsqueeze(1) + offsets
        indices = torch.clamp(indices, min=0, max=L - 1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)
        new_valid_len = actual_k
        pad_count = self.top_k - new_valid_len
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()
        return top_k_tokens, new_padding_mask, indices

    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        B, L, D = x.shape
        if L > self.top_k:
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                head_dim = rope_cos.shape[2]
                cos_expanded = rope_cos.expand(B, -1, -1)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)
            attn_out, _ = self.attn(
                query=q_normed, key=kv_normed, value=kv_normed,
                key_padding_mask=key_padding_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
                q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin,
            )
            out = q + attn_out
        else:
            new_mask = key_padding_mask
            x_normed = self.norm_q(x)
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
            attn_out, _ = self.attn(
                query=x_normed, key=x_normed, value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )
            out = x + attn_out

        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out
        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# MultiSeqHyFormerBlock（融合版：使用 HeteroCrossAttention + HighwayRankMixer）
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSeqHyFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full',
        use_hetero_cross_attn: bool = True,
        hetero_kernel_size: int = 3,
        use_highway_mixer: bool = True,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            ) for _ in range(num_sequences)
        ])

        # 融合点：选择 CrossAttention 或 HeteroCrossAttention
        cross_attn_cls = HeteroCrossAttention if use_hetero_cross_attn else CrossAttention
        self.cross_attns = nn.ModuleList([
            cross_attn_cls(
                d_model=d_model,
                num_heads=num_heads,
                seq_len=top_k if seq_encoder_type == 'longer' else 512,
                kernel_size=hetero_kernel_size,
                dropout=dropout,
                ln_mode='pre',
                use_hetero_branch=use_hetero_cross_attn,
            ) for _ in range(num_sequences)
        ])

        # 融合点：选择 RankMixerBlock 或 HighwayRankMixerBlock
        n_total = num_queries * num_sequences + num_ns
        mixer_cls = HighwayRankMixerBlock if use_highway_mixer else RankMixerBlock
        self.mixer = mixer_cls(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        S = self.num_sequences
        Nq = self.num_queries

        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)
        boosted = self.mixer(combined)

        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizers（完全保留）
# ═══════════════════════════════════════════════════════════════════════════════

class GroupNSTokenizer(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            ) for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class RankMixerNSTokenizer(nn.Module):
    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        chunks = cat_emb.split(self.chunk_dim, dim=-1)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 主模型：PCVRHyFormer-Hetero
# ═══════════════════════════════════════════════════════════════════════════════

class PCVRHyFormerHetero(nn.Module):
    """融合版模型：PCVRHyFormer 底盘 + HeteroFormer 交互引擎。

    新增超参数（相比原版 PCVRHyFormer）：
        - ns_interaction_rank: 三元低秩交互的秩（默认 32）
        - ns_interaction_banks: 子空间银行数（默认 4）
        - use_ns_triplet_enhancer: 是否启用 NS 三元交互增强
        - use_hetero_cross_attn: 是否用 HeteroCrossAttention 替换标准 CrossAttn
        - hetero_kernel_size: Hetero 分支的因果卷积核大小
        - use_highway_mixer: 是否用 HighwayRankMixerBlock 替换 RankMixerBlock
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",
        # NS grouping config
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        # === HeteroFormer 融合参数 ===
        ns_interaction_rank: int = 32,
        ns_interaction_banks: int = 4,
        use_ns_triplet_enhancer: bool = True,
        use_hetero_cross_attn: bool = True,
        hetero_kernel_size: int = 3,
        use_highway_mixer: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type

        # ================== NS Tokenizers ==================
        if ns_tokenizer_type == 'group':
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)
            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens
            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        self.num_user_ns = num_user_ns
        self.num_item_ns = num_item_ns

        # Dense projections
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        self.num_context_ns = int(self.has_user_dense) + int(self.has_item_dense)
        self.num_ns = num_user_ns + num_item_ns + self.num_context_ns

        # === 融合点 1：NS 三元交互增强器 ===
        self.use_ns_triplet_enhancer = use_ns_triplet_enhancer
        if use_ns_triplet_enhancer:
            self.ns_enhancer = NSTripletEnhancer(
                d_model=d_model,
                num_user_ns=num_user_ns,
                num_item_ns=num_item_ns,
                num_context_ns=self.num_context_ns,
                rank=ns_interaction_rank,
                num_banks=ns_interaction_banks,
                dropout=dropout_rate,
            )

        # ================== Check d_model % T ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T={T}. "
                f"Valid T values: {valid_T_values}"
            )

        # ================== Seq Embeddings ==================
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_is_id = {}
        self._seq_vocab_sizes = {}
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Embedding ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # === 融合点 2 & 3：Block 中使用 HeteroCrossAttention + HighwayRankMixer ===
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
                use_hetero_cross_attn=use_hetero_cross_attn,
                hetero_kernel_size=hetero_kernel_size,
                use_highway_mixer=use_highway_mixer,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.emb_dropout = nn.Dropout(dropout_rate)

        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        self._init_params()

        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> "set[int]":
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = F.gelu(proj(cat_emb))
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)
        output = all_q.view(B, -1)
        output = self.output_proj(output)
        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        # 1. NS tokens
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # === 融合点 1：NS 三元交互增强 ===
        if self.use_ns_triplet_enhancer:
            ns_tokens = self.ns_enhancer(ns_tokens)

        # 2. Embed sequences
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate queries
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        # 4. Run blocks
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training
        )

        # 5. Classifier
        logits = self.clsfier(output)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference mode: no dropout, returns (logits, embeddings)."""
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)
        if self.use_ns_triplet_enhancer:
            ns_tokens = self.ns_enhancer(ns_tokens)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
