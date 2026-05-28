import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

# ==============================================================================
# 基础组件（保持原设计，仅微调初始化）
# ==============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=self.padding,
                              dilation=dilation, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        return out.transpose(1, 2)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(x * F.silu(gate))


class ScaledDotCrossAttention(nn.Module):
    """原步骤(2)：softmax(X*W/√d)*Cross，未改动"""
    def __init__(self, d_model: int, num_fields: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.W_attn = nn.Linear(d_model * num_fields, num_fields, bias=False)
        self.scale = d_model ** -0.5

    def forward(self, fields: List[torch.Tensor], cross_feat: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        X = torch.cat(fields, dim=-1)
        attn_logits = self.W_attn(X) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        crossed_fields = []
        for i, field in enumerate(fields):
            crossed = field + attn_weights[:, i:i+1] * cross_feat
            crossed_fields.append(crossed)
        return crossed_fields, attn_weights


# ==============================================================================
# 核心改进 1: ProgressiveLowRankNSInteraction —— 支持 Scaling 的版本
# ==============================================================================

class ProgressiveLowRankNSInteraction(nn.Module):
    """
    改进点：
    1. 引入惰性残差门控 (Highway Gate) 保护原始特征独立性
    2. 对低秩投影使用缩放初始化 (随层深衰减初始化方差)
    3. Pre-Norm 结构 (LayerNorm 在子层前)
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        num_banks: int = 4,
        dropout: float = 0.1,
        layer_idx: int = 0,          # 新增：用于控制初始化强度
        pre_norm: bool = True         # 新增：Pre-Norm
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.pre_norm = pre_norm

        # LayerNorm (Pre-Norm)
        self.norm_user = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.norm_item = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.norm_context = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        # 子空间银行（初始化方差随层深衰减，避免深层信号爆炸）
        init_std = 0.02 / (layer_idx + 1)
        self.subspace_banks = nn.Parameter(
            torch.randn(num_banks, rank, d_model) * init_std
        )

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(d_model * 3, num_banks),
            nn.Softmax(dim=-1)
        )

        # 三元投影
        self.W_Q = nn.Linear(d_model, rank, bias=False)
        self.W_K = nn.Linear(d_model, rank, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # 高阶表示学习
        self.high_order_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )

        # 新增：惰性残差门控 (Highway Gate)
        self.gate_user = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.gate_item = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.gate_context = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5

    def forward(
        self,
        user_cross: torch.Tensor,
        item_cross: torch.Tensor,
        context_cross: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pre-Norm
        u_norm = self.norm_user(user_cross)
        i_norm = self.norm_item(item_cross)
        c_norm = self.norm_context(context_cross)

        # 路由选择子空间
        joint_state = torch.cat([u_norm, i_norm, c_norm], dim=-1)
        router_weights = self.router(joint_state)
        mixed_basis = torch.einsum('bn,nrd->brd', router_weights, self.subspace_banks)

        # 三元交互
        Q_lr = self.W_Q(u_norm)
        K_lr = self.W_K(i_norm)
        V_hr = self.W_V(c_norm)

        attn_scores = torch.einsum('br,br->b', Q_lr, K_lr) * self.scale
        attn_weights = torch.sigmoid(attn_scores).unsqueeze(-1)
        fused = attn_weights * V_hr

        high_order = self.high_order_proj(fused)

        # 改进：惰性残差门控 —— 保护原始特征不被低秩交互完全覆盖
        gate_u = self.gate_user(user_cross)
        gate_i = self.gate_item(item_cross)
        gate_c = self.gate_context(context_cross)

        user_b = gate_u * user_cross + (1 - gate_u) * self.dropout(high_order)
        item_b = gate_i * item_cross + (1 - gate_i) * self.dropout(high_order)
        context_b = gate_c * context_cross + (1 - gate_c) * high_order

        ns_matrix = torch.stack([user_b, item_b, context_b], dim=1)
        return user_b, item_b, context_b, ns_matrix


# ==============================================================================
# MixAllReduce 和 HeteroAttention（保持核心设计，微调超参数自适应）
# ==============================================================================

class MixAllReduce(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_global_tokens: int = 4,
        use_rope: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_global_tokens = num_global_tokens
        self.use_rope = use_rope

        self.reduce_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * num_global_tokens),
            nn.Unflatten(-1, (num_global_tokens, d_model))
        )

        self.mix_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        if use_rope:
            self.register_buffer("rope_freqs", None)

    def _apply_rope(self, x):
        if not self.use_rope or self.rope_freqs is None:
            return x
        B, N, D = x.shape
        if self.rope_freqs is None:
            theta = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device) / D))
            self.rope_freqs = theta
        pos = torch.arange(N, device=x.device)
        freqs = torch.outer(pos, self.rope_freqs)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos, sin = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, ns_matrix: torch.Tensor) -> torch.Tensor:
        B = ns_matrix.size(0)
        flat = ns_matrix.flatten(1)
        global_tokens = self.reduce_proj(flat)
        mixed = self.mix_enhance(global_tokens)
        if self.use_rope:
            mixed = self._apply_rope(mixed)
        return mixed


class HeteroAttention(nn.Module):
    """
    核心交互保留：einsum 显式点积。
    改进：Pre-Norm，头数随 d_model 自适应（在外部配置）。
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pre_norm = pre_norm

        self.norm_ns = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.norm_seq = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

        self.ns_proj = nn.Linear(d_model, d_model)
        self.seq_proj = nn.Linear(d_model, d_model)

        self.time_cnn = CausalConv1d(d_model, d_model, kernel_size, dilation=1)
        self.cnn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.seq_output_proj = nn.Linear(d_model, d_model)
        self.ns_output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ns_global: torch.Tensor,
        seq_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm
        ns_norm = self.norm_ns(ns_global)
        seq_norm = self.norm_seq(seq_feat)

        B, num_tokens, d = ns_norm.shape
        L = seq_norm.size(1)

        ns_broadcast = ns_norm.unsqueeze(1).expand(-1, L, -1, -1)
        ns_proj = self.ns_proj(ns_broadcast)
        seq_proj = self.seq_proj(seq_norm).unsqueeze(2)

        # 显式点积交互（原汁原味）
        interaction = torch.einsum('blnd,bld->bln', ns_proj, seq_proj.squeeze(2))
        interaction = F.silu(interaction)

        weighted_ns = torch.einsum('bln,blnd->bld', interaction, ns_proj)
        fused = seq_feat + self.dropout(weighted_ns)  # 残差加在原始 seq_feat 上

        cnn_out = self.time_cnn(fused)
        cnn_out = self.cnn_norm(cnn_out + fused)

        attn_out, _ = self.self_attn(cnn_out, cnn_out, cnn_out)
        attn_out = self.attn_norm(attn_out + cnn_out)

        seq_out = self.seq_output_proj(attn_out)
        ns_out = self.ns_output_proj(ns_global.mean(dim=1))

        return seq_out, ns_out


# ==============================================================================
# 改进后的 HeteroFormerBlock：支持 Pre-Norm 和层索引传递
# ==============================================================================

class HeteroFormerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        rank: int,
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: int = 4,
        num_banks: int = 4,
        dropout: float = 0.1,
        layer_idx: int = 0,
        pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # 渐进低秩交互（带层索引）
        self.ns_interaction = ProgressiveLowRankNSInteraction(
            d_model, rank, num_banks, dropout, layer_idx, pre_norm
        )

        self.mix_reduce = MixAllReduce(
            d_model, num_global_tokens, use_rope=True, dropout=dropout
        )

        self.hetero_attn = HeteroAttention(
            d_model, seq_len, kernel_size, num_heads, dropout, pre_norm
        )

        # SwiGLU + Norm（Pre-Norm 下这里改为 Post-Norm 形式，但保持简单）
        self.swiglu_seq = SwiGLU(d_model, expand_ratio=2.0)
        self.swiglu_ns = SwiGLU(d_model, expand_ratio=2.0)
        self.norm_seq_out = nn.LayerNorm(d_model)
        self.norm_ns_out = nn.LayerNorm(d_model)

        self.ns_split_proj = nn.Linear(d_model, d_model * 3)

    def forward(
        self,
        user_cross: torch.Tensor,
        item_cross: torch.Tensor,
        context_cross: torch.Tensor,
        seq_feat: torch.Tensor,
        user_residual: Optional[torch.Tensor] = None,
        item_residual: Optional[torch.Tensor] = None,
        context_residual: Optional[torch.Tensor] = None,
        seq_residual: Optional[torch.Tensor] = None
    ):
        # 应用外部残差（来自上一层）
        u_in = user_cross if user_residual is None else user_cross + user_residual
        i_in = item_cross if item_residual is None else item_cross + item_residual
        c_in = context_cross if context_residual is None else context_cross + context_residual
        s_in = seq_feat if seq_residual is None else seq_feat + seq_residual

        user_b, item_b, context_b, ns_matrix = self.ns_interaction(u_in, i_in, c_in)
        ns_global = self.mix_reduce(ns_matrix)
        seq_out, ns_out = self.hetero_attn(ns_global, s_in)

        # 最终残差 + SwiGLU
        seq_b = self.norm_seq_out(s_in + self.swiglu_seq(seq_out))
        ns_b = self.norm_ns_out(ns_out + self.swiglu_ns(ns_out))

        ns_split = self.ns_split_proj(ns_b).chunk(3, dim=-1)

        return {
            'user_b': user_b,
            'item_b': item_b,
            'context_b': context_b,
            'seq_b': seq_b,
            'ns_b': ns_b,
            'ns_split': ns_split,
        }


# ==============================================================================
# 主模型：支持 Scaling 的 HeteroFormer
# ==============================================================================

class HeteroFormer(nn.Module):
    """
    改进后的完整模型，支持深度与宽度扩展。
    
    秩调度策略：采用瓶颈式（U型）或温和衰减，确保深层秩不低于 d_model/4。
    超参数自适应：num_heads、num_banks 根据 d_model 自动缩放。
    """
    def __init__(
        self,
        d_model: int = 64,
        seq_len: int = 50,
        num_layers: int = 4,
        base_rank: Optional[int] = None,
        rank_schedule: str = 'bottleneck',   # 'bottleneck', 'gentle', 'constant'
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: Optional[int] = None,
        num_banks: Optional[int] = None,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # 自动计算超参数
        if num_heads is None:
            num_heads = max(4, d_model // 32)
        if num_banks is None:
            num_banks = max(4, d_model // 16)
        if base_rank is None:
            base_rank = max(16, d_model // 2)

        # 生成各层的秩
        self.ranks = self._generate_ranks(base_rank, num_layers, rank_schedule)
        print(f"[HeteroFormerScaled] d_model={d_model}, layers={num_layers}")
        print(f"[HeteroFormerScaled] Rank schedule: {self.ranks}")
        print(f"[HeteroFormerScaled] num_heads={num_heads}, num_banks={num_banks}")

        # 输入投影
        self.user_proj = nn.Linear(d_model, d_model, bias=False)
        self.item_proj = nn.Linear(d_model, d_model, bias=False)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        self.seq_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_proj = nn.Linear(d_model, d_model, bias=False)

        self.init_cross_attn = ScaledDotCrossAttention(d_model, num_fields=3)

        # 堆叠 Block
        self.blocks = nn.ModuleList([
            HeteroFormerBlock(
                d_model=d_model,
                seq_len=seq_len,
                rank=self.ranks[i],
                num_global_tokens=num_global_tokens,
                kernel_size=kernel_size if i == 0 else 3,
                num_heads=num_heads,
                num_banks=num_banks,
                dropout=dropout,
                layer_idx=i,
                pre_norm=pre_norm
            )
            for i in range(num_layers)
        ])

        # 最终预测头
        self.final_norm = nn.LayerNorm(d_model)
        self.final_ffn = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1)
        )

    def _generate_ranks(self, base_rank: int, num_layers: int, schedule: str) -> List[int]:
        ranks = []
        for i in range(num_layers):
            if schedule == 'constant':
                r = base_rank
            elif schedule == 'gentle':
                r = max(16, int(base_rank / (1 + 0.15 * i)))
            elif schedule == 'bottleneck':
                # U型：首尾高，中间低
                mid = num_layers // 2
                if i <= mid:
                    r = max(16, base_rank - i * (base_rank // (mid+1)))
                else:
                    r = max(16, base_rank - (num_layers-1-i) * (base_rank // (mid+1)))
            else:
                r = base_rank
            ranks.append(r)
        return ranks

    def forward(
        self,
        seq_feat: torch.Tensor,
        user_feat: torch.Tensor,
        item_feat: torch.Tensor,
        context_feat: torch.Tensor,
        cross_feat: torch.Tensor
    ) -> torch.Tensor:
        B = seq_feat.size(0)

        u = self.user_proj(user_feat)
        i = self.item_proj(item_feat)
        c = self.context_proj(context_feat)
        s = self.seq_proj(seq_feat)
        cross = self.cross_proj(cross_feat)

        crossed_fields, _ = self.init_cross_attn([u, i, c], cross)
        u_cross, i_cross, c_cross = crossed_fields

        residuals = {'user': None, 'item': None, 'context': None, 'seq': None}

        for block in self.blocks:
            outputs = block(
                u_cross, i_cross, c_cross, s,
                residuals['user'], residuals['item'],
                residuals['context'], residuals['seq']
            )
            residuals = {
                'user': outputs['user_b'],
                'item': outputs['item_b'],
                'context': outputs['context_b'],
                'seq': outputs['seq_b']
            }
            u_cross, i_cross, c_cross = outputs['ns_split']
            s = outputs['seq_b']

        final_ns = torch.stack([outputs['user_b'], outputs['item_b'], outputs['context_b']], dim=1)
        final_seq = outputs['seq_b'].mean(dim=1)

        final_repr = torch.cat([final_ns.flatten(1), final_seq], dim=-1)
        logit = self.final_ffn(final_repr)
        return torch.sigmoid(logit)