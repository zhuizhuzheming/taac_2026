import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

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
    """
    你的步骤(2)：softmax(X*W/√d)*Cross
    显式建模User/Item/Context与Cross Feature的交互权重
    """
    def __init__(self, d_model: int, num_fields: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields  # User, Item, Context
        
        # X = [User, Item, Context]，学习各自的交叉权重
        self.W_attn = nn.Linear(d_model * num_fields, num_fields, bias=False)
        self.scale = d_model ** -0.5
        
    def forward(self, fields: List[torch.Tensor], cross_feat: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            fields: [User, Item, Context] 每个都是 [B, d]
            cross_feat: [B, d] Cross Feature
        Returns:
            crossed_fields: [User_Cross, Item_Cross, Context_Cross]
        """
        # X = concat(User, Item, Context)
        X = torch.cat(fields, dim=-1)  # [B, 3*d]
        
        # 学习各字段与Cross的交互权重
        attn_logits = self.W_attn(X) * self.scale  # [B, 3]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, 3]
        
        # 加权分配Cross Feature
        crossed_fields = []
        for i, field in enumerate(fields):
            # field + weight_i * cross_feat（残差式）
            crossed = field + attn_weights[:, i:i+1] * cross_feat
            crossed_fields.append(crossed)
            
        return crossed_fields, attn_weights


class ProgressiveLowRankNSInteraction(nn.Module):
    """
    你的步骤(4)升级：Q=User, K=Item, V=Context 的三元交互
    但用渐进低秩替代SVD，支持多层堆叠
    
    设计：
    - 每层秩不同（渐进递减）
    - 动态子空间路由（条件化）
    - 输出User-b/Item-b/Context-b用于下一层
    """
    def __init__(
        self,
        d_model: int,
        rank: int,  # 当前层秩（渐进）
        num_banks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # 可学习低秩子空间（替代SVD的静态分解）
        self.subspace_banks = nn.Parameter(
            torch.randn(num_banks, rank, d_model) * 0.02
        )
        
        # 基于当前NS状态路由选择子空间
        self.router = nn.Sequential(
            nn.Linear(d_model * 3, num_banks),  # 基于User+Item+Context联合路由
            nn.Softmax(dim=-1)
        )
        
        # 三元交互投影（你的Q=User, K=Item, V=Context思想）
        self.W_Q = nn.Linear(d_model, rank)   # User投影到查询
        self.W_K = nn.Linear(d_model, rank)   # Item投影到键
        self.W_V = nn.Linear(d_model, d_model)  # Context投影到值（保持维度）
        
        # 高阶表示学习（你的"w学习高阶层表示"）
        self.high_order_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = rank ** -0.5
        
    def forward(
        self,
        user_cross: torch.Tensor,   # [B, d] 来自步骤2
        item_cross: torch.Tensor,   # [B, d]
        context_cross: torch.Tensor # [B, d]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_b, item_b, context_b: 用于下一层的残差输入 [B, d]
            ns_matrix: NS Feature大矩阵 [B, 3, d]（用于后续Mix-all_reduce）
        """
        # 联合路由条件 [B, num_banks]
        joint_state = torch.cat([user_cross, item_cross, context_cross], dim=-1)
        router_weights = self.router(joint_state)
        
        # 动态混合子空间 [B, rank, d]
        mixed_basis = torch.einsum('bn,nrd->brd', router_weights, self.subspace_banks)
        
        # 三元交互在低秩空间（你的Q=User, K=Item, V=Context）
        Q_lr = self.W_Q(user_cross)      # [B, rank]
        K_lr = self.W_K(item_cross)      # [B, rank]
        V_hr = self.W_V(context_cross)   # [B, d]（值保持高维）
        
        # 低秩空间注意力（复杂度O(B*rank) vs O(B*d)）
        attn_scores = torch.einsum('br,br->b', Q_lr, K_lr) * self.scale  # [B]
        attn_weights = torch.sigmoid(attn_scores).unsqueeze(-1)  # [B, 1]
        
        # 加权融合：Context作为值，被User-Item相关性调制
        fused = attn_weights * V_hr  # [B, d]
        
        # 高阶层表示学习（你的"w学习"）
        high_order = self.high_order_proj(fused)
        
        # 残差输出（用于下一层）
        user_b = user_cross + self.dropout(high_order * 0.3)   # User侧轻量更新
        item_b = item_cross + self.dropout(high_order * 0.3)   # Item侧轻量更新
        context_b = context_cross + high_order                  # Context侧主要更新
        
        # NS Feature大矩阵（用于Mix-all_reduce）
        ns_matrix = torch.stack([user_b, item_b, context_b], dim=1)  # [B, 3, d]
        
        return user_b, item_b, context_b, ns_matrix


class MixAllReduce(nn.Module):
    """
    你的步骤(4)：Mix-all_reduce得到NS Feature大矩阵
    将User-b/Item-b/Context-b聚合成统一的NS表示
    
    升级：支持生成多个Global Tokens，并应用ROPE
    """
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
        
        # 从NS矩阵聚合为Global Tokens
        self.reduce_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model * num_global_tokens),
            nn.Unflatten(-1, (num_global_tokens, d_model))
        )
        
        # 混合增强（你的all-reduce思想）
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
        """
        Args:
            ns_matrix: [B, 3, d] User-b/Item-b/Context-b
        Returns:
            global_tokens: [B, num_global_tokens, d] 用于HeteroAttention
        """
        B = ns_matrix.size(0)
        
        # 展平并投影为多个Global Tokens
        flat = ns_matrix.flatten(1)  # [B, 3*d]
        global_tokens = self.reduce_proj(flat)  # [B, num_tokens, d]
        
        # 混合增强（all-reduce思想：每个token融合所有信息）
        mixed = self.mix_enhance(global_tokens)
        
        # ROPE位置编码
        if self.use_rope:
            mixed = self._apply_rope(mixed)
            
        return mixed

class HeteroAttention(nn.Module):
    """
    你的步骤(4)核心：NS Feature与Seq Feature的显式矩阵交互
    
    设计：
    - NS Feature广播后与Seq Feature点积
    - Time-CNN聚合
    - 双轨输出（0行给Seq，1行给NS）
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # NS→Seq广播点积前的投影
        self.ns_proj = nn.Linear(d_model, d_model)
        self.seq_proj = nn.Linear(d_model, d_model)
        
        # Time-CNN（你的聚合挖掘）
        self.time_cnn = CausalConv1d(d_model, d_model, kernel_size, dilation=1)
        self.cnn_norm = nn.LayerNorm(d_model)
        
        # Self-Attention聚合
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # 双轨输出投影
        # 0行：Seq输出（用于序列侧更新）
        # 1行：NS输出（用于NS侧更新）
        self.seq_output_proj = nn.Linear(d_model, d_model)
        self.ns_output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        ns_global: torch.Tensor,  # [B, num_tokens, d] 来自MixAllReduce
        seq_feat: torch.Tensor    # [B, L, d] 序列特征
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seq_out: [B, L, d] 用于序列侧（你的0行）
            ns_out:  [B, d] 聚合后的NS表示（你的1行）
        """
        B, num_tokens, d = ns_global.shape
        L = seq_feat.size(1)
        
        # NS广播：扩展为与Seq相同长度 [B, L, num_tokens, d]
        ns_broadcast = ns_global.unsqueeze(1).expand(-1, L, -1, -1)
        
        # 投影
        ns_proj = self.ns_proj(ns_broadcast)      # [B, L, num_tokens, d]
        seq_proj = self.seq_proj(seq_feat).unsqueeze(2)  # [B, L, 1, d]
        
        # 显式点积交互（你的核心操作）
        # 每一行NS与Seq点积，得到交互矩阵 [B, L, num_tokens]
        interaction = torch.einsum('blnd,bld->bln', ns_proj, seq_proj.squeeze(2))
        interaction = F.silu(interaction)  # 非线性激活
        
        # 用交互权重加权NS特征 [B, L, d]
        weighted_ns = torch.einsum('bln,blnd->bld', interaction, ns_proj)
        
        # 与原始Seq融合（残差）
        fused = seq_feat + self.dropout(weighted_ns)
        
        # Time-CNN聚合（你的time-CNN）
        cnn_out = self.time_cnn(fused)
        cnn_out = self.cnn_norm(cnn_out + fused)
        
        # Self-Attention聚合
        attn_out, _ = self.self_attn(cnn_out, cnn_out, cnn_out)
        attn_out = self.attn_norm(attn_out + cnn_out)
        
        # 双轨输出
        # 0行：Seq输出（最后时刻或平均池化）
        seq_out = self.seq_output_proj(attn_out)  # [B, L, d]
        
        # 1行：NS输出（聚合所有token）
        ns_out = self.ns_output_proj(ns_global.mean(dim=1))  # [B, d]
        # 或者更复杂的聚合：用attention权重聚合
        
        return seq_out, ns_out


# ==============================================================================
# 完整的可堆叠Block（融合你的所有步骤）
# ==============================================================================

class HeteroFormerBlock(nn.Module):
    """
    你的完整流程的可堆叠版本：
    
    输入: User_cross, Item_cross, Context_cross, Seq_feat
         ↓
    [1] ProgressiveLowRankNSInteraction（替代SVD）
         ↓
    输出: User_b, Item_b, Context_b（残差连接用）
         ↓
    [2] MixAllReduce（你的Mix-all_reduce）
         ↓
    输出: NS Global Tokens
         ↓
    [3] HeteroAttention（你的显式矩阵交互）
         ↓
    输出: Seq_out（0行）, NS_out（1行）
         ↓
    [4] SwiGLU + LayerNorm（你的最后一步）
         ↓
    返回: User_b, Item_b, Context_b, Seq_b, NS_b（全部用于下一层残差）
    """
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        rank: int,  # 渐进低秩
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: int = 4,
        num_banks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        
        # 步骤[1]：渐进低秩NS交互（替代SVD）
        self.ns_interaction = ProgressiveLowRankNSInteraction(
            d_model, rank, num_banks, dropout
        )
        
        # 步骤[2]：Mix-all_reduce
        self.mix_reduce = MixAllReduce(
            d_model, num_global_tokens, use_rope=True, dropout=dropout
        )
        
        # 步骤[3]：HeteroAttention
        self.hetero_attn = HeteroAttention(
            d_model, seq_len, kernel_size, num_heads, dropout
        )
        
        # 步骤[4]：SwiGLU + Norm
        self.swiglu_seq = SwiGLU(d_model, expand_ratio=2.0)
        self.swiglu_ns = SwiGLU(d_model, expand_ratio=2.0)
        self.norm_seq = nn.LayerNorm(d_model)
        self.norm_ns = nn.LayerNorm(d_model)
        
        # 输出投影（将NS_out映射回3个字段用于下一层残差）
        self.ns_split_proj = nn.Linear(d_model, d_model * 3)
        
    def forward(
        self,
        user_cross: torch.Tensor,   # [B, d] 来自上层或初始
        item_cross: torch.Tensor,   # [B, d]
        context_cross: torch.Tensor, # [B, d]
        seq_feat: torch.Tensor,      # [B, L, d]
        # 残差输入（可选）
        user_residual: Optional[torch.Tensor] = None,
        item_residual: Optional[torch.Tensor] = None,
        context_residual: Optional[torch.Tensor] = None,
        seq_residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        完整前向，返回所有输出供下一层残差使用
        """
        # 应用残差（你的步骤5）
        u_in = user_cross if user_residual is None else user_cross + user_residual
        i_in = item_cross if item_residual is None else item_cross + item_residual
        c_in = context_cross if context_residual is None else context_cross + context_residual
        s_in = seq_feat if seq_residual is None else seq_feat + seq_residual
        
        # [1] 渐进低秩NS交互（替代SVD，支持条件化）
        user_b, item_b, context_b, ns_matrix = self.ns_interaction(u_in, i_in, c_in)
        
        # [2] Mix-all_reduce → Global Tokens
        ns_global = self.mix_reduce(ns_matrix)  # [B, num_tokens, d]
        
        # [3] HeteroAttention：NS ↔ Seq显式交互
        seq_out, ns_out = self.hetero_attn(ns_global, s_in)  # 0行, 1行
        
        # [4] SwiGLU + Norm
        seq_b = self.norm_seq(s_in + self.swiglu_seq(seq_out))
        ns_b = self.norm_ns(ns_out + self.swiglu_ns(ns_out))  # ns_out是1行输出
        
        # 将ns_b映射回3个字段用于下一层残差
        ns_split = self.ns_split_proj(ns_b).chunk(3, dim=-1)
        
        return {
            'user_b': user_b,      # 用于下一层
            'item_b': item_b,
            'context_b': context_b,
            'seq_b': seq_b,        # 0行输出
            'ns_b': ns_b,          # 1行输出（聚合NS）
            'ns_split': ns_split,  # 分解后的3个字段
        }

class HeteroFormer(nn.Module):
    """
    你的完整架构的终极版本：
    - 保留所有步骤语义
    - 渐进低秩替代SVD
    - 支持任意层数堆叠（Scaling Law）
    """
    def __init__(
        self,
        d_model: int = 64,
        seq_len: int = 50,
        num_layers: int = 4,        # 可配置！
        base_rank: int = 64,       # 第一层秩
        rank_decay: float = 0.5,   # 秩衰减
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: int = 4,
        num_banks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 计算渐进秩
        self.ranks = [
            max(8, int(base_rank * (rank_decay ** i)))
            for i in range(num_layers)
        ]
        print(f"[HeteroFormerV2] Progressive ranks: {self.ranks}")
        
        # 输入投影（保留）
        self.user_proj = nn.Linear(d_model, d_model, bias=False)
        self.item_proj = nn.Linear(d_model, d_model, bias=False)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        self.seq_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 初始交叉特征交互（你的步骤2）
        self.init_cross_attn = ScaledDotCrossAttention(d_model, num_fields=3)
        
        # 可堆叠的HeteroFormer Blocks（每层不同秩）
        self.blocks = nn.ModuleList([
            HeteroFormerBlock(
                d_model=d_model,
                seq_len=seq_len,
                rank=self.ranks[i],
                num_global_tokens=num_global_tokens,
                kernel_size=kernel_size if i == 0 else 3,  # 仅第一层用指定kernel
                num_heads=num_heads,
                num_banks=num_banks,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # 最终预测（简化版）
        self.final_norm = nn.LayerNorm(d_model)
        self.final_ffn = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),  # 4 = 3个NS字段 + 1个Seq聚合
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1)
        )
        
    def forward(
        self,
        seq_feat: torch.Tensor,
        user_feat: torch.Tensor,
        item_feat: torch.Tensor,
        context_feat: torch.Tensor,
        cross_feat: torch.Tensor
    ) -> torch.Tensor:
        B = seq_feat.size(0)
        
        # 输入投影
        u = self.user_proj(user_feat)
        i = self.item_proj(item_feat)
        c = self.context_proj(context_feat)
        s = self.seq_proj(seq_feat)
        cross = self.cross_proj(cross_feat)
        
        # 步骤2：初始交叉特征交互
        crossed_fields, cross_weights = self.init_cross_attn([u, i, c], cross)
        u_cross, i_cross, c_cross = crossed_fields
        
        # 分层堆叠（支持残差）
        residuals = {
            'user': None, 'item': None, 'context': None, 'seq': None
        }
        
        for block_idx, block in enumerate(self.blocks):
            outputs = block(
                u_cross, i_cross, c_cross, s,
                residuals['user'], residuals['item'], 
                residuals['context'], residuals['seq']
            )
            
            # 更新残差（你的步骤5扩展）
            residuals = {
                'user': outputs['user_b'],
                'item': outputs['item_b'],
                'context': outputs['context_b'],
                'seq': outputs['seq_b']
            }
            
            # 为下一层准备输入
            u_cross, i_cross, c_cross = outputs['ns_split']
            s = outputs['seq_b']
        
        # 最终预测
        final_ns = torch.stack([outputs['user_b'], outputs['item_b'], outputs['context_b']], dim=1)
        final_seq = outputs['seq_b'].mean(dim=1)  # 聚合序列
        
        final_repr = torch.cat([
            final_ns.flatten(1),  # [B, 3*d]
            final_seq               # [B, d]
        ], dim=-1)
        
        logit = self.final_ffn(final_repr)
        return torch.sigmoid(logit)
