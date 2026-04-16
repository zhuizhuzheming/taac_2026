"""
OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer
论文实现 (WWW 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] or [B, D]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


class MixedCausalAttention(nn.Module):
    """
    混合因果注意力: S-tokens共享QKV参数，NS-tokens使用token-specific参数
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_ns_tokens: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_ns_tokens = num_ns_tokens
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # S-tokens共享的QKV参数
        self.qkv_s = nn.Linear(dim, 3 * dim, bias=False)
        
        # NS-tokens的token-specific QKV参数
        self.qkv_ns = nn.ModuleList([
            nn.Linear(dim, 3 * dim, bias=False) for _ in range(num_ns_tokens)
        ])
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 缓存因果mask
        self.register_buffer("causal_mask", None)
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果mask: 每个位置只能attend到之前的位置"""
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            self.register_buffer("causal_mask", mask)
        return self.causal_mask[:seq_len, :seq_len].to(device)
    
    def forward(
        self, 
        x: torch.Tensor,
        s_len: int,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [B, L, D] 统一token序列 (S-tokens + NS-tokens)
            s_len: S-tokens的长度
            kv_cache: 可选的KV缓存，用于推理加速
            use_cache: 是否返回KV缓存
        """
        B, L, D = x.shape
        
        # 分离S-tokens和NS-tokens
        x_s = x[:, :s_len]  # [B, s_len, D]
        x_ns = x[:, s_len:]  # [B, ns_len, D]
        ns_len = x_ns.shape[1]
        
        # 计算S-tokens的QKV (共享参数)
        qkv_s = self.qkv_s(x_s).reshape(B, s_len, 3, self.num_heads, self.head_dim)
        qkv_s = qkv_s.permute(2, 0, 3, 1, 4)  # [3, B, H, s_len, head_dim]
        q_s, k_s, v_s = qkv_s[0], qkv_s[1], qkv_s[2]
        
        # 计算NS-tokens的QKV (token-specific参数)
        q_ns_list, k_ns_list, v_ns_list = [], [], []
        for i in range(ns_len):
            qkv_i = self.qkv_ns[i](x_ns[:, i:i+1])  # [B, 1, 3*D]
            qkv_i = qkv_i.reshape(B, 1, 3, self.num_heads, self.head_dim)
            qkv_i = qkv_i.permute(2, 0, 3, 1, 4)  # [3, B, H, 1, head_dim]
            q_ns_list.append(qkv_i[0])
            k_ns_list.append(qkv_i[1])
            v_ns_list.append(qkv_i[2])
        
        if ns_len > 0:
            q_ns = torch.cat(q_ns_list, dim=2)  # [B, H, ns_len, head_dim]
            k_ns = torch.cat(k_ns_list, dim=2)  # [B, H, ns_len, head_dim]
            v_ns = torch.cat(v_ns_list, dim=2)  # [B, H, ns_len, head_dim]
        else:
            q_ns = k_ns = v_ns = None
        
        # 合并QKV
        if ns_len > 0:
            q = torch.cat([q_s, q_ns], dim=2)  # [B, H, L, head_dim]
            k = torch.cat([k_s, k_ns], dim=2)  # [B, H, L, head_dim]
            v = torch.cat([v_s, v_ns], dim=2)  # [B, H, L, head_dim]
        else:
            q, k, v = q_s, k_s, v_s
        
        # 处理KV缓存 (用于推理时的Cross-Request缓存)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        # 计算注意力
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L']
        
        # 应用因果mask
        if kv_cache is None:
            # 训练模式: 应用完整的因果mask
            mask = self.get_causal_mask(L, x.device)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # 推理模式: 只需要mask新计算的部分
            # S-tokens已经缓存，只需要处理新加入的NS-tokens
            pass
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, v)  # [B, H, L, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, L, D)
        
        out = self.proj(attn_out)
        out = self.dropout(out)
        
        # 返回KV缓存用于后续推理
        new_kv_cache = (k, v) if use_cache else None
        
        return out, new_kv_cache


class MixedFFN(nn.Module):
    """
    混合前馈网络: S-tokens共享FFN参数，NS-tokens使用token-specific参数
    """
    def __init__(
        self,
        dim: int,
        num_ns_tokens: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_ns_tokens = num_ns_tokens
        
        # S-tokens共享的FFN
        self.ffn_s = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # NS-tokens的token-specific FFN
        self.ffn_ns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            ) for _ in range(num_ns_tokens)
        ])
    
    def forward(self, x: torch.Tensor, s_len: int) -> torch.Tensor:
        B, L, D = x.shape
        
        # 分离S-tokens和NS-tokens
        x_s = x[:, :s_len]  # [B, s_len, D]
        x_ns = x[:, s_len:]  # [B, ns_len, D]
        ns_len = x_ns.shape[1]
        
        # S-tokens使用共享FFN
        out_s = self.ffn_s(x_s)
        
        # NS-tokens使用token-specific FFN
        if ns_len > 0:
            out_ns_list = []
            for i in range(ns_len):
                out_i = self.ffn_ns[i](x_ns[:, i:i+1])
                out_ns_list.append(out_i)
            out_ns = torch.cat(out_ns_list, dim=1)
            out = torch.cat([out_s, out_ns], dim=1)
        else:
            out = out_s
        
        return out


class OneTransBlock(nn.Module):
    """
    OneTrans Block: 预归一化Transformer Block，包含混合因果注意力和混合FFN
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_ns_tokens: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_ns_tokens = num_ns_tokens
        
        # 预归一化
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # 混合因果注意力
        self.attn = MixedCausalAttention(dim, num_heads, num_ns_tokens, dropout)
        
        # 混合FFN
        self.ffn = MixedFFN(dim, num_ns_tokens, hidden_dim, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        s_len: int,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 预归一化 + 残差连接
        normed = self.norm1(x)
        attn_out, new_kv_cache = self.attn(normed, s_len, kv_cache, use_cache)
        x = x + attn_out
        
        # 预归一化 + FFN + 残差连接
        normed = self.norm2(x)
        ffn_out = self.ffn(normed, s_len)
        x = x + ffn_out
        
        return x, new_kv_cache


class SequentialTokenizer(nn.Module):
    """
    序列Tokenizer: 处理多行为序列，支持timestamp-aware和timestamp-agnostic融合
    """
    def __init__(
        self,
        input_dims: List[int],  # 每个序列的输入维度
        output_dim: int,
        use_timestamp_aware: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.use_timestamp_aware = use_timestamp_aware
        self.num_sequences = len(input_dims)
        
        # 每个序列的投影MLP
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
            ) for dim in input_dims
        ])
        
        # 序列分隔符 [SEP] token
        self.sep_token = nn.Parameter(torch.randn(1, 1, output_dim))
    
    def forward(
        self,
        sequences: List[torch.Tensor],  # 每个序列: [B, L_i, D_i]
        timestamps: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequences: 多行为序列列表
            timestamps: 每个序列的时间戳，用于timestamp-aware融合
        Returns:
            tokens: [B, L_s, D] 合并后的序列token
        """
        B = sequences[0].shape[0]
        
        # 投影到统一维度
        projected_seqs = []
        for i, seq in enumerate(sequences):
            proj = self.projections[i](seq)  # [B, L_i, D]
            projected_seqs.append(proj)
        
        if self.use_timestamp_aware and timestamps is not None:
            # Timestamp-aware: 按时间戳交错合并
            tokens = self._timestamp_aware_merge(projected_seqs, timestamps)
        else:
            # Timestamp-agnostic: 按重要性顺序拼接 (如: purchase > cart > click)
            tokens = self._concat_with_sep(projected_seqs)
        
        return tokens
    
    def _concat_with_sep(self, seqs: List[torch.Tensor]) -> torch.Tensor:
        """用[SEP]token连接多个序列"""
        B = seqs[0].shape[0]
        result = []
        
        for i, seq in enumerate(seqs):
            result.append(seq)
            # 在每个序列后添加SEP (除了最后一个)
            if i < len(seqs) - 1:
                sep = self.sep_token.expand(B, 1, -1)
                result.append(sep)
        
        return torch.cat(result, dim=1)
    
    def _timestamp_aware_merge(
        self, 
        seqs: List[torch.Tensor], 
        timestamps: List[torch.Tensor]
    ) -> torch.Tensor:
        """按时间戳交错合并序列"""
        # 简化实现: 假设所有序列已经按全局时间排序
        # 实际实现需要更复杂的排序逻辑
        return self._concat_with_sep(seqs)


class NonSequentialTokenizer(nn.Module):
    """
    非序列Tokenizer: 处理用户/物品/上下文特征
    支持Group-wise和Auto-Split两种模式
    """
    def __init__(
        self,
        feature_dims: Dict[str, int],  # 特征名到维度的映射
        output_dim: int,
        num_tokens: int,
        mode: str = "auto_split",  # "auto_split" 或 "group_wise"
    ):
        super().__init__()
        self.mode = mode
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        
        if mode == "group_wise":
            # Group-wise: 手动分组，每组一个token
            self.group_mlps = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.LayerNorm(output_dim),
                ) for name, dim in feature_dims.items()
            })
        else:
            # Auto-Split: 全部拼接后一次性投影再分割
            total_dim = sum(feature_dims.values())
            self.projection = nn.Sequential(
                nn.Linear(total_dim, output_dim * num_tokens),
                nn.LayerNorm(output_dim * num_tokens),
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 特征字典，每个特征: [B, D_i]
        Returns:
            tokens: [B, L_ns, D] NS-tokens
        """
        if self.mode == "group_wise":
            # 每组特征生成一个token
            tokens = []
            for name, feat in features.items():
                token = self.group_mlps[name](feat).unsqueeze(1)  # [B, 1, D]
                tokens.append(token)
            return torch.cat(tokens, dim=1)  # [B, num_groups, D]
        else:
            # Auto-Split: 拼接所有特征，投影后分割
            concat_feat = torch.cat(list(features.values()), dim=-1)  # [B, total_D]
            projected = self.projection(concat_feat)  # [B, num_tokens * D]
            tokens = projected.reshape(-1, self.num_tokens, self.output_dim)
            return tokens


class OneTrans(nn.Module):
    """
    OneTrans: 统一的Transformer推荐模型
    """
    def __init__(
        self,
        # 序列特征配置
        seq_input_dims: List[int],  # 每个行为序列的输入维度
        # 非序列特征配置
        ns_feature_dims: Dict[str, int],  # 非序列特征维度
        # 模型结构配置
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        num_ns_tokens: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        # 金字塔配置
        pyramid_schedule: Optional[List[int]] = None,  # 每层保留的S-token数量
        # Tokenizer配置
        use_timestamp_aware: bool = True,
        tokenizer_mode: str = "auto_split",
    ):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        self.num_ns_tokens = num_ns_tokens
        
        # Tokenizers
        self.seq_tokenizer = SequentialTokenizer(
            seq_input_dims, dim, use_timestamp_aware
        )
        self.ns_tokenizer = NonSequentialTokenizer(
            ns_feature_dims, dim, num_ns_tokens, tokenizer_mode
        )
        
        # 位置编码
        self.max_seq_len = 2048  # 最大序列长度
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, dim) * 0.02)
        
        # 金字塔结构配置
        if pyramid_schedule is None:
            # 默认线性递减: 从初始长度递减到num_ns_tokens
            initial_s_len = sum([50] * len(seq_input_dims))  # 假设每个序列50个行为
            pyramid_schedule = [
                max(num_ns_tokens, initial_s_len - (initial_s_len - num_ns_tokens) * i // num_layers)
                for i in range(num_layers)
            ]
        self.pyramid_schedule = pyramid_schedule
        
        # OneTrans Blocks
        hidden_dim = int(dim * ffn_ratio)
        self.blocks = nn.ModuleList([
            OneTransBlock(dim, num_heads, num_ns_tokens, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层 (多任务)
        self.norm_final = RMSNorm(dim)
        self.task_head = nn.Linear(dim, 1)  # 可以扩展为多任务
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        seq_features: List[torch.Tensor],  # 多行为序列
        ns_features: Dict[str, torch.Tensor],  # 非序列特征
        timestamps: Optional[List[torch.Tensor]] = None,
        kv_cache: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """
        Args:
            seq_features: 列表，每个元素是 [B, L_i, D_i] 的行为序列
            ns_features: 字典，每个值是 [B, D_i] 的非序列特征
            timestamps: 可选的时间戳列表
            kv_cache: 可选的KV缓存列表，用于推理加速
            use_cache: 是否返回KV缓存
        Returns:
            logits: [B, 1] 预测分数
            new_kv_cache: 新的KV缓存 (如果use_cache=True)
        """
        B = seq_features[0].shape[0]
        
        # Tokenization
        s_tokens = self.seq_tokenizer(seq_features, timestamps)  # [B, L_s, D]
        ns_tokens = self.ns_tokenizer(ns_features)  # [B, L_ns, D]
        
        # 合并token序列: [S-tokens; NS-tokens]
        x = torch.cat([s_tokens, ns_tokens], dim=1)  # [B, L, D]
        L = x.shape[1]
        
        # 添加位置编码
        x = x + self.pos_emb[:, :L, :]
        
        # 通过OneTrans Blocks (金字塔结构)
        new_kv_cache_list = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            s_len = min(self.pyramid_schedule[i], s_tokens.shape[1])
            
            # 金字塔剪枝: 只保留最近的s_len个S-tokens用于query
            if s_len < s_tokens.shape[1]:
                # 保留最近的s_len个S-tokens + 所有NS-tokens
                x_pruned = torch.cat([x[:, :s_len], x[:, s_tokens.shape[1]:]], dim=1)
            else:
                x_pruned = x
            
            # 当前层的KV缓存
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            
            # 通过block
            x_out, layer_new_kv = block(
                x_pruned, 
                s_len=s_len,
                kv_cache=layer_kv_cache,
                use_cache=use_cache,
            )
            
            # 更新完整序列 (用于下一层)
            if s_len < s_tokens.shape[1]:
                x = torch.cat([x_out[:, :s_len], x_out[:, s_len:]], dim=1)
            else:
                x = x_out
            
            if use_cache:
                new_kv_cache_list.append(layer_new_kv)
        
        # 最终归一化和预测
        x = self.norm_final(x)
        
        # 使用NS-tokens进行预测 (它们聚合了所有序列信息)
        ns_repr = x[:, -self.num_ns_tokens:, :].mean(dim=1)  # [B, D]
        logits = self.task_head(ns_repr)  # [B, 1]
        
        return logits, new_kv_cache_list
    
    @torch.no_grad()
    def predict_with_cache(
        self,
        seq_features: List[torch.Tensor],
        ns_features: Dict[str, torch.Tensor],
        kv_cache: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        推理模式: 使用KV缓存加速
        适用于Cross-Request场景 (S-tokens相同，NS-tokens变化)
        """
        self.eval()
        logits, new_cache = self.forward(
            seq_features, ns_features, 
            kv_cache=kv_cache, use_cache=True
        )
        return torch.sigmoid(logits), new_cache
