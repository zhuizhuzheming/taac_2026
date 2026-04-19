"""
HyFormer: 最终修复版 - 解决token数量不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class RMSNorm(nn.Module):
    """RMSNorm for pre-normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))


class QueryGeneration(nn.Module):
    """Query Generation模块"""
    def __init__(
        self,
        ns_feature_dims: Dict[str, int],
        seq_hidden_dim: int,
        num_queries: int,
        dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim
        
        total_ns_dim = sum(ns_feature_dims.values())
        self.ns_proj = nn.Sequential(
            nn.Linear(total_ns_dim, dim * num_queries),
            nn.Dropout(dropout),
        )
        
        self.seq_pool_proj = nn.Sequential(
            nn.Linear(seq_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        self.query_compress = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim),
        )
    
    def forward(self, ns_features: Dict[str, torch.Tensor], seq_pooled: torch.Tensor) -> torch.Tensor:
        B = seq_pooled.shape[0]
        ns_concat = torch.cat(list(ns_features.values()), dim=-1)
        base_queries = self.ns_proj(ns_concat).reshape(B, self.num_queries, self.dim)
        seq_info = self.seq_pool_proj(seq_pooled).unsqueeze(1)
        queries = base_queries + seq_info
        queries = self.query_compress(queries)
        return queries


class SequenceEncoder(nn.Module):
    """序列表示编码"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        mode: str = "longer",
        num_short_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_short_tokens = num_short_tokens
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if mode == "full":
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = RMSNorm(hidden_dim)
            self.norm2 = RMSNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        elif mode == "longer":
            self.short_queries = nn.Parameter(torch.randn(1, num_short_tokens, hidden_dim))
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm = RMSNorm(hidden_dim)
        else:
            self.swiglu = SwiGLU(hidden_dim, hidden_dim * 4, dropout)
            self.norm = RMSNorm(hidden_dim)
    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(seq)
        
        if self.mode == "full":
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x = x + attn_out
            normed = self.norm2(x)
            ffn_out = self.ffn(normed)
            x = x + ffn_out
        elif self.mode == "longer":
            B = x.shape[0]
            short_q = self.short_queries.expand(B, -1, -1)
            normed = self.norm(x)
            attn_out, _ = self.cross_attn(short_q, normed, normed, need_weights=False)
            x = attn_out
        else:
            normed = self.norm(x)
            x = x + self.swiglu(normed)
        
        return x


class QueryDecoding(nn.Module):
    """Query Decoding模块"""
    def __init__(self, dim: int, seq_hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.k_proj = nn.Linear(seq_hidden_dim, dim)
        self.v_proj = nn.Linear(seq_hidden_dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries: torch.Tensor, seq_encoded: torch.Tensor) -> torch.Tensor:
        K = self.k_proj(seq_encoded)
        V = self.v_proj(seq_encoded)
        Q = self.q_proj(queries)
        normed_q = self.norm(Q)
        attn_out, _ = self.cross_attn(normed_q, K, V, need_weights=False)
        output = queries + self.dropout(attn_out)
        return output


class QueryBoosting(nn.Module):
    """
    Query Boosting模块 - 修复版
    关键修复: 动态适应实际的token数量，而不是固定配置值
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        
        # 不再固定total_tokens，而是在forward中动态处理
        # Token Mixing: 使用1D卷积实现跨token混合，避免固定维度问题
        # 或者使用MultiheadAttention作为token mixing
        
        # 方案: 使用Attention-based Token Mixing (更灵活)
        self.token_mixing_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=dropout, batch_first=True)
        self.token_mixing_norm = RMSNorm(dim)
        
        # Channel Mixing (FFN)
        self.channel_mixing = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        
        # Per-Token FFN (使用共享参数，避免固定token数量问题)
        self.per_token_norm = RMSNorm(dim)
        self.per_token_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, decoded_queries: torch.Tensor, ns_tokens: torch.Tensor) -> torch.Tensor:
        """
        修复: 动态处理任意数量的queries和ns_tokens
        """
        # 拼接
        unified_tokens = torch.cat([decoded_queries, ns_tokens], dim=1)  # [B, T, D]
        B, T, D = unified_tokens.shape
        
        # Token Mixing: 使用Self-Attention让每个token看到其他所有token
        # 这比MLP-Mixer更灵活，不依赖固定token数量
        normed = self.token_mixing_norm(unified_tokens)
        mixed, _ = self.token_mixing_attn(normed, normed, normed, need_weights=False)
        unified_tokens = unified_tokens + mixed
        
        # Channel Mixing
        unified_tokens = unified_tokens + self.channel_mixing(unified_tokens)
        
        # Per-Token FFN (共享参数，对所有token应用相同变换)
        # 论文要求per-token FFN，但为简化使用共享参数版本
        # 如需严格遵循论文，可使用ModuleList，但会固定token数量
        normed = self.per_token_norm(unified_tokens)
        per_token_out = self.per_token_ffn(normed)
        unified_tokens = unified_tokens + per_token_out
        
        return unified_tokens


class HyFormerLayer(nn.Module):
    """单个HyFormer层"""
    def __init__(self, dim: int, seq_hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.decoding = QueryDecoding(dim, seq_hidden_dim, num_heads, dropout)
        self.boosting = QueryBoosting(dim, dropout)
    
    def forward(self, queries: torch.Tensor, seq_encoded: torch.Tensor, ns_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded_queries = self.decoding(queries, seq_encoded)
        boosted_unified = self.boosting(decoded_queries, ns_tokens)
        
        # 分离
        num_queries = queries.shape[1]
        next_queries = boosted_unified[:, :num_queries, :]
        next_ns_tokens = boosted_unified[:, num_queries:, :]
        
        return next_queries, next_ns_tokens


class HyFormer(nn.Module):
    """HyFormer: 完整的混合Transformer架构 - 最终修复版"""
    def __init__(
        self,
        seq_configs: List[Dict],
        ns_feature_dims: Dict[str, int],
        dim: int = 256,
        num_layers: int = 6,
        num_queries_per_seq: int = 3,
        num_heads: int = 8,
        seq_encoder_mode: str = "longer",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        self.num_sequences = len(seq_configs)
        self.num_queries_per_seq = num_queries_per_seq
        
        # 实际的NS token数量由特征数量决定
        self.actual_ns_tokens = len(ns_feature_dims)
        total_queries = num_queries_per_seq * self.num_sequences
        
        # NS Tokenizer
        self.ns_tokenizer = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim_in, dim), RMSNorm(dim))
            for name, dim_in in ns_feature_dims.items()
        })
        
        # 序列编码器
        self.seq_encoders = nn.ModuleList([
            SequenceEncoder(
                input_dim=cfg['input_dim'],
                hidden_dim=cfg.get('hidden_dim', dim),
                num_heads=num_heads,
                mode=seq_encoder_mode,
                num_short_tokens=cfg.get('num_short_tokens', 16),
                dropout=dropout,
            ) for cfg in seq_configs
        ])
        
        # Query Generation
        first_seq_hidden = seq_configs[0].get('hidden_dim', dim)
        self.query_generation = QueryGeneration(
            ns_feature_dims=ns_feature_dims,
            seq_hidden_dim=first_seq_hidden,
            num_queries=total_queries,
            dim=dim,
            dropout=dropout,
        )
        
        # HyFormer Layers (使用动态token数量的版本)
        self.layers = nn.ModuleList([
            HyFormerLayer(dim, first_seq_hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 预测头
        self.norm_final = RMSNorm(dim)
        self.predictor = nn.Sequential(
            nn.Linear(dim * total_queries, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, seq_features: List[torch.Tensor], ns_features: Dict[str, torch.Tensor], return_layer_outputs: bool = False):
        B = seq_features[0].shape[0]
        
        # Tokenize NS特征
        ns_tokens_list = []
        for name, feat in ns_features.items():
            token = self.ns_tokenizer[name](feat).unsqueeze(1)
            ns_tokens_list.append(token)
        ns_tokens = torch.cat(ns_tokens_list, dim=1)  # [B, actual_ns_tokens, D]
        
        # 编码所有序列
        seq_encoded_list = []
        seq_pooled_list = []
        for seq, encoder in zip(seq_features, self.seq_encoders):
            encoded = encoder(seq)
            seq_encoded_list.append(encoded)
            pooled = encoded.mean(dim=1)
            seq_pooled_list.append(pooled)
        
        # 生成初始Global Queries
        queries = self.query_generation(ns_features, seq_pooled_list[0])
        
        # 通过HyFormer Layers
        layer_outputs = []
        for layer in self.layers:
            current_seq_encoded = seq_encoded_list[0]
            queries, ns_tokens = layer(queries, current_seq_encoded, ns_tokens)
            if return_layer_outputs:
                layer_outputs.append((queries.clone(), ns_tokens.clone()))
        
        # 最终预测
        queries_flat = queries.reshape(B, -1)
        logits = self.predictor(queries_flat)
        
        if return_layer_outputs:
            return logits, layer_outputs
        return logits
