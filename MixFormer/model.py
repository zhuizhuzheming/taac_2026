"""
MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders
论文实现 (2026)

核心架构:
- 统一Transformer backbone，共享参数处理序列和dense特征
- Query Mixer: HeadMixing + Per-head SwiGLU (替代Self-Attention)
- Cross Attention: NS heads作为query，序列作为KV
- Output Fusion: Per-head SwiGLU深度融合
- UI-MixFormer: User-Item解耦支持RLB优化
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


class HeadMixing(nn.Module):
    """
    HeadMixing模块 (论文3.3.1)
    关键要求: D必须能被N整除
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # 确保D能被N整除
        assert D % N == 0, f"D={D} must be divisible by N={N} for HeadMixing"
        
        # [B, N, D] -> [B, N, N, D/N] -> transpose -> [B, N, D]
        x_reshaped = x.reshape(B, N, N, D // N)
        x_transposed = x_reshaped.transpose(1, 2)
        mixed = x_transposed.reshape(B, N, D)
        
        return mixed


class QueryMixer(nn.Module):
    """Query Mixer模块"""
    def __init__(
        self,
        num_heads: int,
        dim_per_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        
        self.norm1 = RMSNorm(dim_per_head)
        self.head_mixing = HeadMixing()
        
        # Per-head SwiGLU
        self.per_head_ffn = nn.ModuleList([
            SwiGLU(dim_per_head, dim_per_head * 2, dropout)
            for _ in range(num_heads)
        ])
        
        self.norm2 = RMSNorm(dim_per_head)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        mixed = self.head_mixing(normed)
        x = x + mixed
        
        normed = self.norm2(x)
        ffn_outputs = []
        for i in range(self.num_heads):
            head_i = normed[:, i, :]
            ffn_out = self.per_head_ffn[i](head_i)
            ffn_outputs.append(ffn_out.unsqueeze(1))
        
        ffn_out = torch.cat(ffn_outputs, dim=1)
        output = x + ffn_out
        
        return output


class CrossAttention(nn.Module):
    """Cross Attention模块"""
    def __init__(
        self,
        num_heads: int,
        dim_per_head: int,
        seq_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        
        # Per-layer SwiGLU for sequence
        self.seq_swiglu = SwiGLU(seq_dim, seq_dim * 2, dropout)
        self.seq_norm = RMSNorm(seq_dim)
        
        # 投影序列到每个head的KV
        self.k_projs = nn.ModuleList([
            nn.Linear(seq_dim, dim_per_head) for _ in range(num_heads)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(seq_dim, dim_per_head) for _ in range(num_heads)
        ])
        
        self.q_norm = RMSNorm(dim_per_head)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, ns_heads: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        B, N, D = ns_heads.shape
        
        seq = self.seq_norm(seq)
        seq_processed = seq + self.seq_swiglu(seq)
        
        outputs = []
        for i in range(N):
            q = ns_heads[:, i, :]
            q = self.q_norm(q)
            
            k = self.k_projs[i](seq_processed)
            v = self.v_projs[i](seq_processed)
            
            scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / math.sqrt(D)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            out = torch.matmul(attn_weights, v).squeeze(1)
            outputs.append(out.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        output = ns_heads + output
        
        return output


class OutputFusion(nn.Module):
    """Output Fusion模块"""
    def __init__(
        self,
        num_heads: int,
        dim_per_head: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        self.per_head_ffn = nn.ModuleList([
            nn.Sequential(
                RMSNorm(dim_per_head),
                SwiGLU(dim_per_head, dim_per_head * 2, dropout)
            ) for _ in range(num_heads)
        ])
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, N, D = z.shape
        
        outputs = []
        for i in range(N):
            z_i = z[:, i, :]
            out = z_i + self.per_head_ffn[i](z_i)
            outputs.append(out.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output


class MixFormerBlock(nn.Module):
    """MixFormer Block"""
    def __init__(
        self,
        num_heads: int,
        dim_per_head: int,
        seq_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.query_mixer = QueryMixer(num_heads, dim_per_head, dropout)
        self.cross_attn = CrossAttention(num_heads, dim_per_head, seq_dim, dropout)
        self.output_fusion = OutputFusion(num_heads, dim_per_head, dropout)
    
    def forward(self, ns_heads: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        ns_heads = self.query_mixer(ns_heads)
        z = self.cross_attn(ns_heads, seq)
        ns_heads = self.output_fusion(z)
        return ns_heads


class MixFormer(nn.Module):
    """
    MixFormer: 统一Transformer架构 - 最终修复版
    关键: 确保dim_per_head能被num_heads整除
    """
    def __init__(
        self,
        ns_feature_dims: Dict[str, int],
        seq_input_dim: int,
        seq_hidden_dim: int,
        num_heads: int = 16,
        dim_per_head: int = 384,  # 修复: 384能被16整除 (384/16=24)
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
        self.num_layers = num_layers
        
        # 验证: dim_per_head必须能被num_heads整除 (HeadMixing要求)
        assert dim_per_head % num_heads == 0, \
            f"dim_per_head={dim_per_head} must be divisible by num_heads={num_heads}"
        
        # NS特征embedding
        self.ns_embeddings = nn.ModuleDict({
            name: nn.Linear(dim, dim) for name, dim in ns_feature_dims.items()
        })
        
        # 投影到固定维度 (确保能被num_heads整除)
        total_ns_dim = sum(ns_feature_dims.values())
        self.ns_proj_dim = num_heads * 64  # 1024 for num_heads=16
        
        self.ns_proj = nn.Sequential(
            nn.Linear(total_ns_dim, self.ns_proj_dim),
            nn.LayerNorm(self.ns_proj_dim),
        )
        
        self.split_dim = self.ns_proj_dim // num_heads  # 64
        
        # 投影到dim_per_head
        self.head_projs = nn.ModuleList([
            nn.Linear(self.split_dim, dim_per_head) for _ in range(num_heads)
        ])
        
        # 序列编码
        self.seq_proj = nn.Linear(seq_input_dim, seq_hidden_dim)
        
        # MixFormer Blocks
        self.blocks = nn.ModuleList([
            MixFormerBlock(num_heads, dim_per_head, seq_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 预测头
        total_dim = num_heads * dim_per_head
        self.task_head = nn.Sequential(
            RMSNorm(total_dim),
            nn.Linear(total_dim, dim_per_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_per_head, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _split_ns_features(self, ns_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        ns_list = []
        for name, feat in ns_features.items():
            emb = self.ns_embeddings[name](feat)
            ns_list.append(emb)
        
        ns_concat = torch.cat(ns_list, dim=-1)
        B = ns_concat.shape[0]
        
        ns_proj = self.ns_proj(ns_concat)
        ns_splits = ns_proj.reshape(B, self.num_heads, self.split_dim)
        
        heads = []
        for i in range(self.num_heads):
            head_i = self.head_projs[i](ns_splits[:, i, :])
            heads.append(head_i.unsqueeze(1))
        
        ns_heads = torch.cat(heads, dim=1)
        return ns_heads
    
    def forward(self, seq_features: torch.Tensor, ns_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        ns_heads = self._split_ns_features(ns_features)
        seq = self.seq_proj(seq_features)
        
        for block in self.blocks:
            ns_heads = block(ns_heads, seq)
        
        ns_flat = ns_heads.reshape(ns_heads.shape[0], -1)
        logits = self.task_head(ns_flat)
        
        return logits
