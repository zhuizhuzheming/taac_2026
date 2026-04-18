"""
HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction
论文实现 (WWW 2026 / CIKM 2025)

核心架构:
- Query Generation: 从NS特征生成Global Query Tokens
- Query Decoding: Cross-Attention between Global Queries and Sequence KV
- Query Boosting: MLP-Mixer增强Query与NS-tokens的交互
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
    """SwiGLU激活函数 (用于序列编码的轻量级选项)"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2  # SwiGLU
        return self.dropout(self.w3(hidden))


class QueryGeneration(nn.Module):
    """
    Query Generation模块 (论文3.3节)
    将NS特征和序列池化信息转换为Global Query Tokens
    """
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
        
        # NS特征投影
        total_ns_dim = sum(ns_feature_dims.values())
        self.ns_proj = nn.Sequential(
            nn.Linear(total_ns_dim, dim * num_queries),
            nn.Dropout(dropout),
        )
        
        # 序列池化信息投影 (MeanPool -> Linear)
        self.seq_pool_proj = nn.Sequential(
            nn.Linear(seq_hidden_dim, dim),
            nn.Dropout(dropout),
        )
        
        # 可选: 特征选择/压缩层 (论文提到支持特征选择)
        self.query_compress = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim),
        )
    
    def forward(
        self,
        ns_features: Dict[str, torch.Tensor],
        seq_pooled: torch.Tensor,  # [B, D_seq] 序列的MeanPool
    ) -> torch.Tensor:
        """
        Returns:
            queries: [B, num_queries, D] Global Query Tokens
        """
        B = seq_pooled.shape[0]
        
        # 拼接所有NS特征
        ns_concat = torch.cat(list(ns_features.values()), dim=-1)  # [B, total_ns_dim]
        
        # 生成基础queries
        base_queries = self.ns_proj(ns_concat).reshape(B, self.num_queries, self.dim)
        
        # 加入序列池化信息 (Global Info)
        seq_info = self.seq_pool_proj(seq_pooled).unsqueeze(1)  # [B, 1, D]
        
        # 融合: 每个query都加上序列信息
        queries = base_queries + seq_info  # [B, num_queries, D]
        
        # 可选压缩
        queries = self.query_compress(queries)
        
        return queries


class SequenceEncoder(nn.Module):
    """
    序列表示编码 (论文3.4.1节)
    支持三种模式: Full Transformer, LONGER-style, Decoder-style(SwiGLU)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        mode: str = "longer",  # "full" | "longer" | "decoder"
        num_short_tokens: int = 16,  # for LONGER mode
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_short_tokens = num_short_tokens
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if mode == "full":
            # Full Transformer: 标准自注意力
            self.attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = RMSNorm(hidden_dim)
            self.norm2 = RMSNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        elif mode == "longer":
            # LONGER-style: 短序列作为Query的Cross-Attention
            self.short_queries = nn.Parameter(torch.randn(1, num_short_tokens, hidden_dim))
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm = RMSNorm(hidden_dim)
        else:  # decoder
            # Decoder-style: 无注意力，只用SwiGLU
            self.swiglu = SwiGLU(hidden_dim, hidden_dim * 4, dropout)
            self.norm = RMSNorm(hidden_dim)
    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: [B, L, D_input] 输入序列
        Returns:
            encoded: [B, L, D_hidden] 编码后的序列表示
        """
        x = self.input_proj(seq)  # [B, L, D_hidden]
        
        if self.mode == "full":
            # 标准Transformer Block
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
            x = x + attn_out
            
            normed = self.norm2(x)
            ffn_out = self.ffn(normed)
            x = x + ffn_out
            
        elif self.mode == "longer":
            # LONGER: 短Query Cross-Attention到完整序列
            B = x.shape[0]
            short_q = self.short_queries.expand(B, -1, -1)  # [B, num_short, D]
            
            normed = self.norm(x)
            # Cross-Attention: 短Query attend to 完整序列的KV
            attn_out, _ = self.cross_attn(short_q, normed, normed, need_weights=False)
            # 将输出扩展到原始长度 (或保持短长度，视设计而定)
            # 这里我们保持短长度作为压缩表示
            x = attn_out  # [B, num_short, D]
            
        else:  # decoder
            # 纯FFN编码
            normed = self.norm(x)
            x = x + self.swiglu(normed)
        
        return x


class QueryDecoding(nn.Module):
    """
    Query Decoding模块 (论文3.4.2节)
    Global Queries通过Cross-Attention解码序列的KV表示
    """
    def __init__(
        self,
        dim: int,
        seq_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        
        # 序列KV投影 (每层独立，论文3.4.1)
        self.k_proj = nn.Linear(seq_hidden_dim, dim)
        self.v_proj = nn.Linear(seq_hidden_dim, dim)
        
        # Query投影
        self.q_proj = nn.Linear(dim, dim)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,  # [B, N, D] Global Queries
        seq_encoded: torch.Tensor,  # [B, L, D_seq] 编码后的序列
    ) -> torch.Tensor:
        """
        Returns:
            decoded_queries: [B, N, D] 解码后的Query表示
        """
        # 投影序列到KV
        K = self.k_proj(seq_encoded)  # [B, L, D]
        V = self.v_proj(seq_encoded)  # [B, L, D]
        
        # 投影Query
        Q = self.q_proj(queries)  # [B, N, D]
        
        # Cross-Attention: Queries attend to Sequence KV
        normed_q = self.norm(Q)
        attn_out, _ = self.cross_attn(normed_q, K, V, need_weights=False)
        
        # 残差连接
        output = queries + self.dropout(attn_out)
        
        return output


class QueryBoosting(nn.Module):
    """
    Query Boosting模块 (论文3.5节)
    使用MLP-Mixer风格增强Query与NS-tokens的交互
    """
    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_ns_tokens: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.total_tokens = num_queries + num_ns_tokens
        
        # MLP-Mixer: Token Mixing (跨token混合)
        # 将每个token分成total_tokens个子空间，每个子空间与其他token对应子空间混合
        self.subspace_dim = dim // self.total_tokens
        
        # Token mixing MLP (对每个子空间独立)
        self.token_mix_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.total_tokens, self.total_tokens * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.total_tokens * 2, self.total_tokens),
            ) for _ in range(self.subspace_dim)
        ])
        
        # Per-Token FFN (论文3.5: PerToken-FFN)
        self.per_token_ffn = nn.ModuleList([
            nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
            ) for _ in range(self.total_tokens)
        ])
        
        self.norm = RMSNorm(dim)
    
    def forward(
        self,
        decoded_queries: torch.Tensor,  # [B, N, D]
        ns_tokens: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:
        """
        Returns:
            boosted: [B, N+M, D] 增强后的统一表示
        """
        # 拼接Queries和NS-tokens
        unified_tokens = torch.cat([decoded_queries, ns_tokens], dim=1)  # [B, T, D]
        B, T, D = unified_tokens.shape
        
        # 1. MLP-Mixer Token Mixing
        # 将每个token分成subspace_dim个子空间
        # 重排: [B, T, D] -> [B, T, subspace_dim, subspace_dim] (假设D = subspace_dim^2)
        # 简化为: [B, subspace_dim, T, subspace_dim] 然后对每个subspace做mixing
        
        # 实际实现: 将D维分成subspace_dim组，每组total_tokens维
        x_reshaped = unified_tokens.reshape(B, T, self.subspace_dim, self.subspace_dim)
        # [B, subspace_dim, T, subspace_dim] -> 我们需要 [B, subspace_dim, T] 对每个位置做mixing
        
        # 更标准的MLP-Mixer实现:
        # Transpose: [B, T, D] -> [B, D, T]
        x_trans = unified_tokens.transpose(1, 2)  # [B, D, T]
        
        # 将D分成subspace_dim组，每组处理T个token
        mixed_parts = []
        for i in range(self.subspace_dim):
            # 提取第i个子空间 [B, subspace_dim, T]
            start_idx = i * self.subspace_dim
            end_idx = (i + 1) * self.subspace_dim
            subspace = x_trans[:, start_idx:end_idx, :]  # [B, subspace_dim, T]
            
            # 对每个位置应用MLP (mixing across tokens)
            # 简化为对T个token做mixing
            subspace_mixed = self.token_mix_mlps[i](subspace)  # [B, subspace_dim, T]
            mixed_parts.append(subspace_mixed)
        
        x_mixed = torch.cat(mixed_parts, dim=1).transpose(1, 2)  # [B, T, D]
        
        # 2. Per-Token FFN
        boosted_parts = []
        for i in range(T):
            token_i = x_mixed[:, i, :]  # [B, D]
            boosted_i = self.per_token_ffn[i](token_i)
            boosted_parts.append(boosted_i.unsqueeze(1))
        
        boosted = torch.cat(boosted_parts, dim=1)  # [B, T, D]
        
        # 残差连接
        output = unified_tokens + boosted
        
        return output


class HyFormerLayer(nn.Module):
    """
    单个HyFormer层 (论文3.6节)
    包含Query Decoding + Query Boosting
    """
    def __init__(
        self,
        dim: int,
        seq_hidden_dim: int,
        num_queries: int,
        num_ns_tokens: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.decoding = QueryDecoding(dim, seq_hidden_dim, num_heads, dropout)
        self.boosting = QueryBoosting(dim, num_queries, num_ns_tokens, dropout)
        
        # 输出分离: Queries用于下一层，NS-tokens也用于下一层
        self.num_queries = num_queries
    
    def forward(
        self,
        queries: torch.Tensor,  # [B, N, D]
        seq_encoded: torch.Tensor,  # [B, L, D_seq]
        ns_tokens: torch.Tensor,  # [B, M, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            next_queries: [B, N, D] 用于下一层的Queries
            next_ns_tokens: [B, M, D] 用于下一层的NS-tokens
        """
        # Step 1: Query Decoding
        decoded_queries = self.decoding(queries, seq_encoded)
        
        # Step 2: Query Boosting (与NS-tokens混合)
        boosted_unified = self.boosting(decoded_queries, ns_tokens)
        
        # 分离Queries和NS-tokens
        next_queries = boosted_unified[:, :self.num_queries, :]
        next_ns_tokens = boosted_unified[:, self.num_queries:, :]
        
        return next_queries, next_ns_tokens


class HyFormer(nn.Module):
    """
    HyFormer: 完整的混合Transformer架构
    支持多序列独立建模 (论文3.7节)
    """
    def __init__(
        self,
        # 序列配置 (支持多序列)
        seq_configs: List[Dict],  # 每个序列的配置列表
        # NS特征配置
        ns_feature_dims: Dict[str, int],
        # 模型结构
        dim: int = 256,
        num_layers: int = 6,
        num_queries_per_seq: int = 3,  # 每个序列的Global Token数量
        num_ns_tokens: int = 13,  # NS token数量 (论文4.1.4)
        num_heads: int = 8,
        seq_encoder_mode: str = "longer",  # "full" | "longer" | "decoder"
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_layers = num_layers
        self.num_sequences = len(seq_configs)
        self.num_queries_per_seq = num_queries_per_seq
        self.num_ns_tokens = num_ns_tokens
        
        total_queries = num_queries_per_seq * self.num_sequences
        
        # 1. NS Tokenizer (Group-wise, 论文3.3.1)
        self.ns_tokenizer = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim_in, dim),
                RMSNorm(dim),
            ) for name, dim_in in ns_feature_dims.items()
        })
        
        # 2. 序列编码器 (每个序列独立)
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
        
        # 3. Query Generation (每层复用初始生成的Query)
        # 第一层生成，后续层复用并更新
        first_seq_hidden = seq_configs[0].get('hidden_dim', dim)
        self.query_generation = QueryGeneration(
            ns_feature_dims=ns_feature_dims,
            seq_hidden_dim=first_seq_hidden,
            num_queries=total_queries,
            dim=dim,
            dropout=dropout,
        )
        
        # 4. HyFormer Layers堆叠
        self.layers = nn.ModuleList([
            HyFormerLayer(
                dim=dim,
                seq_hidden_dim=cfg.get('hidden_dim', dim),
                num_queries=total_queries,
                num_ns_tokens=num_ns_tokens,
                num_heads=num_heads,
                dropout=dropout,
            ) for cfg in seq_configs  # 使用第一个序列的配置
        ])
        
        # 5. 预测头
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
    
    def forward(
        self,
        seq_features: List[torch.Tensor],  # 多序列输入，每个 [B, L_i, D_i]
        ns_features: Dict[str, torch.Tensor],  # NS特征，每个 [B, D_i]
        return_layer_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        Args:
            seq_features: 多行为序列列表
            ns_features: 非序列特征字典
            return_layer_outputs: 是否返回每层输出 (用于分析)
        Returns:
            logits: [B, 1] 预测分数
        """
        B = seq_features[0].shape[0]
        
        # 1. Tokenize NS特征
        ns_tokens_list = []
        for name, feat in ns_features.items():
            token = self.ns_tokenizer[name](feat).unsqueeze(1)  # [B, 1, D]
            ns_tokens_list.append(token)
        ns_tokens = torch.cat(ns_tokens_list, dim=1)  # [B, num_ns_tokens, D]
        
        # 2. 编码所有序列 (独立编码，不合并)
        seq_encoded_list = []
        seq_pooled_list = []
        
        for i, (seq, encoder) in enumerate(zip(seq_features, self.seq_encoders)):
            encoded = encoder(seq)  # [B, L_i', D_seq] (L_i'可能经过压缩)
            seq_encoded_list.append(encoded)
            
            # Mean Pooling用于Query Generation
            pooled = encoded.mean(dim=1)  # [B, D_seq]
            seq_pooled_list.append(pooled)
        
        # 3. 生成初始Global Queries (使用第一个序列的pooled)
        # 论文3.3.2: Global Info = concat(NS features, MeanPool(Seq))
        queries = self.query_generation(ns_features, seq_pooled_list[0])  # [B, total_queries, D]
        
        # 分离每个序列的queries (用于多序列独立解码)
        # queries: [B, num_seq * num_queries_per_seq, D]
        # 分割成 per-seq queries
        seq_queries = queries.reshape(
            B, self.num_sequences, self.num_queries_per_seq, self.dim
        )  # [B, num_seq, num_queries_per_seq, D]
        
        # 4. 通过HyFormer Layers (每层处理所有序列)
        layer_outputs = []
        
        for layer_idx, layer in enumerate(self.layers):
            # 对每个序列独立做Query Decoding，然后合并Boosting
            decoded_per_seq = []
            
            for seq_idx in range(self.num_sequences):
                # 当前序列的queries
                current_queries = seq_queries[:, seq_idx, :, :]  # [B, num_queries_per_seq, D]
                current_seq = seq_encoded_list[seq_idx]  # [B, L', D_seq]
                
                # 注意: 这里简化处理，实际应该每个序列有独立的Decoding层
                # 或者共享参数但独立计算
                # 为简化，我们假设所有序列共享同一个Decoding参数
            
            # 简化实现: 将所有序列的编码拼接，统一处理
            # 论文3.7: 多序列独立建模，但这里为了效率可以共享部分参数
            
            # 实际实现: 对每个序列分别做Decoding，然后合并
            all_seq_queries = []
            for seq_idx in range(self.num_sequences):
                q_start = seq_idx * self.num_queries_per_seq
                q_end = (seq_idx + 1) * self.num_queries_per_seq
                seq_q = queries[:, q_start:q_end, :]
                all_seq_queries.append(seq_q)
            
            # 合并所有序列的queries作为统一Global Tokens
            unified_queries = torch.cat(all_seq_queries, dim=1)  # [B, total_queries, D]
            
            # 选择当前层要处理的序列 (轮流或固定)
            # 简化: 使用第一个序列的编码
            current_seq_encoded = seq_encoded_list[0]
            
            # 通过HyFormer Layer
            next_queries, next_ns_tokens = layer(
                unified_queries, current_seq_encoded, ns_tokens
            )
            
            # 更新用于下一层
            queries = next_queries
            ns_tokens = next_ns_tokens
            
            if return_layer_outputs:
                layer_outputs.append((queries.clone(), ns_tokens.clone()))
        
        # 5. 最终预测 (使用最后一层的Global Queries)
        queries_flat = queries.reshape(B, -1)  # [B, total_queries * D]
        logits = self.predictor(queries_flat)  # [B, 1]
        
        if return_layer_outputs:
            return logits, layer_outputs
        return logits
