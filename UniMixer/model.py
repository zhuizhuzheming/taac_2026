"""
UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems
论文实现 (2026)

核心创新:
1. 统一理论框架: 将Attention-based、TokenMixer-based、FM-based方法统一
2. UniMixing模块: 参数化的TokenMixer，通过Kronecker分解实现Global/Local mixing
3. UniMixing-Lite: 低秩近似+基矩阵动态生成，进一步提升效率
4. SiameseNorm: 解决Pre/Post-Norm的冲突，支持深度扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math


class RMSNorm(nn.Module):
    """RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SiameseNorm(nn.Module):
    """
    SiameseNorm (论文4.4节)
    双耦合流设计，解决Pre-Norm和Post-Norm的冲突
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm_x = RMSNorm(dim)
        self.norm_y = RMSNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,  # 主分支
        y: torch.Tensor,  # 辅助分支
        mixer_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        论文公式(19):
        Y_tilde = RMSNorm(Y_t)
        O_t = UniMixer(X_t + Y_tilde)
        X_{t+1} = RMSNorm(X_t + O_t)
        Y_{t+1} = Y_t + O_t
        """
        # 更新X分支
        x_next = self.norm_x(x + mixer_output)
        # 更新Y分支
        y_next = y + mixer_output
        
        return x_next, y_next
    
    def fuse(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """最终融合: X_M + RMSNorm(Y_M)"""
        return x + self.norm_y(y)


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp迭代归一化 (论文4.3节)
    确保矩阵满足双随机性(doubly stochastic): 行列和均为1
    """
    def __init__(self, num_iters: int = 5, eps: float = 1e-6):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
    
    def forward(self, matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            matrix: 输入矩阵 [N, N]
            temperature: 温度系数τ，控制稀疏性
        Returns:
            归一化后的双随机矩阵
        """
        # 应用温度系数
        matrix = matrix / temperature
        
        # 指数化确保正数
        matrix = torch.exp(matrix)
        
        # Sinkhorn-Knopp迭代
        for _ in range(self.num_iters):
            # 行归一化
            matrix = matrix / (matrix.sum(dim=1, keepdim=True) + self.eps)
            # 列归一化
            matrix = matrix / (matrix.sum(dim=0, keepdim=True) + self.eps)
        
        return matrix


class UniMixing(nn.Module):
    """
    UniMixing模块 (论文4.3节)
    通过Kronecker分解实现参数化的TokenMixer
    
    Global Mixing: W_G 控制块间交互
    Local Mixing: W_B^i 控制块内交互
    """
    def __init__(
        self,
        dim: int,
        block_size: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        
        assert dim % block_size == 0, f"dim={dim} must be divisible by block_size={block_size}"
        
        # Global mixing矩阵 W_G: [num_blocks, num_blocks]
        self.W_G_raw = nn.Parameter(torch.randn(self.num_blocks, self.num_blocks))
        
        # Local mixing矩阵 W_B^i: 每个块一个 [block_size, block_size]
        self.W_B_list = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size))
            for _ in range(self.num_blocks)
        ])
        
        self.sinkhorn = SinkhornKnopp()
        self.temperature = temperature
        
        # 对称化约束: (W + W^T) / 2
        self._apply_symmetry = True
    
    def _get_constrained_weights(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """获取约束后的权重 (双随机性、稀疏性、对称性)"""
        # Global权重对称化
        W_G_sym = (self.W_G_raw + self.W_G_raw.t()) / 2 if self._apply_symmetry else self.W_G_raw
        
        # Sinkhorn-Knopp归一化
        W_G = self.sinkhorn(W_G_sym, self.temperature)
        
        # Local权重处理
        W_B_constrained = []
        for W_B in self.W_B_list:
            W_B_sym = (W_B + W_B.t()) / 2 if self._apply_symmetry else W_B
            W_B_norm = self.sinkhorn(W_B_sym, self.temperature)
            W_B_constrained.append(W_B_norm)
        
        return W_G, W_B_constrained
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 输入token序列
        Returns:
            mixed: [B, T, D] 混合后的输出
        """
        B, T, D = x.shape
        
        # 获取约束后的权重
        W_G, W_B_list = self._get_constrained_weights()
        
        # 按照论文优化后的计算流程
        # 输入维度 L = T * D (总嵌入维度)
        # 但UniMixing是在每个token的维度D上进行的（跨token混合）
        
        # 实际上，根据论文图2和公式，UniMixing是在token维度上进行的
        # 输入x: [B, T, D]，其中D = dim_per_token
        # 我们将D分成num_blocks个块，每块block_size维
        
        # Reshape: [B, T, D] -> [B, T, num_blocks, block_size]
        x_blocks = x.reshape(B, T, self.num_blocks, self.block_size)
        
        # Step 1: Local mixing (块内交互)
        # 对每个block应用对应的W_B^i
        local_mixed = []
        for i in range(self.num_blocks):
            # x_blocks[:, :, i, :]: [B, T, block_size]
            # W_B_list[i]: [block_size, block_size]
            mixed_i = torch.matmul(x_blocks[:, :, i, :], W_B_list[i])  # [B, T, block_size]
            local_mixed.append(mixed_i.unsqueeze(2))  # [B, T, 1, block_size]
        
        local_mixed = torch.cat(local_mixed, dim=2)  # [B, T, num_blocks, block_size]
        
        # Step 2: Global mixing (块间交互)
        # 转置: [B, T, num_blocks, block_size] -> [B, T, block_size, num_blocks]
        local_mixed_t = local_mixed.transpose(2, 3)  # [B, T, block_size, num_blocks]
        
        # 应用W_G: [B, T, block_size, num_blocks] @ [num_blocks, num_blocks]^T
        # W_G: [num_blocks, num_blocks]
        global_mixed = torch.matmul(local_mixed_t, W_G.t())  # [B, T, block_size, num_blocks]
        
        # 转置回来: [B, T, num_blocks, block_size]
        global_mixed = global_mixed.transpose(2, 3)
        
        # Step 3: Reshape回原始形状 [B, T, D]
        output = global_mixed.reshape(B, T, D)
        
        return output


class UniMixingLite(nn.Module):
    """
    UniMixing-Lite (论文4.3节)
    使用低秩近似和基矩阵动态生成，减少参数
    
    W_G ≈ A_G * B_G (低秩分解)
    W_B^i = Σ ω^i_l * Z_l (基矩阵线性组合)
    """
    def __init__(
        self,
        dim: int,
        block_size: int = 16,
        num_bases: int = 4,  # 基矩阵数量b
        rank: int = 16,       # 低秩r
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        self.num_bases = num_bases
        self.rank = rank
        
        # Global mixing低秩分解: W_G ≈ A_G * B_G
        self.A_G = nn.Parameter(torch.randn(self.num_blocks, rank))
        self.B_G = nn.Parameter(torch.randn(rank, self.num_blocks))
        
        # Local mixing基矩阵
        self.basis_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(block_size, block_size))
            for _ in range(num_bases)
        ])
        
        # 每个块的基组合权重
        self.omega = nn.Parameter(torch.randn(self.num_blocks, num_bases))
        
        self.sinkhorn = SinkhornKnopp()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # 构建低秩W_G
        W_G_raw = torch.matmul(self.A_G, self.B_G)  # [num_blocks, num_blocks]
        W_G_sym = (W_G_raw + W_G_raw.t()) / 2
        W_G = self.sinkhorn(W_G_sym, self.temperature)
        
        # 动态生成W_B^i
        W_B_list = []
        for i in range(self.num_blocks):
            # 加权组合基矩阵
            W_B_i = sum(
                self.omega[i, l] * self.basis_matrices[l]
                for l in range(self.num_bases)
            )
            W_B_sym = (W_B_i + W_B_i.t()) / 2
            W_B_norm = self.sinkhorn(W_B_sym, self.temperature)
            W_B_list.append(W_B_norm)
        
        # 后续计算与UniMixing相同
        x_blocks = x.reshape(B, T, self.num_blocks, self.block_size)
        
        local_mixed = []
        for i in range(self.num_blocks):
            mixed_i = torch.matmul(x_blocks[:, :, i, :], W_B_list[i])
            local_mixed.append(mixed_i.unsqueeze(2))
        
        local_mixed = torch.cat(local_mixed, dim=2)
        local_mixed_t = local_mixed.transpose(2, 3)
        global_mixed = torch.matmul(local_mixed_t, W_G.t())
        global_mixed = global_mixed.transpose(2, 3)
        
        output = global_mixed.reshape(B, T, D)
        
        return output


class PerTokenSwiGLU(nn.Module):
    """
    Per-token SwiGLU (论文4.3节)
    每个token独立的SwiGLU，建模特征异构性
    """
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        # 每个token独立的投影参数
        self.W_up = nn.Linear(dim, hidden_dim)
        self.W_gate = nn.Linear(dim, hidden_dim)
        self.W_down = nn.Linear(hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (W_up * x) ⊗ Swish(W_gate * x)
        up = self.W_up(x)
        gate = F.silu(self.W_gate(x))  # Swish = SiLU
        hidden = up * gate
        output = self.W_down(hidden)
        return self.dropout(output)


class UniMixerBlock(nn.Module):
    """
    UniMixer Block (论文4.3节，图2)
    结构: RMSNorm -> UniMixing -> Add -> RMSNorm -> PerTokenSwiGLU -> Add
    配合SiameseNorm使用
    """
    def __init__(
        self,
        dim: int,
        block_size: int = 16,
        use_lite: bool = False,
        num_bases: int = 4,
        rank: int = 16,
        expansion_factor: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        # UniMixing或UniMixing-Lite
        if use_lite:
            self.mixer = UniMixingLite(
                dim=dim,
                block_size=block_size,
                num_bases=num_bases,
                rank=rank,
                temperature=temperature,
            )
        else:
            self.mixer = UniMixing(
                dim=dim,
                block_size=block_size,
                temperature=temperature,
            )
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Per-token SwiGLU
        self.ffn = PerTokenSwiGLU(dim, expansion_factor, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准Pre-Norm结构
        """
        # UniMixing with residual
        normed = self.norm1(x)
        mixed = self.mixer(normed)
        x = x + mixed
        
        # PerTokenSwiGLU with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class UniMixer(nn.Module):
    """
    UniMixer完整架构 (论文4.1-4.3节，图2)
    
    结构: 
    1. Feature Tokenization (Embedding + Split + Projection)
    2. M个UniMixer Blocks (配合SiameseNorm)
    3. Task Towers
    """
    def __init__(
        self,
        # 特征配置
        feature_dims: Dict[str, int],  # 各特征域维度
        num_tokens: int = 16,          # T: token数量 (论文N=16)
        dim_per_token: int = 768,      # D: 每个token维度
        num_blocks: int = 2,           # M: UniMixer block数量
        # Block配置
        block_size: int = 16,
        use_lite: bool = False,
        num_bases: int = 4,
        rank: int = 16,
        expansion_factor: float = 2.0,
        dropout: float = 0.1,
        temperature: float = 1.0,
        # 训练策略
        use_siamese_norm: bool = True,
    ):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.dim_per_token = dim_per_token
        self.num_blocks = num_blocks
        self.use_siamese_norm = use_siamese_norm
        
        # 1. Feature Tokenization (论文4.2节)
        # 修复：使用明确的特征类型配置，而不是根据维度判断
        # 默认所有特征都使用Linear投影（连续特征）
        self.feature_embeddings = nn.ModuleDict()
        for name, dim in feature_dims.items():
            # 统一使用Linear投影到dim_per_token维度
            self.feature_embeddings[name] = nn.Linear(dim, dim_per_token)
        
        # 计算总维度并投影到num_tokens * dim_per_token
        # 每个特征域输出 dim_per_token，总维度 = len(feature_dims) * dim_per_token
        total_feature_dim = len(feature_dims) * dim_per_token
        self.token_proj = nn.Sequential(
            nn.Linear(total_feature_dim, num_tokens * dim_per_token),
            nn.LayerNorm(num_tokens * dim_per_token),
        )
        
        # 2. UniMixer Blocks
        self.blocks = nn.ModuleList([
            UniMixerBlock(
                dim=dim_per_token,
                block_size=block_size,
                use_lite=use_lite,
                num_bases=num_bases,
                rank=rank,
                expansion_factor=expansion_factor,
                dropout=dropout,
                temperature=temperature,
            ) for _ in range(num_blocks)
        ])
        
        # SiameseNorm (如果使用)
        if use_siamese_norm:
            self.siamese_norms = nn.ModuleList([
                SiameseNorm(dim_per_token) for _ in range(num_blocks)
            ])
        
        # 3. Task Towers (多任务)
        self.task_norm = RMSNorm(dim_per_token)
        total_dim = num_tokens * dim_per_token
        self.task_head = nn.Sequential(
            nn.Linear(total_dim, dim_per_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_per_token, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _tokenize_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        特征token化 (论文4.2节)
        """
        # 收集所有特征embedding
        embeddings = []
        for name, feat in features.items():
            emb_layer = self.feature_embeddings[name]
            # 统一使用Linear层处理
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1).float()
            elif feat.dim() == 2:
                feat = feat.float()
            
            emb = emb_layer(feat)
            
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
        
        # 拼接并投影
        concat = torch.cat(embeddings, dim=-1)  # [B, len(features) * dim_per_token]
        tokens_flat = self.token_proj(concat)  # [B, num_tokens * dim_per_token]
        
        # Reshape为token序列: [B, num_tokens, dim_per_token]
        B = tokens_flat.shape[0]
        tokens = tokens_flat.reshape(B, self.num_tokens, self.dim_per_token)
        
        return tokens
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 特征字典，每个特征 [B, dim] 或 [B]
        Returns:
            logits: [B, 1] 预测分数
        """
        # Tokenization
        x = self._tokenize_features(features)  # [B, T, D]
        
        # 通过UniMixer Blocks
        if self.use_siamese_norm:
            # SiameseNorm双分支
            y = x.clone()  # 辅助分支Y_0 = X_0
            
            for i, (block, siamese_norm) in enumerate(zip(self.blocks, self.siamese_norms)):
                # 按照论文公式(19): O_l = UniMixer(X_l + RMSNorm(Y_l))
                y_normed = siamese_norm.norm_y(y)
                mixed_input = x + y_normed
                mixer_output = block.mixer(mixed_input)
                
                # SiameseNorm更新
                x, y = siamese_norm(x, y, mixer_output)
                
                # 继续通过FFN (在X分支上)
                x = x + block.ffn(block.norm2(x))
            
            # 最终融合
            x = siamese_norm.fuse(x, y)
        else:
            # 标准结构
            for block in self.blocks:
                x = block(x)
        
        # Task Tower
        x_norm = self.task_norm(x)
        x_flat = x_norm.reshape(x_norm.shape[0], -1)
        logits = self.task_head(x_flat)
        
        return logits
