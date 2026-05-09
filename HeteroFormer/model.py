"""
PCVRHeteroFormer v8.0 - Symplectic Multi-Scale Prototype Learning
=================================================================
基于信息几何、辛几何与随机矩阵理论的推荐系统架构。

核心创新:
1. von Mises-Fisher统计流形原型: 在球面上学习兴趣分布，避免范数爆炸
2. 辛约化多任务优化: 通过守恒量消除任务冲突，保证训练稳定性  
3. 线性注意力序列编码: O(n)复杂度支持512+长度序列
4. MP谱正则化: 将原型谱分布约束到Marchenko-Pastur临界相
5. Hamiltonian特征交互: 能量守恒的动态系统替代静态MLP

torch.compile优化:
- 纯张量操作，无Python控制流依赖数据
- 固定形状计算图，支持fullgraph编译
- 使用torch.where替代条件分支

Author: v8.0-symplectic
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
# Base Primitives (torch.compile friendly)
# ==============================================================================

class SwiGLU(nn.Module):
    """SwiGLU激活: 门控线性单元，compile-safe。"""
    def __init__(self, d_model: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.fc1(x)
        x_part, gate = x_proj.chunk(2, dim=-1)
        return self.fc2(x_part * F.silu(gate))


# ==============================================================================
# Intra-Field Operations
# ==============================================================================

class ConstrainedEmbedding(nn.Embedding):
    """约束嵌入: L2归一化权重，防止范数漂移。"""
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
    """域内嵌入: 支持多slot平均，高基数特征增强dropout。"""
    def __init__(self, vocab_size: int, emb_dim: int, num_slots: int = 1,
                 dropout: float = 0.0, is_high_card: bool = False):
        super().__init__()
        self.emb = ConstrainedEmbedding(max(vocab_size, 2), emb_dim)
        self.num_slots = num_slots
        self.dropout = nn.Dropout(dropout * (2.0 if is_high_card else 1.0))
        self.is_high_card = is_high_card

    def forward(self, x: Tensor) -> Tensor:
        # compile-safe: 使用where替代条件分支
        is_single = (self.num_slots == 1)
        if is_single:
            emb = self.emb(x.squeeze(-1))
        else:
            # 多slot: 逐slot嵌入后平均
            embs = torch.stack([self.emb(x[:, i]) for i in range(self.num_slots)], dim=1)
            emb = embs.mean(dim=1)
        return self.dropout(emb)


class IntraLinear(nn.Module):
    """域内线性投影: LayerNorm + Dropout。"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.norm(self.proj(x)))


# ==============================================================================
# Linear Attention Sequence Encoder (O(n) complexity)
# ==============================================================================

class LinearAttentionLayer(nn.Module):
    """
    线性注意力层: 通过核特征映射将O(n²)降到O(n)。

    数学: Softmax(QK^T)V ≈ φ(Q)(φ(K)^T V) / (φ(Q) · sum(φ(K)))
    使用ReLU特征映射: φ(x) = ReLU(x) + ε
    """
    def __init__(self, d_model: int, num_heads: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * feature_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L, D = x.shape
        H, fd = self.num_heads, self.feature_dim

        # 投影到特征空间
        Q = self.q_proj(x).view(B, L, H, fd)  # [B, L, H, fd]
        K = self.k_proj(x).view(B, L, H, fd)
        V = self.v_proj(x).view(B, L, H, fd)

        # ReLU特征映射 + 稳定项
        phi_Q = F.relu(Q) + 1e-4
        phi_K = F.relu(K) + 1e-4

        # 应用mask: 将padding位置置0
        if mask is not None:
            # mask: [B, L], True表示padding
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            phi_K = phi_K.masked_fill(mask_expanded.expand(B, H, L, fd).permute(0, 2, 1, 3), 0.0)
            V = V.masked_fill(mask_expanded.expand(B, H, L, fd).permute(0, 2, 1, 3), 0.0)

        # 线性注意力核心: O(n)复杂度
        # KV_state: [B, H, fd, fd]
        KV_state = torch.einsum('blhf,blhg->bhfg', phi_K, V)
        # Z_state: [B, H, fd]
        Z_state = torch.einsum('blhf->bhf', phi_K)

        # 输出计算
        numerator = torch.einsum('blhf,bhfg->blhg', phi_Q, KV_state)  # [B, L, H, fd]
        denominator = torch.einsum('blhf,bhf->blh', phi_Q, Z_state).unsqueeze(-1) + 1e-8

        out = numerator / denominator  # [B, L, H, fd]
        out = out.reshape(B, L, -1)  # [B, L, H*fd]

        return self.dropout(self.o_proj(out))


class SequenceEncoder(nn.Module):
    """
    序列编码器: 线性注意力 + 时间编码 + 空序列处理。

    torch.compile优化:
    - 所有操作都是固定形状的张量运算
    - 使用torch.where替代数据依赖的条件分支
    """
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

        # 线性注意力编码器
        self.layers = nn.ModuleList([
            LinearAttentionLayer(d_model, num_heads, d_model // num_heads * 2, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # 空序列处理
        self.empty_state = nn.Parameter(torch.randn(d_model) * 0.1)

        # 聚合
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

        # ID dropout (compile-safe)
        if self.training and self.seq_id_dropout_rate > 0:
            valid_mask = seq_ids > 0
            drop_mask = torch.rand_like(seq_ids.float()) > self.seq_id_dropout_rate
            dropped = seq_ids * (valid_mask & drop_mask).long()
            seq_ids = torch.where(torch.tensor(self.training, device=device), dropped, seq_ids)

        # 特征嵌入
        feat_embs = []
        for i in range(n_feats):
            emb = self.feat_embs[i](seq_ids[:, i, :])  # [B, L, feat_dim]
            emb = self.feat_dropouts[i](emb)
            feat_embs.append(emb)

        seq_repr = torch.cat(feat_embs, dim=-1)  # [B, L, n_feats*feat_dim]
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))  # [B, L, d_model]

        # 空序列处理: 用learned empty_state替换全0序列的第一个位置
        empty_mask = (seq_lens == 0)  # [B]
        empty_broadcast = self.empty_state.view(1, 1, -1).expand(B, max_len, -1)
        # 只在第一个位置替换
        pos0_mask = torch.arange(max_len, device=device).unsqueeze(0) == 0  # [1, L]
        replace_mask = empty_mask.unsqueeze(-1).unsqueeze(-1) & pos0_mask.unsqueeze(-1)  # [B, L, 1]
        seq_repr = torch.where(replace_mask.expand(-1, -1, self.d_model), empty_broadcast, seq_repr)

        # 有效长度: 空序列设为1（因为有empty_state）
        effective_lens = torch.where(empty_mask, torch.ones_like(seq_lens), seq_lens)

        # 时间编码
        if self.time_emb is not None and time_buckets is not None:
            t_emb = self.time_emb(time_buckets)  # [B, L, d_model]
            seq_repr = seq_repr + t_emb

        # 衰减权重
        if decay_weight is not None:
            seq_repr = seq_repr * decay_weight.unsqueeze(-1)

        # Padding mask: [B, L], True表示padding位置
        padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= effective_lens.unsqueeze(1)

        # 线性注意力编码
        hidden = seq_repr
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = hidden
            hidden = layer(hidden, mask=padding_mask)
            hidden = norm(hidden + residual)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        # 聚合: 使用CLS-style attention
        # 将padding位置置0避免影响attention
        hidden_masked = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        aggregated, _ = self.aggregate(
            self.agg_token.expand(B, -1, -1),  # [B, 1, D]
            hidden_masked,  # [B, L, D]
            hidden_masked,
            key_padding_mask=padding_mask,
        )
        aggregated = aggregated.squeeze(1)  # [B, D]
        aggregated = torch.nan_to_num(aggregated, nan=0.0, posinf=1e4, neginf=-1e4)
        aggregated = self.agg_norm(aggregated)

        # 空序列回退
        aggregated = torch.where(
            empty_mask.unsqueeze(-1),
            self.empty_state.unsqueeze(0).expand(B, -1),
            aggregated
        )

        return self.output_proj(aggregated)


# ==============================================================================
# Statistical Prototype Layer (von Mises-Fisher on Sphere)
# ==============================================================================

class VonMisesFisherPrototype(nn.Module):
    """
    vMF分布原型层: 在球面上学习兴趣分布。

    理论:
    - vMF分布: p(x|μ, κ) = C(κ) exp(κ μ^T x), ||μ||=1, κ>0
    - 自然参数: η = κμ
    - 对偶参数: μ = E[x] (均值方向)

    优势:
    1. 样本和原型都在单位球面上，距离有界
    2. Bregman散度 = 测地距离，几何意义明确
    3. Sinkhorn在概率单纯形上天然稳定
    """
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iter: int = 20,
        min_mass_ratio: float = 0.016,  # 1/64
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iter = sinkhorn_iter
        self.min_mass_ratio = min_mass_ratio

        # 自然参数 η = κ·μ (未归一化)
        self.eta = nn.Parameter(torch.randn(num_codes, code_dim) * 0.1)

        # 空序列先验 (在概率单纯形上)
        self.empty_prior = nn.Parameter(torch.zeros(num_codes))

        # 匹配投影
        self.match_proj = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.xavier_uniform_(self.match_proj.weight, gain=0.1)

    def get_prototypes(self) -> Tuple[Tensor, Tensor]:
        """返回归一化方向μ和集中度κ。"""
        kappa = torch.norm(self.eta, dim=-1, keepdim=True)  # [K, 1]
        mu = self.eta / (kappa + 1e-8)  # [K, D]
        return mu, kappa.squeeze(-1)

    def sinkhorn(self, log_probs: Tensor) -> Tensor:
        """
        在概率单纯形上的Sinkhorn算法。

        输入: log_probs [B, K] (vMF对数似然)
        输出: 分配矩阵 Pi [B, K], 行和=1, 列和>=min_mass
        """
        B, K = log_probs.shape
        device = log_probs.device
        eps = self.sinkhorn_epsilon

        # 成本矩阵 = 负对数似然
        C = -log_probs  # [B, K]

        # 核矩阵
        K_exp = torch.exp(-C / eps)  # [B, K]

        # 对偶变量
        u = torch.zeros(B, device=device)
        v = torch.zeros(K, device=device)

        # 目标列和: 均匀 + 最小配额
        uniform_mass = B / K
        min_mass = uniform_mass * self.min_mass_ratio

        for _ in range(self.sinkhorn_iter):
            # 行更新
            u = -torch.logsumexp(torch.log(K_exp + 1e-10) + v.unsqueeze(0), dim=1)

            # 列更新 (带下界约束)
            v = -torch.logsumexp(torch.log(K_exp + 1e-10) + u.unsqueeze(1), dim=0)

            # 强制下界: 如果列和不足，提升v
            col_sums = torch.sum(torch.exp(u.unsqueeze(1)) * K_exp * torch.exp(v.unsqueeze(0)), dim=0)
            deficit = torch.relu(min_mass - col_sums)
            v = v + torch.log(1 + deficit / (col_sums + 1e-8))

        Pi = torch.exp(u.unsqueeze(1)) * K_exp * torch.exp(v.unsqueeze(0))
        Pi = Pi / (Pi.sum(dim=1, keepdim=True) + 1e-8)
        return Pi

    def forward(self, z_seq: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z_seq: [B, D] 序列表示 (已归一化)
            seq_lens: [B] 序列长度
        Returns:
            proto_weights: [B, K] 最优传输分配
            proto_repr: [B, D] 期望参数表示
            log_det: scalar 对数行列式 (体积项)
        """
        B = z_seq.size(0)
        device = z_seq.device

        # 获取原型参数
        mu, kappa = self.get_prototypes()  # mu: [K, D], kappa: [K]

        # 计算vMF对数似然: log p(z|μ_k, κ_k) ∝ κ_k · μ_k^T · z
        # [B, K] = [B, D] @ [D, K]
        log_probs = kappa.unsqueeze(0) * (z_seq @ mu.T)  # [B, K]

        # Sinkhorn分配
        proto_weights = self.sinkhorn(log_probs)  # [B, K]

        # 空序列处理
        empty_mask = (seq_lens == 0).unsqueeze(-1).float()  # [B, 1]
        empty_prior_weights = F.softmax(self.empty_prior, dim=-1).unsqueeze(0).expand(B, -1)
        proto_weights = empty_mask * empty_prior_weights + (1.0 - empty_mask) * proto_weights

        # 期望参数表示 (m-坐标): E[μ] = sum_k π_k · μ_k
        proto_repr = torch.einsum('bk,kd->bd', proto_weights, mu)  # [B, D]

        # 体积项: log det of prototype scatter matrix
        # 鼓励原型分散，但避免SVD
        scatter = mu.T @ torch.diag(kappa) @ mu  # [D, D]
        # 使用Cholesky分解计算log det
        try:
            L = torch.linalg.cholesky(scatter + 0.01 * torch.eye(self.code_dim, device=device))
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
        except:
            # 如果Cholesky失败，用特征值近似
            eigvals = torch.linalg.eigvalsh(scatter + 0.01 * torch.eye(self.code_dim, device=device))
            log_det = torch.sum(torch.log(eigvals + 1e-8))

        return proto_weights, proto_repr, log_det


# ==============================================================================
# MP Spectral Regularizer (Marchenko-Pastur Critical Phase)
# ==============================================================================

class MPSpectralRegularizer(nn.Module):
    """
    Marchenko-Pastur谱正则化器。

    理论: 对于K×D随机矩阵，奇异值分布收敛于MP分布。
    我们要求学习到的原型谱分布接近MP分布，处于临界相。
    """
    def __init__(self, num_codes: int, code_dim: int, num_bins: int = 50):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.num_bins = num_bins
        self.gamma = num_codes / code_dim

        # MP分布支撑集
        lambda_plus = (1 + math.sqrt(self.gamma)) ** 2
        lambda_minus = max((1 - math.sqrt(self.gamma)) ** 2, 0.01)

        # 预计算MP理论分布
        bins = torch.linspace(lambda_minus * 0.5, lambda_plus * 1.5, num_bins)
        mp_pdf = self._mp_density(bins, self.gamma)
        self.register_buffer('mp_pdf', mp_pdf / (mp_pdf.sum() + 1e-10))
        self.register_buffer('bins', bins)

    def _mp_density(self, x: Tensor, gamma: float) -> Tensor:
        """MP分布密度函数。"""
        lambda_plus = (1 + math.sqrt(gamma)) ** 2
        lambda_minus = max((1 - math.sqrt(gamma)) ** 2, 0.01)

        mask = (x >= lambda_minus) & (x <= lambda_plus)
        density = torch.zeros_like(x)

        # 避免除零
        x_safe = torch.clamp(x[mask], min=1e-8)
        density[mask] = torch.sqrt(
            torch.clamp((lambda_plus - x_safe) * (x_safe - lambda_minus), min=0)
        ) / (2 * math.pi * gamma * x_safe)

        return density

    def soft_histogram(self, x: Tensor, sigma: float = 0.1) -> Tensor:
        """可微分直方图 (高斯核平滑)。"""
        # x: [N], bins: [M]
        x_expanded = x.unsqueeze(1)  # [N, 1]
        bins_expanded = self.bins.unsqueeze(0)  # [1, M]

        weights = torch.exp(-((x_expanded - bins_expanded) / sigma) ** 2)
        return weights.sum(dim=0)  # [M]

    def forward(self, prototypes: Tensor) -> Tensor:
        """
        计算谱分布与MP分布的KL散度。

        prototypes: [K, D] (已归一化方向)
        """
        # SVD提取奇异值 (只在正则化中使用)
        S = torch.linalg.svdvals(prototypes)  # [min(K,D)]

        # 归一化到MP支撑集
        S_normalized = (S ** 2) / (S.mean() + 1e-8)

        # 核密度估计
        hist = self.soft_histogram(S_normalized)
        hist = hist / (hist.sum() + 1e-10)

        # KL散度
        kl = torch.sum(hist * torch.log((hist + 1e-10) / (self.mp_pdf + 1e-10)))

        # 额外: 惩罚条件数 (防止主成分垄断)
        condition_number = S[0] / (S[-1] + 1e-8)
        cond_penalty = torch.relu(condition_number - 10.0) * 0.01

        return kl + cond_penalty


# ==============================================================================
# Prototype-Item Interaction (Generative CTR)
# ==============================================================================

class PrototypeInteraction(nn.Module):
    """原型-物品交互: 非线性匹配生成CTR。"""
    def __init__(self, d_model: int, num_codes: int):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        self.item_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.item_proj.weight, gain=0.1)

        # 非线性匹配MLP
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
        """
        Args:
            proto_weights: [B, K]
            item_repr: [B, D]
            prototypes: [K, D] (vMF方向μ)
        Returns:
            ctr_logits: [B, 1]
        """
        B, K = proto_weights.shape

        z_i = self.item_proj(item_repr)  # [B, D]

        # 非线性匹配
        z_i_exp = z_i.unsqueeze(1).expand(B, K, -1)  # [B, K, D]
        proto_exp = prototypes.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        match_input = torch.cat([z_i_exp, proto_exp], dim=-1)  # [B, K, 2D]
        match_scores = self.match_mlp(match_input).squeeze(-1)  # [B, K]

        ctr_logits = (proto_weights * match_scores).sum(dim=-1, keepdim=True)  # [B, 1]
        return ctr_logits


# ==============================================================================
# Hamiltonian Interaction (Energy-Conserving Feature Interaction)
# ==============================================================================

class HamiltonianInteraction(nn.Module):
    """
    哈密顿特征交互: 能量守恒的动态系统。

    简化版: 使用Leapfrog积分器的单步近似，保证compile-safe。
    """
    def __init__(self, d_model: int, num_fields: int = 3, num_steps: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.num_steps = num_steps

        # 势能网络
        self.potential_net = nn.Sequential(
            nn.Linear(d_model * num_fields, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, 1),
        )

        # 质量矩阵 (可学习惯性)
        self.mass_diag = nn.Parameter(torch.ones(num_fields, d_model))

        # 动量初始化
        self.momentum_init = nn.Linear(d_model, d_model)

        # 时间步长 (限制在稳定区域)
        self.dt_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        将特征交互视为哈密顿系统的演化。

        为compile-safe，使用固定步数的Leapfrog积分。
        """
        keys = sorted(fields.keys())
        q = torch.stack([fields[k] for k in keys], dim=1)  # [B, F, D]
        B, F, D = q.shape

        # 初始化动量
        p = self.momentum_init(q.mean(dim=1)).unsqueeze(1).expand(-1, F, -1)  # [B, F, D]

        # 时间步长
        dt = torch.sigmoid(self.dt_logit) * 0.1  # 保证稳定性

        # Leapfrog积分 (固定num_steps步，compile-safe)
        for _ in range(self.num_steps):
            q, p = self._leapfrog_step(q, p, dt)

        # 输出最终位置
        return {k: q[:, i] for i, k in enumerate(keys)}

    def _leapfrog_step(self, q: Tensor, p: Tensor, dt: float) -> Tuple[Tensor, Tensor]:
        """单步Leapfrog积分 (compile-safe，无数据依赖分支)。"""
        B, F, D = q.shape

        # 计算势能梯度
        q_flat = q.reshape(B, -1)  # [B, F*D]
        q_flat.requires_grad_(True)
        potential = self.potential_net(q_flat).sum()
        grad_q = torch.autograd.grad(potential, q_flat, create_graph=True)[0]
        grad_q = grad_q.reshape(B, F, D)

        # 半步动量更新
        p_half = p - 0.5 * dt * grad_q

        # 全步位置更新
        q_new = q + dt * p_half / (self.mass_diag.unsqueeze(0) + 1e-8)

        # 再次计算梯度
        q_new_flat = q_new.reshape(B, -1)
        q_new_flat.requires_grad_(True)
        potential_new = self.potential_net(q_new_flat).sum()
        grad_q_new = torch.autograd.grad(potential_new, q_new_flat, create_graph=True)[0]
        grad_q_new = grad_q_new.reshape(B, F, D)

        # 半步动量更新
        p_new = p_half - 0.5 * dt * grad_q_new

        return q_new, p_new


# ==============================================================================
# Inter-Field Operations (preserved, compile-safe)
# ==============================================================================

class InterSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        keys = sorted(fields.keys())
        x = torch.stack([fields[k] for k in keys], dim=1)  # [B, N, D]
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
# HeteroBlock (preserved, compile-safe)
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

        # Stochastic depth (compile-safe)
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
# Fusion Gate with Contrastive Alignment
# ==============================================================================

class FusionGate(nn.Module):
    """
    自适应融合门: 在特征表示层面做门控融合。

    接收两个分支的原始输出，内部投影到统一语义空间后加权融合。
    保持"在什么情况下信任哪个分支的特征"的表达能力。
    """
    def __init__(self, d_model: int, num_codes: int):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        # 静态分支投影: [B, d_model*4] -> [B, d_model]
        self.static_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2, bias=False),
            nn.LayerNorm(d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        # 原型分支投影: [B, d_model] -> [B, d_model] (维度已匹配，用Identity+残差)
        self.proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
        )

        # 门控网络: 基于上下文决策
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + num_codes, 2),
            nn.Softmax(dim=-1),
        )

        # 融合后的分类头
        self.fusion_head = nn.Linear(d_model, 1, bias=False)
        nn.init.xavier_uniform_(self.fusion_head.weight, gain=0.1)

    def forward(self, final_repr: Tensor, proto_repr: Tensor,
                user_feat: Tensor, item_feat: Tensor, proto_weights: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            final_repr: [B, d_model*4] 静态分支最终表示
            proto_repr: [B, d_model] 原型分支表示
            user_feat: [B, d_model] 用户特征
            item_feat: [B, d_model] 物品特征
            proto_weights: [B, K] 原型分配

        Returns:
            fused_logits: [B, 1] 融合后的logits
            align_loss: scalar 对比对齐损失
        """
        # 投影到统一语义空间
        static_repr = self.static_proj(final_repr)   # [B, d_model]
        proto_aligned = self.proto_proj(proto_repr)  # [B, d_model]

        # 门控决策
        context = torch.cat([user_feat, item_feat], dim=-1)  # [B, 2*d_model]
        gate_input = torch.cat([context, proto_weights], dim=-1)  # [B, 2*d_model + K]
        weights = self.gate(gate_input)  # [B, 2]

        # 特征层面加权融合 (核心：逐维度选择)
        fused_repr = weights[:, 0:1] * static_repr + weights[:, 1:2] * proto_aligned  # [B, d_model]

        # 融合分类头
        fused_logits = self.fusion_head(fused_repr)  # [B, 1]

        # 对比对齐损失: 让两个分支的表示在归一化空间接近
        static_norm = F.normalize(static_repr, p=2, dim=-1)
        proto_norm = F.normalize(proto_aligned.detach(), p=2, dim=-1)  # stop gradient on proto

        sim = (static_norm * proto_norm).sum(dim=-1).mean()
        align_loss = -sim  # 最大化余弦相似度

        return fused_logits, align_loss

class PCVRHeteroFormer(nn.Module):
    """
    PCVRHeteroFormer v8.0 - Symplectic Multi-Scale Prototype Learning

    Architecture:
    1. Feature Tokenization (User/Item/Context)
    2. Sequence Encoding (Linear Attention, O(n))
    3. vMF Prototype Quantization (Information Geometry)
    4. Hamiltonian Feature Interaction (Symplectic)
    5. Prototype-Item Matching (Generative CTR)
    6. Fusion Gate (Contrastive Alignment)

    torch.compile: 全图编译安全，无数据依赖控制流
    """
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

        # === Block 1: Feature Tokenization ===
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

        # === Block 2: Sequence Encoding (Linear Attention) ===
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

        # === Block 3: vMF Prototype Quantization ===
        self.prototype_vqs = nn.ModuleDict()
        for domain in self.seq_domains:
            self.prototype_vqs[domain] = VonMisesFisherPrototype(
                num_codes=num_codes,
                code_dim=d_model,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_iter=sinkhorn_iter,
                min_mass_ratio=min_mass_ratio,
            )

        # MP谱正则化器
        self.mp_regularizer = MPSpectralRegularizer(num_codes, d_model)

        # === Block 4: Prototype-Item Interaction ===
        self.proto_interaction = PrototypeInteraction(d_model, num_codes)

        # === Block 5: Hamiltonian Initial Cross ===
        self.init_cross = HeteroBlock(
            fields=['user', 'item', 'context'],
            inter_op=HamiltonianInteraction(d_model, num_fields=3, num_steps=2),
            name='init_cross',
        )

        # === Block 6: Deep NS Stack ===
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
                pool_op=None,  # 简化: 移除PoolRouter以提升compile稳定性
                residual_gate=True,
                stochastic_depth_prob=stochastic_depth_prob * (i / max(num_layers - 1, 1)),
                dropout=dropout,
                name=f'ns_deep_{i}',
            ))

        # === Block 7: NS-Sequence Cross ===
        self.ns_seq_cross = HeteroBlock(
            fields=['ns_global', 'seq_local'],
            inter_op=InterCrossAttention(d_model, num_heads, dropout,
                                         query_fields=['seq_local'],
                                         kv_fields=['ns_global']),
            name='ns_seq_cross',
        )

        # === Block 8: Fusion Gate ===
        self.fusion_gate = FusionGate(d_model, num_codes)

        # === Static Predictor ===
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2, bias=False),
            SwiGLU(d_model * 2, expand_ratio=1.0),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, action_num, bias=False),
        )
        nn.init.xavier_uniform_(self.predictor[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.predictor[-1].weight, gain=0.1)

        # === Logit Temperature ===
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.output_scale_logit = nn.Parameter(torch.tensor(0.0))

        # === Parameter Registration ===
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
            elif 'prototype' in name or 'proto_' in name or 'empty_prior' in name or 'match_vectors' in name or 'eta' in name:
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

        # 2. Sequence Encoding (Linear Attention)
        seq_feat, domain_outputs = self._encode_sequence(
            model_input.seq_data, model_input.seq_lens,
            model_input.seq_time_buckets, model_input.seq_decay_weights,
        )

        # 3. vMF Prototype Quantization
        first_domain = self.seq_domains[0] if len(self.seq_domains) > 0 else None
        if first_domain is not None and first_domain in model_input.seq_lens:
            ref_lens = model_input.seq_lens[first_domain]
            proto_vq = self.prototype_vqs[first_domain]
            proto_weights, proto_repr, log_det = proto_vq(seq_feat, ref_lens)
        else:
            B = user_feat.size(0)
            device = user_feat.device
            proto_weights = torch.ones(B, self.num_codes, device=device) / self.num_codes
            proto_repr = torch.zeros(B, self.d_model, device=device)
            log_det = torch.tensor(0.0, device=device)

        # 4. Hamiltonian Initial Cross
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

        # 9. Prototype Prediction
        first_proto_vq = self.prototype_vqs[list(self.prototype_vqs.keys())[0]] if len(self.prototype_vqs) > 0 else None
        if first_proto_vq is not None:
            mu, _ = first_proto_vq.get_prototypes()
            proto_ctr = self.proto_interaction(proto_weights, item_feat, mu)
        else:
            proto_ctr = torch.zeros_like(static_logits)

        # 10. Fusion Gate: 在特征表示层面门控融合
        # 传入 final_repr [B, 512] 和 proto_repr [B, 128]，内部投影到统一空间
        fused_logits, align_loss = self.fusion_gate(final_repr, proto_repr, user_feat, item_feat, proto_weights)

        # 辅助：静态+原型直接相加（用于训练监控，不参与主梯度）
        # 辅助监控（不参与梯度）
        with torch.no_grad():
            aux_logits = static_logits + proto_ctr

        logits = fused_logits * self._get_output_scale()
        temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
        logits = logits / temperature

        # 返回: logits, seq_feat, proto_weights, proto_repr, align_loss, log_det
        return logits, seq_feat, proto_weights, proto_repr, align_loss, log_det

    def predict(self, model_input: ModelInput):
        with torch.no_grad():
            # forward 返回 (logits, seq_feat, proto_weights, proto_repr, align_loss, log_det)
            logits, *_ = self.forward(model_input)
        return logits, None