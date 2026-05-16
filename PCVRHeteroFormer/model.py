"""
PCVRHeteroFormer v9.3 - Generative Fusion & Overfit-Aware MetaAligner
======================================================================
v9.3核心修改：
1. MetaAligner: 基于valid/train AUC过拟合检测的动态任务权重，移除硬编码step_ratio冻结
2. DiffusionHead/EnergyHead: 新增get_gen_repr()路径，输出生成式语义表示（不detach）
3. GenerativeFusion: 融合原型/扩散/能量三种表示，门控机制供CTR使用
4. CTR路径: 训练时使用融合表示，eval时回退到原始proto_repr（保证确定性）
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


def create_noise_like(x: Tensor, is_training: bool) -> Tensor:
    noise = torch.randn_like(x)
    return noise * torch.tensor(is_training, dtype=x.dtype, device=x.device)


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    x_masked = x * mask.unsqueeze(-1)
    sum_x = x_masked.sum(dim=dim)
    count = mask.sum(dim=dim).clamp(min=1.0)
    return sum_x / count.unsqueeze(-1)


def check_nan_tensor(tensor: Tensor, name: str = "tensor") -> bool:
    has_nan = torch.isnan(tensor).any() or torch.isinf(tensor).any()
    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logging.warning(f"【NaN检测】{name}: NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
    return has_nan


# ==============================================================================
# Base Primitives (preserved)
# ==============================================================================

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden = int(d_model * expand_ratio)
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = self.fc1(x)
        x_part, gate = x_proj.chunk(2, dim=-1)
        return self.fc2(x_part * F.silu(gate))


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
    def __init__(self, vocab_size: int, emb_dim: int, num_slots: int = 1, is_high_card: bool = False):
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


class IntraLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))


class PoolConcatLinear(nn.Module):
    def __init__(self, field_dims: Dict[str, int], out_dim: int):
        super().__init__()
        total_in = sum(field_dims.values())
        self.proj = nn.Linear(total_in, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-5)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)

    def forward(self, fields: Dict[str, Tensor]) -> Tensor:
        x = torch.cat(list(fields.values()), dim=-1)
        return self.norm(self.proj(x))


# ==============================================================================
# v9.3: MultiViewEncoder (preserved)
# ==============================================================================

class MultiViewEncoder(nn.Module):
    def __init__(self, user_int_feature_specs, item_int_feature_specs, user_dense_dim, item_dense_dim,
                 d_model=128, emb_dim=16, id_vocab_threshold=10000, emb_skip_threshold=0, task_dims=None):
        super().__init__()
        self.d_model = d_model
        self.user_int_feature_specs = user_int_feature_specs
        self.item_int_feature_specs = item_int_feature_specs
        task_dims = task_dims or {'ctr': d_model, 'diff': d_model, 'energy': d_model}
        self.task_names = sorted(task_dims.keys())

        user_intra = {}
        user_dims = {}
        for idx, (vs, offset, length) in enumerate(user_int_feature_specs):
            if vs > 1 and (emb_skip_threshold <= 0 or vs <= emb_skip_threshold):
                fname = f'ufeat_{idx}'
                user_intra[fname] = IntraEmbedding(vs, emb_dim, length, vs > id_vocab_threshold)
                user_dims[fname] = emb_dim

        item_intra = {}
        item_dims = {}
        for idx, (vs, offset, length) in enumerate(item_int_feature_specs):
            if vs > 1 and (emb_skip_threshold <= 0 or vs <= emb_skip_threshold):
                fname = f'ifeat_{idx}'
                item_intra[fname] = IntraEmbedding(vs, emb_dim, length, vs > id_vocab_threshold)
                item_dims[fname] = emb_dim

        self.user_tokenize = HeteroBlock(
            fields=sorted(user_intra.keys()),
            intra_ops=user_intra,
            pool_op=PoolConcatLinear(user_dims, d_model) if user_dims else None,
            name='user_tokenize',
        )

        self.item_tokenize = HeteroBlock(
            fields=sorted(item_intra.keys()),
            intra_ops=item_intra,
            pool_op=PoolConcatLinear(item_dims, d_model) if item_dims else None,
            name='item_tokenize',
        )

        self.user_dense_proj = IntraLinear(user_dense_dim, d_model) if user_dense_dim > 0 else None
        self.item_dense_proj = IntraLinear(item_dense_dim, d_model) if item_dense_dim > 0 else None
        self.fusion_norm = nn.LayerNorm(d_model)

        self.view_projs = nn.ModuleDict()
        for task_name in self.task_names:
            dim = task_dims[task_name]
            self.view_projs[task_name] = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, dim), nn.SiLU(), nn.Linear(dim, dim),
            )
            for layer in self.view_projs[task_name]:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.1)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def forward(self, user_int, item_int, user_dense, item_dense):
        B = user_int.size(0)
        device = user_int.device

        user_fields = {}
        for i, (_, offset, length) in enumerate(self.user_int_feature_specs):
            fname = f'ufeat_{i}'
            if fname in self.user_tokenize.fields:
                user_fields[fname] = user_int[:, offset:offset+length]

        user_emb = self.user_tokenize(user_fields) if user_fields else None
        user_feat = user_emb[sorted(user_emb.keys())[0]] if user_emb else torch.zeros(B, self.d_model, device=device)
        if self.user_dense_proj is not None and user_dense.size(-1) > 0:
            user_feat = user_feat + self.user_dense_proj(user_dense)

        item_fields = {}
        for i, (_, offset, length) in enumerate(self.item_int_feature_specs):
            fname = f'ifeat_{i}'
            if fname in self.item_tokenize.fields:
                item_fields[fname] = item_int[:, offset:offset+length]

        item_emb = self.item_tokenize(item_fields) if item_fields else None
        item_feat = item_emb[sorted(item_emb.keys())[0]] if item_emb else torch.zeros(B, self.d_model, device=device)
        if self.item_dense_proj is not None and item_dense.size(-1) > 0:
            item_feat = item_feat + self.item_dense_proj(item_dense)

        context_feat = user_feat + item_feat
        z_shared = self.fusion_norm(user_feat + item_feat + context_feat)

        views = {task_name: self.view_projs[task_name](z_shared) for task_name in self.task_names}

        return {'shared': z_shared, 'user': user_feat, 'item': item_feat, 'context': context_feat, 'views': views}


# ==============================================================================
# v9.1: SSMCell (preserved)
# ==============================================================================

class SSMCell(nn.Module):
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        A_real = -torch.exp(torch.linspace(0, 3, state_dim))
        self.register_buffer('A_real', A_real)
        self.B_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(state_dim, input_dim, bias=False)
        self.D_skip = nn.Parameter(torch.ones(1) * 0.5)
        self.delta_proj = nn.Sequential(nn.Linear(1, state_dim), nn.Softplus())
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.1)

    def forward(self, x_seq: Tensor, time_deltas: Tensor) -> Tensor:
        B, L, D = x_seq.shape
        device = x_seq.device
        delta_scaled = self.delta_proj(time_deltas)
        delta_scaled = torch.clamp(delta_scaled, min=1e-4, max=1.0)
        A_discrete = torch.exp(self.A_real.unsqueeze(0).unsqueeze(0) * delta_scaled)
        Bx = self.B_proj(x_seq)
        Bx = torch.clamp(Bx, min=-10.0, max=10.0)

        logA = torch.log(A_discrete.clamp(min=1e-10))
        cumsum_logA = torch.cumsum(logA, dim=1)
        prefix_A = torch.exp(cumsum_logA)
        prefix_A_safe = prefix_A.clamp(min=1e-10)
        weighted_Bx = Bx / prefix_A_safe
        cumsum_weighted = torch.cumsum(weighted_Bx, dim=1)
        h_states = prefix_A * cumsum_weighted

        h_states = torch.where(
            torch.isnan(h_states) | torch.isinf(h_states),
            torch.zeros_like(h_states), h_states
        )
        h_states = torch.clamp(h_states, min=-100.0, max=100.0)

        y = self.C_proj(h_states) + self.D_skip * x_seq
        y = torch.clamp(y, min=-10.0, max=10.0)

        return y

class BilinearCrossLayer(nn.Module):
    """
    轻量级双线性交叉层：捕获域间二阶交互
    只保留上三角交互，避免参数量爆炸
    """
    def __init__(self, d_model: int, num_fields: int = 4, cross_ratio: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.cross_ratio = cross_ratio  # 交叉项残差比例，避免主导
        
        # 上三角交互参数：[i, j, d] for i < j
        num_pairs = num_fields * (num_fields - 1) // 2
        self.cross_weights = nn.Parameter(torch.randn(num_pairs, d_model) * 0.01)
        
        # 输出投影（稳定训练）
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, fields: List[Tensor]) -> List[Tensor]:
        """
        Args:
            fields: List[Tensor]，每个 [B, D]，如 [user_feat, item_feat, context_feat, seq_feat]
        Returns:
            交叉增强后的场表示，残差形式
        """
        num_actual = len(fields)
        if num_actual < 2:
            return fields
        
        out = [f.clone() for f in fields]
        pair_idx = 0
        
        for i in range(num_actual):
            for j in range(i + 1, num_actual):
                # 逐元素交互：[B, D] * [D] * [B, D] -> [B, D]
                interaction = fields[i] * self.cross_weights[pair_idx] * fields[j]
                
                # 残差叠加，避免主导原始表示
                out[i] = out[i] + self.cross_ratio * interaction
                out[j] = out[j] + self.cross_ratio * interaction
                pair_idx += 1
        
        # 归一化稳定
        out = [self.output_norm(o) for o in out]
        return out

# ==============================================================================
# v9.1: ContinuousSequenceEncoder (preserved)
# ==============================================================================

class ContinuousSequenceEncoder(nn.Module):
    def __init__(self, vocab_sizes, d_model=128, state_dim=64, num_layers=2, max_seq_len=512, id_threshold=10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        n_feats = len(vocab_sizes)
        feat_dim = max(d_model // n_feats, 1)
        self.feat_embs = nn.ModuleList()
        for vs in vocab_sizes:
            self.feat_embs.append(nn.Embedding(max(vs, 2), feat_dim))
            nn.init.normal_(self.feat_embs[-1].weight, std=0.01)
        self.seq_proj = nn.Linear(feat_dim * n_feats, d_model, bias=False)
        self.seq_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.seq_proj.weight, gain=0.1)
        self.ssm_layers = nn.ModuleList([SSMCell(d_model, state_dim) for _ in range(num_layers)])
        self.ssm_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.empty_mu = nn.Parameter(torch.randn(d_model) * 0.1)
        self.empty_log_sigma = nn.Parameter(torch.randn(d_model) * 0.01 - 2.0)
        self.short_proj = nn.Linear(d_model, d_model)
        self.long_proj = nn.Linear(d_model, d_model)
        self.static_proj = nn.Linear(d_model * 3, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, seq_ids, seq_lens, time_buckets=None, decay_weight=None, timestamps_raw=None):
        B, n_feats, max_len = seq_ids.shape
        device = seq_ids.device
        feat_embs = [self.feat_embs[i](seq_ids[:, i, :]) for i in range(n_feats)]
        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))

        if timestamps_raw is not None:
            time_deltas = torch.zeros(B, max_len, 1, device=device)
            if max_len > 1:
                time_deltas[:, 1:, 0] = timestamps_raw[:, 1:] - timestamps_raw[:, :-1]
                time_deltas = torch.clamp(time_deltas, min=1.0)
            else:
                time_deltas[:, :, 0] = 1.0
        else:
            time_deltas = torch.ones(B, max_len, 1, device=device)

        hidden = seq_repr
        for ssm, norm in zip(self.ssm_layers, self.ssm_norms):
            residual = hidden
            hidden = ssm(hidden, time_deltas)
            hidden = norm(hidden + residual)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)

        positions = torch.arange(max_len, device=device).unsqueeze(0)
        lens_expanded = seq_lens.unsqueeze(1)
        valid_mask = (positions < lens_expanded).float()

        empty_sigma = torch.exp(self.empty_log_sigma)
        empty_noise = create_noise_like(self.empty_mu, self.training)
        empty_state = self.empty_mu + empty_sigma * empty_noise

        short_weights = torch.exp(-torch.arange(max_len, device=device).float() / 10.0).unsqueeze(0) * valid_mask
        short_weights = short_weights / (short_weights.sum(dim=1, keepdim=True) + 1e-8)
        s_short = torch.sum(hidden * short_weights.unsqueeze(-1), dim=1)
        s_short = self.short_proj(s_short)

        long_weights = valid_mask / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        s_long = torch.sum(hidden * long_weights.unsqueeze(-1), dim=1)
        s_long = self.long_proj(s_long)

        seq_mean = masked_mean(hidden, valid_mask, dim=1)
        sq_mean = masked_mean(hidden ** 2, valid_mask, dim=1)
        seq_std = torch.sqrt(torch.clamp(sq_mean - seq_mean ** 2, min=0.0) + 1e-8)
        last_idx = (seq_lens - 1).clamp(min=0).long()
        seq_last = hidden[torch.arange(B, device=device), last_idx]
        s_static = self.static_proj(torch.cat([seq_mean, seq_std, seq_last], dim=-1))

        empty_mask = (seq_lens == 0).unsqueeze(-1).float()
        for s in [s_short, s_long, s_static]:
            s = torch.where(empty_mask.bool(), empty_state.unsqueeze(0), s)

        return {'short': self.output_norm(s_short), 'long': self.output_norm(s_long),
                'static': self.output_norm(s_static), 'full_seq': hidden}

class MultiSeqFusion(nn.Module):
    def __init__(self, d_model: int, num_domains: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_domains = num_domains
        
        self.domain_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            ) for _ in range(num_domains)
        ])
        
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq_reprs: List[Tensor], seq_lens_list: Optional[List[Tensor]] = None):
        if len(seq_reprs) == 0:
            B = 1
            return torch.zeros(B, self.d_model, device=next(self.parameters()).device), None, torch.tensor(0.0)
        
        B = seq_reprs[0].size(0)
        device = seq_reprs[0].device
        
        if len(seq_reprs) == 1:
            proj = self.domain_projs[0](seq_reprs[0])
            return self.output_proj(proj), torch.ones(B, 1, device=device), torch.tensor(0.0)
        
        projected = []
        for proj, r in zip(self.domain_projs[:len(seq_reprs)], seq_reprs):
            p = proj(r)
            projected.append(p)
        
        stacked = torch.stack(projected, dim=1)
        query = self.query.expand(B, -1, -1)
        attn_out, attn_weights = self.attn(query, stacked, stacked)
        
        gates = attn_weights.squeeze(1)
        fused = attn_out.squeeze(1)
        fused = self.output_proj(fused)
        
        return fused, gates, torch.tensor(0.0, device=device)


# ==============================================================================
# v9.1: CayleyRotation (preserved)
# ==============================================================================

class CayleyRotation(nn.Module):
    def __init__(self, d_model: int, user_cond_dim: int, rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.scale_net = nn.Sequential(nn.Linear(user_cond_dim, rank), nn.Tanh())

    def rotate(self, eta: Tensor, user_feat: Tensor) -> Tensor:
        K, D = eta.shape
        B = user_feat.size(0)
        device = eta.device
        if torch.isnan(eta).any() or torch.isnan(user_feat).any():
            logging.warning("【NaN防护】CayleyRotation输入含NaN，返回恒等映射")
            return eta.unsqueeze(0).expand(B, -1, -1)
        w = self.scale_net(user_feat)
        w = torch.clamp(w, min=-2.0, max=2.0)
        U_w = torch.einsum('Dr,br->bDr', self.U, w)
        V_w = torch.einsum('Dr,br->bDr', self.V, w)
        eta_b = eta.unsqueeze(0).expand(B, -1, -1)
        Vt_eta = torch.einsum('bDr,bkD->brk', V_w, eta_b)
        Ut_eta = torch.einsum('bDr,bkD->brk', U_w, eta_b)
        G_eta = torch.einsum('bDr,brk->bDk', U_w, Vt_eta) - torch.einsum('bDr,brk->bDk', V_w, Ut_eta)
        G_eta = G_eta.transpose(1, 2)
        G_eta = torch.clamp(G_eta, min=-10.0, max=10.0)
        eta_rotated = eta_b + 0.5 * G_eta
        return safe_normalize(eta_rotated)


# ==============================================================================
# v9.2-final: LangevinSinkhorn (preserved)
# ==============================================================================

class LangevinSinkhorn(nn.Module):
    def __init__(self, epsilon=0.05, num_iter=20, langevin_steps=3, noise_scale=0.01):
        super().__init__()
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.langevin_steps = langevin_steps
        self.noise_scale = noise_scale

    def forward(self, log_probs: Tensor, is_training: bool) -> Tuple[Tensor, Tensor]:
        B, K = log_probs.shape
        device = log_probs.device
        C = -log_probs
        exp_input = -C / self.epsilon
        exp_input = torch.clamp(exp_input, min=-80.0, max=80.0)
        K_exp = torch.exp(exp_input)
        u = torch.zeros(B, device=device)
        v = torch.zeros(K, device=device)
        det_steps = self.num_iter - self.langevin_steps
        for _ in range(det_steps):
            u = -torch.logsumexp(torch.log(K_exp + 1e-10) + v.unsqueeze(0), dim=1)
            v = -torch.logsumexp(torch.log(K_exp + 1e-10) + u.unsqueeze(1), dim=0)
        if is_training:
            for _ in range(self.langevin_steps):
                u = -torch.logsumexp(torch.log(K_exp + 1e-10) + v.unsqueeze(0), dim=1)
                v = -torch.logsumexp(torch.log(K_exp + 1e-10) + u.unsqueeze(1), dim=0)
                u = u + torch.randn_like(u) * self.noise_scale
                v = v + torch.randn_like(v) * self.noise_scale
        Pi = torch.exp(u.unsqueeze(1)) * K_exp * torch.exp(v.unsqueeze(0))
        Pi = Pi / (Pi.sum(dim=1, keepdim=True) + 1e-8)
        Pi = torch.nan_to_num(Pi, nan=1.0/K, posinf=1.0/K, neginf=0.0)
        entropy = -(Pi * torch.log(Pi.clamp(min=1e-10))).sum(dim=-1)
        return Pi, entropy


class DynamicPrototypeManifold(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, user_cond_dim: int,
                 kappa_base: float = 2.0, kappa_min: float = 0.5,
                 sinkhorn_epsilon: float = 0.05, sinkhorn_iter: int = 20,
                 min_mass_ratio: float = 0.005, coherence_threshold: float = 0.15,
                 lie_rank: int = 8):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.kappa_base = kappa_base
        self.kappa_min = kappa_min
        self.eta = nn.Parameter(torch.randn(num_codes, code_dim))
        self.rotation = CayleyRotation(code_dim, user_cond_dim, lie_rank)
        self.kappa_log = nn.Parameter(torch.tensor(math.log(kappa_base - kappa_min)))
        self.kappa_user_proj = nn.Linear(user_cond_dim, num_codes)
        self.kappa_time_decay = nn.Parameter(torch.tensor(0.1))
        self.empty_prior = nn.Parameter(torch.zeros(num_codes))
        self.task_proj = nn.ModuleDict({
            'ctr': nn.Sequential(nn.Linear(code_dim, code_dim), nn.LayerNorm(code_dim)),
            'diff': nn.Sequential(nn.Linear(code_dim, code_dim), nn.LayerNorm(code_dim)),
            'energy': nn.Sequential(nn.Linear(code_dim, code_dim), nn.LayerNorm(code_dim)),
        })
        self.sinkhorn = LangevinSinkhorn(sinkhorn_epsilon, sinkhorn_iter)
        self.coherence_threshold = coherence_threshold

    def get_global_prototypes(self) -> Tensor:
        return safe_normalize(self.eta)

    def get_kappa(self, user_feat: Tensor, time_span=None) -> Tensor:
        B = user_feat.size(0)
        kappa_base = torch.exp(torch.clamp(self.kappa_log, min=-2.0, max=4.0)) + self.kappa_min
        user_offset = self.kappa_user_proj(user_feat)
        user_offset = torch.clamp(user_offset, min=-2.0, max=2.0)
        if time_span is not None:
            time_days = time_span.unsqueeze(1) / 86400.0
            decay = torch.exp(-F.softplus(self.kappa_time_decay) * time_days)
        else:
            decay = torch.ones(B, 1, device=user_feat.device)
        kappa = (kappa_base + user_offset) * decay
        kappa = F.softplus(kappa) + self.kappa_min
        kappa = torch.clamp(kappa, min=self.kappa_min, max=10.0)
        kappa = torch.where(torch.isnan(kappa) | torch.isinf(kappa),
                           torch.full_like(kappa, self.kappa_min), kappa)
        return kappa

    def _check_and_recover_nan(self):
        nan_found = False
        for name, p in self.named_parameters():
            if p.requires_grad and torch.isnan(p).any():
                nan_found = True
                nan_count = torch.isnan(p).sum().item()
                logging.warning(f"【NaN恢复】{name}: {nan_count}/{p.numel()} NaN values, reinitializing")
                if 'eta' in name: nn.init.normal_(p, std=0.01)
                elif 'kappa_log' in name: nn.init.normal_(p, mean=0.4, std=0.2)
                elif 'kappa' in name or 'U' in name or 'V' in name: nn.init.normal_(p, std=0.01)
                elif 'proj' in name or 'weight' in name: nn.init.xavier_uniform_(p, gain=0.1)
                else: nn.init.normal_(p, std=0.01)
                if p.grad is not None: p.grad.zero_()
        return nan_found

    def forward(self, z_seq: Tensor, seq_meta: Dict[str, Tensor], user_feat: Tensor,
                task_id: str, is_training: bool):
        self._check_and_recover_nan()
        B = z_seq.size(0)
        mu_u = self.rotation.rotate(self.get_global_prototypes(), user_feat)
        if torch.isnan(mu_u).any():
            logging.warning("【NaN防护】CayleyRotation输出NaN，使用未旋转原型")
            mu_u = self.get_global_prototypes().unsqueeze(0).expand(B, -1, -1)
        time_span = seq_meta.get('time_span')
        kappa = self.get_kappa(user_feat, time_span)
        z_norm = safe_normalize(z_seq)
        cosine_sim = torch.einsum('bd,bkd->bk', z_norm, mu_u)
        cosine_sim = torch.tanh(cosine_sim)
        log_probs = cosine_sim * kappa
        log_probs = torch.clamp(log_probs, min=-15.0, max=15.0)
        proto_weights, assignment_entropy = self.sinkhorn(log_probs, is_training)
        seq_lens = seq_meta.get('len', torch.ones(B, device=z_seq.device) * self.num_codes)
        empty_mask = (seq_lens == 0).unsqueeze(-1).float()
        empty_prior = F.softmax(self.empty_prior, dim=-1).unsqueeze(0).expand(B, -1)
        proto_weights = empty_mask * empty_prior + (1.0 - empty_mask) * proto_weights
        proto_weights = torch.where(torch.isnan(proto_weights),
                                     torch.ones_like(proto_weights) / self.num_codes,
                                     proto_weights)
        proto_weights = proto_weights / (proto_weights.sum(dim=1, keepdim=True) + 1e-8)
        mu_task = self.task_proj[task_id](mu_u.reshape(-1, self.code_dim)).reshape(B, self.num_codes, self.code_dim)
        proto_repr = torch.einsum('bk,bkd->bd', proto_weights, mu_task)
        proto_repr = safe_normalize(proto_repr)
        if torch.isnan(proto_repr).any():
            logging.warning("【NaN防护】proto_repr NaN，回退到z_seq")
            proto_repr = z_seq
        return proto_weights, proto_repr, kappa.mean(), assignment_entropy

    def packing_loss(self) -> Tensor:
        mu = self.get_global_prototypes()
        gram = torch.mm(mu, mu.t())
        mask = 1.0 - torch.eye(self.num_codes, device=mu.device)
        off_diag = gram * mask
        excess = F.relu(torch.abs(off_diag) - self.coherence_threshold)
        packing = (excess ** 2).sum()
        UUt = torch.mm(self.rotation.U.t(), self.rotation.U)
        VVt = torch.mm(self.rotation.V.t(), self.rotation.V)
        I = torch.eye(self.rotation.rank, device=UUt.device)
        rot_reg = ((UUt - I) ** 2).sum() + ((VVt - I) ** 2).sum()
        kappa_base = torch.exp(torch.clamp(self.kappa_log, min=-2.0, max=4.0)) + self.kappa_min
        target_kappa = 8.0
        kappa_reg = F.relu(target_kappa - kappa_base) * 0.5
        return packing + 0.01 * rot_reg + kappa_reg


# ==============================================================================
# v9.0: LangevinForceField (preserved)
# ==============================================================================

class LangevinForceField(nn.Module):
    def __init__(self, d_model: int, num_fields: int = 3, num_steps: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.num_steps = num_steps
        self.force_net = nn.Sequential(
            nn.Linear(d_model * num_fields, d_model * num_fields),
            nn.LayerNorm(d_model * num_fields), nn.SiLU(),
            nn.Linear(d_model * num_fields, d_model * num_fields),
        )
        self.mass_log = nn.Parameter(torch.zeros(num_fields, d_model))
        self.gamma_log = nn.Parameter(torch.tensor(0.0))
        self.temperature_log = nn.Parameter(torch.tensor(0.0))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model * num_fields, d_model * num_fields), nn.SiLU(),
            nn.Linear(d_model * num_fields, num_fields * d_model),
        )

    def forward(self, fields: Dict[str, Tensor], is_training: bool) -> Tuple[Dict[str, Tensor], Tensor]:
        keys = sorted(fields.keys())
        assert len(keys) == self.num_fields
        q = torch.stack([fields[k] for k in keys], dim=1)
        B, F, D = q.shape
        p = torch.zeros(B, F, D, device=q.device)
        M = torch.exp(self.mass_log.unsqueeze(0))
        gamma = F.softplus(self.gamma_log)
        T = F.softplus(self.temperature_log)
        dt = torch.tensor(0.1, device=q.device)
        for _ in range(self.num_steps):
            q_half = q + 0.5 * dt * p / M
            q_flat = q_half.reshape(B, -1)
            force_det = self.force_net(q_flat).reshape(B, F, D)
            noise_scale = torch.sqrt(2 * gamma * T * dt)
            noise = create_noise_like(force_det, is_training) * noise_scale
            force = force_det + noise
            p = p * (1 - gamma * dt) + force * dt
            q = q_half + 0.5 * dt * p / M
        uncertainty = self.uncertainty_head(q_flat)
        uncertainty = F.softplus(uncertainty).reshape(B, F, D)
        return {k: q[:, i] for i, k in enumerate(keys)}, uncertainty


# ==============================================================================
# v9.0: CrossFieldNet (preserved)
# ==============================================================================

class SeqFiLM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.beta_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gamma_proj[0].bias)
        nn.init.constant_(self.gamma_proj[0].weight, 0.0)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: Tensor, seq_cond: Tensor) -> Tensor:
        gamma = self.gamma_proj(seq_cond).unsqueeze(1)
        beta = self.beta_proj(seq_cond).unsqueeze(1)
        return x * gamma + beta


class CrossFieldLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.film = SeqFiLM(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.ffn[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.ffn[2].weight, gain=0.1)

    def forward(self, x: Tensor, seq_cond: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)
        x = self.film(x, seq_cond)
        x = self.ffn_norm(x + self.ffn(x))
        return x


class CrossFieldNet(nn.Module):
    def __init__(self, fields: List[str], d_model: int = 128, num_layers: int = 4,
                 num_heads: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.fields = fields
        self.num_fields = len(fields)
        self.d_model = d_model
        self.layers = nn.ModuleList([CrossFieldLayer(d_model, num_heads, ffn_ratio) for _ in range(num_layers)])
        self.field_emb = nn.Parameter(torch.randn(self.num_fields, d_model) * 0.02)

    def forward(self, fields: Dict[str, Tensor], seq_cond: Tensor) -> Dict[str, Tensor]:
        x = torch.stack([fields[f] for f in self.fields], dim=1)
        x = x + self.field_emb.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, seq_cond)
        return {f: x[:, i] for i, f in enumerate(self.fields)}


# ==============================================================================
# v9.3: Isolated Task Heads — 新增生成表示路径
# ==============================================================================

class CTRHead(nn.Module):
    def __init__(self, d_model: int, num_codes: int):
        super().__init__()
        self.static_proj = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.LayerNorm(d_model))
        self.proto_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.gate = nn.Sequential(nn.Linear(d_model * 2 + num_codes, 2), nn.Softmax(dim=-1))
        self.output = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, final_repr, proto_repr, proto_weights, user_feat, item_feat):
        static = self.static_proj(final_repr)
        proto = self.proto_proj(proto_repr)
        context = torch.cat([user_feat, item_feat], dim=-1)
        gate_input = torch.cat([context, proto_weights], dim=-1)
        g = self.gate(gate_input)
        fused = g[:, 0:1] * static + g[:, 1:2] * proto
        return self.output(fused)


class DiffusionHead(nn.Module):
    def __init__(self, d_model: int, num_codes: int, num_steps: int = 5):
        super().__init__()
        self.num_steps = num_steps
        self.noise_pred = nn.Sequential(
            nn.Linear(d_model * 4 + 1, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))
        # 【v9.3】生成表示投影：将去噪后的原型映射到语义空间
        self.gen_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        betas = torch.linspace(1e-4, 0.02, num_steps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', torch.cumprod(alphas, dim=0))
        nn.init.xavier_uniform_(self.noise_pred[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.noise_pred[2].weight, gain=0.1)
        nn.init.xavier_uniform_(self.gen_proj[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.gen_proj[3].weight, gain=0.1)

    def get_gen_repr(
        self,
        proto_repr: Tensor,
        user_feat: Tensor,
        item_feat: Tensor,
        context_feat: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        返回: (gen_repr, diff_residual)
        deterministic=True 时用于 eval，固定 t 和 zero noise
        """
        B, D = proto_repr.shape
        device = proto_repr.device

        if deterministic:
            t = torch.full((B,), self.num_steps // 2, device=device, dtype=torch.long)
            noise = torch.zeros_like(proto_repr)
        else:
            t = torch.randint(0, self.num_steps, (B,), device=device)
            noise = torch.randn_like(proto_repr)

        alpha_bar_t = self.alpha_bars[t].view(B, 1)
        noisy_proto = torch.sqrt(alpha_bar_t) * proto_repr + torch.sqrt(1 - alpha_bar_t) * noise

        cond = torch.cat([user_feat, item_feat, context_feat, noisy_proto], dim=-1)
        t_emb = (t.float() / self.num_steps).view(B, 1)

        with torch.no_grad():
            pred_noise = self.noise_pred(torch.cat([cond, t_emb], dim=-1))

        # 扩散残差作为 uncertainty 信号
        diff_residual = F.mse_loss(pred_noise, noise).detach()

        denoised = (noisy_proto - torch.sqrt(1 - alpha_bar_t) * pred_noise.detach()) / torch.sqrt(alpha_bar_t)
        denoised = torch.clamp(denoised, min=-10.0, max=10.0)
        gen_repr = self.gen_proj(torch.cat([denoised, proto_repr], dim=-1))

        return F.normalize(gen_repr, p=2, dim=-1) * math.sqrt(D), diff_residual

    def forward(self, proto_repr, user_feat, item_feat, context_feat):
        """原始扩散路径（用于独立loss计算，输入由caller detach）"""
        B, D = proto_repr.shape
        device = proto_repr.device
        t = torch.randint(0, self.num_steps, (B,), device=device)
        alpha_bar_t = self.alpha_bars[t].view(B, 1)
        noise = torch.randn_like(proto_repr)
        noisy_proto = torch.sqrt(alpha_bar_t) * proto_repr + torch.sqrt(1 - alpha_bar_t) * noise
        cond = torch.cat([user_feat, item_feat, context_feat, noisy_proto], dim=-1)
        t_emb = (t.float() / self.num_steps).view(B, 1)
        pred_noise = self.noise_pred(torch.cat([cond, t_emb], dim=-1))
        return pred_noise, noise, t


class EnergyHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 1))
        # 【v9.3】生成表示投影
        self.gen_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.xavier_uniform_(self.gen_proj[0].weight, gain=0.1)
        nn.init.xavier_uniform_(self.gen_proj[3].weight, gain=0.1)

    def get_gen_repr(self, proto_repr, user_feat, item_feat):
        """【v9.3】能量生成表示：基于能量函数学习判别性表示，梯度流向共享层"""
        joint = torch.cat([proto_repr, user_feat, item_feat], dim=-1)
        gen_repr = self.gen_proj(joint)
        return F.normalize(gen_repr, p=2, dim=-1) * math.sqrt(proto_repr.size(-1))

    def forward(self, proto_repr, user_feat, item_feat):
        joint = torch.cat([proto_repr, user_feat, item_feat], dim=-1)
        energy = self.net(joint).squeeze(-1)
        return energy


# ==============================================================================
# v9.3: GenerativeFusion — 新增模块
# ==============================================================================

class GenerativeFusion(nn.Module):
    """
    【v9.3】生成式表示融合模块：借鉴HSTU/TIGER范式，
    将扩散去噪表示、能量判别表示与原型表示通过门控融合，
    输出语义更丰富的联合表示供CTR预测使用。
    v9.3-RS: 残差融合，eval/train 行为一致，仅随机性不同
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        for m in self.fusion_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.residual_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        proto_repr: Tensor,
        diff_repr: Tensor,
        energy_repr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        delta_diff = diff_repr - proto_repr
        delta_energy = energy_repr - proto_repr
        gate_input = torch.cat([proto_repr, diff_repr, energy_repr], dim=-1)
        gates = self.fusion_gate(gate_input)

        # 残差形式：eval 回退到 proto 时损失可控
        fused = proto_repr + gates[:, 1:2] * delta_diff + gates[:, 2:3] * delta_energy
        out = self.residual_proj(fused)
        return out, gates


# ==============================================================================
# v9.0: FokkerPlanckRegularizer (preserved)
# ==============================================================================

class FokkerPlanckRegularizer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.drift_mlp = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.log_diffusion = nn.Parameter(torch.zeros(d_model))

    def forward(self, z_t, z_t_next, delta_t=1.0):
        z_mean = z_t.mean(dim=0, keepdim=True)
        z_std = z_t.std(dim=0, keepdim=True).clamp(min=1e-8)
        score = (z_mean - z_t) / (z_std ** 2)
        drift_input = torch.cat([z_t, score], dim=-1)
        f_z = self.drift_mlp(drift_input)
        D_z = torch.exp(self.log_diffusion).unsqueeze(0)
        dz_dt = (z_t_next - z_t) / delta_t
        residual = dz_dt - f_z - 0.5 * D_z * score
        loss_fp = torch.mean((residual ** 2) / (D_z + 1e-8)) + torch.mean(torch.log(D_z + 1e-8))
        return loss_fp


# ==============================================================================
# v9.3: MetaAligner — 过拟合感知动态调度
# ==============================================================================

class MetaAligner(nn.Module):
    """
    MetaAligner v2: 协同自适应正则化控制器
    核心改进：
    1. 单一 aux_weight 信源，无 alpha/diff/energy 分离
    2. 泄漏积分器：历史自动衰减，无债务累积
    3. Valid 间插值：无 valid 时向保守值回归，而非 health 放飞
    4. 自适应 EMA：根据 gap 变化率调整平滑度
    5. 辅助任务只输出 loss 权重，不输出表示（物理隔离）
    """

    def __init__(
        self,
        num_codes: int = 128,
        warmup_steps: int = 300,
        probe_burn_in: int = 200,
        kp: float = 2.0,
        ki: float = 0.05,      # 降低，配合泄漏积分器
        kd: float = 1.0,
        max_aux_weight: float = 0.3,   # 从 0.5 降到 0.3
        leak_rate: float = 0.01,       # 积分器泄漏率
        valid_decay_rate: float = 0.995,  # Valid 间 aux_weight 衰减
        ema_fast: float = 0.8,         # gap 变化快时用
        ema_slow: float = 0.95,        # gap 稳定时用
    ):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # 保持 nn.Module 身份
        
        self.warmup_steps = warmup_steps
        self.probe_burn_in = probe_burn_in
        self.max_aux_weight = max_aux_weight
        self.leak_rate = leak_rate
        self.valid_decay_rate = valid_decay_rate
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        
        self.kp, self.ki, self.kd = kp, ki, kd
        
        # 泄漏积分器：带自动衰减
        self.register_buffer('integral', torch.tensor(0.0))
        self.register_buffer('integral_momentum', torch.tensor(0.0))  # 二阶矩，检测变化
        
        # Valid 间状态缓存
        self.register_buffer('last_valid_gap', torch.tensor(0.0))
        self.register_buffer('last_valid_aux', torch.tensor(0.0))
        self.register_buffer('steps_since_valid', torch.tensor(0))
        
        # EMA 状态
        self.register_buffer('ema_aux', torch.tensor(0.0))
        self.register_buffer('ema_gap', torch.tensor(0.0))
        
        # 历史（仅用于诊断，不参与计算）
        self.gap_history = deque(maxlen=10)
        self.aux_history = deque(maxlen=10)

    def _leaky_integrate(self, gap: float) -> float:
        """
        泄漏积分：每一步自动衰减，防止历史债务
        integral_t = (1 - leak) * integral_{t-1} + gap_t
        """
        self.integral = (1 - self.leak_rate) * self.integral + gap
        # 软裁剪，非硬边界
        self.integral = self.integral * torch.sigmoid(self.integral)  # 自限幅
        return self.integral.item()

    def _adaptive_ema(self, new_val: float, old_ema: float) -> float:
        """
        自适应 EMA：根据变化速度选择快/慢衰减
        """
        delta = abs(new_val - old_ema)
        # 变化大时用 fast（快速响应），变化小时用 slow（稳定）
        ema_decay = self.ema_fast if delta > 0.1 else self.ema_slow
        return ema_decay * old_ema + (1 - ema_decay) * new_val

    def _interpolate_without_valid(self, base_aux: float) -> float:
        """
        Valid 间插值：向保守值回归，而非 health 放飞
        """
        steps = int(self.steps_since_valid.item())
        # 无 valid 越久，aux_weight 越向 0 衰减（保守策略）
        decay = self.valid_decay_rate ** steps
        # 基础值与保守值的插值
        conservative = 0.05  # 最低辅助权重
        return conservative + (base_aux - conservative) * decay

    def forward(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float],
        valid_auc: Optional[float] = None,
        train_auc: Optional[float] = None,
        global_step: int = 0,
    ) -> Dict[str, Any]:
        
        # === Phase 1: 计算当前 gap ===
        gap = 0.0
        if train_auc is not None and valid_auc is not None and valid_auc > 0:
            gap = max(0.0, train_auc - valid_auc)
            self.gap_history.append(gap)
            self.last_valid_gap = torch.tensor(gap)
            self.steps_since_valid = torch.tensor(0)
        else:
            self.steps_since_valid += 1
            # 无 valid 时，用上次 valid 的 gap 估计（保守假设 gap 不变或微增）
            gap = float(self.last_valid_gap.item()) * 1.01  # 微增假设
        
        # === Phase 2: PID 控制（泄漏积分器） ===
        integral = self._leaky_integrate(gap)
        
        # 变化率（导数）
        gap_trend = 0.0
        if len(self.gap_history) >= 2:
            gap_trend = (self.gap_history[-1] - self.gap_history[0]) / len(self.gap_history)
        
        P = self.kp * gap
        I = self.ki * integral
        D = self.kd * (-gap_trend)
        
        raw_aux = max(0.0, P + I + D)
        
        # === Phase 3: Valid 间插值 ===
        if valid_auc is None or valid_auc <= 0:
            raw_aux = self._interpolate_without_valid(raw_aux)
        
        # === Phase 4: 自适应 EMA 平滑 ===
        new_ema = self._adaptive_ema(raw_aux, float(self.ema_aux.item()))
        self.ema_aux = torch.tensor(new_ema)
        
        # 最终 aux_weight：EMA 后硬上限
        aux_weight = min(self.max_aux_weight, new_ema)
        
        # === Phase 5: 辅助任务分配（单一信源） ===
        # 不再维护独立的 alpha['diff'] / alpha['energy']
        # 直接返回 aux_weight，由调用方按比例分配
        diff_ratio = 0.6  # 固定比例，或基于 probe 动态调整
        energy_ratio = 0.4
        
        result = {
            'aux_weight': aux_weight,
            'diff_weight': aux_weight * diff_ratio,
            'energy_weight': aux_weight * energy_ratio,
            'ctr_weight': 1.0 - aux_weight,
            'gap': gap,
            'integral': integral,
            'mode': 'valid_pid' if (valid_auc and valid_auc > 0) else 'interpolated',
            'ema_aux': new_ema,
            'steps_since_valid': int(self.steps_since_valid.item()),
        }
        
        self.aux_history.append(aux_weight)
        return result


# ==============================================================================
# v9.3: PCVRHeteroFormer (main model)
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
        seq_len: int = 50,
        num_layers: int = 4,
        base_rank: Optional[int] = None,
        rank_schedule: str = 'bottleneck',
        num_global_tokens: int = 4,
        kernel_size: int = 3,
        num_heads: Optional[int] = None,
        num_banks: Optional[int] = None,
        dropout: float = 0.0,
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
        id_dropout_rate: float = 0.0,
        seq_id_dropout_rate: float = 0.0,
        id_vocab_threshold: int = 10000,
        shrinkage: float = 0.05,
        cross_network_layers: int = 2,
        num_codes: int = 128,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iter: int = 20,
        min_mass_ratio: float = 0.005,
        coherence_threshold: float = 0.15,
        kappa_base: float = 2.0,
        kappa_min: float = 0.5,
        lie_rank: int = 8,
        use_domain_adversarial: bool = False,
        use_generative_fusion: bool = True,  # 【v9.3】新增
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes
        self.use_domain_adversarial = use_domain_adversarial
        self.use_generative_fusion = use_generative_fusion

        self.encoder = MultiViewEncoder(
            user_int_feature_specs=user_int_feature_specs,
            item_int_feature_specs=item_int_feature_specs,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            d_model=d_model,
            emb_dim=emb_dim,
            id_vocab_threshold=id_vocab_threshold,
            emb_skip_threshold=emb_skip_threshold,
        )

        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_encoders = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            self.seq_encoders[domain] = ContinuousSequenceEncoder(
                vocab_sizes=vocab_sizes, d_model=d_model, state_dim=max(64, d_model // 2),
                num_layers=2, max_seq_len=512, id_threshold=seq_id_threshold,
            )

        # 【手术1】多序列融合模块
        self.num_seq_domains = len(self.seq_domains)
        if self.num_seq_domains > 1:
            self.seq_fusion_short = MultiSeqFusion(d_model, self.num_seq_domains, num_heads=4)                             
            self.seq_fusion_long = MultiSeqFusion(d_model, self.num_seq_domains, num_heads=4)
            self.seq_fusion_static = MultiSeqFusion(d_model, self.num_seq_domains, num_heads=4)
        else:
            self.seq_fusion_short = None
            self.seq_fusion_long = None
            self.seq_fusion_static = None

        self.prototype = DynamicPrototypeManifold(
            num_codes=num_codes, code_dim=d_model, user_cond_dim=d_model,
            kappa_base=kappa_base, kappa_min=kappa_min,
            sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iter=sinkhorn_iter,
            min_mass_ratio=min_mass_ratio, coherence_threshold=coherence_threshold,
            lie_rank=lie_rank,
        )

        self.explicit_cross = BilinearCrossLayer(d_model, num_fields=4, cross_ratio=0.1)
        self.cross_field = CrossFieldNet(
            fields=['user', 'item', 'context', 'seq'],
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads or max(4, d_model // 32), ffn_ratio=2.0,
        )

        self.ctr_head = CTRHead(d_model, num_codes)
        self.diff_head = DiffusionHead(d_model, num_codes)
        self.energy_head = EnergyHead(d_model)
        self.fp_regularizer = FokkerPlanckRegularizer(d_model)

        # 【v9.3】生成式表示融合
        if use_generative_fusion:
            self.gen_fusion = GenerativeFusion(d_model)
        else:
            self.gen_fusion = None

        if use_domain_adversarial:
            self.domain_disc = nn.Sequential(nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 1))

        self.meta_aligner = MetaAligner(
            num_codes=num_codes,
            warmup_steps=300,
            probe_burn_in=200,
            kp=2.0,
            ki=0.05,              # V2 降低
            kd=1.0,
            max_aux_weight=0.3,     # V2 从 0.5 降到 0.3
            leak_rate=0.01,        # V2 新增
            valid_decay_rate=0.995,  # V2 新增
            ema_fast=0.8,          # V2 新增
            ema_slow=0.95,         # V2 新增
        )
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        self.seq_fusion = MultiSeqFusion(d_model, len(self.seq_domains))

        self._register_param_groups()

    def _register_param_groups(self):
        """【v9.3】参数分组，新增gen_fusion归入ctr_head_params"""
        self._sparse_params = []
        self._shared_encoder_params = []
        self._seq_encoder_params = []
        self._proto_geo_params = []
        self._proto_cond_params = []
        self._cross_field_params = []
        self._ctr_head_params = []
        self._diff_head_params = []
        self._energy_head_params = []
        self._task_proj_ctr_params = []
        self._task_proj_diff_params = []
        self._task_proj_energy_params = []
        self._fp_params = []
        self._meta_params = []
        self._disc_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if 'embedding' in name or ('emb' in name.lower() and 'num_embeddings' in str(type(p))):
                self._sparse_params.append(p)
            elif 'seq_encoders' in name:
                self._seq_encoder_params.append(p)
            elif any(x in name for x in ['prototype.eta', 'prototype.rotation', 
                                           'prototype.kappa_log', 'prototype.kappa_time_decay']):
                self._proto_geo_params.append(p)
            elif 'prototype.task_proj.ctr' in name:
                self._task_proj_ctr_params.append(p)
            elif 'prototype.task_proj.diff' in name:
                self._task_proj_diff_params.append(p)
            elif 'prototype.task_proj.energy' in name:
                self._task_proj_energy_params.append(p)
            elif 'prototype' in name:
                self._proto_cond_params.append(p)
            elif 'cross_field' in name:
                self._cross_field_params.append(p)
            elif 'ctr_head' in name:
                self._ctr_head_params.append(p)
            elif 'diff_head' in name:
                self._diff_head_params.append(p)
            elif 'energy_head' in name:
                self._energy_head_params.append(p)
            elif 'gen_fusion' in name:
                self._ctr_head_params.append(p)
            elif 'fp_regularizer' in name:
                self._fp_params.append(p)
            elif 'meta_aligner' in name:
                self._meta_params.append(p)
            elif 'domain_disc' in name:
                self._disc_params.append(p)
            elif 'logit_temperature' in name or 'logit_bias' in name:
                self._ctr_head_params.append(p)
            elif 'encoder' in name:
                self._shared_encoder_params.append(p)
            else:
                self._shared_encoder_params.append(p)

        all_assigned = (
            self._sparse_params + self._shared_encoder_params + 
            self._seq_encoder_params + self._proto_geo_params +
            self._proto_cond_params + self._cross_field_params +
            self._ctr_head_params + self._diff_head_params +
            self._energy_head_params + self._task_proj_ctr_params +
            self._task_proj_diff_params + self._task_proj_energy_params +
            self._fp_params + self._meta_params + self._disc_params
        )
        total_params = sum(1 for _ in self.parameters() if _.requires_grad)
        assigned_count = len(all_assigned)

        if assigned_count != total_params:
            logging.warning(f"【参数分组警告】{assigned_count}/{total_params} 参数被分配，存在遗漏！")
        else:
            logging.info(f"【参数分组v9.3】所有{total_params}个参数已正确分配")

        group_counts = {
            'sparse': len(self._sparse_params), 'shared_encoder': len(self._shared_encoder_params),
            'seq_encoder': len(self._seq_encoder_params), 'proto_geo': len(self._proto_geo_params),
            'proto_cond': len(self._proto_cond_params), 'cross_field': len(self._cross_field_params),
            'ctr_head': len(self._ctr_head_params), 'diff_head': len(self._diff_head_params),
            'energy_head': len(self._energy_head_params), 'task_proj_ctr': len(self._task_proj_ctr_params),
            'task_proj_diff': len(self._task_proj_diff_params), 'task_proj_energy': len(self._task_proj_energy_params),
            'fp': len(self._fp_params), 'meta': len(self._meta_params), 'disc': len(self._disc_params),
        }
        for gname, gcount in group_counts.items():
            if gcount > 0:
                logging.info(f"  {gname}: {gcount} params")

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            'sparse': self._sparse_params, 'shared_encoder': self._shared_encoder_params,
            'seq_encoder': self._seq_encoder_params, 'proto_geo': self._proto_geo_params,
            'proto_cond': self._proto_cond_params, 'cross_field': self._cross_field_params,
            'ctr_head': self._ctr_head_params, 'diff_head': self._diff_head_params,
            'energy_head': self._energy_head_params, 'task_proj_ctr': self._task_proj_ctr_params,
            'task_proj_diff': self._task_proj_diff_params, 'task_proj_energy': self._task_proj_energy_params,
            'fp': self._fp_params, 'meta': self._meta_params, 'disc': self._disc_params,
        }

    def forward(self, model_input: ModelInput, task_id: str = 'ctr'):
        B = model_input.user_int_feats.size(0)
        device = model_input.user_int_feats.device

        # 1. Feature encoding
        enc_out = self.encoder(
            model_input.user_int_feats, model_input.item_int_feats,
            model_input.user_dense_feats, model_input.item_dense_feats,
        )
        z_shared = enc_out['shared']
        user_feat = enc_out['user']
        item_feat = enc_out['item']
        context_feat = enc_out['context']

        for name, tensor in [('z_shared', z_shared), ('user_feat', user_feat),
                             ('item_feat', item_feat), ('context_feat', context_feat)]:
            if torch.isnan(tensor).any():
                logging.warning(f"【NaN防护】Encoder输出{name}含NaN，执行零填充恢复")
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

        # 2. Sequence encoding
        seq_outputs = {}
        for domain in self.seq_domains:
            if domain in model_input.seq_data:
                timestamps_raw = (model_input.seq_timestamps_raw or {}).get(domain)
                seq_out = self.seq_encoders[domain](
                    model_input.seq_data[domain], model_input.seq_lens[domain],
                    timestamps_raw=timestamps_raw,
                )
                seq_outputs[domain] = seq_out

        # 【手术1】多序列门控融合，替代单一序列硬选择
        seq_meta = {'len': torch.zeros(B, device=device), 'time_span': torch.ones(B, device=device) * 86400.0}
        
        if len(seq_outputs) > 0:
            # 收集所有可用序列域的输出
            short_list = []
            long_list = []
            static_list = []
            len_list = []
            
            for domain in self.seq_domains:
                if domain in seq_outputs:
                    short_list.append(seq_outputs[domain]['short'])
                    long_list.append(seq_outputs[domain]['long'])
                    static_list.append(seq_outputs[domain]['static'])
                    len_list.append(model_input.seq_lens[domain])
            
            # 计算主序列长度（取各域最大长度作为保守估计）
            if len_list:
                stacked_lens = torch.stack(len_list, dim=1)  # [B, num_domains]
                seq_meta['len'] = stacked_lens.max(dim=1)[0]
                # 时间跨度：取第一个域的保守估计
                seq_meta['time_span'] = torch.ones(B, device=device) * 86400.0
            
            # 多序列融合
            if self.num_seq_domains > 1 and len(short_list) > 1:
                s_short, gate_short, ent_loss_short = self.seq_fusion_short(short_list)
                s_long, gate_long, ent_loss_long = self.seq_fusion_long(long_list)
                s_static, gate_static, ent_loss_static = self.seq_fusion_static(static_list)
                # 缓存熵 loss（供 trainer 使用）
                self._cached_entropy_loss = (ent_loss_short + ent_loss_long + ent_loss_static) / 3.0
                
                # 缓存门控权重供诊断（可选）
                self._cached_seq_gate_short = gate_short
                self._cached_seq_gate_long = gate_long
                self._cached_seq_gate_static = gate_static
            else:
                # 单域时直接取
                s_short = short_list[0] if short_list else torch.zeros(B, self.d_model, device=device)
                s_long = long_list[0] if long_list else torch.zeros(B, self.d_model, device=device)
                s_static = static_list[0] if static_list else torch.zeros(B, self.d_model, device=device)
                self._cached_entropy_loss = torch.tensor(0.0, device=device)
        else:
            # 无任何序列域
            s_short = torch.zeros(B, self.d_model, device=device)
            s_long = torch.zeros(B, self.d_model, device=device)
            s_static = torch.zeros(B, self.d_model, device=device)

        if self.num_seq_domains > 1 and len(short_list) > 1:
            s_short, gate_short, ent_loss_short = self.seq_fusion_short(short_list)
            s_long, gate_long, ent_loss_long = self.seq_fusion_long(long_list)
            s_static, gate_static, ent_loss_static = self.seq_fusion_static(static_list)
            
            # 【缓存熵 loss】供 trainer 使用
            # 三个熵 loss 取平均，避免过度正则
            self._cached_entropy_loss = (ent_loss_short + ent_loss_long + ent_loss_static) / 3.0
            
            # 缓存门控权重供诊断
            self._cached_seq_gate_short = gate_short
        else:
            # ... 单域情况 ...
            self._cached_entropy_loss = torch.tensor(0.0, device=device)

        # 3. Prototype
        proto_weights, proto_repr, kappa_mean, assign_entropy = self.prototype(
            s_short, seq_meta, user_feat, task_id=task_id, is_training=self.training
        )

        # 3. 【手术2】显式二阶交叉
        fields_list = [user_feat, item_feat, context_feat, s_short]
        crossed_fields = self.explicit_cross(fields_list)
        user_feat, item_feat, context_feat, s_short = crossed_fields

        # 4. Cross-field
        fields = {'user': user_feat, 'item': item_feat, 'context': context_feat, 'seq': s_short}
        cross_out = self.cross_field(fields, seq_cond=s_short)

        # 5. Final representation
        final_repr = torch.cat([cross_out['user'], cross_out['item'],
                                cross_out['context'], cross_out['seq']], dim=-1)

        if torch.isnan(final_repr).any():
            logging.warning("【NaN防护】final_repr NaN，使用context_feat替代")
            final_repr = torch.cat([context_feat] * 4, dim=-1)

        # 6. Task-specific output
        if task_id == 'ctr':
            gen_gates = None
            uncertainty_pkg = {}
            fused_repr = proto_repr

            # 【v9.3-RS】始终启用 fusion，eval 时 deterministic
            if self.use_generative_fusion:
                # 【修复3】完全 detach，切断辅助任务梯度流向共享层
                gen_diff_repr, diff_residual = self.diff_head.get_gen_repr(
                    proto_repr.detach(), 
                    user_feat.detach(), 
                    item_feat.detach(), 
                    context_feat.detach(),
                    deterministic=not self.training,
                )
                gen_energy_repr = self.energy_head.get_gen_repr(
                    proto_repr.detach(), 
                    user_feat.detach(), 
                    item_feat.detach(),
                )

                # 计算 uncertainty 信号（no_grad，轻量）
                with torch.no_grad():
                    energy_batch = self.energy_head(proto_repr, user_feat, item_feat)
                    energy_flatness = energy_batch.std()

                uncertainty_pkg = {
                    'diff_residual': diff_residual.item() if torch.is_tensor(diff_residual) else float(diff_residual),
                    'energy_flatness': energy_flatness.item() if torch.is_tensor(energy_flatness) else float(energy_flatness),
                }

                fused_repr, gen_gates = self.gen_fusion(
                    proto_repr, gen_diff_repr, gen_energy_repr
                )

            logits = self.ctr_head(final_repr, fused_repr, proto_weights, user_feat, item_feat)

            if torch.isnan(logits).any():
                logging.warning("【NaN防护】CTR logits NaN，输出零")
                logits = torch.zeros(B, 1, device=device)

            logits = torch.clamp(logits, min=-20.0, max=20.0)
            temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
            logits = logits / temperature + self.logit_bias
            logits = torch.clamp(logits, min=-20.0, max=20.0)

            # 缓存中间变量供 Trainer probe 复用
            self._cached_user_feat = user_feat
            self._cached_item_feat = item_feat
            self._cached_context_feat = context_feat
            self._cached_proto_repr = proto_repr

            return logits, proto_weights, proto_repr, kappa_mean, assign_entropy, gen_gates, uncertainty_pkg

        elif task_id == 'diff':
            pred_noise, target_noise, t = self.diff_head(
                proto_repr.detach(), user_feat.detach(), item_feat.detach(), context_feat.detach())
            return pred_noise, target_noise, t

        elif task_id == 'energy':
            energy = self.energy_head(
                proto_repr.detach(), user_feat.detach(), item_feat.detach())
            return energy
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    def predict(self, model_input: ModelInput):
        with torch.no_grad():
            out = self.forward(model_input, task_id='ctr')
        # 兼容 7 元素返回
        if isinstance(out, tuple) and len(out) >= 6:
            return out[0], None
        return out[0], None

    def get_packing_loss(self) -> Tensor:
        return self.prototype.packing_loss()

    def get_fp_loss(self, z_t: Tensor, z_t_next: Tensor) -> Tensor:
        return self.fp_regularizer(z_t, z_t_next)


# ==============================================================================
# Preserved HeteroBlock
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