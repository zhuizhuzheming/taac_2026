"""PCVRHeteroFormer v10 - Generative Semantics & Proto-Conditioned Interaction
================================================================================
v10 核心架构（严格遵循修改意见框架）：
1. 基础编码层：MultiViewEncoder + ContinuousSequenceEncoder + MultiSeqFusion
2. 生成式语义层：DynamicPrototypeManifold + DiffusionExplainer + EnergyCalibrator
   - 【自监督训练】IB loss + recon loss + ortho loss + packing loss
   - 【不直接参与 CTR 梯度】所有生成输出进入 CTR 路径前均 detach()
3. 特征交互层：BilinearCrossLayer + ProtoConditionedCrossFieldNet
   - proto_weights → 场注意力偏置（温和注入）
   - diff_explain  → FFN 场门控（可选）
   - energy_score   → 不参与训练，仅 eval 校准
4. 预测层：CalibratedCTRHead
   - eval 时以 energy_score 做动态 logit 校准
5. MetaAligner：过拟合感知，仅输出 aux_weight 控制生成模块总权重
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
# v10: MultiViewEncoder (preserved)
# ==============================================================================

class MultiViewEncoder(nn.Module):
    def __init__(self, user_int_feature_specs, item_int_feature_specs, user_dense_dim, item_dense_dim,
                 d_model=128, emb_dim=16, id_vocab_threshold=10000, emb_skip_threshold=0, task_dims=None, dropout_rate=0.15):
        super().__init__()
        self.d_model = d_model
        self.user_int_feature_specs = user_int_feature_specs
        self.item_int_feature_specs = item_int_feature_specs
        task_dims = task_dims or {'ctr': d_model, 'diff': d_model, 'energy': d_model}
        self.task_names = sorted(task_dims.keys())
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

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
        z_shared = self.dropout(z_shared)

        views = {task_name: self.view_projs[task_name](z_shared) for task_name in self.task_names}

        return {'shared': z_shared, 'user': user_feat, 'item': item_feat, 'context': context_feat, 'views': views}


# ==============================================================================
# v10: SSMCell (preserved)
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
    轻量级双线性交叉层：捕获域间二阶交互（上三角，避免参数量爆炸）
    """
    def __init__(self, d_model: int, num_fields: int = 4, cross_ratio: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_fields = num_fields
        self.cross_ratio = cross_ratio
        num_pairs = num_fields * (num_fields - 1) // 2
        self.cross_weights = nn.Parameter(torch.randn(num_pairs, d_model) * 0.01)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, fields: List[Tensor]) -> List[Tensor]:
        num_actual = len(fields)
        if num_actual < 2:
            return fields

        out = [f.clone() for f in fields]
        pair_idx = 0
        for i in range(num_actual):
            for j in range(i + 1, num_actual):
                interaction = fields[i] * self.cross_weights[pair_idx] * fields[j]
                out[i] = out[i] + self.cross_ratio * interaction
                out[j] = out[j] + self.cross_ratio * interaction
                pair_idx += 1

        out = [self.output_norm(o) for o in out]
        return out


# ==============================================================================
# v10: ContinuousSequenceEncoder (preserved)
# ==============================================================================

class ContinuousSequenceEncoder(nn.Module):
    def __init__(self, vocab_sizes, d_model=128, state_dim=64, num_layers=2, max_seq_len=512, id_threshold=10000, dropout_rate=0.15):
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
        self.seq_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, seq_ids, seq_lens, time_buckets=None, decay_weight=None, timestamps_raw=None):
        B, n_feats, max_len = seq_ids.shape
        device = seq_ids.device
        feat_embs = [self.feat_embs[i](seq_ids[:, i, :]) for i in range(n_feats)]
        seq_repr = torch.cat(feat_embs, dim=-1)
        seq_repr = self.seq_norm(self.seq_proj(seq_repr))
        seq_repr = self.seq_dropout(seq_repr)

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
            hidden = self.seq_dropout(hidden)

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
# v10: CayleyRotation (preserved)
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
# v10: LangevinSinkhorn (preserved)
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


# ==============================================================================
# v10: DynamicPrototypeManifold（移除 task_proj，添加 get_rotated_prototypes）
# ==============================================================================

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
        # v10: 删除 task_proj，原型表示保持纯语义，不偏向任何下游任务
        self.sinkhorn = LangevinSinkhorn(sinkhorn_epsilon, sinkhorn_iter)
        self.coherence_threshold = coherence_threshold

    def get_global_prototypes(self) -> Tensor:
        return safe_normalize(self.eta)

    def get_rotated_prototypes(self, user_feat: Tensor) -> Tensor:
        """获取用户条件旋转后的原型矩阵 [B, K, D]，供 DiffusionExplainer 使用"""
        mu_u = self.rotation.rotate(self.get_global_prototypes(), user_feat)
        if torch.isnan(mu_u).any():
            logging.warning("【NaN防护】CayleyRotation输出NaN，使用未旋转原型")
            mu_u = self.get_global_prototypes().unsqueeze(0).expand(user_feat.size(0), -1, -1)
        return mu_u

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
        mu_u = self.get_rotated_prototypes(user_feat)
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
        # v10: 直接使用旋转原型，无任务投影
        proto_repr = torch.einsum('bk,bkd->bd', proto_weights, mu_u)
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
# v10: DiffusionExplainer（信息瓶颈编码器）
# ==============================================================================

class DiffusionExplainer(nn.Module):
    """信息瓶颈编码器：将原型-序列联合分布编码到瓶颈空间，输出可解释语义 diff_explain"""

    def __init__(self, d_model: int, num_codes: int, bottleneck_dim: Optional[int] = None, dropout_rate: float = 0.15):
        super().__init__()
        bottleneck_dim = bottleneck_dim or max(16, d_model // 4)
        self.bottleneck_dim = bottleneck_dim

        # 编码器：z_seq + proto_res_pooled + proto_weights
        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2 + num_codes, d_model),
            nn.LayerNorm(d_model), nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(d_model, bottleneck_dim * 2),  # mu, logvar
        )

        # 可解释表示投影
        self.explain_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model), nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model),
        )

        # 重建解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_seq: Tensor, proto_weights: Tensor, proto_res: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # proto_res: [B, K, D]（旋转后的原型矩阵）
        B = z_seq.size(0)
        # 池化原型矩阵为 [B, D]
        proto_pooled = torch.einsum('bk,bkd->bd', proto_weights, proto_res)
        x = torch.cat([z_seq, proto_pooled, proto_weights], dim=-1)

        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = mu + std * eps if self.training else mu

        # IB loss: KL(q(z|x) || N(0,1))
        ib_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        ib_loss = torch.clamp(ib_loss, max=20.0)

        diff_explain = self.explain_proj(z)
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, z_seq.detach(), reduction='mean')

        with torch.no_grad():
            err = F.mse_loss(recon, z_seq, reduction='none').mean(dim=-1)
            sig = (z_seq ** 2).mean(dim=-1).clamp(min=1e-8)
            diff_quality = torch.exp(-err / sig * 5)

        return diff_explain, diff_quality, ib_loss, recon_loss


# ==============================================================================
# v10: EnergyCalibrator（能量校准器，eval 时校准 CTR）
# ==============================================================================

class EnergyCalibrator(nn.Module):
    def __init__(self, d_model: int, num_codes: int, dropout_rate: float = 0.15):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(num_codes + d_model * 2, d_model),
            nn.LayerNorm(d_model), nn.SiLU(), nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2), nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, proto_weights: Tensor, user_feat: Tensor, item_feat: Tensor) -> Tensor:
        x = torch.cat([proto_weights, user_feat, item_feat], dim=-1)
        energy = self.energy_net(x).squeeze(-1)
        return torch.clamp(energy, min=0.0, max=10.0)

    def compute_ranking_loss(self, energy: Tensor, labels: Tensor, margin: float = 1.0) -> Tensor:
        """自监督对比排序：正样本能量应低于负样本"""
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]
        n_pos = pos_idx.numel()
        n_neg = neg_idx.numel()
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(0.0, device=energy.device)
        n_sample = min(5, n_neg)
        if n_neg > n_sample:
            neg_sample_idx = neg_idx[torch.randperm(n_neg, device=energy.device)[:n_sample]]
        else:
            neg_sample_idx = neg_idx
        pos_e = energy[pos_idx]
        neg_e = energy[neg_sample_idx]
        diff = pos_e.unsqueeze(1) - neg_e.unsqueeze(0) + margin
        return F.relu(diff).mean()


# ==============================================================================
# v10: ProtoConditionedCrossFieldNet（特征交互层核心）
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


class ProtoConditionedCrossFieldLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, ffn_ratio: float = 2.0,
                 dropout_rate: float = 0.15, num_codes: int = 128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        # proto_weights → 场注意力偏置
        self.proto_bias_proj = nn.Linear(num_codes, d_model)
        self.film = SeqFiLM(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        # diff_explain → 场门控（可选）
        self.diff_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, seq_cond: Tensor,
                proto_weights: Optional[Tensor] = None,
                diff_explain: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        if proto_weights is not None:
            # 温和场偏置：避免主导原始注意力
            proto_bias = self.proto_bias_proj(proto_weights).unsqueeze(1)  # [B, 1, D]
            attn_out = attn_out + 0.05 * proto_bias
        x = self.attn_norm(x + attn_out)
        x = self.dropout(x)
        x = self.film(x, seq_cond)
        ffn_out = self.ffn(x)
        if diff_explain is not None:
            gate = self.diff_gate(diff_explain).unsqueeze(1)  # [B, 1, D]
            ffn_out = ffn_out * gate
        x = self.ffn_norm(x + ffn_out)
        x = self.dropout(x)
        return x


class ProtoConditionedCrossFieldNet(nn.Module):
    def __init__(self, fields: List[str], d_model: int = 128, num_layers: int = 4,
                 num_heads: int = 4, num_codes: int = 128, ffn_ratio: float = 2.0,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.fields = fields
        self.num_fields = len(fields)
        self.d_model = d_model
        self.layers = nn.ModuleList([
            ProtoConditionedCrossFieldLayer(d_model, num_heads, ffn_ratio, dropout_rate, num_codes)
            for _ in range(num_layers)
        ])
        self.field_emb = nn.Parameter(torch.randn(self.num_fields, d_model) * 0.02)

    def forward(self, fields: Dict[str, Tensor], seq_cond: Tensor,
                proto_weights: Optional[Tensor] = None,
                diff_explain: Optional[Tensor] = None) -> Dict[str, Tensor]:
        x = torch.stack([fields[f] for f in self.fields], dim=1)  # [B, F, D]
        x = x + self.field_emb.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, seq_cond, proto_weights=proto_weights, diff_explain=diff_explain)
        return {f: x[:, i] for i, f in enumerate(self.fields)}


# ==============================================================================
# v10: CalibratedCTRHead（预测层）
# ==============================================================================

class CalibratedCTRHead(nn.Module):
    def __init__(self, d_model: int, num_codes: int, dropout_rate: float = 0.15):
        super().__init__()
        self.static_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model), nn.LayerNorm(d_model)
        )
        self.proto_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )
        # 双尺度门控：static / proto
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + num_codes, d_model),
            nn.LayerNorm(d_model), nn.SiLU(),
            nn.Linear(d_model, 2), nn.Softmax(dim=-1),
        )
        # 正样本残差聚焦
        self.pos_residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, d_model)
        )
        self.output = nn.Linear(d_model, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        # CVR ≈ 5% 先验 log-odds ≈ -2.94，给模型合理起点
        nn.init.constant_(self.output.bias, -2.0)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)

    def forward(self, final_repr: Tensor, proto_repr: Tensor, proto_weights: Tensor,
                user_feat: Tensor, item_feat: Tensor, energy_score: Optional[Tensor] = None) -> Tensor:
        static = self.static_proj(final_repr)
        proto = self.proto_proj(proto_repr)
        # 正样本模式注入：防止全员预测负类躺平
        proto_activations = proto_weights.max(dim=-1, keepdim=True)[0]
        pos_boost = torch.sigmoid(proto_activations - 0.5) * self.pos_residual(proto)
        context = torch.cat([user_feat, item_feat], dim=-1)
        gate_input = torch.cat([context, proto_weights], dim=-1)
        g = self.gate(gate_input)  # [B, 2]
        fused = g[:, 0:1] * static + g[:, 1:2] * (proto + pos_boost)
        fused = self.dropout(fused)
        logits = self.output(fused)
        # eval 时 energy_score 校准：高不确定性降低 logit
        if energy_score is not None and not self.training:
            with torch.no_grad():
                cal = -0.1 * (energy_score - energy_score.mean()).unsqueeze(-1)
                logits = logits + cal
        # 动态偏置校正：防止 logits 无限下钻
        if self.training:
            with torch.no_grad():
                lm = logits.mean()
                if lm < -3.0:
                    self.output.bias.data += 0.01 * (-3.0 - lm)
        return logits


# ==============================================================================
# v10: MetaAligner（过拟合感知，仅输出 aux_weight）
# ==============================================================================

class MetaAligner(nn.Module):
    def __init__(
        self,
        num_codes: int = 128,
        warmup_steps: int = 300,
        probe_burn_in: int = 200,
        kp: float = 2.0,
        ki: float = 0.02,
        kd: float = 1.0,
        max_aux_weight: float = 0.15,
        leak_rate: float = 0.03,
        valid_decay_rate: float = 0.98,
        ema_fast: float = 0.7,
        ema_slow: float = 0.95,
        integral_clip: float = 2.0,
    ):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

        self.warmup_steps = warmup_steps
        self.probe_burn_in = probe_burn_in
        self.max_aux_weight = max_aux_weight
        self.leak_rate = leak_rate
        self.valid_decay_rate = valid_decay_rate
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.integral_clip = integral_clip

        self.kp, self.ki, self.kd = kp, ki, kd

        self.register_buffer('integral', torch.tensor(0.0))
        self.register_buffer('last_valid_gap', torch.tensor(0.0))
        self.register_buffer('steps_since_valid', torch.tensor(0))
        self.register_buffer('ema_aux', torch.tensor(0.0))

        self.gap_history = deque(maxlen=10)
        self.aux_history = deque(maxlen=10)

    def _leaky_integrate(self, gap: float) -> float:
        self.integral = (1 - self.leak_rate) * self.integral + gap
        self.integral = torch.clamp(
            self.integral,
            min=-self.integral_clip,
            max=self.integral_clip
        )
        return self.integral.item()

    def _adaptive_ema(self, new_val: float, old_ema: float) -> float:
        delta = abs(new_val - old_ema)
        ema_decay = self.ema_fast if delta > 0.1 else self.ema_slow
        return ema_decay * old_ema + (1 - ema_decay) * new_val

    def _interpolate_without_valid(self, base_aux: float) -> float:
        steps = int(self.steps_since_valid.item())
        decay = self.valid_decay_rate ** steps
        conservative = 0.05
        return conservative + (base_aux - conservative) * decay

    def forward(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float],
        valid_auc: Optional[float] = None,
        train_auc: Optional[float] = None,
        global_step: int = 0,
        ctr_loss: Optional[float] = None,
        uncertainty_mean: Optional[float] = None,
        diff_quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        # 计算当前 gap
        gap = 0.0
        if train_auc is not None and valid_auc is not None and valid_auc > 0:
            gap = max(0.0, train_auc - valid_auc)
            self.gap_history.append(gap)
            self.last_valid_gap = torch.tensor(gap)
            self.steps_since_valid = torch.tensor(0)
        else:
            self.steps_since_valid += 1
            gap = float(self.last_valid_gap.item()) * 1.01

        # CTR loss 趋势信号
        ctr_trend = 0.0
        if ctr_loss is not None:
            if not hasattr(self, '_ctr_loss_history'):
                self._ctr_loss_history = deque(maxlen=5)
            if len(self._ctr_loss_history) > 0:
                last_ctr = self._ctr_loss_history[-1]
                ctr_trend = max(0.0, (ctr_loss - last_ctr) / max(abs(last_ctr), 1e-8))
            self._ctr_loss_history.append(ctr_loss)

        # 不确定性信号
        uncertainty_signal = 0.0
        if uncertainty_mean is not None:
            uncertainty_signal = max(0.0, uncertainty_mean - 1.0)

        # PID 控制
        integral = self._leaky_integrate(gap + 0.5 * ctr_trend + 0.3 * uncertainty_signal)
        gap_trend = 0.0
        if len(self.gap_history) >= 2:
            gap_trend = (self.gap_history[-1] - self.gap_history[0]) / len(self.gap_history)

        P = self.kp * gap
        I = self.ki * integral
        D = self.kd * (-gap_trend)
        raw_aux = max(0.0, P + I + D)

        # Valid 间插值
        if valid_auc is None or valid_auc <= 0:
            raw_aux = self._interpolate_without_valid(raw_aux)

        # 自适应 EMA
        new_ema = self._adaptive_ema(raw_aux, float(self.ema_aux.item()))
        self.ema_aux = torch.tensor(new_ema)
        aux_weight = min(self.max_aux_weight, new_ema)

        result = {
            'aux_weight': aux_weight,
            'ctr_weight': 1.0,
            'gap': gap,
            'integral': integral,
            'mode': 'valid_pid' if (valid_auc and valid_auc > 0) else 'interpolated',
            'ema_aux': new_ema,
            'steps_since_valid': int(self.steps_since_valid.item()),
        }
        self.aux_history.append(aux_weight)
        return result


# ==============================================================================
# v10: PCVRHeteroFormer (main model)
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
        **kwargs,  # 吸收旧版兼容参数（use_diffusion / use_energy 等）
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes

        self.encoder = MultiViewEncoder(
            user_int_feature_specs=user_int_feature_specs,
            item_int_feature_specs=item_int_feature_specs,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            d_model=d_model,
            emb_dim=emb_dim,
            id_vocab_threshold=id_vocab_threshold,
            emb_skip_threshold=emb_skip_threshold,
            dropout_rate=dropout,
        )

        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_encoders = nn.ModuleDict()
        for domain, vocab_sizes in seq_vocab_sizes.items():
            self.seq_encoders[domain] = ContinuousSequenceEncoder(
                vocab_sizes=vocab_sizes, d_model=d_model, state_dim=max(64, d_model // 2),
                num_layers=2, max_seq_len=512, id_threshold=seq_id_threshold,
                dropout_rate=dropout,
            )

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

        # v10: 生成式语义层
        self.diffusion_explainer = DiffusionExplainer(d_model, num_codes, dropout_rate=dropout)
        self.energy_calibrator = EnergyCalibrator(d_model, num_codes, dropout_rate=dropout)

        # v10: 特征交互层
        self.explicit_cross = BilinearCrossLayer(d_model, num_fields=4, cross_ratio=0.1)
        self.cross_field = ProtoConditionedCrossFieldNet(
            fields=['user', 'item', 'context', 'seq'],
            d_model=d_model, num_layers=num_layers,
            num_heads=num_heads or max(4, d_model // 32), num_codes=num_codes,
            ffn_ratio=2.0, dropout_rate=dropout,
        )

        # v10: 预测层
        self.ctr_head = CalibratedCTRHead(d_model, num_codes, dropout_rate=dropout)

        self.meta_aligner = MetaAligner(
            num_codes=num_codes,
            warmup_steps=300,
            probe_burn_in=200,
            kp=2.0,
            ki=0.05,
            kd=1.0,
            max_aux_weight=0.3,
            leak_rate=0.01,
            valid_decay_rate=0.995,
            ema_fast=0.8,
            ema_slow=0.95,
        )
        self.logit_temperature = nn.Parameter(torch.tensor(0.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        self.seq_fusion = MultiSeqFusion(d_model, len(self.seq_domains))

        self._register_param_groups()

    def _register_param_groups(self):
        """v10 参数分组：shared(CTR路径) / gen(生成模块自监督) / sparse / meta"""
        self._sparse_params = []
        self._shared_encoder_params = []
        self._seq_encoder_params = []
        self._proto_geo_params = []
        self._cross_field_params = []
        self._ctr_head_params = []
        self._gen_module_params = []   # diffusion_explainer + energy_calibrator
        self._meta_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if 'embedding' in name or ('emb' in name.lower() and 'num_embeddings' in str(type(p))):
                self._sparse_params.append(p)
            elif 'seq_encoders' in name:
                self._seq_encoder_params.append(p)
            elif any(x in name for x in ['prototype.eta', 'prototype.rotation',
                                           'prototype.kappa_log', 'prototype.kappa_time_decay',
                                           'prototype.empty_prior', 'prototype.U', 'prototype.V',
                                           'prototype.kappa_user_proj']):
                self._proto_geo_params.append(p)
            elif 'prototype' in name:
                self._proto_geo_params.append(p)
            elif 'diffusion_explainer' in name or 'energy_calibrator' in name:
                self._gen_module_params.append(p)
            elif 'cross_field' in name or 'explicit_cross' in name:
                self._cross_field_params.append(p)
            elif 'ctr_head' in name:
                self._ctr_head_params.append(p)
            elif 'meta_aligner' in name:
                self._meta_params.append(p)
            elif 'logit_temperature' in name or 'logit_bias' in name:
                self._ctr_head_params.append(p)
            elif 'encoder' in name:
                self._shared_encoder_params.append(p)
            else:
                self._shared_encoder_params.append(p)

        # 生成模块统一包含所有 prototype 几何参数
        self._gen_module_params.extend(self._proto_geo_params)

        all_assigned = (
            self._sparse_params + self._shared_encoder_params +
            self._seq_encoder_params + self._cross_field_params +
            self._ctr_head_params + self._gen_module_params + self._meta_params
        )
        total_params = sum(1 for _ in self.parameters() if _.requires_grad)
        assigned_count = len(all_assigned)

        if assigned_count != total_params:
            logging.warning(f"【参数分组警告】{assigned_count}/{total_params} 参数被分配，存在遗漏！")
        else:
            logging.info(f"【参数分组v10】所有{total_params}个参数已正确分配")

        group_counts = {
            'sparse': len(self._sparse_params),
            'shared_encoder': len(self._shared_encoder_params),
            'seq_encoder': len(self._seq_encoder_params),
            'cross_field': len(self._cross_field_params),
            'ctr_head': len(self._ctr_head_params),
            'gen_module': len(self._gen_module_params),
            'meta': len(self._meta_params),
        }
        for gname, gcount in group_counts.items():
            if gcount > 0:
                logging.info(f"  {gname}: {gcount} params")

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            'sparse': self._sparse_params,
            'shared_encoder': self._shared_encoder_params,
            'seq_encoder': self._seq_encoder_params,
            'cross_field': self._cross_field_params,
            'ctr_head': self._ctr_head_params,
            'gen_module': self._gen_module_params,
            'meta': self._meta_params,
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

        # 3. Multi-seq fusion
        seq_meta = {'len': torch.zeros(B, device=device), 'time_span': torch.ones(B, device=device) * 86400.0}
        z_seq_fused = torch.zeros(B, self.d_model, device=device)

        if len(seq_outputs) > 0:
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

            if len_list:
                stacked_lens = torch.stack(len_list, dim=1)
                seq_meta['len'] = stacked_lens.max(dim=1)[0]

            if self.num_seq_domains > 1 and len(short_list) > 1:
                s_short, gate_short, ent_loss_short = self.seq_fusion_short(short_list)
                s_long, gate_long, ent_loss_long = self.seq_fusion_long(long_list)
                s_static, gate_static, ent_loss_static = self.seq_fusion_static(static_list)
                self._cached_entropy_loss = (ent_loss_short + ent_loss_long + ent_loss_static) / 3.0
                self._cached_seq_gate_short = gate_short
            else:
                s_short = short_list[0] if short_list else torch.zeros(B, self.d_model, device=device)
                s_long = long_list[0] if long_list else torch.zeros(B, self.d_model, device=device)
                s_static = static_list[0] if static_list else torch.zeros(B, self.d_model, device=device)
                self._cached_entropy_loss = torch.tensor(0.0, device=device)

            z_seq_fused = s_short
        else:
            s_short = torch.zeros(B, self.d_model, device=device)
            s_long = torch.zeros(B, self.d_model, device=device)
            s_static = torch.zeros(B, self.d_model, device=device)

        # 4. Prototype（生成式语义基座）
        proto_weights, proto_repr, kappa_mean, assign_entropy = self.prototype(
            z_seq_fused, seq_meta, user_feat, task_id='ctr', is_training=self.training
        )
        mu_u = self.prototype.get_rotated_prototypes(user_feat)  # [B, K, D]

        # 5. 生成式语义层（自监督，与 CTR 梯度解耦）
        diff_explain, diff_quality, ib_loss, recon_loss = self.diffusion_explainer(
            z_seq_fused.detach(), proto_weights.detach(), mu_u.detach()
        )
        energy_score = self.energy_calibrator(
            proto_weights.detach(), user_feat.detach(), item_feat.detach()
        )

        # 6. 显式二阶交叉（基础交互）
        fields_list = [user_feat, item_feat, context_feat, z_seq_fused]
        crossed_fields = self.explicit_cross(fields_list)
        u_c, i_c, c_c, s_c = crossed_fields

        # 7. Proto条件化场交互（核心预测能力来源，接收 CTR 梯度）
        fields = {'user': u_c, 'item': i_c, 'context': c_c, 'seq': s_c}
        cross_out = self.cross_field(
            fields, seq_cond=s_c,
            proto_weights=proto_weights.detach(),
            diff_explain=diff_explain.detach(),
        )
        final_repr = torch.cat([cross_out['user'], cross_out['item'],
                                cross_out['context'], cross_out['seq']], dim=-1)

        if torch.isnan(final_repr).any():
            logging.warning("【NaN防护】final_repr NaN，使用context_feat替代")
            final_repr = torch.cat([context_feat] * 4, dim=-1)

        # 8. 正交约束（生成模块内部自监督）
        ortho_loss = torch.abs(F.cosine_similarity(
            F.normalize(proto_repr, dim=-1),
            F.normalize(diff_explain, dim=-1),
            dim=-1
        )).mean()

        # 9. 预测层（eval 时 energy_score 校准）
        energy_cal = energy_score.detach() if not self.training else None
        logits = self.ctr_head(
            final_repr, proto_repr.detach(), proto_weights.detach(),
            user_feat, item_feat, energy_score=energy_cal
        )

        if torch.isnan(logits).any():
            logging.warning("【NaN防护】CTR logits NaN，输出零")
            logits = torch.zeros(B, 1, device=device)

        logits = torch.clamp(logits, min=-20.0, max=20.0)
        temperature = torch.clamp(torch.exp(self.logit_temperature), min=0.1, max=5.0)
        logits = logits / temperature + self.logit_bias
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        # 缓存（供 trainer 诊断使用）
        self._cached_user_feat = user_feat
        self._cached_item_feat = item_feat
        self._cached_context_feat = context_feat
        self._cached_proto_repr = proto_repr
        self._cached_z_seq = z_seq_fused
        self._cached_energy_score = energy_score

        # 10. packing loss（原型几何正则）
        packing_loss = self.prototype.packing_loss()

        # 返回 12 元素（严格对齐 v10 框架）
        return (
            logits, proto_weights, proto_repr, kappa_mean, assign_entropy,
            diff_explain, diff_quality, ib_loss, energy_score, recon_loss, ortho_loss, packing_loss
        )

    def predict(self, model_input: ModelInput):
        with torch.no_grad():
            out = self.forward(model_input, task_id='ctr')
        return out[0], None

    def get_packing_loss(self) -> Tensor:
        return self.prototype.packing_loss()


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