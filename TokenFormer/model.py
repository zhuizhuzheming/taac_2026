# ==================== model.py (TokenFormer DEFUSE 完整版) ====================

"""PCVRTokenFormer with Action-Aware Embedding and Multi-Task Output."""

import logging
import math
from typing import List, NamedTuple, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict
    seq_time_decay: Optional[dict] = None
    user_id: Optional[List[str]] = None
    action_type: Optional[torch.Tensor] = None  # 【新增】[B] 0=曝光,1=点击,2=转化


# ── RoPE (unchanged) ──
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        return self.cos_cached[:, :seq_len, :].to(device), self.sin_cached[:, :seq_len, :].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    out = x * cos_ + rotate_half(x) * sin_
    return out.to(orig_dtype)


# ── NLIR Gate (unchanged) ──
class NLIRGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_model, bias=False)

    def forward(self, residual: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W_g(residual))
        return gate * attn_out


# ── SwiGLU FFN (unchanged) ──
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        self.W12 = nn.Linear(d_model, 2 * hidden_dim)
        self.W3 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x12 = self.W12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        x = self.W3(x)
        return residual + x


# ── Unified Interaction Block (unchanged) ──
class UnifiedInteractionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 hidden_mult: int = 4, use_nlir: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_nlir = use_nlir
        assert d_model % num_heads == 0

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        if use_nlir:
            self.nlr = NLIRGate(d_model)
        self.ffn = SwiGLUFFN(d_model, hidden_mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm_attn(x)

        qkv = self.W_qkv(x_norm)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            Q = apply_rope_to_tensor(Q, rope_cos, rope_sin)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)

        if self.use_nlir:
            out = self.nlr(residual, out)
        out = residual + self.dropout(out)
        out = self.ffn(out)
        return out


# ── CrossDomainQueryInteraction (unchanged) ──
class CrossDomainQueryInteraction(nn.Module):
    def __init__(self, d_model: int, num_sequences: int, num_queries: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.SiLU(),
            ) for _ in range(num_sequences)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_sequences)
        ])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_sequences)])

    def forward(self, q_tokens_list: List[torch.Tensor]) -> List[torch.Tensor]:
        q_all = torch.cat(q_tokens_list, dim=1)
        global_q = self.global_pool(q_all.transpose(1, 2)).transpose(1, 2)
        
        result = []
        for i, q_domain in enumerate(q_tokens_list):
            attn_out, _ = self.cross_attn[i](q_domain, global_q, global_q)
            attn_out = self.norm[i](q_domain + attn_out)
            fused = self.fusion[i](torch.cat([q_domain, global_q.expand(-1, q_domain.shape[1], -1)], dim=-1))
            result.append(fused + attn_out)
        
        return result


# ── NS Tokenizers (unchanged) ──
class GroupNSTokenizer(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            ) for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class RankMixerNSTokenizer(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 num_ns_tokens: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            ) for _ in range(num_ns_tokens)
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)
        cat_emb = torch.cat(all_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))
        chunks = cat_emb.split(self.chunk_dim, dim=-1)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


# ── NS Compressor (unchanged) ──
class NSCompressor(nn.Module):
    def __init__(self, d_model: int, num_summary_tokens: int = 4):
        super().__init__()
        self.summary_tokens = num_summary_tokens
        self.query = nn.Parameter(torch.randn(1, num_summary_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        B = ns_tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.cross_attn(q, ns_tokens, ns_tokens)
        return self.norm(out)


# ── MultiSeqQueryGenerator (unchanged) ──
class MultiSeqQueryGenerator(nn.Module):
    def __init__(
        self,
        d_model: int, num_ns: int, num_queries: int,
        num_sequences: int, hidden_mult: int = 4,
        use_time_diff: bool = False, time_emb_dim: int = 0,
        use_time_decay: bool = False):
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.use_time_diff = use_time_diff
        self.time_emb_dim = time_emb_dim
        self.use_time_decay = use_time_decay

        self.ns_flat_dim = num_ns * d_model
        self.seq_pool_dim = d_model
        self.base_input_dim = self.ns_flat_dim + self.seq_pool_dim
        self.has_time_diff = use_time_diff and time_emb_dim > 0
        
        self.input_proj = nn.Linear(self.base_input_dim, d_model)
        
        if self.has_time_diff:
            self.time_diff_proj = nn.Linear(time_emb_dim, d_model // 2)
            self.fusion_proj = nn.Linear(d_model + d_model // 2, d_model)
        else:
            self.fusion_proj = None
        
        self.global_info_norm = nn.LayerNorm(d_model)
        
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                ) for _ in range(num_queries)
            ]) for _ in range(num_sequences)
        ])

    def forward(self, ns_tokens: torch.Tensor, seq_tokens_list: list,
                seq_padding_masks: list, 
                seq_time_embs: Optional[List[torch.Tensor]] = None,
                seq_time_decay: Optional[List[torch.Tensor]] = None):
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)
        q_tokens_list = []
        
        for i in range(self.num_sequences):
            mask = ~seq_padding_masks[i]
            valid_mask = mask.unsqueeze(-1).float()
            
            seq_sum = (seq_tokens_list[i] * valid_mask).sum(dim=1)
            seq_count = valid_mask.sum(dim=1).clamp(min=1)
            seq_pooled = seq_sum / seq_count

            if self.use_time_decay and seq_time_decay is not None and i < len(seq_time_decay):
                decay = seq_time_decay[i].unsqueeze(-1)
                weighted_sum = (seq_tokens_list[i] * decay * valid_mask).sum(dim=1)
                weight_sum = (decay * valid_mask).sum(dim=1).clamp(min=1)
                seq_pooled = weighted_sum / weight_sum

            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            global_info = self.input_proj(global_info)
            
            if self.has_time_diff and seq_time_embs is not None and i < len(seq_time_embs):
                time_feat = self.time_diff_proj(seq_time_embs[i])
                global_info = self.fusion_proj(torch.cat([global_info, time_feat], dim=-1))
            
            global_info = self.global_info_norm(global_info)
            
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)
            q_tokens_list.append(q_tokens)
        
        return q_tokens_list


# ── Adaptive SWA Config (unchanged) ──
class AdaptiveSWAConfig:
    def __init__(self, default_window: int = 50):
        self.default_window = default_window
        self.domain_windows: Dict[str, int] = {}

    def set_window(self, domain: str, max_len: int, ratio: float = 0.2):
        window = max(int(max_len * ratio), self.default_window)
        self.domain_windows[domain] = window
        return window

    def get_window(self, domain: str) -> int:
        return self.domain_windows.get(domain, self.default_window)


# ═══════════════════════════════════════════════════════
# Main Model - TokenFormer DEFUSE Edition
# ═══════════════════════════════════════════════════════
class PCVRTokenFormer(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        d_model: int = 128,           # 【修改】64→128
        emb_dim: int = 128,           # 【修改】64→128
        num_queries: int = 4,         # 【修改】1→4
        num_blocks: int = 3,          # 【修改】4→3
        num_heads: int = 4,
        hidden_mult: int = 4,
        dropout_rate: float = 0.1,    # 【修改】0.01→0.1
        action_num: int = 1,
        num_time_buckets: int = 65,
        use_rope: bool = True,        # 【修改】False→True
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        bfts_l_full: int = 2,
        bfts_swa_window: int = 50,
        use_nlir: bool = True,
        ns_token_dropout: float = 0.1,  # 【修改】0.0→0.1
        use_time_diff: bool = False,
        use_time_decay: bool = True,    # 【修改】默认开启
        ns_summary_tokens: int = 4,
        cache_ns_during_training: bool = False,
        ns_cache_size: int = 10000,
        seq_max_lens: Optional[Dict[str, int]] = None,
        use_cross_domain: bool = False,  # 【修改】默认关闭，简化
        predict_conversion_time: bool = False,
        multi_task: bool = False,        # 【新增】多任务开关
        num_action_types: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.use_rope = use_rope
        self.bfts_l_full = bfts_l_full
        self.use_nlir = use_nlir
        self.num_blocks = num_blocks
        self.ns_token_dropout = ns_token_dropout
        self.use_time_diff = use_time_diff
        self.use_time_decay = use_time_decay
        self.ns_summary_tokens = ns_summary_tokens
        self.cache_ns_during_training = cache_ns_during_training
        self.predict_conversion_time = predict_conversion_time
        self.multi_task = multi_task      # 【新增】
        self._global_ns = None

        # Adaptive SWA
        self.swa_config = AdaptiveSWAConfig(default_window=bfts_swa_window)
        if seq_max_lens is not None:
            for domain in self.seq_domains:
                max_len = seq_max_lens.get(domain, 256)
                self.swa_config.set_window(domain, max_len, ratio=0.2)
            logging.info(f"Adaptive SWA: {self.swa_config.domain_windows}")

        # NS tokenizers
        if ns_tokenizer_type == 'group':
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs, groups=user_ns_groups,
                emb_dim=emb_dim, d_model=d_model, emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)
            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs, groups=item_ns_groups,
                emb_dim=emb_dim, d_model=d_model, emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            if user_ns_tokens <= 0: user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0: item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs, groups=user_ns_groups,
                emb_dim=emb_dim, d_model=d_model, num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens
            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs, groups=item_ns_groups,
                emb_dim=emb_dim, d_model=d_model, num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model), nn.LayerNorm(d_model))
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model), nn.LayerNorm(d_model))

        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        self.ns_compressor = NSCompressor(d_model, ns_summary_tokens)

        if cache_ns_during_training:
            self.register_buffer('_ns_cache_keys', torch.full((ns_cache_size,), -1, dtype=torch.long))
            self.register_buffer('_ns_cache_values', torch.zeros(ns_cache_size, ns_summary_tokens, d_model))
            self._ns_cache_ptr = 0
            self._ns_cache_hits = 0
            self._ns_cache_misses = 0

        # 【新增】Action Type Embedding
        self.action_emb = nn.Embedding(num_action_types, emb_dim, padding_idx=0)  # 0=曝光,1=点击,2=转化
        nn.init.xavier_normal_(self.action_emb.weight.data)
        self.action_emb.weight.data[0, :] = 0

        # Sequence embeddings
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}
        self._seq_is_id = {}
        self._seq_vocab_sizes = {}
        self._seq_proj = nn.ModuleDict()

        def _make_seq_embs(vocab_sizes):
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        time_diff_dim = d_model if (use_time_diff and num_time_buckets > 0) else 0

        if use_time_decay:
            self.time_decay_proj = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, d_model),
            )

        self.query_generator = MultiSeqQueryGenerator(
            d_model=self.d_model, num_ns=self.num_ns, num_queries=num_queries,
            num_sequences=self.num_sequences, hidden_mult=hidden_mult,
            use_time_diff=use_time_diff, time_emb_dim=time_diff_dim,
            use_time_decay=use_time_decay,
        )

        self.use_cross_domain = use_cross_domain
        if use_cross_domain:
            self.cross_domain_interaction = CrossDomainQueryInteraction(
                d_model=d_model, num_sequences=self.num_sequences, 
                num_queries=num_queries, num_heads=num_heads, dropout=dropout_rate
            )

        self.sep_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList([
            UnifiedInteractionBlock(
                d_model=d_model, num_heads=num_heads, dropout=dropout_rate,
                hidden_mult=hidden_mult, use_nlir=use_nlir,
            ) for _ in range(num_blocks)
        ])

        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Static masks
        self._has_static_masks = False
        if seq_max_lens is not None:
            self._register_static_masks(seq_max_lens, num_blocks)

        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.emb_dropout = nn.Dropout(dropout_rate)
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        # 【修改】分类器：支持多任务
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )
        
        # 【新增】多任务输出头
        if multi_task:
            self.cvr_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, 1)
            )
            self.ctr_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, 1)
            )
            self.ctcvr_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model // 2, 1)
            )
        
        if predict_conversion_time:
            self.time_predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, 1)
            )

        self._init_params()
        self._init_weights()

    def _register_static_masks(self, seq_max_lens: Dict[str, int], num_layers: int):
        ns_len = self.ns_summary_tokens
        total_len = ns_len
        boundaries = []
        
        for domain in self.seq_domains:
            max_len = seq_max_lens.get(domain, 256)
            start = total_len + 1
            end = start + max_len
            boundaries.append((start, end))
            total_len = end + 1
        
        total_len += 1
        query_start = total_len
        for _ in self.seq_domains:
            total_len += self.num_queries
        
        l_full = min(self.bfts_l_full, num_layers)
        
        for l_idx in range(num_layers):
            mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
            
            if l_idx >= l_full:
                for domain_idx, (start, end) in enumerate(boundaries):
                    if end <= start:
                        continue
                    domain = self.seq_domains[domain_idx]
                    L_seq = end - start
                    window = self.swa_config.get_window(domain)
                    
                    idx = torch.arange(L_seq)
                    win_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) >= 0) & \
                               (idx.unsqueeze(1) - idx.unsqueeze(0) < window)
                    mask[start:end, start:end] = win_mask
            
            self.register_buffer(f'_static_mask_l{l_idx}', mask, persistent=False)
        
        self._has_static_masks = True
        self._static_total_len = total_len
        self._static_query_start = query_start
        self._static_ns_len = ns_len
        self._static_boundaries = boundaries

    def _init_params(self):
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0
        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0
        nn.init.normal_(self.sep_embedding, std=0.02)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def set_global_ns(self, global_ns: torch.Tensor):
        with torch.no_grad():
            summary = self.ns_compressor(global_ns.unsqueeze(0))
        self._global_ns_summary = summary.squeeze(0)

    def _get_cached_ns(self, user_ids, ns_tokens_raw):
        if not hasattr(self, '_ns_cache_keys'):
            return self.ns_compressor(ns_tokens_raw)
        B = len(user_ids)
        device = ns_tokens_raw.device
        result = torch.zeros(B, self.ns_summary_tokens, self.d_model, device=device)
        need_compute_idx = []
        need_compute_cache_pos = []
        
        for i, uid in enumerate(user_ids):
            if uid is None:
                need_compute_idx.append(i)
                need_compute_cache_pos.append(-1)
                continue
            cache_idx = hash(str(uid)) % self._ns_cache_keys.shape[0]
            cached_key = self._ns_cache_keys[cache_idx].item()
            if cached_key == hash(str(uid)):
                result[i] = self._ns_cache_values[cache_idx].to(device)
                self._ns_cache_hits += 1
            else:
                need_compute_idx.append(i)
                need_compute_cache_pos.append(cache_idx)
                self._ns_cache_misses += 1
        
        if need_compute_idx:
            computed = self.ns_compressor(ns_tokens_raw[need_compute_idx])
            result[need_compute_idx] = computed
            for idx, cache_pos in zip(need_compute_idx, need_compute_cache_pos):
                if cache_pos >= 0:
                    self._ns_cache_keys[cache_pos] = hash(str(user_ids[idx]))
                    self._ns_cache_values[cache_pos] = computed[idx].detach().cpu()
        
        return result

    # 【核心修改】_embed_seq_domain 增加 action_types 参数
    def _embed_seq_domain(self, seq, sideinfo_embs, proj, is_id, emb_index, 
                         time_bucket_ids, time_decay=None, action_types=None):
        """
        【修改】增加 action_types 参数，注入行为类型信号
        
        action_types: [B] — 每个样本的行为类型（0/1/2）
        """
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        
        # 基础 item embedding
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = F.gelu(proj(cat_emb))  # [B, L, d_model]
        
        # 【核心新增】Action Type Gating
        if action_types is not None and hasattr(self, 'action_emb'):
            # action_types: [B] → [B, 1] → [B, L]
            action_types_expanded = action_types.unsqueeze(1).expand(-1, L)  # [B, L]
            
            # 查询 action embedding: [B, L] → [B, L, emb_dim]
            action_emb = self.action_emb(action_types_expanded)
            
            # 轻量投影到 d_model 维度（如果 emb_dim != d_model）
            if action_emb.shape[-1] != token_emb.shape[-1]:
                action_proj = nn.Linear(action_emb.shape[-1], token_emb.shape[-1], device=action_emb.device)
                action_emb = action_proj(action_emb)
            
            # 门控融合：action_emb 调制 token_emb
            # 点击(1)和转化(2)增强，曝光(0)中性
            gate = torch.sigmoid(action_emb.mean(dim=-1, keepdim=True))  # [B, L, 1]
            # 中心化：gate 从 [0,1] 映射到 [0.5, 1.5] 的调制范围
            modulation = 1.0 + 0.3 * (gate - 0.5)  # 点击/转化增强 ~1.15，曝光抑制 ~0.85
            token_emb = token_emb * modulation
        
        # Time Bucket
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        
        # Time Decay
        if time_decay is not None and hasattr(self, 'time_decay_proj'):
            decay_emb = self.time_decay_proj(time_decay.unsqueeze(-1))
            token_emb = token_emb + decay_emb
        
        return token_emb

    def _make_padding_mask(self, seq_len, max_len):
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    def _forward_impl(self, inputs: ModelInput, apply_dropout: bool):
        B = inputs.user_int_feats.shape[0]
        device = inputs.user_int_feats.device

        # 1. Raw NS tokens
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)
        ns_parts = [user_ns]
        if self.has_user_dense:
            ns_parts.append(F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1))
        ns_parts.append(item_ns)
        if self.has_item_dense:
            ns_parts.append(F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1))
        ns_tokens_raw = torch.cat(ns_parts, dim=1)

        if self.training and self.ns_token_dropout > 0:
            mask = torch.rand(B, ns_tokens_raw.shape[1], 1, device=device) >= self.ns_token_dropout
            ns_tokens_raw = ns_tokens_raw * mask

        # 2. NS Compression
        if self.training and self.cache_ns_during_training and inputs.user_id is not None:
            ns_summary = self._get_cached_ns(inputs.user_id, ns_tokens_raw)
        elif not self.training and self._global_ns is not None:
            ns_summary = self._global_ns_summary.unsqueeze(0).expand(B, -1, -1)
        else:
            ns_summary = self.ns_compressor(ns_tokens_raw)
        ns_len = ns_summary.shape[1]

        # 3. Embed sequences —— 【修改】传入 action_type
        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
        seq_time_decay_list = []
        
        # 获取 action_type（可能为 None）
        action_types = getattr(inputs, 'action_type', None)
        
        for domain in self.seq_domains:
            time_decay = None
            if self.use_time_decay and inputs.seq_time_decay is not None:
                time_decay = inputs.seq_time_decay.get(domain, None)
            
            # 【修改】传入 action_types
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                time_decay=time_decay,
                action_types=action_types)
            
            mask = self._make_padding_mask(inputs.seq_lens[domain], tokens.shape[1])
            seq_tokens_list.append(tokens)
            seq_masks_list.append(mask)
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])
            if time_decay is not None:
                seq_time_decay_list.append(time_decay)

        # 4. Time embeddings
        seq_time_embs = None
        if self.use_time_diff and self.num_time_buckets > 0:
            seq_time_embs = []
            for time_bucket_ids in seq_time_buckets_list:
                valid_mask = (time_bucket_ids != 0).float()
                sum_bucket = (time_bucket_ids.float() * valid_mask).sum(dim=1)
                count = valid_mask.sum(dim=1).clamp(min=1)
                mean_bucket = (sum_bucket / count).long()
                mean_bucket = torch.clamp(mean_bucket, 1, self.num_time_buckets - 1)
                time_emb = self.time_embedding(mean_bucket)
                seq_time_embs.append(time_emb)

        # 5. Query tokens
        q_tokens_list = self.query_generator(
            ns_tokens_raw, seq_tokens_list, seq_masks_list, 
            seq_time_embs=seq_time_embs,
            seq_time_decay=seq_time_decay_list if self.use_time_decay else None)

        # Cross-domain interaction
        if self.use_cross_domain:
            q_tokens_list = self.cross_domain_interaction(q_tokens_list)

        # 6. Build unified stream
        sep = self.sep_embedding.expand(B, -1, -1)
        parts = [ns_summary]
        boundaries = []
        current = ns_len
        for tokens in seq_tokens_list:
            parts.append(sep)
            current += 1
            start = current
            parts.append(tokens)
            current += tokens.shape[1]
            end = current
            boundaries.append((start, end))
        parts.append(sep)
        current += 1
        query_start = current
        for q in q_tokens_list:
            parts.append(q)
            current += q.shape[1]

        x = torch.cat(parts, dim=1)
        total_len = x.shape[1]

        if apply_dropout:
            x = self.emb_dropout(x)

        # 7. Run blocks with static masks
        l_full = min(self.bfts_l_full, self.num_blocks)
        
        if self._has_static_masks and total_len == self._static_total_len:
            for l_idx in range(self.num_blocks):
                mask = getattr(self, f'_static_mask_l{l_idx}')
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
                rope_cos, rope_sin = None, None
                if self.rotary_emb is not None:
                    rope_cos, rope_sin = self.rotary_emb(total_len, device)
                x = self.blocks[l_idx](x, attn_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
                if torch.isnan(x).any():
                    logging.warning(f"NaN after block {l_idx}, resetting")
                    x = torch.nan_to_num(x, nan=0.0)
        else:
            for l_idx in range(self.num_blocks):
                mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1).clone()
                if l_idx >= l_full:
                    for domain_idx, (start, end) in enumerate(boundaries):
                        if end <= start: 
                            continue
                        domain = self.seq_domains[domain_idx]
                        L_seq = end - start
                        window = self.swa_config.get_window(domain)
                        
                        idx = torch.arange(L_seq, device=device)
                        win_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) >= 0) & \
                                   (idx.unsqueeze(1) - idx.unsqueeze(0) < window)
                        mask[:, :, start:end, start:end] = win_mask.unsqueeze(0).unsqueeze(0)
                rope_cos, rope_sin = None, None
                if self.rotary_emb is not None:
                    rope_cos, rope_sin = self.rotary_emb(total_len, device)
                x = self.blocks[l_idx](x, attn_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)

        # 8. Extract outputs
        query_outputs = x[:, query_start:, :]
        flat_q = query_outputs.reshape(B, -1)
        out_repr = self.output_proj(flat_q)
        
        # 【修改】多任务输出
        if self.multi_task:
            cvr_logit = self.cvr_head(out_repr)
            ctr_logit = self.ctr_head(out_repr)
            ctcvr_logit = self.ctcvr_head(out_repr)
            return cvr_logit, ctr_logit, ctcvr_logit, out_repr
        
        logits = self.clsfier(out_repr)
        
        time_pred = None
        if self.predict_conversion_time and hasattr(self, 'time_predictor'):
            time_pred = self.time_predictor(out_repr).squeeze(-1)
        
        return logits, out_repr, time_pred

    # 【修改】forward 支持多任务
    def forward(self, inputs: ModelInput):
        if self.multi_task:
            cvr_logit, ctr_logit, ctcvr_logit, out_repr = self._forward_impl(inputs, apply_dropout=self.training)
            if self.predict_conversion_time and self.training:
                time_pred = self.time_predictor(out_repr).squeeze(-1) if hasattr(self, 'time_predictor') else None
                return cvr_logit, ctr_logit, ctcvr_logit, time_pred
            return cvr_logit, ctr_logit, ctcvr_logit
        
        logits, _, time_pred = self._forward_impl(inputs, apply_dropout=self.training)
        if self.predict_conversion_time and self.training:
            return logits, time_pred
        return logits

    # 【修改】predict 支持多任务
    def predict(self, inputs: ModelInput):
        with torch.no_grad():
            if self.multi_task:
                cvr_logit, ctr_logit, ctcvr_logit, out_repr = self._forward_impl(inputs, apply_dropout=False)
                return cvr_logit, ctr_logit, ctcvr_logit, out_repr
            
            logits, out_repr, _ = self._forward_impl(inputs, apply_dropout=False)
        return logits, out_repr

    def get_sparse_params(self):
        sparse_ptrs = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_ptrs.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_ptrs]

    def get_dense_params(self):
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, threshold=10000):
        reinit_ptrs = set()
        for domain in self.seq_domains:
            for i, vs in enumerate(self._seq_vocab_sizes[domain]):
                real_idx = self._seq_emb_index[domain][i]
                if real_idx == -1: 
                    continue
                if int(vs) > threshold:
                    emb = self._seq_embs[domain][real_idx]
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for i, (vs, offset, length) in enumerate(tokenizer.feature_specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1: 
                    continue
                if int(vs) > threshold:
                    emb = tokenizer.embs[real_idx]
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        logging.info(f"Re-initialised {len(reinit_ptrs)} high‑card embeddings")
        return reinit_ptrs
