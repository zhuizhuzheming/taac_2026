"""PCVRHyFormer → TokenFormer hybrid with static shape & computational graph optimisations.

New features:
- NS compressor: reduces variable-length NS tokens → fixed number of summary tokens,
  enabling static sequence length throughout the model.
- Fused QKV projection: single Linear(d, 3d) for efficiency.
- Template‑based BFTS mask cache: pre‑built lower‑triangular and sliding window masks,
  avoiding per‑layer Python loops.
"""

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


# ══════════════════════════════ RoPE (stability) ══════════════════════════════
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
    """RoPE computed in float32 to prevent fp16 overflow."""
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float()
    sin = sin.float()
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (1, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    out = x * cos_ + rotate_half(x) * sin_
    return out.to(orig_dtype)


# ═══════════════════════════ NLIR Gate ═══════════════════════════
class NLIRGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_model, bias=False)

    def forward(self, residual: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W_g(residual))
        return gate * attn_out


# ═══════════════════════════ SwiGLU FFN ═══════════════════════════
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.norm = nn.LayerNorm(d_model)
        # Fused W1 and W2 into a single output = 2 * hidden_dim
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


# ═══════════════════ Unified Interaction Block ═══════════════════
class UnifiedInteractionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 hidden_mult: int = 4, use_nlir: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_nlir = use_nlir
        assert d_model % num_heads == 0

        # Fused Q, K, V projection
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

        qkv = self.W_qkv(x_norm)  # (B, L, 3*D)
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


# ═══════════════ BFTS Mask Cache (static templates) ═══════════════
class BFTSMaskCache:
    def __init__(self, max_len: int, swa_window: int):
        # Pre-compute causal matrix once
        self.causal = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))  # (L, L)
        self.swa_window = swa_window
        self.max_len = max_len

        # Pre-compute sliding window mask (non‑causal part)
        idx = torch.arange(max_len).unsqueeze(1) - torch.arange(max_len).unsqueeze(0)
        self.win_mask = (idx >= 0) & (idx < swa_window)  # lower triangular & within window
        self.win_mask = self.win_mask.bool()

    def get_full_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return self.causal[:L, :L].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, L, L)

    def get_swa_mask(self, L: int, start: int, device: torch.device) -> torch.Tensor:
        """Return boolean mask that limits token i (in [start, start+L)) to only
        attend to j in [start, i] within window."""
        win = self.win_mask[:L, :L].to(device)
        mask = torch.zeros(1, 1, L, L, dtype=torch.bool, device=device)
        mask[:, :, :, :] = win  # will be placed later by indexing into the full mask
        return mask


def build_bfts_mask_static(
    B: int, total_len: int, ns_len: int,
    seq_boundaries: List[Tuple[int, int]],
    query_start: int,
    l_full: int, layer_idx: int,
    cache: BFTSMaskCache,
    device: torch.device,
) -> torch.Tensor:
    """Construct BFTS mask using pre‑cached static templates.

    Shallow layers (layer_idx < l_full): full causal attention.
    Deep layers:
    - Behaviour tokens: sliding window within their own sequence,
      plus they CAN attend to NS summary tokens and queries.
    - NS summary & query tokens: full causal attention.
    """
    # Start from full causal mask (template)
    mask = cache.get_full_mask(total_len, device)  # (1, 1, L, L)
    mask = mask.expand(B, -1, -1, -1)               # (B, 1, L, L)

    if layer_idx < l_full:
        return mask

    # Deep layers: restrict behaviour tokens to sliding window
    for start, end in seq_boundaries:
        if end <= start:
            continue
        L_seq = end - start
        # Get sliding window mask for this segment
        swa_mask = cache.get_swa_mask(L_seq, start, device)  # (1, 1, L_seq, L_seq)
        # Insert into the full mask at positions start:end, start:end
        mask[:, :, start:end, start:end] = swa_mask

        # Behaviour tokens can still attend to NS summary and queries
        # (already true because causal mask kept those columns, we only restricted self‑window)

    return mask


# ═══════════════ NS Tokenizers (unchanged) ═══════════════
class GroupNSTokenizer(nn.Module):
    # ... (keep your existing code, unchanged)
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
    # ... (unchanged)
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


# ═══════════════ NS Compressor (new) ═══════════════
class NSCompressor(nn.Module):
    """Compresses variable‑length NS tokens into a fixed number of summary tokens."""
    def __init__(self, d_model: int, num_summary_tokens: int = 4):
        super().__init__()
        self.summary_tokens = num_summary_tokens
        # Simple learnable pooling queries
        self.query = nn.Parameter(torch.randn(1, num_summary_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ns_tokens: torch.Tensor) -> torch.Tensor:
        """Input: ns_tokens (B, N, D). Output: (B, S, D) with S fixed."""
        B = ns_tokens.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, S, D)
        # Cross attention: summary queries attend to NS tokens
        out, _ = self.cross_attn(q, ns_tokens, ns_tokens)
        return self.norm(out)


# ═══════════════ MultiSeqQueryGenerator (unchanged) ═══════════════
class MultiSeqQueryGenerator(nn.Module):
    def __init__(self, d_model: int, num_ns: int, num_queries: int,
                 num_sequences: int, hidden_mult: int = 4,
                 use_time_diff: bool = False, time_emb_dim: int = 0):
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.use_time_diff = use_time_diff
        self.time_emb_dim = time_emb_dim

        global_info_dim = num_ns * d_model + d_model
        if use_time_diff and time_emb_dim > 0:
            global_info_dim += time_emb_dim

        self.global_info_norm = nn.LayerNorm(global_info_dim)
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                ) for _ in range(num_queries)
            ]) for _ in range(num_sequences)
        ])

    def forward(self, ns_tokens: torch.Tensor, seq_tokens_list: list,
                seq_padding_masks: list, seq_time_embs: Optional[List[torch.Tensor]] = None):
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)
        q_tokens_list = []
        for i in range(self.num_sequences):
            mask = ~seq_padding_masks[i]
            valid_mask = mask.unsqueeze(-1).float()
            seq_sum = (seq_tokens_list[i] * valid_mask).sum(dim=1)
            seq_count = valid_mask.sum(dim=1).clamp(min=1)
            seq_pooled = seq_sum / seq_count

            global_info = [ns_flat, seq_pooled]
            if self.use_time_diff and seq_time_embs is not None and i < len(seq_time_embs):
                global_info.append(seq_time_embs[i])

            global_info = torch.cat(global_info, dim=-1)
            global_info = self.global_info_norm(global_info)
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)
            q_tokens_list.append(q_tokens)
        return q_tokens_list


# ═══════════════════════ Main Model ═══════════════════════
class PCVRHyFormer(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_blocks: int = 4,
        num_heads: int = 4,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        num_time_buckets: int = 65,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        bfts_l_full: int = 2,
        bfts_swa_window: int = 50,
        use_nlir: bool = True,
        ns_token_dropout: float = 0.0,
        use_time_diff: bool = False,
        ns_summary_tokens: int = 4,  # new: number of compressed NS summary tokens
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
        self.bfts_swa_window = bfts_swa_window
        self.use_nlir = use_nlir
        self.num_blocks = num_blocks
        self.ns_token_dropout = ns_token_dropout
        self.use_time_diff = use_time_diff
        self.ns_summary_tokens = ns_summary_tokens
        self._global_ns = None

        # ---- NS tokenizers (unchanged) ----
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

        # ---- NS Compressor ----
        self.ns_compressor = NSCompressor(d_model, ns_summary_tokens)

        # ---- Sequence embedding tables ----
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

        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model, num_ns=self.num_ns, num_queries=num_queries,
            num_sequences=self.num_sequences, hidden_mult=hidden_mult,
            use_time_diff=use_time_diff, time_emb_dim=time_diff_dim,
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

        # Mask cache for BFTS (max length approximated)
        self.mask_cache = BFTSMaskCache(max_len=2048, swa_window=bfts_swa_window)

        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.emb_dropout = nn.Dropout(dropout_rate)
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # ---- Custom weight initialisation ----
        self._init_params()
        self._init_weights()

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
        """Small‑scale normal init for all Linear layers (excluding embeddings)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def set_global_ns(self, global_ns: torch.Tensor):
        # global_ns is the uncompressed NS mean; we compress it once and cache
        with torch.no_grad():
            summary = self.ns_compressor(global_ns.unsqueeze(0))  # (1, S, D)
        self._global_ns_summary = summary.squeeze(0)  # (S, D)

    # -- embedding helpers unchanged --
    def _embed_seq_domain(self, seq, sideinfo_embs, proj, is_id, emb_index, time_bucket_ids):
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
        cat_emb = torch.cat(emb_list, dim=-1)
        token_emb = F.gelu(proj(cat_emb))
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
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
        ns_tokens_raw = torch.cat(ns_parts, dim=1)  # (B, Nns, D)

        # NS token dropout (training)
        if self.training and self.ns_token_dropout > 0:
            mask = torch.rand(B, ns_tokens_raw.shape[1], 1, device=device) >= self.ns_token_dropout
            ns_tokens_raw = ns_tokens_raw * mask

        # 2. Compress NS to fixed number of summary tokens
        if not self.training and self._global_ns is not None:
            # Use pre‑compressed summary from global mean (if set)
            ns_summary = self._global_ns_summary.unsqueeze(0).expand(B, -1, -1)
        else:
            ns_summary = self.ns_compressor(ns_tokens_raw)  # (B, S, D)
        ns_len = ns_summary.shape[1]  # fixed

        # 3. Embed sequences
        seq_tokens_list = []
        seq_masks_list = []
        seq_time_buckets_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            mask = self._make_padding_mask(inputs.seq_lens[domain], tokens.shape[1])
            seq_tokens_list.append(tokens)
            seq_masks_list.append(mask)
            seq_time_buckets_list.append(inputs.seq_time_buckets[domain])

        # 4. Time embeddings for query generator (NaN‑safe)
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

        # 5. Query tokens (still uses raw NS for generation)
        q_tokens_list = self.query_generator(
            ns_tokens_raw, seq_tokens_list, seq_masks_list, seq_time_embs)

        # 6. Build unified stream with fixed length
        sep = self.sep_embedding.expand(B, -1, -1)
        parts = [ns_summary]            # start with summary tokens
        boundaries: List[Tuple[int, int]] = []
        first_seq_start = ns_len + 1    # after first sep
        current = ns_len
        for tokens in seq_tokens_list:
            parts.append(sep)
            current += 1  # sep position
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

        x = torch.cat(parts, dim=1)  # (B, total_len, D), length is fixed per batch
        total_len = x.shape[1]

        if apply_dropout:
            x = self.emb_dropout(x)

        # 7. Run blocks with static BFTS mask
        l_full = min(self.bfts_l_full, self.num_blocks)
        for l_idx in range(self.num_blocks):
            # Mask derived from static cache
            mask = build_bfts_mask_static(
                B=B, total_len=total_len, ns_len=ns_len,
                seq_boundaries=boundaries, query_start=query_start,
                l_full=l_full, layer_idx=l_idx,
                cache=self.mask_cache, device=device,
            )
            rope_cos, rope_sin = None, None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(total_len, device)
            x = self.blocks[l_idx](x, attn_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
            if torch.isnan(x).any():
                logging.warning(f"NaN after block {l_idx}, resetting")
                x = torch.nan_to_num(x, nan=0.0)

        # 8. Extract query outputs
        query_outputs = x[:, query_start:, :]
        flat_q = query_outputs.reshape(B, -1)
        out_repr = self.output_proj(flat_q)
        logits = self.clsfier(out_repr)
        return logits, out_repr

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        logits, _ = self._forward_impl(inputs, apply_dropout=self.training)
        return logits

    def predict(self, inputs: ModelInput):
        with torch.no_grad():
            logits, out_repr = self._forward_impl(inputs, apply_dropout=False)
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
                if real_idx == -1: continue
                if int(vs) > threshold:
                    emb = self._seq_embs[domain][real_idx]
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for i, (vs, offset, length) in enumerate(tokenizer.feature_specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1: continue
                if int(vs) > threshold:
                    emb = tokenizer.embs[real_idx]
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
        logging.info(f"Re-initialised {len(reinit_ptrs)} high‑card embeddings")
        return reinit_ptrs
