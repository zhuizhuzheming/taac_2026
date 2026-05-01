"""PCVR Parquet dataset module (performance-tuned) with delay-feedback awareness.

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Key safety feature: Column-name-based feature detection to prevent data leakage.
Instead of relying solely on fid mapping (which may be incorrect), we inspect
actual Parquet column names to determine feature types.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.

Delay-feedback enhancements:
- Proper handling of label_time as Unix timestamp for delay modeling.
- Fake-negative weighting based on time since click (computed offline).
- Dynamic negative sampling with time-aware sampling.
- Time-ordered train/valid split to simulate real prediction scenario.
- Three-class sample differentiation: not-clicked, clicked-not-converted, clicked-converted.
"""

import os
import logging
import random
import json
import gc
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.
    """

    def __init__(self) -> None:
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        return {'entries': self.entries, 'total_dim': self.total_dim}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)

# Use filesystem-based tensor sharing
torch.multiprocessing.set_sharing_strategy('file_system')

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


# ─────────────────────────── Column Name Patterns ────────────────────────────

# Regex patterns for detecting feature column types from Parquet column names
COLUMN_PATTERNS = {
    'user_int_scalar': re.compile(r'^user_int_feats_(\d+)$'),
    'user_int_array': re.compile(r'^user_int_array_feats_(\d+)$'),
    'item_int_scalar': re.compile(r'^item_int_feats_(\d+)$'),
    'item_int_array': re.compile(r'^item_int_array_feats_(\d+)$'),
    'user_dense': re.compile(r'^user_dense_feats_(\d+)$'),
    'item_dense': re.compile(r'^item_dense_feats_(\d+)$'),
    'seq_domain': re.compile(r'^domain_([a-zA-Z])_seq_(\d+)$'),
    'seq_legacy': re.compile(r'^seq_([a-zA-Z])_(\d+)$'),
}

# Metadata columns that must NEVER be used as features
METADATA_COLUMN_NAMES = {
    'user_id', 'item_id', 'label_type', 'label_time', 'timestamp',
    'user_id_hash', 'item_id_hash',  # possible variants
}


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset with automatic column-type detection from Parquet schema.

    Instead of relying solely on schema.json fid mapping, we inspect actual
    Parquet column names to build feature plans. This prevents data leakage
    when schema.json has incorrect fid-to-column mappings.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        bucket_by_domain: Optional[str] = None,
        delay_decay_window: float = 604800.0,
        dynamic_negative_sampling: bool = False,
        neg_sample_ratio: float = 3.0,
        global_current_time: Optional[int] = None,
    ) -> None:
        super().__init__()

        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.bucket_by_domain = bucket_by_domain
        self.delay_decay_window = delay_decay_window
        self.dynamic_negative_sampling = dynamic_negative_sampling
        self.neg_sample_ratio = neg_sample_ratio
        self.global_current_time = global_current_time
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build Row Groups list
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load schema.json (for vocab sizes, etc.)
        self._load_schema(schema_path, seq_max_lens or {})

        # Inspect actual Parquet column names
        pf = pq.ParquetFile(self._parquet_files[0])
        self._parquet_columns = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(self._parquet_columns)}

        # CRITICAL: Auto-detect feature columns from Parquet names
        self._detect_columns_from_parquet()

        # Validate no metadata columns leaked into features
        self._validate_no_leakage()

        # Build feature plans BEFORE allocating buffers
        self._build_feature_plans()

        # Pre-allocate buffers (now schema.total_dim is correctly populated)
        B = batch_size
        self._buf_user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        self._buf_item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        self._buf_user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        self._buf_seq_ts = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)
            self._buf_seq_ts[domain] = np.zeros((B, max_len), dtype=np.int64)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows, "
            f"batch_size={batch_size}, shuffle={shuffle}, "
            f"user_int_features={len(self._user_int_cols)}, "
            f"item_int_features={len(self._item_int_cols)}, "
            f"user_dense_features={len(self._user_dense_cols)}, "
            f"item_dense_features={len(self._item_dense_cols)}, "
            f"seq_domains={self.seq_domains}")

    def _detect_columns_from_parquet(self) -> None:
        """Auto-detect feature columns from actual Parquet column names.
        
        FIX: Map short domain names (e.g., 'a') to schema.json domain names (e.g., 'seq_a').
        """
        self._detected_user_int = []
        self._detected_item_int = []
        self._detected_user_dense = []
        self._detected_item_dense = []
        self._detected_seq = {}

        # Build mapping from short domain to schema domain
        schema_short_to_full = {}
        for cfg_domain in (getattr(self, '_raw_seq_cfg', None) or {}).keys():
            short = cfg_domain.replace('seq_', '')
            schema_short_to_full[short] = cfg_domain
            # Also handle the case where schema already uses short name
            schema_short_to_full[cfg_domain] = cfg_domain

        # Track which columns we've assigned
        assigned_cols = set()

        # 1. Detect user_int features
        for name in self._parquet_columns:
            if name in assigned_cols:
                continue
            m = COLUMN_PATTERNS['user_int_scalar'].match(name)
            if m:
                fid = int(m.group(1))
                if name in METADATA_COLUMN_NAMES:
                    logging.warning(f"SAFETY: Column '{name}' matches user_int pattern but is in metadata list. Skipping.")
                    continue
                vocab_size = self._lookup_vocab_size('user_int', fid, default=0)
                self._detected_user_int.append((fid, name, vocab_size, 1))
                assigned_cols.add(name)
                continue

            m = COLUMN_PATTERNS['user_int_array'].match(name)
            if m:
                fid = int(m.group(1))
                vocab_size = self._lookup_vocab_size('user_int', fid, default=0)
                dim = self._lookup_dim('user_int', fid, default=1)
                self._detected_user_int.append((fid, name, vocab_size, dim))
                assigned_cols.add(name)

        # 2. Detect item_int features
        for name in self._parquet_columns:
            if name in assigned_cols:
                continue
            m = COLUMN_PATTERNS['item_int_scalar'].match(name)
            if m:
                fid = int(m.group(1))
                vocab_size = self._lookup_vocab_size('item_int', fid, default=0)
                self._detected_item_int.append((fid, name, vocab_size, 1))
                assigned_cols.add(name)
                continue

            m = COLUMN_PATTERNS['item_int_array'].match(name)
            if m:
                fid = int(m.group(1))
                vocab_size = self._lookup_vocab_size('item_int', fid, default=0)
                dim = self._lookup_dim('item_int', fid, default=1)
                self._detected_item_int.append((fid, name, vocab_size, dim))
                assigned_cols.add(name)

        # 3. Detect user_dense features
        for name in self._parquet_columns:
            if name in assigned_cols:
                continue
            m = COLUMN_PATTERNS['user_dense'].match(name)
            if m:
                fid = int(m.group(1))
                dim = self._lookup_dim('user_dense', fid, default=1)
                self._detected_user_dense.append((fid, name, dim))
                assigned_cols.add(name)

        # 4. Detect item_dense features
        for name in self._parquet_columns:
            if name in assigned_cols:
                continue
            m = COLUMN_PATTERNS['item_dense'].match(name)
            if m:
                fid = int(m.group(1))
                dim = self._lookup_dim('item_dense', fid, default=1)
                self._detected_item_dense.append((fid, name, dim))
                assigned_cols.add(name)

        # 5. Detect sequence features
        for name in self._parquet_columns:
            if name in assigned_cols:
                continue
            
            # Try official format: domain_a_seq_38
            m = COLUMN_PATTERNS['seq_domain'].match(name)
            if not m:
                # Try legacy format: seq_a_38
                m = COLUMN_PATTERNS['seq_legacy'].match(name)
            
            if m:
                raw_domain = m.group(1)  # e.g., 'a'
                fid = int(m.group(2))
                
                # Map to schema.json domain name
                domain = schema_short_to_full.get(raw_domain, raw_domain)
                
                if domain not in self._detected_seq:
                    self._detected_seq[domain] = {}
                
                vocab_size = self._lookup_vocab_size('seq', domain, fid, default=0)
                self._detected_seq[domain][fid] = (name, vocab_size)
                assigned_cols.add(name)

        # 6. Check for unassigned columns that look suspicious
        unassigned = [n for n in self._parquet_columns if n not in assigned_cols]
        metadata_found = [n for n in unassigned if n in METADATA_COLUMN_NAMES]
        suspicious = [n for n in unassigned 
                     if any(p in n.lower() for p in ['label', 'convert', 'target'])]
        
        if suspicious:
            logging.warning(f"SAFETY: Suspicious unassigned columns: {suspicious}")
        if metadata_found:
            logging.info(f"Metadata columns correctly isolated: {metadata_found}")

    def _lookup_vocab_size(self, group: str, *keys, default: int = 0) -> int:
        """Look up vocab size from schema.json with fallback."""
        try:
            if group == 'user_int':
                fid = keys[0]
                for f, vs, dim in self._raw_user_int:
                    if f == fid:
                        return vs
            elif group == 'item_int':
                fid = keys[0]
                for f, vs, dim in self._raw_item_int:
                    if f == fid:
                        return vs
            elif group == 'seq':
                domain, fid = keys
                cfg = self._raw_seq_cfg.get(domain, {})
                for f, vs in cfg.get('features', []):
                    if f == fid:
                        return vs
        except (AttributeError, IndexError):
            pass
        return default

    def _lookup_dim(self, group: str, fid: int, default: int = 1) -> int:
        """Look up feature dimension from schema.json."""
        try:
            if group == 'user_int':
                for f, vs, dim in self._raw_user_int:
                    if f == fid:
                        return dim
            elif group == 'item_int':
                for f, vs, dim in self._raw_item_int:
                    if f == fid:
                        return dim
            elif group == 'user_dense':
                for f, dim in self._raw_user_dense:
                    if f == fid:
                        return dim
            elif group == 'item_dense':
                for f, dim in self._raw_item_dense:
                    if f == fid:
                        return dim
        except AttributeError:
            pass
        return default

    def _validate_no_leakage(self) -> None:
        """Ensure no metadata columns are in feature lists."""
        all_feature_cols = (
            [n for _, n, _, _ in self._detected_user_int] +
            [n for _, n, _, _ in self._detected_item_int] +
            [n for _, n, _ in self._detected_user_dense] +
            [n for _, n, _ in self._detected_item_dense]
        )
        for domain, feats in self._detected_seq.items():
            all_feature_cols.extend([n for n, _ in feats.values()])
        
        leaked = [n for n in all_feature_cols if n in METADATA_COLUMN_NAMES]
        if leaked:
            raise RuntimeError(
                f"CRITICAL: Metadata columns leaked into features: {leaked}. "
                f"This would cause data leakage and invalid AUC."
            )
        
        # Also check: no feature column has 'label' in its name
        suspicious = [n for n in all_feature_cols if 'label' in n.lower()]
        if suspicious:
            logging.warning(f"SAFETY: Feature columns with 'label' in name: {suspicious}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Load schema.json. Store raw data for lookup; actual column detection
        happens in _detect_columns_from_parquet()."""
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Store raw for lookups
        self._raw_user_int = raw.get('user_int', [])
        self._raw_item_int = raw.get('item_int', [])
        self._raw_user_dense = raw.get('user_dense', [])
        self._raw_item_dense = raw.get('item_dense', [])
        self._raw_seq_cfg = raw.get('seq', {})

        # Build schemas from DETECTED columns (will be populated after detection)
        self.user_int_schema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        self._user_int_cols: List[Tuple[int, str, int, int]] = []

        self.item_int_schema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        self._item_int_cols: List[Tuple[int, str, int, int]] = []

        self.user_dense_schema = FeatureSchema()
        self._user_dense_cols: List[Tuple[int, str, int]] = []

        self.item_dense_schema = FeatureSchema()
        self._item_dense_cols: List[Tuple[int, str, int]] = []

        self.seq_domains: List[str] = []
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in sorted(self._raw_seq_cfg.keys()):
            cfg = self._raw_seq_cfg[domain]
            self._seq_prefix[domain] = cfg.get('prefix', f'domain_{domain}_seq')
            self.ts_fids[domain] = cfg.get('ts_fid')
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def _build_feature_plans(self) -> None:
        """Build execution plans from detected columns."""
        
        # user_int plan: (col_idx, dim, offset, vocab_size)
        offset = 0
        self._user_int_plan = []
        for fid, col_name, vocab, dim in sorted(self._detected_user_int, key=lambda x: x[0]):
            ci = self._col_idx[col_name]
            self._user_int_plan.append((ci, dim, offset, vocab))
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vocab] * dim)
            self._user_int_cols.append((fid, col_name, vocab, dim))
            offset += dim

        # item_int plan
        offset = 0
        self._item_int_plan = []
        for fid, col_name, vocab, dim in sorted(self._detected_item_int, key=lambda x: x[0]):
            ci = self._col_idx[col_name]
            self._item_int_plan.append((ci, dim, offset, vocab))
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vocab] * dim)
            self._item_int_cols.append((fid, col_name, vocab, dim))
            offset += dim

        # user_dense plan
        offset = 0
        self._user_dense_plan = []
        for fid, col_name, dim in sorted(self._detected_user_dense, key=lambda x: x[0]):
            ci = self._col_idx[col_name]
            self._user_dense_plan.append((ci, dim, offset))
            self.user_dense_schema.add(fid, dim)
            self._user_dense_cols.append((fid, col_name, dim))
            offset += dim

        # item_dense plan
        offset = 0
        self._item_dense_plan = []
        for fid, col_name, dim in sorted(self._detected_item_dense, key=lambda x: x[0]):
            ci = self._col_idx[col_name]
            self._item_dense_plan.append((ci, dim, offset))
            self.item_dense_schema.add(fid, dim)
            self._item_dense_cols.append((fid, col_name, dim))
            offset += dim

        # Sequence plans
        self._seq_plan = {}
        for domain in sorted(self._detected_seq.keys()):
            self.seq_domains.append(domain)
            feats = self._detected_seq[domain]
            
            all_fids = sorted(feats.keys())
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: feats[fid][1] for fid in all_fids}
            
            ts_fid = self.ts_fids.get(domain)
            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]
            
            # Build side plan
            side_plan = []
            for slot, fid in enumerate(sideinfo):
                col_name, vocab = feats[fid]
                ci = self._col_idx[col_name]
                side_plan.append((ci, slot, vocab))
            
            # ts plan
            ts_ci = None
            if ts_fid is not None and ts_fid in feats:
                ts_col_name, _ = feats[ts_fid]
                ts_ci = self._col_idx.get(ts_col_name)
            
            self._seq_plan[domain] = (side_plan, ts_ci)

    def __len__(self) -> int:
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Concatenate buffered batches, shuffle, re-slice.
        
        FIX: Correctly handle both tensor and non-tensor data.
        FIX: Dynamic negative sampling must filter ALL data types consistently.
        FIX: _seq_domains is not indexable, treat as constant.
        """
        # Step 1: Merge all batches
        merged: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if k == '_seq_domains':
                # All batches have the same _seq_domains, keep first
                merged[k] = buffer[0][k]
            elif isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            elif isinstance(buffer[0][k], list):
                merged[k] = []
                for b in buffer:
                    merged[k].extend(b[k])
            elif isinstance(buffer[0][k], (np.ndarray,)):
                merged[k] = np.concatenate([b[k] for b in buffer], axis=0)
            else:
                merged[k] = buffer[0][k]
        
        total_rows = merged['label'].shape[0]

        # ===== Dynamic negative sampling with three-class awareness =====
        if self.dynamic_negative_sampling and self.is_training:
            labels = merged['label']           # 0/1 (binary)
            label_type = merged['label_type']   # 0/1/2 (three-class)
            
            # Three-class indices
            pos_mask = (labels == 1).squeeze()                      # clicked-converted
            click_not_conv_mask = (label_type == 1).squeeze()       # clicked-not-converted
            not_click_mask = (label_type == 0).squeeze()            # not-clicked
            
            pos_indices = torch.where(pos_mask)[0]
            click_not_conv_indices = torch.where(click_not_conv_mask)[0]
            not_click_indices = torch.where(not_click_mask)[0]
            
            n_pos = pos_indices.shape[0]
            
            # Strategy: Keep all positives + all clicked-not-converted + sampled not-clicked
            n_click_not_conv = click_not_conv_indices.shape[0]
            
            # not-clicked: sample according to neg_sample_ratio relative to positives
            n_not_click_target = min(int(n_pos * self.neg_sample_ratio), not_click_indices.shape[0])
            
            if n_not_click_target < not_click_indices.shape[0]:
                not_click_timestamps = merged['timestamp'][not_click_indices]
                _, selected_not_click = torch.topk(not_click_timestamps, n_not_click_target)
                selected_not_click_indices = not_click_indices[selected_not_click]
            else:
                selected_not_click_indices = not_click_indices
            
            # Combine all selected indices
            all_indices = torch.cat([
                pos_indices, 
                click_not_conv_indices,
                selected_not_click_indices
            ])
            
            # Shuffle
            rand_perm = torch.randperm(all_indices.shape[0])
            all_indices = all_indices[rand_perm]
            
            # Slice into batches
            for i in range(0, all_indices.shape[0], self.batch_size):
                end = min(i + self.batch_size, all_indices.shape[0])
                batch_idx = all_indices[i:end]
                yield self._slice_merged_data(merged, batch_idx)
            
            del merged
            buffer.clear()
            return

        # Standard shuffle or bucketing
        if self.bucket_by_domain is not None and self.bucket_by_domain in merged:
            lengths = merged[f'{self.bucket_by_domain}_len'].numpy()
            sorted_idx = torch.from_numpy(np.argsort(lengths))
            window = self.batch_size * 2
            indices = []
            for start in range(0, total_rows, window):
                end = min(start + window, total_rows)
                window_idx = sorted_idx[start:end]
                perm = torch.randperm(len(window_idx))
                indices.append(window_idx[perm])
            rand_idx = torch.cat(indices)
        elif self.shuffle:
            rand_idx = torch.randperm(total_rows)
        else:
            rand_idx = torch.arange(total_rows)

        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch_idx = rand_idx[i:end]
            yield self._slice_merged_data(merged, batch_idx)
        
        del merged
        buffer.clear()

    def _slice_merged_data(self, merged: Dict[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
        """Slice merged data by indices, handling both tensor and non-tensor types correctly.
        
        FIX: _seq_domains is not indexable, must be passed through as-is.
        FIX: All list data must be indexed with bounds checking.
        """
        batch: Dict[str, Any] = {}
        idx_list = indices.tolist()
        
        for k, v in merged.items():
            if k == '_seq_domains':
                # _seq_domains is the same for all samples, just copy
                batch[k] = v
            elif isinstance(v, torch.Tensor):
                batch[k] = v[indices]
            elif isinstance(v, list):
                # FIX: Bounds checking to prevent IndexError
                valid_indices = [j for j in idx_list if 0 <= j < len(v)]
                if len(valid_indices) != len(idx_list):
                    logging.warning_once(
                        f"Index out of bounds in _slice_merged_data for key '{k}': "
                        f"requested {len(idx_list)} items but list has {len(v)}. "
                        f"This may indicate inconsistent data lengths after sampling."
                    )
                batch[k] = [v[j] for j in valid_indices]
            elif isinstance(v, np.ndarray):
                batch[k] = v[idx_list]
            else:
                # Non-indexable data (scalars, etc.)
                batch[k] = v
        
        return batch

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max'] = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.fill_null(0).to_numpy()
    
        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)
    
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len
    
        padded[padded <= 0] = 0
        return padded, lengths
    
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.fill_null(0.0).to_numpy()
    
        padded = np.zeros((B, max_dim), dtype=np.float32)
    
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]
    
        return padded

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert Arrow RecordBatch to training dict with three-class label awareness."""
        B = batch.num_rows
    
        # Metadata columns
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
    
        # FIX: Preserve original label_type (0=not-clicked, 1=clicked-not-converted, 2=clicked-converted)
        label_type = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64))
        
        # Binary label for BCE loss
        labels = (label_type == 2).astype(np.int64)
        is_clicked_not_converted = (label_type == 1).astype(np.int64)
    
        if 'label_time' in self._col_idx:
            label_times = batch.column(self._col_idx['label_time']).to_numpy().astype(np.int64)
        else:
            label_times = np.zeros(B, dtype=np.int64)
    
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()
    
        # ===== Delay feedback weighting with three-class awareness =====
        current_time = (self.global_current_time 
                        if self.global_current_time is not None 
                        else timestamps.max())
        if self.global_current_time is None:
            logging.warning_once(
                "global_current_time not provided; falling back to batch-local max. "
                "This may cause fake_negative_weight to be near-zero for all negatives."
            )
        
        if self.is_training:
            # Time since event for all samples
            time_since_event = np.maximum(current_time - timestamps, 0).astype(np.float32)
            
            # Three-class differentiated weights:
            # - clicked-converted (type=2): weight = 1.0 (positive)
            # - clicked-not-converted (type=1): delay-aware down-weighting
            #   Recent clicks are more likely to convert later -> lower weight
            #   Distant clicks are likely true negatives -> higher weight
            # - not-clicked (type=0): weight = 1.0 (true negative)
            
            fake_negative_weight = np.ones(B, dtype=np.float32)
            
            # clicked-converted: positive samples
            fake_negative_weight = np.where(label_type == 2, 1.0, fake_negative_weight)
            
            # clicked-not-converted: delay-aware weighting
            click_not_conv_mask = (label_type == 1)
            # Sigmoid smoothing: recent -> ~0.1, distant -> ~0.9
            # Window is centered at delay_decay_window
            sigmoid_input = (time_since_event - self.delay_decay_window) / (self.delay_decay_window / 4)
            click_weight = 0.1 + 0.8 / (1.0 + np.exp(-sigmoid_input))
            fake_negative_weight = np.where(click_not_conv_mask, click_weight, fake_negative_weight)
            
            # not-clicked: remains 1.0 (true negative)
            
            # observed_delay: only meaningful for converted positives
            observed_delay = np.where(label_type == 2, np.maximum(label_times - timestamps, 0), 0)
            
        else:
            # Validation: no delay weighting
            observed_delay = np.where(label_type == 2, np.maximum(label_times - timestamps, 0), 0)
            fake_negative_weight = np.ones(B, dtype=np.float32)
    
        # user_int features
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if pa.types.is_list(col.type):
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset:offset + dim] = padded
            else:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
    
        # item_int features
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if pa.types.is_list(col.type):
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset:offset + dim] = padded
            else:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
    
        # user_dense features
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset + dim] = padded

        # item_dense features
        item_dense = np.zeros((B, self.item_dense_schema.total_dim), dtype=np.float32)
        for ci, dim, offset in self._item_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            item_dense[:, offset:offset + dim] = padded
    
        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.from_numpy(item_dense.copy()),
            'label': torch.from_numpy(labels),
            'label_type': torch.from_numpy(label_type),  # FIX: preserve three-class label
            'is_clicked_not_converted': torch.from_numpy(is_clicked_not_converted),
            'label_time': torch.from_numpy(label_times),
            'observed_delay': torch.from_numpy(observed_delay.astype(np.float32)),
            'fake_negative_weight': torch.from_numpy(fake_negative_weight.astype(np.float32)),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }
    
        # Sequence features
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]
    
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0
            seq_ts = self._buf_seq_ts[domain][:B]
            seq_ts[:] = 0
    
            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.fill_null(0).to_numpy(), vs, ci))
    
            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul
    
            out[out <= 0] = 0
    
            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0
    
            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())
    
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.fill_null(0).to_numpy()
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]
                    seq_ts[i, :ul] = ts_vals[s:s + ul]
    
                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets
    
            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_timestamps'] = torch.from_numpy(seq_ts.copy())
    
        return result


def _get_global_max_timestamp(data_dir: str) -> Optional[int]:
    """Scan parquet column statistics to obtain a global max timestamp.
    
    This is used as the reference time for delay-feedback weighting.
    Using a batch-local max causes all weights to collapse to ~0.
    """
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    if not files:
        return None
    global_max = 0
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            if 'timestamp' not in names:
                continue
            col_idx = names.index('timestamp')
            for rg_idx in range(pf.metadata.num_row_groups):
                stats = pf.metadata.row_group(rg_idx).column(col_idx).statistics
                if stats and stats.max is not None:
                    global_max = max(global_max, int(stats.max))
        except Exception:
            continue
    return global_max if global_max > 0 else None


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    bucket_by_domain: Optional[str] = None,
    split_by_time: bool = True,
    delay_decay_window: float = 604800.0,
    dynamic_negative_sampling: bool = False,
    neg_sample_ratio: float = 3.0,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    if split_by_time:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs
        
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
            logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")
        
        train_rg_list = rg_info[:n_train_rgs]
        valid_rg_list = rg_info[n_train_rgs:]
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = total_rgs - n_valid_rgs
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
            logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")
        train_rg_list = rg_info[:n_train_rgs]
        valid_rg_list = rg_info[n_train_rgs:]

    train_rows = sum(r[2] for r in train_rg_list)
    valid_rows = sum(r[2] for r in valid_rg_list)

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows), "
                 f"split_by_time={split_by_time}")

    # FIX: Obtain global reference time for correct delay weighting
    global_current_time = _get_global_max_timestamp(data_dir)
    if global_current_time is not None:
        logging.info(f"Global max timestamp for delay weighting: {global_current_time}")
    else:
        logging.warning("Could not determine global max timestamp from Parquet stats. "
                        "Delay weights may be inaccurate.")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        is_training=True,
        bucket_by_domain=bucket_by_domain,
        delay_decay_window=delay_decay_window,
        dynamic_negative_sampling=dynamic_negative_sampling,
        neg_sample_ratio=neg_sample_ratio,
        global_current_time=global_current_time,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=use_cuda, **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        is_training=False,
        global_current_time=global_current_time,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
                 f"bucket_by_domain={bucket_by_domain}, "
                 f"dynamic_neg_sampling={dynamic_negative_sampling}")

    return train_loader, valid_loader, train_dataset