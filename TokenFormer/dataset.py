# ==================== dataset.py (TokenFormer DEFUSE 完整版) ====================

"""PCVR Parquet dataset with DEFUSE continuous labels and user-level sampling."""

import os
import logging
import random
import json
import gc
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ═════════════ Numba threading safety ═════════════
import numba
from numba import njit, prange

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

# ────────────── Numba JIT Kernels ──────────────
@njit(cache=True, fastmath=True)
def _fill_sequences_numba(
    offsets_list, values_list, out, lengths, max_len, B, n_feats
):
    for c in prange(n_feats):
        offsets = offsets_list[c]
        values = values_list[c]
        for i in range(B):
            s = offsets[i]
            e = offsets[i + 1]
            rl = e - s
            if rl <= 0:
                continue
            ul = rl if rl < max_len else max_len
            for j in range(ul):
                out[i, c, j] = values[s + j]
            if ul > lengths[i]:
                lengths[i] = ul
    return out, lengths

@njit(cache=True, fastmath=True)
def _fill_timestamps_numba(offsets, values, out, max_len, B):
    for i in prange(B):
        s = offsets[i]
        e = offsets[i + 1]
        rl = e - s
        if rl > 0:
            ul = rl if rl < max_len else max_len
            for j in range(ul):
                out[i, j] = values[s + j]
    return out

@njit(cache=True)
def _compute_time_decay_weights(time_diff, out, max_len, B, decay_factor):
    for i in prange(B):
        for j in range(max_len):
            if time_diff[i, j] <= 0:
                out[i, j] = 0.0
            else:
                out[i, j] = np.exp(-time_diff[i, j] / decay_factor)
    return out

# ────────────── Constants ──────────────
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

# ────────────── Feature Schema ──────────────
class FeatureSchema:
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

# ────────────── 主数据集类 ──────────────
class PCVRParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 10,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        time_decay_factor: float = 604800.0,
        use_time_decay: bool = True,
        neg_downsample_ratio: Optional[float] = None,
        min_neg_per_batch: int = 0,
        pos_upsample_ratio: Optional[float] = None,
        min_batch_size_ratio: float = 0.0,
        label_mode: str = 'defuse',
        soft_label_decay_hours: float = 24.0,
        global_click_conv_prob: float = 0.3,
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
        self.time_decay_factor = time_decay_factor
        self.use_time_decay = use_time_decay
        self.neg_downsample_ratio = neg_downsample_ratio
        self.min_neg_per_batch = min_neg_per_batch
        self.pos_upsample_ratio = pos_upsample_ratio
        self.min_batch_size = int(batch_size * min_batch_size_ratio)
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        self.label_mode = label_mode
        self.soft_label_decay_hours = soft_label_decay_hours
        self.global_click_conv_prob = global_click_conv_prob

        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)
        self._load_schema(schema_path, seq_max_lens or {})

        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        self._has_label_type = 'label_type' in self._col_idx
        self._has_label_time = 'label_time' in self._col_idx
        if not self._has_label_type:
            logging.warning("label_type column not found! Labels will be all zeros.")
        logging.info(f"Label source: label_type={self._has_label_type}, label_time={self._has_label_time}")

        self._sample_stats = {
            'total': 0, 'pos': 0, 'neg_kept': 0, 'neg_dropped': 0,
            'dropped_too_small': 0, 'users_kept': 0, 'users_dropped': 0,
        }

        self._user_int_plan = []
        offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim

        self._user_dense_plan = []
        offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, dim, offset))
            offset += dim

        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows, batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, time_decay={use_time_decay}, "
            f"neg_downsample={neg_downsample_ratio}, pos_upsample={pos_upsample_ratio}, "
            f"min_batch_size={self.min_batch_size}, label_mode={label_mode}, "
            f"numba_threading={numba.config.THREADING_LAYER}")

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self._user_int_cols = raw['user_int']
        self.user_int_schema = FeatureSchema()
        self.user_int_vocab_sizes = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        self._item_int_cols = raw['item_int']
        self.item_int_schema = FeatureSchema()
        self.item_int_vocab_sizes = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        self._user_dense_cols = raw['user_dense']
        self.user_dense_schema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        self.item_dense_schema = FeatureSchema()

        self._seq_cfg = raw['seq']
        self.seq_domains = sorted(self._seq_cfg.keys())
        self.seq_feature_ids = {}
        self.seq_vocab_sizes = {}
        self.seq_domain_vocab_sizes = {}
        self.ts_fids = {}
        self.sideinfo_fids = {}
        self._seq_prefix = {}
        self._seq_maxlen = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

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
            read_batch_size = max(self.batch_size * 8, 2048)
            for batch in pf.iter_batches(batch_size=read_batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch_with_sampling(batch)
                if batch_dict is None:
                    continue
                    
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

    def _convert_batch_with_sampling(self, batch: "pa.RecordBatch") -> Optional[Dict[str, Any]]:
        """按用户级别采样，保持序列完整性"""
        try:
          B_raw = batch.num_rows
          
          if not self._has_label_type:
              result = self._convert_batch(batch)
              if result['label'].shape[0] < self.min_batch_size:
                  self._sample_stats['dropped_too_small'] += 1
                  return None
              return result
          
          label_type = batch.column(self._col_idx['label_type']).to_numpy(zero_copy_only=False).astype(np.int64)
          user_ids = np.array(batch.column(self._col_idx['user_id']).to_pylist())
          
          unique_users = np.unique(user_ids)
          kept_users = []
          
          for user in unique_users:
              user_mask = (user_ids == user)
              user_labels = label_type[user_mask]
              n_pos = int((user_labels == 2).sum())
              n_click = int((user_labels == 1).sum())
              
              if n_pos > 0:
                  kept_users.append(user)
              elif n_click > 0:
                  if self.neg_downsample_ratio is None or random.random() < self.neg_downsample_ratio:
                      kept_users.append(user)
              else:
                  if self.neg_downsample_ratio is not None and random.random() < self.neg_downsample_ratio * 0.5:
                      kept_users.append(user)
          
          self._sample_stats['users_kept'] += len(kept_users)
          self._sample_stats['users_dropped'] += (len(unique_users) - len(kept_users))
          
          if len(kept_users) == 0:
              return None
          
          keep_mask = np.isin(user_ids, kept_users)
          batch = batch.filter(pa.array(keep_mask.tolist()))
          
          result = self._convert_batch(batch)
          if result is not None and result['label'].shape[0] < self.min_batch_size:
              self._sample_stats['dropped_too_small'] += 1
              return None
              
          return result
        except Exception as e:
          logging.error(f"Error in _convert_batch_with_sampling: {e}, skipping batch")
          return None

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]   # 如 _seq_domains 等不变
    
        total_rows = merged['label'].shape[0]
        rand_idx = torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
    
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            if end - i < self.min_batch_size:
                self._sample_stats['dropped_too_small'] += 1
                continue
    
            batch = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    def _record_oob(self, group: str, col_idx: int, arr: np.ndarray, vocab_size: int) -> None:
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
            raise ValueError(f"{group} col_idx={col_idx}: {n} values OOB")

    def _pad_varlen_float_column(self, arrow_col, max_dim: int, B: int) -> np.ndarray:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
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
        """DEFUSE 连续标签版本"""
        B = batch.num_rows
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)

        label_type = batch.column(self._col_idx['label_type']).to_numpy(zero_copy_only=False).astype(np.int64)
        
        labels, sample_weights, action_types, soft_labels = self._build_defuse_labels(
            label_type, timestamps, batch
        )
        
        user_ids = batch.column(self._col_idx['user_id']).to_pylist()

        user_int = np.zeros((B, self.user_int_schema.total_dim), dtype=np.int64)
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                offsets_arr = col.offsets.to_numpy()
                values_arr = col.values.to_numpy()
                for i in range(B):
                    s, e = int(offsets_arr[i]), int(offsets_arr[i+1])
                    rl = e - s
                    if rl > 0:
                        ul = min(rl, dim)
                        user_int[i, offset:offset+ul] = values_arr[s:s+ul]
                if vs > 0:
                    self._record_oob('user_int', ci, user_int[:, offset:offset+dim], vs)
                else:
                    user_int[:, offset:offset+dim] = 0

        item_int = np.zeros((B, self.item_int_schema.total_dim), dtype=np.int64)
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                offsets_arr = col.offsets.to_numpy()
                values_arr = col.values.to_numpy()
                for i in range(B):
                    s, e = int(offsets_arr[i]), int(offsets_arr[i+1])
                    rl = e - s
                    if rl > 0:
                        ul = min(rl, dim)
                        item_int[i, offset:offset+ul] = values_arr[s:s+ul]
                if vs > 0:
                    self._record_oob('item_int', ci, item_int[:, offset:offset+dim], vs)
                else:
                    item_int[:, offset:offset+dim] = 0

        user_dense = np.zeros((B, self.user_dense_schema.total_dim), dtype=np.float32)
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset:offset+dim] = padded

        result = {
            'user_int_feats': torch.from_numpy(user_int),
            'user_dense_feats': torch.from_numpy(user_dense),
            'item_int_feats': torch.from_numpy(item_int),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'sample_weights': torch.from_numpy(sample_weights),
            'action_type': torch.from_numpy(action_types),
            'soft_label': torch.from_numpy(soft_labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': torch.tensor(user_ids, dtype=torch.long),
            '_seq_domains': self.seq_domains,
        }

        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]
            n_feats = len(side_plan)

            out = np.zeros((B, n_feats, max_len), dtype=np.int64)
            lengths = np.zeros(B, dtype=np.int64)
            time_diff = np.zeros((B, max_len), dtype=np.int64)
            time_decay = np.zeros((B, max_len), dtype=np.float32)
            time_bucket = np.zeros((B, max_len), dtype=np.int64)

            offsets_list = []
            values_list = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                offsets_list.append(col.offsets.to_numpy())
                values_list.append(col.values.to_numpy())

            _fill_sequences_numba(offsets_list, values_list, out, lengths, max_len, B, n_feats)

            for c, (ci, slot, vs) in enumerate(side_plan):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            out[out <= 0] = 0
            result[domain] = torch.from_numpy(out)
            result[f'{domain}_len'] = torch.from_numpy(lengths)

            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                _fill_timestamps_numba(ts_offs, ts_vals, ts_padded, max_len, B)

                ts_expanded = timestamps.reshape(-1, 1)
                diff = ts_expanded - ts_padded
                diff = np.maximum(diff, 0)
                time_diff[:] = diff

                flat_diff = diff.ravel()
                raw_buckets = np.searchsorted(BUCKET_BOUNDARIES, flat_diff)
                raw_buckets = np.clip(raw_buckets, 0, len(BUCKET_BOUNDARIES) - 1)
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                if self.use_time_decay:
                    _compute_time_decay_weights(time_diff, time_decay, max_len, B, self.time_decay_factor)

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket)
            result[f'{domain}_time_diff'] = torch.from_numpy(time_diff)
            result[f'{domain}_time_decay'] = torch.from_numpy(time_decay)

        return result

    def _build_defuse_labels(
        self, 
        label_type: np.ndarray, 
        timestamps: np.ndarray,
        batch: "pa.RecordBatch"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """DEFUSE 连续标签构建"""
        B = len(label_type)
        labels = np.zeros(B, dtype=np.float32)
        weights = np.ones(B, dtype=np.float32)
        action_types = label_type.copy()
        soft_labels = np.zeros(B, dtype=np.float32)
        
        conv_mask = (label_type == 2)
        labels[conv_mask] = 1.0
        weights[conv_mask] = 1.0
        soft_labels[conv_mask] = 1.0
        
        neg_mask = (label_type == 0)
        labels[neg_mask] = 0.0
        weights[neg_mask] = 1.0
        soft_labels[neg_mask] = 0.0
        
        click_mask = (label_type == 1)
        if click_mask.any() and self._has_label_time:
            label_time = batch.column(self._col_idx['label_time']).to_numpy(zero_copy_only=False).astype(np.int64)
            
            future_conv = click_mask & (label_time > timestamps)
            if future_conv.any():
                time_diff = np.maximum(label_time[future_conv] - timestamps[future_conv], 0).astype(np.float32)
                decay = self.soft_label_decay_hours * 3600.0
                
                soft_vals = np.exp(-time_diff / decay) * 0.8 + 0.1
                soft_vals = np.clip(soft_vals, 0.05, 0.95)
                
                labels[future_conv] = soft_vals
                soft_labels[future_conv] = soft_vals
                weights[future_conv] = np.exp(-time_diff / decay)
                weights[future_conv] = np.clip(weights[future_conv], 0.1, 1.0)
            
            uncertain = click_mask & (label_time <= timestamps)
            if uncertain.any():
                labels[uncertain] = self.global_click_conv_prob
                soft_labels[uncertain] = self.global_click_conv_prob
                weights[uncertain] = 0.5
        
        elif click_mask.any():
            labels[click_mask] = self.global_click_conv_prob
            soft_labels[click_mask] = self.global_click_conv_prob
            weights[click_mask] = 0.5
        
        return labels, weights, action_types, soft_labels

    def get_sampling_stats(self) -> Dict[str, int]:
        return dict(self._sample_stats)


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 512,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 4,
    buffer_batches: int = 10,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    time_decay_factor: float = 604800.0,
    use_time_decay: bool = True,
    neg_downsample_ratio: Optional[float] = None,
    pos_upsample_ratio: Optional[float] = None,
    min_batch_size_ratio: float = 0.0,
    label_mode: str = 'defuse',
    soft_label_decay_hours: float = 24.0,
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

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

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
        time_decay_factor=time_decay_factor,
        use_time_decay=use_time_decay,
        neg_downsample_ratio=neg_downsample_ratio,
        pos_upsample_ratio=pos_upsample_ratio,
        min_batch_size_ratio=min_batch_size_ratio,
        label_mode=label_mode,
        soft_label_decay_hours=soft_label_decay_hours,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 4

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
        time_decay_factor=time_decay_factor,
        use_time_decay=use_time_decay,
        neg_downsample_ratio=None,
        pos_upsample_ratio=None,
        min_batch_size_ratio=0.0,
        label_mode='hard',
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=max(1, num_workers // 2),
        pin_memory=use_cuda,
        persistent_workers=True if num_workers > 0 else False,
    )

    logging.info(f"Data loaders ready: train={train_rows}, valid={valid_rows}, "
                 f"workers={num_workers}, prefetch=4")

    return train_loader, valid_loader, train_dataset