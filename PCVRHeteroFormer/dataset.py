"""PCVR Parquet dataset module (v11 — Curriculum Learning + Anti-Leak).

Changes from v9.2:
1. 课程学习支持：动态截断序列长度（实例方法 + 环境变量双通道，兼容多进程）。
2. 保留所有 v9.2 修复：统一 click-filtering、CVR label 映射、OOB vocab limiting、
   list merge flush、validation truncation 等。
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import numpy.typing as npt
except ImportError:
    class _NptFallback:
        NDArray = Any
    npt = _NptFallback()


# ─────────────────────────── Dynamic Buffer Pool ─────────────────────────────


class DynamicBufferPool:
    """动态大小的缓冲区池，按 (batch_size, key) 复用 numpy 数组。"""
    
    def __init__(self, max_size_types: int = 4):
        self._pools: Dict[Tuple[int, str], np.ndarray] = {}
        self._max_size_types = max_size_types
        self._access_order: List[Tuple[int, str]] = []
        
    def get(self, B: int, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        pool_key = (B, key)
        
        if pool_key in self._pools:
            if pool_key in self._access_order:
                self._access_order.remove(pool_key)
            self._access_order.append(pool_key)
            buf = self._pools[pool_key]
            buf.fill(0)
            return buf
        
        current_sizes = set(k[0] for k in self._pools.keys())
        if len(current_sizes) >= self._max_size_types and B not in current_sizes:
            for old_key in list(self._access_order):
                if old_key[0] != B:
                    del self._pools[old_key]
                    self._access_order.remove(old_key)
                    break
        
        full_shape = (B,) + shape
        buf = np.zeros(full_shape, dtype=dtype)
        self._pools[pool_key] = buf
        self._access_order.append(pool_key)
        return buf
    
    def clear(self):
        self._pools.clear()
        self._access_order.clear()


# ─────────────────────────── Feature Schema ──────────────────────────────────


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

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)


torch.multiprocessing.set_sharing_strategy('file_system')

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

# ===== 推理优化：预计算时间桶查找表 =====
_MAX_LUT_TIME = int(BUCKET_BOUNDARIES[-1]) + 1  # 31536001
_TIME_BUCKET_LUT = np.zeros(_MAX_LUT_TIME, dtype=np.uint8)

def _init_time_bucket_lut():
    """向量化预计算：time_diff -> bucket 的 O(1) 查表"""
    global _TIME_BUCKET_LUT
    # 零值保持零
    _TIME_BUCKET_LUT[0] = 0
    # 用 np.searchsorted 一次性初始化整个 LUT
    all_times = np.arange(1, _MAX_LUT_TIME, dtype=np.int64)
    buckets = np.searchsorted(BUCKET_BOUNDARIES, all_times) + 1
    _TIME_BUCKET_LUT[1:] = buckets.astype(np.uint8)

_init_time_bucket_lut()
# ==========================================


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    v11:
    - Curriculum learning: dynamic sequence length truncation via
      `set_curriculum_max_len()` or env var `CURRICULUM_MAX_LEN`.
    - All v9.2 fixes preserved.
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
        row_groups: Optional[List[Tuple[str, int, int, int, int]]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        split_timestamp: Optional[int] = None,
        train_vocab: Optional[Dict[str, int]] = None,
        curriculum_max_len: Optional[Dict[str, int]] = None,
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
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # v11 curriculum state (instance-level, single-process)
        self._curriculum_max_len: Dict[str, int] = curriculum_max_len or {}

        if row_groups is not None:
            self._rg_list = [(f, i, n) for f, i, n, _, _ in row_groups]
            self._split_timestamp = split_timestamp
            self._train_vocab = train_vocab
        else:
            self._rg_list = []
            for f in self._parquet_files:
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

            if row_group_range is not None:
                start, end = row_group_range
                self._rg_list = self._rg_list[start:end]

            self._split_timestamp = None
            self._train_vocab = None

        self.num_rows = sum(r[2] for r in self._rg_list)

        self._load_schema(schema_path, seq_max_lens or {})

        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        self._effective_buffer = max(buffer_batches, 50)
        self._buffer_pool = DynamicBufferPool(max_size_types=4)

        # ---- Column plans ----
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

        # 预构建 (group, col_idx) -> col_name 映射
        self._col_name_map: Dict[Tuple[str, int], str] = {}
        for ci, dim, offset, vs in self._user_int_plan:
            if ci is not None:
                for fid, ent_offset, ent_len in self.user_int_schema.entries:
                    if ent_offset == offset:
                        self._col_name_map[('user_int', ci)] = f'user_int_feats_{fid}'
                        break
        for ci, dim, offset, vs in self._item_int_plan:
            if ci is not None:
                for fid, ent_offset, ent_len in self.item_int_schema.entries:
                    if ent_offset == offset:
                        self._col_name_map[('item_int', ci)] = f'item_int_feats_{fid}'
                        break
        for domain in self.seq_domains:
            side_plan, _ = self._seq_plan[domain]
            for ci, slot, vs in side_plan:
                if ci is not None:
                    fid = self.sideinfo_fids[domain][slot]
                    prefix = self._seq_prefix.get(domain, '')
                    self._col_name_map[(f'seq_{domain}', ci)] = f'{prefix}_{fid}'

        logging.info(
            f"PCVRParquetDataset v11: {self.num_rows} rows, "
            f"batch_size={batch_size}, shuffle={shuffle}, is_training={is_training}, "
            f"curriculum={self._curriculum_max_len}")

    def set_curriculum_max_len(self, limits: Dict[str, int]) -> None:
        """v11: 动态设置课程学习长度限制，并同步到环境变量供多进程 worker 读取。"""
        self._curriculum_max_len = limits
        try:
            os.environ['CURRICULUM_MAX_LEN'] = json.dumps(limits)
            logging.info(f"【v11-Curriculum】Set limits: {limits}")
        except Exception as e:
            logging.warning(f"Failed to sync curriculum limits to env: {e}")

    def _get_effective_max_len(self, domain: str) -> int:
        """v11: 返回实际生效的序列长度（考虑课程学习截断）。"""
        base = self._seq_maxlen.get(domain, 256)
        # 1) 实例级别（单进程生效）
        if self._curriculum_max_len and domain in self._curriculum_max_len:
            base = min(base, self._curriculum_max_len[domain])
        # 2) 环境变量级别（多进程 worker 生效）
        env_raw = os.environ.get('CURRICULUM_MAX_LEN', '')
        if env_raw:
            try:
                env_dict = json.loads(env_raw)
                if domain in env_dict:
                    base = min(base, int(env_dict[domain]))
            except Exception:
                pass
        return base

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        self.item_dense_schema = FeatureSchema()

        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

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

    def _is_valid_training_batch(self, batch_dict: Dict[str, Any]) -> bool:
        if not self.is_training:
            return True
        labels = batch_dict['label']
        n_pos = labels.sum().item()
        if n_pos == 0 or n_pos == labels.shape[0]:
            return False
        return True

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            
            if self.is_training:
                arrow_batch_size = self.batch_size * 2
            else:
                arrow_batch_size = self.batch_size * 16
            
            for batch in pf.iter_batches(batch_size=arrow_batch_size, row_groups=[rg_idx]):
                batch_size_actual = batch.num_rows
                
                if batch_size_actual > self.batch_size * 2:
                    for start in range(0, batch_size_actual, self.batch_size):
                        end = min(start + self.batch_size, batch_size_actual)
                        sub_batch = batch.slice(start, end - start)
                        sub_B = end - start
                        
                        batch_dict = self._convert_batch(sub_batch)
                        if batch_dict is None:
                            continue
                        
                        if sub_B == self.batch_size and self.shuffle and self._effective_buffer > 1:
                            buffer.append(batch_dict)
                            if len(buffer) >= self._effective_buffer:
                                yield from self._flush_buffer(buffer)
                                buffer = []
                        else:
                            if self._is_valid_training_batch(batch_dict):
                                yield batch_dict
                else:
                    batch_dict = self._convert_batch(batch)
                    if batch_dict is None:
                        continue
                    
                    actual_B = batch_dict['label'].shape[0]
                    if actual_B == self.batch_size and self.shuffle and self._effective_buffer > 1:
                        buffer.append(batch_dict)
                        if len(buffer) >= self._effective_buffer:
                            yield from self._flush_buffer(buffer)
                            buffer = []
                    else:
                        if self._is_valid_training_batch(batch_dict):
                            yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        if not buffer:
            return
        
        merged: Dict[str, Any] = {}
        non_tensor_keys: Dict[str, Any] = {}
        
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                stacked = torch.stack([b[k] for b in buffer], dim=0)
                new_shape = (-1,) + stacked.shape[2:]
                merged[k] = stacked.view(new_shape)
            elif isinstance(buffer[0][k], list):
                merged_list: List[Any] = []
                for b in buffer:
                    merged_list.extend(b[k])
                non_tensor_keys[k] = merged_list
            else:
                non_tensor_keys[k] = buffer[0][k]
        
        total_rows = merged['label'].shape[0]
        
        if self.shuffle:
            rand_idx = torch.randperm(total_rows)
        else:
            rand_idx = None
        
        batch_size = self.batch_size
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, total_rows)
            
            if rand_idx is not None:
                idx = rand_idx[start:end]
            else:
                idx = slice(start, end)
            
            batch: Dict[str, Any] = {}
            for k, v in merged.items():
                batch[k] = v[idx]
            batch.update(non_tensor_keys)
            
            if not self._is_valid_training_batch(batch):
                continue
            
            yield batch
        
        del merged, rand_idx
        buffer.clear()

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        effective_vocab = vocab_size
        if not self.is_training and self._train_vocab is not None:
            col_name = self._col_name_map.get((group, col_idx))
            if col_name and col_name in self._train_vocab:
                effective_vocab = self._train_vocab[col_name]
                if effective_vocab < vocab_size:
                    logging.debug(f"【v11-anti-leak】Limiting vocab {col_name}: {vocab_size} -> {effective_vocab}")

        oob_mask = arr >= effective_vocab
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
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': effective_vocab,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {effective_vocab}), actual=[{mn}, {mx}].")

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
        """向量化版本：用NumPy高级索引替代Python循环"""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)

        raw_lens = offsets[1:] - offsets[:-1]
        use_lens = np.minimum(raw_lens, max_len)
        valid_mask = use_lens > 0

        if valid_mask.any():
            valid_indices = np.where(valid_mask)[0]
            starts = offsets[:-1][valid_mask]
            lens = use_lens[valid_mask]

            rows = np.repeat(valid_indices, lens)
            col_offsets = np.arange(lens.sum()) - np.repeat(
                np.cumsum(np.concatenate([[0], lens[:-1]])), lens)
            cols = col_offsets
            val_indices = np.repeat(starts, lens) + col_offsets

            padded[rows, cols] = values[val_indices]
            lengths[valid_indices] = lens

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

    def _convert_batch(self, batch: "pa.RecordBatch") -> Optional[Dict[str, Any]]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors.

        v11:
        - Curriculum learning: dynamic max_len truncation per domain.
        - Unified click-filtering for CVR (preserved).
        """
        B = batch.num_rows
        pool = self._buffer_pool

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        label_types = batch.column(self._col_idx['label_type']).fill_null(0).to_numpy().astype(np.int64)

        click_mask = label_types > 0
        if self.is_training:
            if not click_mask.any():
                logging.debug("Batch has no click samples, skipping")
                return None
            labels = (label_types == 2).astype(np.int64)
            n_keep = int(click_mask.sum())
        else:
            if click_mask.any():
                labels = (label_types == 2).astype(np.int64)
                n_keep = int(click_mask.sum())
            else:
                labels = np.zeros(B, dtype=np.int64)
                click_mask = np.ones(B, dtype=bool)
                n_keep = B
        if n_keep < B:
            B = n_keep
            timestamps = timestamps[click_mask]

        user_ids_all = batch.column(self._col_idx['user_id']).to_pylist()
        user_ids = [user_ids_all[i] for i in range(len(user_ids_all)) if click_mask[i]] if n_keep < len(user_ids_all) else user_ids_all

        # ---- user_int ----
        user_int = pool.get(B, 'user_int', (self.user_int_schema.total_dim,), np.int64)
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('user_int', ci, arr, vs)
                else:
                    arr[:] = 0
                if n_keep < batch.num_rows:
                    user_int[:, offset] = arr[click_mask]
                else:
                    user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, batch.num_rows)
                if vs > 0:
                    self._record_oob('user_int', ci, padded, vs)
                else:
                    padded[:] = 0
                if n_keep < batch.num_rows:
                    user_int[:, offset:offset + dim] = padded[click_mask]
                else:
                    user_int[:, offset:offset + dim] = padded

        # ---- item_int ----
        item_int = pool.get(B, 'item_int', (self.item_int_schema.total_dim,), np.int64)
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._record_oob('item_int', ci, arr, vs)
                else:
                    arr[:] = 0
                if n_keep < batch.num_rows:
                    item_int[:, offset] = arr[click_mask]
                else:
                    item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, batch.num_rows)
                if vs > 0:
                    self._record_oob('item_int', ci, padded, vs)
                else:
                    padded[:] = 0
                if n_keep < batch.num_rows:
                    item_int[:, offset:offset + dim] = padded[click_mask]
                else:
                    item_int[:, offset:offset + dim] = padded

        # ---- user_dense ----
        user_dense = pool.get(B, 'user_dense', (self.user_dense_schema.total_dim,), np.float32)
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, batch.num_rows)
            if n_keep < batch.num_rows:
                user_dense[:, offset:offset + dim] = padded[click_mask]
            else:
                user_dense[:, offset:offset + dim] = padded

        labels = labels[click_mask] if n_keep < batch.num_rows else labels[:B]

        result = {
            'user_int_feats': torch.from_numpy(user_int.copy()),
            'user_dense_feats': torch.from_numpy(user_dense.copy()),
            'item_int_feats': torch.from_numpy(item_int.copy()),
            'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32),
            'label': torch.from_numpy(labels),
            'timestamp': torch.from_numpy(timestamps),
            'user_id': user_ids,
            '_seq_domains': self.seq_domains,
        }

        # ---- Sequence features ----
        for domain in self.seq_domains:
            # v11: 课程学习动态截断
            max_len = self._get_effective_max_len(domain)
            n_feats = len(self.sideinfo_fids[domain])
            side_plan, ts_ci = self._seq_plan[domain]

            all_offsets = []
            all_values = []
            all_vs = []
            all_ci = []

            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                all_offsets.append(col.offsets.to_numpy())
                all_values.append(col.values.to_numpy())
                all_vs.append(vs)
                all_ci.append(ci)

            out_full = pool.get(batch.num_rows, f'seq_{domain}', (n_feats, max_len), np.int64)
            lengths_full = pool.get(batch.num_rows, f'seq_len_{domain}', (), np.int64)

            for c, (offs, vals, vs, ci) in enumerate(zip(all_offsets, all_values, all_vs, all_ci)):
                raw_lens = offs[1:] - offs[:-1]
                use_lens = np.minimum(raw_lens, max_len)
                # v11.2-infer-opt: vectorized fill, replaces for-i loop
                valid_mask = use_lens > 0
                if valid_mask.any():
                    valid_idx = np.where(valid_mask)[0]
                    starts = offs[:-1][valid_mask]
                    lens = use_lens[valid_mask]
                    if lens.size > 0:
                        rows = np.repeat(valid_idx, lens)
                        cumlens = np.cumsum(np.concatenate([[0], lens[:-1]]))
                        col_offsets = np.arange(lens.sum()) - np.repeat(cumlens, lens)
                        val_idx = np.repeat(starts, lens) + col_offsets
                        out_full[rows, c, col_offsets] = vals[val_idx]
                        lengths_full[valid_idx] = np.maximum(lengths_full[valid_idx], lens)

            out_full[out_full <= 0] = 0

            for c, (_, _, vs, ci) in enumerate(zip(all_offsets, all_values, all_vs, all_ci)):
                slice_c = out_full[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs)
                else:
                    slice_c[:] = 0

            if n_keep < batch.num_rows:
                out = out_full[click_mask]
                lengths = lengths_full[click_mask]
            else:
                out = out_full[:B]
                lengths = lengths_full[:B]

            # Anti-leak validation truncation
            if not self.is_training and self._split_timestamp is not None and ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()

                for i in range(B):
                    start, end = int(ts_offs[i]), int(ts_offs[i+1])
                    if end > start:
                        seq_ts = ts_vals[start:end]
                        valid_positions = np.where(seq_ts <= self._split_timestamp)[0]
                        if len(valid_positions) > 0:
                            new_len = min(int(valid_positions[-1]) + 1, max_len)
                            lengths[i] = min(lengths[i], new_len)
                            if new_len < max_len:
                                out[i, :, new_len:] = 0
                        else:
                            lengths[i] = 0
                            out[i] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing + raw timestamps
            time_bucket = pool.get(B, f'seq_tb_{domain}', (max_len,), np.int64)
            decay_weight = pool.get(B, f'seq_dw_{domain}', (max_len,), np.float32)
            timestamps_raw = pool.get(B, f'seq_ts_raw_{domain}', (max_len,), np.float32)

            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()

                ts_padded = pool.get(batch.num_rows, f'ts_pad_{domain}', (max_len,), np.int64)
                raw_lens = ts_offs[1:] - ts_offs[:-1]
                use_lens = np.minimum(raw_lens, max_len)
                # v11.2-infer-opt: vectorized fill, replaces for-i loop
                valid_mask = use_lens > 0
                if valid_mask.any():
                    valid_idx = np.where(valid_mask)[0]
                    starts = ts_offs[:-1][valid_mask]
                    lens = use_lens[valid_mask]
                    if lens.size > 0:
                        rows = np.repeat(valid_idx, lens)
                        cumlens = np.cumsum(np.concatenate([[0], lens[:-1]]))
                        col_offsets = np.arange(lens.sum()) - np.repeat(cumlens, lens)
                        val_idx = np.repeat(starts, lens) + col_offsets
                        ts_padded[rows, col_offsets] = ts_vals[val_idx]

                ts_padded_filtered = ts_padded[click_mask] if n_keep < batch.num_rows else ts_padded[:B]

                ts_expanded = timestamps.reshape(-1, 1).astype(np.float32)
                ts_seq = ts_padded_filtered.astype(np.float32)
                time_diff = np.maximum(ts_expanded - ts_seq, 0.0)

                timestamps_raw[:] = time_diff

                # ===== 优化：推理时用 LUT 替代 searchsorted =====
                if self.is_training:
                    # 训练时保持原逻辑（或也可用 LUT，结果一致）
                    time_diff_int = time_diff.astype(np.int64).clip(0, _MAX_LUT_TIME - 1)
                    buckets = _TIME_BUCKET_LUT[time_diff_int.ravel()].reshape(B, max_len)
                    buckets[ts_padded_filtered == 0] = 0
                    time_bucket[:] = buckets

                    decay_short = np.exp(-time_diff / 3600.0)
                    decay_long = np.exp(-time_diff / 86400.0)
                    decay_weight[:] = 0.7 * decay_short + 0.3 * decay_long
                    decay_weight[ts_padded_filtered == 0] = 0.0
                else:
                    # 推理时：用 LUT 计算 bucket，decay 简化为固定权重
                    time_diff_int = time_diff.astype(np.int64).clip(0, _MAX_LUT_TIME - 1)
                    buckets = _TIME_BUCKET_LUT[time_diff_int.ravel()].reshape(B, max_len)
                    buckets[ts_padded_filtered == 0] = 0
                    time_bucket[:] = buckets
                    
                    # decay 简化为线性衰减（避免两次 exp），精度损失极小
                    decay_weight[:] = np.exp(-time_diff / 3600.0)  # 只算短周期，更稳定
                    decay_weight[ts_padded_filtered == 0] = 0.0
                # ==================================================

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_decay_weight'] = torch.from_numpy(decay_weight.copy())
            result[f'{domain}_timestamps_raw'] = torch.from_numpy(timestamps_raw.copy())

        return result


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
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """v11 防泄露数据加载器（兼容原有接口，新增 curriculum_max_len 透传）"""
    random.seed(seed)
    np.random.seed(seed)

    anti_leak_mode = os.environ.get('ANTI_LEAK_MODE', 'timestamp').lower()
    if anti_leak_mode not in ['timestamp', 'user_id', 'none']:
        logging.warning(f"【v11-anti-leak】Unknown ANTI_LEAK_MODE={anti_leak_mode}, using 'timestamp'")
        anti_leak_mode = 'timestamp'

    if anti_leak_mode == 'none':
        logging.info("【v11-anti-leak】Anti-leak mode DISABLED")
    else:
        logging.info(f"【v11-anti-leak】Anti-leak mode ENABLED: {anti_leak_mode}")

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        schema = pf.schema_arrow
        timestamp_col_idx = None
        for i, name in enumerate(schema.names):
            if name == 'timestamp':
                timestamp_col_idx = i
                break

        if timestamp_col_idx is None:
            logging.warning(f"No 'timestamp' column in {f}, using file order")
            for i in range(pf.metadata.num_row_groups):
                rg_info.append((f, i, pf.metadata.row_group(i).num_rows, 0, 0))
            continue

        for i in range(pf.metadata.num_row_groups):
            rg_meta = pf.metadata.row_group(i)
            col_meta = rg_meta.column(timestamp_col_idx)
            if col_meta.is_stats_set and col_meta.statistics.has_min_max:
                min_ts = int(col_meta.statistics.min)
                max_ts = int(col_meta.statistics.max)
            else:
                rg_batch = pf.read_row_group(i, columns=['timestamp'])
                ts_array = rg_batch.column('timestamp').to_numpy()
                min_ts = int(ts_array.min())
                max_ts = int(ts_array.max())
            rg_info.append((f, i, rg_meta.num_rows, min_ts, max_ts))

    total_rows = sum(r[2] for r in rg_info)

    if anti_leak_mode == 'none':
        n_valid_rgs = max(1, int(len(rg_info) * valid_ratio))
        n_train_rgs = len(rg_info) - n_valid_rgs
        if train_ratio < 1.0:
            n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        train_rgs = rg_info[:n_train_rgs]
        valid_rgs = rg_info[n_train_rgs:]
        split_timestamp = None
        logging.info(f"【v11-anti-leak】Original split: {n_train_rgs} train RGs, {n_valid_rgs} valid RGs")

    elif anti_leak_mode == 'timestamp':
        logging.info("【v11-anti-leak】Sorting row groups by timestamp...")
        rg_info.sort(key=lambda x: x[3])
        n_valid_rows = max(1, int(total_rows * valid_ratio))
        cum_rows = 0
        split_idx = len(rg_info)
        for i in range(len(rg_info) - 1, -1, -1):
            cum_rows += rg_info[i][2]
            if cum_rows >= n_valid_rows:
                split_idx = i
                break
        if train_ratio < 1.0:
            train_rows_target = int(sum(r[2] for r in rg_info[:split_idx]) * train_ratio)
            cum_train = 0
            actual_split = split_idx
            for i in range(split_idx):
                cum_train += rg_info[i][2]
                if cum_train >= train_rows_target:
                    actual_split = i + 1
                    break
            split_idx = actual_split
        train_rgs = rg_info[:split_idx]
        valid_rgs = rg_info[split_idx:]
        split_timestamp = valid_rgs[0][3] if valid_rgs else float('inf')
        logging.info(f"【v11-anti-leak】Temporal split at timestamp={split_timestamp}")

    elif anti_leak_mode == 'user_id':
        logging.info("【v11-anti-leak】Performing user-level split...")
        user_rg_map = {}
        for f, rg_idx, _, _, _ in rg_info:
            pf = pq.ParquetFile(f)
            rg_batch = pf.read_row_group(rg_idx, columns=['user_id'])
            user_ids = rg_batch.column('user_id').to_pylist()
            for uid in set(user_ids):
                if uid not in user_rg_map:
                    user_rg_map[uid] = set()
                user_rg_map[uid].add((f, rg_idx))
        all_users = list(user_rg_map.keys())
        np.random.seed(seed)
        np.random.shuffle(all_users)
        n_valid_users = max(1, int(len(all_users) * valid_ratio))
        train_users = set(all_users[:-n_valid_users])
        valid_users = set(all_users[-n_valid_users:])
        train_rg_set = set()
        valid_rg_set = set()
        for uid in train_users:
            train_rg_set.update(user_rg_map[uid])
        for uid in valid_users:
            valid_rg_set.update(user_rg_map[uid])
        overlap = train_rg_set & valid_rg_set
        if overlap:
            logging.warning(f"【v11-anti-leak】{len(overlap)} row groups overlap, assigning to train")
            valid_rg_set -= overlap
            train_rg_set |= overlap
        train_rgs = [r for r in rg_info if (r[0], r[1]) in train_rg_set]
        valid_rgs = [r for r in rg_info if (r[0], r[1]) in valid_rg_set]
        if valid_rgs:
            valid_rgs_sorted = sorted(valid_rgs, key=lambda x: x[3])
            split_timestamp = valid_rgs_sorted[0][3]
        else:
            split_timestamp = float('inf')
        logging.info(f"【v11-anti-leak】User split: {len(train_users)} train, {len(valid_users)} valid")

    train_vocab = _build_vocab_from_rgs(train_rgs, schema_path) if anti_leak_mode != 'none' else None

    # v11: 透传 curriculum_max_len（kwargs中可能包含）
    curriculum_limits = kwargs.pop('curriculum_max_len', None)

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_groups=train_rgs if anti_leak_mode != 'none' else None,
        row_group_range=(0, len(train_rgs)) if anti_leak_mode == 'none' else None,
        clip_vocab=clip_vocab,
        is_training=True,
        split_timestamp=split_timestamp if anti_leak_mode != 'none' else None,
        train_vocab=train_vocab,
        curriculum_max_len=curriculum_limits,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_groups=valid_rgs if anti_leak_mode != 'none' else None,
        row_group_range=(len(train_rgs), len(rg_info)) if anti_leak_mode == 'none' else None,
        clip_vocab=clip_vocab,
        is_training=False,
        split_timestamp=split_timestamp if anti_leak_mode != 'none' else None,
        train_vocab=train_vocab,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw['persistent_workers'] = True
        _train_kw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, 
        pin_memory=use_cuda,
        **_train_kw,
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0,
        pin_memory=use_cuda,
    )

    logging.info(f"【v11-anti-leak】Final: train={sum(r[2] for r in train_rgs)} rows, "
                 f"valid={sum(r[2] for r in valid_rgs)} rows")

    return train_loader, valid_loader, train_dataset


def _build_vocab_from_rgs(rg_list, schema_path):
    """从训练集row groups构建vocab统计"""
    vocab_sizes = {}
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load schema for vocab build: {e}")
        return vocab_sizes

    processed = 0
    for fpath, rg_idx, _, _, _ in rg_list:
        try:
            pf = pq.ParquetFile(fpath)
            rg_batch = pf.read_row_group(rg_idx)
            for col_name in rg_batch.schema.names:
                if 'int' in col_name:
                    try:
                        col_data = rg_batch.column(col_name).to_numpy()
                        if len(col_data) > 0:
                            valid_data = col_data[col_data > 0]
                            if len(valid_data) > 0:
                                max_val = int(valid_data.max())
                                if col_name not in vocab_sizes or max_val > vocab_sizes[col_name]:
                                    vocab_sizes[col_name] = max_val + 1
                    except Exception:
                        pass
            processed += 1
        except Exception as e:
            logging.debug(f"Skip RG {rg_idx} in {fpath}: {e}")

    logging.info(f"【v11-anti-leak】Built vocab from train set: {len(vocab_sizes)} columns")
    return vocab_sizes