"""PCVR Parquet dataset module (performance-tuned v6.2-pool).

Optimizations (v6.2-pool):
- DynamicBufferPool: 按 (B, key) 复用已分配的 numpy 缓冲区
- Worker-local buffer shuffle
- Arrow batch 切割适配固定 batch size
"""

import os
import logging
import random
import json
import gc
import threading

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
    """
    动态大小的缓冲区池，按 (batch_size, key) 复用 numpy 数组。
    
    设计原则:
    1. 线程安全: 每个 worker 独立实例，无需锁
    2. 惰性分配: 首次请求某 (B, key) 时才分配
    3. 零初始化: 返回前自动清零，保证数据干净
    4. 容量控制: 限制 pool 中保留的最大 batch_size 种类数
    """
    
    def __init__(self, max_size_types: int = 4):
        self._pools: Dict[Tuple[int, str], np.ndarray] = {}
        self._max_size_types = max_size_types
        self._access_order: List[Tuple[int, str]] = []  # LRU 淘汰
        
    def get(self, B: int, key: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """
        获取大小为 (B, *shape) 的缓冲区。
        
        Args:
            B: batch size (第一维度)
            key: 缓冲区标识 (如 'user_int', 'seq_a')
            shape: 除第一维外的形状
            dtype: 数据类型
            
        Returns:
            已清零的 numpy 数组，形状为 (B, *shape)
        """
        pool_key = (B, key)
        
        if pool_key in self._pools:
            # 命中缓存，更新访问顺序
            if pool_key in self._access_order:
                self._access_order.remove(pool_key)
            self._access_order.append(pool_key)
            buf = self._pools[pool_key]
            buf.fill(0)  # 清零复用
            return buf
        
        # 未命中，分配新缓冲区
        # LRU 淘汰: 如果 size types 过多，移除最久未使用的
        current_sizes = set(k[0] for k in self._pools.keys())
        if len(current_sizes) >= self._max_size_types and B not in current_sizes:
            # 找到最久未使用的、不同 size 的 key 淘汰
            for old_key in list(self._access_order):
                if old_key[0] != B:
                    del self._pools[old_key]
                    self._access_order.remove(old_key)
                    break
        
        # 分配新缓冲区
        full_shape = (B,) + shape
        buf = np.zeros(full_shape, dtype=dtype)
        self._pools[pool_key] = buf
        self._access_order.append(pool_key)
        
        return buf
    
    def clear(self):
        """清空 pool，释放内存"""
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


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    v6.2-pool: DynamicBufferPool for efficient memory reuse.
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

        self._effective_buffer = max(buffer_batches, 50)

        # v6.2-pool: 每个 dataset 实例一个 buffer pool
        # 由于每个 worker 会序列化/反序列化 dataset，每个 worker 会有独立 pool
        self._buffer_pool = DynamicBufferPool(max_size_types=4)

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
            f"PCVRParquetDataset v6.2-pool: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={self._effective_buffer}, shuffle={shuffle}")

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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            arrow_batch_size = self.batch_size * 2
            for batch in pf.iter_batches(batch_size=arrow_batch_size, row_groups=[rg_idx]):
                batch_size_actual = batch.num_rows
                
                if batch_size_actual > self.batch_size * 2:
                    # 大 batch 切割
                    for start in range(0, batch_size_actual, self.batch_size):
                        end = min(start + self.batch_size, batch_size_actual)
                        sub_batch = batch.slice(start, end - start)
                        sub_B = end - start
                        
                        batch_dict = self._convert_batch(sub_batch)
                        if batch_dict is None:
                            continue
                        
                        # v6.2-fix: 只有满 batch 才进入 shuffle buffer
                        if sub_B == self.batch_size and self.shuffle and self._effective_buffer > 1:
                            buffer.append(batch_dict)
                            if len(buffer) >= self._effective_buffer:
                                yield from self._flush_buffer(buffer)
                                buffer = []
                        else:
                            # 不满 batch_size 的直接 yield（不 shuffle）
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
                        yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Stack+view 替代 cat，预分块"""
        if not buffer:
            return
        
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                stacked = torch.stack([b[k] for b in buffer], dim=0)
                new_shape = (-1,) + stacked.shape[2:]
                merged[k] = stacked.view(new_shape)
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
            
            if self.is_training and batch['label'].sum() == 0:
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
                f"[0, {vocab_size}), actual=[{mn}, {mx}].")

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
        values = arrow_col.values.to_numpy()

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

        v6.2-pool: 使用 DynamicBufferPool 复用缓冲区。
        """
        B = batch.num_rows
        pool = self._buffer_pool

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        label_types = batch.column(self._col_idx['label_type']).fill_null(0).to_numpy().astype(np.int64)

        if self.is_training:
            click_mask = label_types > 0
            if not click_mask.any():
                logging.debug("Batch has no click samples, skipping")
                return None
            labels = (label_types == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
            click_mask = np.ones(B, dtype=bool)

        n_keep = int(click_mask.sum())
        if n_keep < B:
            B = n_keep
            timestamps = timestamps[click_mask]

        user_ids_all = batch.column(self._col_idx['user_id']).to_pylist()
        user_ids = [user_ids_all[i] for i in range(len(user_ids_all)) if click_mask[i]] if n_keep < len(user_ids_all) else user_ids_all

        # v6.2-pool: 使用 pool 获取缓冲区
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
            max_len = self._seq_maxlen[domain]
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

            # v6.2-pool: 动态获取序列缓冲区
            out_full = pool.get(batch.num_rows, f'seq_{domain}', (n_feats, max_len), np.int64)
            lengths_full = pool.get(batch.num_rows, f'seq_len_{domain}', (), np.int64)

            for c, (offs, vals, vs, ci) in enumerate(zip(all_offsets, all_values, all_vs, all_ci)):
                raw_lens = offs[1:] - offs[:-1]
                use_lens = np.minimum(raw_lens, max_len)
                for i in range(batch.num_rows):
                    if use_lens[i] > 0:
                        s = int(offs[i])
                        e = s + int(use_lens[i])
                        out_full[i, c, :use_lens[i]] = vals[s:e]
                        lengths_full[i] = max(lengths_full[i], use_lens[i])

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

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            # Time bucketing + decay weights
            time_bucket = pool.get(B, f'seq_tb_{domain}', (max_len,), np.int64)
            decay_weight = pool.get(B, f'seq_dw_{domain}', (max_len,), np.float32)

            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()

                ts_padded = pool.get(batch.num_rows, f'ts_pad_{domain}', (max_len,), np.int64)
                raw_lens = ts_offs[1:] - ts_offs[:-1]
                use_lens = np.minimum(raw_lens, max_len)
                for i in range(batch.num_rows):
                    if use_lens[i] > 0:
                        s = int(ts_offs[i])
                        e = s + int(use_lens[i])
                        ts_padded[i, :use_lens[i]] = ts_vals[s:e]

                ts_expanded = timestamps.reshape(-1, 1)
                ts_padded_filtered = ts_padded[click_mask] if n_keep < batch.num_rows else ts_padded[:B]
                time_diff = np.maximum(ts_expanded - ts_padded_filtered, 0)

                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded_filtered == 0] = 0
                time_bucket[:] = buckets

                time_diff_f = time_diff.astype(np.float32)
                decay_short = np.exp(-time_diff_f / 3600.0)
                decay_long = np.exp(-time_diff_f / 86400.0)
                decay_weight[:] = 0.7 * decay_short + 0.3 * decay_long
                decay_weight[ts_padded_filtered == 0] = 0.0

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())
            result[f'{domain}_decay_weight'] = torch.from_numpy(decay_weight.copy())

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

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0,
        pin_memory=use_cuda,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset