"""PCVRHeteroFormer inference script (v10 aligned — Improved).

Improvements over base version:
  1. NaN/Inf detection & fallback on logits (learned from training instability).
  2. Optional diagnostic dump (energy_score, uncertainty, base_logits, etc.).
  3. Streaming JSON writing to avoid OOM on large test sets.
  4. tqdm progress bar for batch-wise inference.
  5. Duplicate user_id detection & warning (dict-key overwrite awareness).
  6. Optional torch.compile for inference acceleration.
  7. Graceful degradation when train_config.json is missing.

Environment variables:
    MODEL_OUTPUT_PATH   Checkpoint directory (contains model.pt & train_config.json).
    EVAL_DATA_PATH      Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH    Directory for predictions.json (and optional diagnostics.json).
    INFER_DIAGNOSTICS   Set to "1" to export per-sample diagnostic vectors.
    INFER_COMPILE       Set to "1" to enable torch.compile on the model.
"""

import glob
import os
import json
import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHeteroFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ─────────────────────────── Fallback Configs ──────────────────────────────────

_FALLBACK_MODEL_CFG = {
    'd_model': 128,
    'emb_dim': 16,
    'seq_len': 50,
    'num_layers': 4,
    'base_rank': None,
    'rank_schedule': 'bottleneck',
    'num_global_tokens': 4,
    'kernel_size': 3,
    'num_heads': None,
    'num_banks': None,
    'dropout': 0.2,
    'pre_norm': True,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'emb_skip_threshold': 0,
    'action_num': 1,
    'gate_anneal_steps': 2000,
    'stochastic_depth_prob': 0.1,
    'progressive_layer_training': False,
    'id_dropout_rate': 0.05,
    'seq_id_dropout_rate': 0.05,
    'id_vocab_threshold': 10000,
    'shrinkage': 0.05,
    'cross_network_layers': 2,
    'num_codes': 128,
    'sinkhorn_epsilon': 0.05,
    'sinkhorn_iter': 20,
    'min_mass_ratio': 0.005,
    'coherence_threshold': 0.15,
    'kappa_base': 2.0,
    'kappa_min': 0.5,
    'lie_rank': 8,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 8

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())

_TRAIN_KEY_MAPPING = {
    'dropout': 'dropout_rate',
    'progressive_layer_training': 'enable_progressive_layers',
}


# ─────────────────────────── Utility Functions ─────────────────────────────

def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}. "
        f"Will use hardcoded fallback defaults — shape mismatch may occur "
        f"if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        train_key = _TRAIN_KEY_MAPPING.get(key, key)
        if train_key in train_config:
            cfg[key] = train_config[train_key]
        elif key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}' (tried '{train_key}'), "
                f"using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHeteroFormer:
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]

    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHeteroFormer with cfg: {model_cfg}")
    model = PCVRHeteroFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state(
    model: nn.Module,
    ckpt_path: str,
    device: str,
    strict: bool = True,
) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)

    has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    if has_orig_mod:
        logging.info("Detected torch.compile checkpoint (_orig_mod. prefix), stripping...")
        stripped_state_dict: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_k = k[len('_orig_mod.'):]
                stripped_state_dict[new_k] = v
            else:
                stripped_state_dict[k] = v
        state_dict = stripped_state_dict

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict. This usually means the model "
            "constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and "
            "matches the training hyperparameters.")
        raise e


def get_ckpt_path(model_dir: str) -> Optional[str]:
    if not model_dir or not os.path.isdir(model_dir):
        return None

    best_pattern = os.path.join(model_dir, "global_step*.best_model", "model.pt")
    best_matches = glob.glob(best_pattern)
    if best_matches:
        def _step_from_path(p: str) -> int:
            dirname = os.path.basename(os.path.dirname(p))
            return int(dirname.replace("global_step", "").replace(".best_model", ""))
        best_matches.sort(key=_step_from_path)
        logging.info(f"Found best model checkpoint: {best_matches[-1]}")
        return best_matches[-1]

    all_ckpts: List[str] = []
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt"):
                all_ckpts.append(os.path.join(root, f))

    if all_ckpts:
        all_ckpts.sort(key=os.path.getmtime)
        ckpt = all_ckpts[-1]
        logging.info(f"Found checkpoint: {ckpt}")
        return ckpt

    return None


# ─────────────────────────── Batch → ModelInput ──────────────────────────────

def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    required_keys = [
        'user_int_feats', 'item_int_feats', 'user_dense_feats',
        'item_dense_feats', '_seq_domains',
    ]
    for k in required_keys:
        if k not in batch:
            raise KeyError(f"Batch missing required key for ModelInput: '{k}'")

    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_decay_weights: Dict[str, torch.Tensor] = {}
    seq_timestamps_raw: Dict[str, torch.Tensor] = {}

    for domain in seq_domains:
        if domain not in device_batch:
            raise KeyError(
                f"Domain '{domain}' listed in _seq_domains but missing from batch")
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        if f'{domain}_decay_weight' in device_batch:
            seq_decay_weights[domain] = device_batch[f'{domain}_decay_weight']
        if f'{domain}_timestamps_raw' in device_batch:
            seq_timestamps_raw[domain] = device_batch[f'{domain}_timestamps_raw']

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_decay_weights=seq_decay_weights if seq_decay_weights else None,
        seq_timestamps_raw=seq_timestamps_raw if seq_timestamps_raw else None,
    )


# ─────────────────────────── Streaming JSON Writer ─────────────────────────

class StreamingJsonWriter:
    """Write a large JSON dict in streaming fashion to avoid memory spikes."""

    def __init__(self, path: str, key: str = "predictions"):
        self.path = path
        self.key = key
        self._first = True
        self._fp = open(path, 'w', encoding='utf-8')
        self._fp.write(f'{{"{key}": {{\n')

    def write_entry(self, k: str, v: Any):
        import json
        prefix = "" if self._first else ",\n"
        self._first = False
        self._fp.write(f'{prefix}  {json.dumps(k)}: {json.dumps(v)}')

    def close(self):
        self._fp.write("\n}}\n")
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────── Main Inference ─────────────────────────────────

def main() -> None:
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')
    do_diagnostics = os.environ.get('INFER_DIAGNOSTICS', '0') == '1'
    do_compile = os.environ.get('INFER_COMPILE', '0') == '1'

    if not model_dir:
        raise ValueError("Environment variable MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("Environment variable EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("Environment variable EVAL_RESULT_PATH is not set")

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)
    logging.info(f"Inference device: {device}")

    # ---- Schema ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
        if os.path.exists(schema_path):
            dst_schema = os.path.join(model_dir, 'schema.json')
            try:
                shutil.copy2(schema_path, dst_schema)
                logging.info(f"Copied schema.json to {model_dir} for future use")
                schema_path = dst_schema
            except Exception as e:
                logging.warning(f"Failed to copy schema.json to model_dir: {e}")
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model ----
    model_cfg = resolve_model_cfg(train_config)

    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Optional torch.compile ----
    if do_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead', dynamic=False)
            logging.info("torch.compile enabled for inference")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}, using eager mode")

    # ---- Load weights ----
    ckpt_path = get_ckpt_path(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"Directory contains: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state(model, ckpt_path, device, strict=True)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Inference loop ----
    output_path = os.path.join(result_dir, 'predictions.json')
    diag_path = os.path.join(result_dir, 'diagnostics.json') if do_diagnostics else None

    total_processed = 0
    nan_count = 0
    duplicate_count = 0
    seen_uids: set = set()

    # Diagnostics accumulator (only if enabled; flushed periodically)
    diag_buffer: List[Dict[str, Any]] = []
    diag_flush_every = 5000

    iterator = test_loader
    if tqdm is not None:
        # Approximate total batches
        est_batches = max(1, (total_test_samples + batch_size - 1) // batch_size)
        iterator = tqdm(iterator, total=est_batches, desc="Inference")

    with StreamingJsonWriter(output_path, key="predictions") as writer:
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                model_input = _batch_to_model_input(batch, device)
                user_ids = batch.get('user_id', [])

                # Use forward() instead of predict() when diagnostics needed
                if do_diagnostics:
                    out = model(model_input, task_id='ctr')
                    logits = out[0]
                    # Extract auxiliary tensors (all on CPU for safety)
                    energy_score = out[8].cpu().numpy() if out[8] is not None else None
                    uncertainty = out[6].cpu().numpy() if out[6] is not None else None
                    base_logits = out[10].cpu().numpy() if out[10] is not None else None
                else:
                    logits, _ = model.predict(model_input)
                    energy_score = None
                    uncertainty = None
                    base_logits = None

                logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)

                # NaN / Inf detection
                nan_mask = torch.isnan(probs) | torch.isinf(probs)
                if nan_mask.any():
                    nan_count += int(nan_mask.sum().item())
                    logging.warning(
                        f"Batch {batch_idx}: {int(nan_mask.sum().item())} NaN/Inf probs detected. "
                        f"Replacing with 0.5 (random guess)."
                    )
                    probs = torch.where(nan_mask, torch.tensor(0.5, device=probs.device), probs)

                batch_probs = [float(p) for p in probs.cpu().numpy().tolist()]
                batch_uids = [
                    int(uid) if hasattr(uid, 'item') else uid
                    for uid in user_ids
                ]

                # Write predictions & detect duplicates
                for uid, prob in zip(batch_uids, batch_probs):
                    key = str(uid)
                    if key in seen_uids:
                        duplicate_count += 1
                    else:
                        seen_uids.add(key)
                    writer.write_entry(key, prob)

                # Collect diagnostics
                if do_diagnostics:
                    for i, uid in enumerate(batch_uids):
                        entry: Dict[str, Any] = {"user_id": str(uid), "prob": batch_probs[i]}
                        if energy_score is not None:
                            entry["energy_score"] = float(energy_score[i])
                        if uncertainty is not None:
                            entry["uncertainty"] = float(uncertainty[i])
                        if base_logits is not None:
                            entry["base_logits"] = float(base_logits[i])
                        diag_buffer.append(entry)

                    if len(diag_buffer) >= diag_flush_every:
                        _flush_diagnostics(diag_buffer, diag_path, batch_idx == 0)
                        diag_buffer.clear()

                total_processed += len(batch_probs)

                if (batch_idx + 1) % 100 == 0:
                    logging.info(
                        f"  Processed {total_processed} samples "
                        f"(batch {batch_idx + 1}, NaN={nan_count}, Dup={duplicate_count})"
                    )

        # Final diagnostics flush
        if do_diagnostics and diag_buffer:
            _flush_diagnostics(diag_buffer, diag_path, False)

    logging.info(
        f"Inference complete: {total_processed} predictions written to {output_path}. "
        f"NaN/Inf replacements: {nan_count}. Duplicate user_id overwrites: {duplicate_count}."
    )

    if duplicate_count > 0:
        logging.warning(
            f"Detected {duplicate_count} duplicate user_ids — later predictions overwrite earlier ones. "
            f"If your task requires per-sample (not per-user) outputs, consider switching to a list format."
        )


def _flush_diagnostics(
    buffer: List[Dict[str, Any]],
    path: Optional[str],
    is_first: bool,
) -> None:
    if not path or not buffer:
        return
    mode = 'w' if is_first else 'a'
    with open(path, mode, encoding='utf-8') as f:
        if is_first:
            f.write('[\n')
        else:
            f.write(',\n')
        for i, entry in enumerate(buffer):
            suffix = '' if i == len(buffer) - 1 else ','
            f.write(json.dumps(entry) + suffix + '\n')
    if not is_first:
        # Rewrite trailing ] to keep JSON valid for incremental writes
        # (simplistic approach: append then fix on final call)
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Inference failed")
        raise
