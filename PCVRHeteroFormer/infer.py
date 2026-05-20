"""PCVRHeteroFormer inference script (v10 aligned).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
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

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHeteroFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
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


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHeteroFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())

# Mapping from model constructor arg name -> train_config.json key name
# (when they differ).
_TRAIN_KEY_MAPPING = {
    'dropout': 'dropout_rate',
    'progressive_layer_training': 'enable_progressive_layers',
}


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
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
    """Construct a ``PCVRHeteroFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
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
    """Load ``state_dict`` with automatic torch.compile prefix handling."""
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
            "Failed to load state_dict. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path(model_dir: str) -> Optional[str]:
    """Locate the checkpoint file inside ``model_dir``.

    Priority:
      1. ``global_step*.best_model/model.pt`` (best model saved by EarlyStopping)
      2. Any ``*.pt`` file found recursively under ``model_dir``, preferring
         the most recently modified one.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None

    # 1. Best model directory
    best_pattern = os.path.join(model_dir, "global_step*.best_model", "model.pt")
    best_matches = glob.glob(best_pattern)
    if best_matches:
        def _step_from_path(p: str) -> int:
            dirname = os.path.basename(os.path.dirname(p))
            return int(dirname.replace("global_step", "").replace(".best_model", ""))
        best_matches.sort(key=_step_from_path)
        logging.info(f"Found best model checkpoint: {best_matches[-1]}")
        return best_matches[-1]

    # 2. Recursively look for any .pt file, prefer latest mtime
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


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
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
            raise KeyError(f"Domain '{domain}' listed in _seq_domains but missing from batch")
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


def main() -> None:
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    if not model_dir:
        raise ValueError("Environment variable MODEL_OUTPUT_PATH is not set")
    if not data_dir:
        raise ValueError("Environment variable EVAL_DATA_PATH is not set")
    if not result_dir:
        raise ValueError("Environment variable EVAL_RESULT_PATH is not set")

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_grad_enabled(False)

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

    # ---- Load weights ----
    ckpt_path = get_ckpt_path(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}."
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

    all_probs: List[float] = []
    all_user_ids: List[Any] = []
    total_processed = 0
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            batch_probs = [float(p) for p in probs.tolist()]
            batch_uids = [
                int(uid) if hasattr(uid, 'item') else uid
                for uid in user_ids
            ]

            all_probs.extend(batch_probs)
            all_user_ids.extend(batch_uids)
            total_processed += len(batch_probs)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {total_processed} samples (batch {batch_idx + 1})")

    # ---- 长度一致性校验 ----
    if len(all_user_ids) != len(all_probs):
        raise RuntimeError(
            f"Mismatch after inference: {len(all_user_ids)} user_ids vs "
            f"{len(all_probs)} probs."
        )

    logging.info(f"Inference complete: {total_processed} predictions")

    # 【修复】恢复为 dict 格式，与 score.py 兼容
    # 注意：如果同一 user_id 出现多次，后面的 score 会覆盖前面的
    predictions_dict: Dict[str, float] = {}
    for uid, prob in zip(all_user_ids, all_probs):
        # JSON key 必须是字符串
        key = str(uid)
        predictions_dict[key] = prob

    predictions = {
        "predictions": predictions_dict,
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"Saved {len(predictions_dict)} predictions to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Inference failed")
        raise