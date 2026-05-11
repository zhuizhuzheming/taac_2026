"""PCVRHeteroFormer inference script (v8.1-patch2 compatible).

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
    'dropout': 0.1,
    'pre_norm': True,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'emb_skip_threshold': 0,
    'action_num': 1,
    'gate_anneal_steps': 2000,
    'stochastic_depth_prob': 0.0,
    'progressive_layer_training': False,
    'id_dropout_rate': 0.15,
    'seq_id_dropout_rate': 0.10,
    'id_vocab_threshold': 10000,
    'shrinkage': 0.05,
    'cross_network_layers': 2,
    'num_codes': 64,
    'sinkhorn_epsilon': 0.05,
    'sinkhorn_iter': 20,
    'min_mass_ratio': 0.016,
    'coherence_threshold': 0.1,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())

_TRAIN_KEY_MAPPING = {
    'dropout': 'dropout_rate',
    'progressive_layer_training': 'enable_progressive_layers',
}


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
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults.")
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
                f"present in the checkpoint's schema.json."
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


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load state_dict, automatically handling torch.compile prefix."""
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # ============================================================
    # FIX: torch.compile wraps model with _orig_mod, so saved keys
    # look like "_orig_mod.logit_temperature". Strip the prefix.
    # Also handle DDP "module." prefix for completeness.
    # ============================================================
    needs_strip = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    needs_ddp_strip = any(k.startswith("module.") for k in state_dict.keys())
    
    if needs_strip or needs_ddp_strip:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
                new_state_dict[new_k] = v
            elif k.startswith("module."):
                new_k = k[len("module."):]
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        if needs_strip:
            logging.info("Stripped '_orig_mod.' prefix from checkpoint (torch.compile)")
        if needs_ddp_strip:
            logging.info("Stripped 'module.' prefix from checkpoint (DDP)")
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
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

    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".pt"):
                ckpt = os.path.join(root, f)
                logging.info(f"Found checkpoint: {ckpt}")
                return ckpt

    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
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

    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        if f'{domain}_decay_weight' in device_batch:
            seq_decay_weights[domain] = device_batch[f'{domain}_decay_weight']

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_decay_weights=seq_decay_weights if seq_decay_weights else None,
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

    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)

    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

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

    ckpt_path = get_ckpt_path(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"Directory contains: {os.listdir(model_dir) if os.path.isdir(model_dir) else 'N/A'}."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()