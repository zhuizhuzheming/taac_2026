"""HeteroFormer training entry point — v9.1 (NaN-Resilient)."""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHeteroFormer, ModelInput
from trainer import PCVRHeteroFormerTrainer


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: List[int]) -> List[Tuple[int, int, int]]:
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HeteroFormer Training v9.1")

    # Data paths
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--buffer_batches', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--eval_every_n_steps', type=int, default=0)
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--base_rank', type=int, default=None)
    parser.add_argument('--rank_schedule', type=str, default='bottleneck')
    parser.add_argument('--num_global_tokens', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--num_banks', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--pre_norm', action='store_true', default=True)
    parser.add_argument('--action_num', type=int, default=1)

    parser.add_argument('--use_time_buckets', action='store_true', default=True)
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false')
    parser.add_argument('--seq_id_threshold', type=int, default=10000)

    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer')
    parser.add_argument('--user_ns_tokens', type=int, default=0)
    parser.add_argument('--item_ns_tokens', type=int, default=0)
    parser.add_argument('--emb_skip_threshold', type=int, default=0)

    # Loss and optimization
    parser.add_argument('--loss_type', type=str, default='focal', choices=['bce', 'focal'])
    parser.add_argument('--focal_alpha', type=float, default=0.1)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--sparse_lr', type=float, default=0.05)
    parser.add_argument('--sparse_weight_decay', type=float, default=1e-4)
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1)
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0)

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups)

    parser.add_argument('--use_adaptive_focal', action='store_true', default=True)
    parser.add_argument('--enable_progressive_layers', action='store_true', default=False)
    parser.add_argument('--stochastic_depth_prob', type=float, default=0.0)
    parser.add_argument('--id_dropout_rate', type=float, default=0.0)
    parser.add_argument('--seq_id_dropout_rate', type=float, default=0.0)
    parser.add_argument('--id_vocab_threshold', type=int, default=10000)

    parser.add_argument('--label_smoothing_strategy', type=str, default='hybrid')
    parser.add_argument('--label_smoothing_max_eps', type=float, default=0.05)
    parser.add_argument('--label_smoothing_min_eps', type=float, default=0.001)
    parser.add_argument('--label_smoothing_anneal_steps', type=int, default=2000)

    parser.add_argument('--shrinkage', type=float, default=0.05)
    parser.add_argument('--cross_network_layers', type=int, default=2)

    # v9.1: Prototype Architecture 【修改】超参数调整
    parser.add_argument('--num_codes', type=int, default=128)
    parser.add_argument('--sinkhorn_epsilon', type=float, default=0.02)  # 【修改】从0.05降低
    parser.add_argument('--sinkhorn_iter', type=int, default=20)
    parser.add_argument('--min_mass_ratio', type=float, default=0.005)
    parser.add_argument('--coherence_threshold', type=float, default=0.15)
    parser.add_argument('--kappa_base', type=float, default=8.0)  # 【修改】从2.0提高
    parser.add_argument('--kappa_min', type=float, default=0.5)
    parser.add_argument('--lie_rank', type=int, default=8)

    # v9.1: Multi-Task Isolation 【修改】超参数调整
    parser.add_argument('--use_diffusion', action='store_true', default=True)
    parser.add_argument('--use_energy', action='store_true', default=True)
    parser.add_argument('--use_domain_adversarial', action='store_true', default=False)
    parser.add_argument('--packing_weight', type=float, default=0.1)  # 【v9.2】保持0.1
    parser.add_argument('--energy_margin', type=float, default=1.0)
    parser.add_argument('--energy_weight', type=float, default=0.1)
    parser.add_argument('--diff_weight', type=float, default=0.05)
    parser.add_argument('--meta_update_interval', type=int, default=50)  # 【v9.2】从100降到50
    parser.add_argument('--curriculum_warmup', type=int, default=5000)
    parser.add_argument('--diffusion_warmup', type=int, default=500)  # 【修改】从1000降低

    # Compilation
    parser.add_argument('--compile_backend', type=str, default='inductor')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead')
    parser.add_argument('--compile_dynamic', action='store_true', default=True)
    parser.add_argument('--no-compile_dynamic', dest='compile_dynamic', action='store_false')
    parser.add_argument('--compile_suppress_errors', action='store_true', default=True)

    # Focal loss parameters
    parser.add_argument('--focal_alpha_pos', type=float, default=0.6)
    parser.add_argument('--focal_alpha_neg', type=float, default=0.4)
    parser.add_argument('--focal_max_gamma', type=float, default=4.0)
    parser.add_argument('--global_ctr', type=float, default=0.01)

    # v9.3: Generative Fusion
    parser.add_argument('--use_generative_fusion', action='store_true', default=True)
    parser.add_argument('--no-generative_fusion', dest='use_generative_fusion', action='store_false')

    args = parser.parse_args()

    # Environment variable overrides
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.tf_events_dir:
        Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir) if args.tf_events_dir else None

    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "seq_len": args.seq_len,
        "num_layers": args.num_layers,
        "base_rank": args.base_rank,
        "rank_schedule": args.rank_schedule,
        "num_global_tokens": args.num_global_tokens,
        "kernel_size": args.kernel_size,
        "num_heads": args.num_heads,
        "num_banks": args.num_banks,
        "dropout": args.dropout_rate,
        "pre_norm": args.pre_norm,
        "id_dropout_rate": args.id_dropout_rate,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "emb_skip_threshold": args.emb_skip_threshold,
        "stochastic_depth_prob": args.stochastic_depth_prob,
        "progressive_layer_training": args.enable_progressive_layers,
        "seq_id_dropout_rate": args.seq_id_dropout_rate,
        "id_vocab_threshold": args.id_vocab_threshold,
        "shrinkage": args.shrinkage,
        "cross_network_layers": args.cross_network_layers,
        "num_codes": args.num_codes,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "sinkhorn_iter": args.sinkhorn_iter,
        "min_mass_ratio": args.min_mass_ratio,
        "coherence_threshold": args.coherence_threshold,
        "kappa_base": args.kappa_base,
        "kappa_min": args.kappa_min,
        "lie_rank": args.lie_rank,
        "use_domain_adversarial": args.use_domain_adversarial,
        "use_generative_fusion": args.use_generative_fusion,
    }

    model = PCVRHeteroFormer(**model_args).to(args.device)

    if args.compile_suppress_errors:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        logging.info("Dynamo suppress_errors enabled")

    if args.device == 'cuda' and hasattr(torch, 'compile'):
        try:
            compile_kwargs = {
                "backend": args.compile_backend,
                "fullgraph": False,
                "dynamic": args.compile_dynamic,
                "mode": args.compile_mode,
            }
            model = torch.compile(model, **compile_kwargs)
            logging.info(f"torch.compile enabled: {compile_kwargs}")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}, falling back to eager mode")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for n, p in model.named_parameters()
                     if 'embedding' in n or 'emb' in n.lower())
    dense_params = total_params - emb_params

    logging.info(f"PCVRHeteroFormer v9.1 (NaN-Resilient)")
    logging.info(f"Total parameters: {total_params:,} | Embedding: {emb_params:,} | Dense: {dense_params:,}")

    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_layers,
        "head": args.num_heads or max(4, args.d_model // 32),
        "hidden": args.d_model,
    }

    trainer = PCVRHeteroFormerTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        warmup_steps=args.warmup_steps,
        use_adaptive_focal=args.use_adaptive_focal,
        enable_progressive_layers=args.enable_progressive_layers,
        focal_alpha_pos=args.focal_alpha_pos,
        focal_alpha_neg=args.focal_alpha_neg,
        focal_max_gamma=args.focal_max_gamma,
        global_ctr=args.global_ctr,
        use_diffusion=args.use_diffusion,
        use_energy=args.use_energy,
        use_domain_adversarial=args.use_domain_adversarial,
        packing_weight=args.packing_weight,
        energy_margin=args.energy_margin,
        energy_weight=args.energy_weight,
        diff_weight=args.diff_weight,
        meta_update_interval=args.meta_update_interval,
        curriculum_warmup=args.curriculum_warmup,
        diffusion_warmup=args.diffusion_warmup,
    )

    trainer.train()
    if writer:
        writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()