# ==================== train.py (TokenFormer DEFUSE 完整版) ====================

"""PCVRTokenFormer Training Entry Point with DEFUSE continuous labels."""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRTokenFormer
from trainer import PCVRTokenFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TokenFormer DEFUSE Training")

    # Paths
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_accumulation_steps', type=int, default=4)  # 8→4
    parser.add_argument('--lr', type=float, default=1e-4)  # 1e-4→5e-5
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Data pipeline
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--buffer_batches', type=int, default=50)  # 10→50
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--eval_every_n_steps', type=int, default=500)
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')

    # 【核心修改】采样策略：默认不采样
    parser.add_argument('--neg_downsample_ratio', type=float, default=None)  # None=不采样
    parser.add_argument('--pos_upsample_ratio', type=float, default=None)    # None=不过采样
    parser.add_argument('--min_neg_per_batch', type=int, default=0)
    parser.add_argument('--min_batch_size_ratio', type=float, default=0.0)

    # 【核心新增】DEFUSE 标签模式
    parser.add_argument('--label_mode', type=str, default='defuse',
                        choices=['hard', 'soft_time', 'defuse', 'multi_task'])
    parser.add_argument('--soft_label_decay_hours', type=float, default=24.0)

    # Model architecture
    parser.add_argument('--d_model', type=int, default=128)  # 64→128
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--num_queries', type=int, default=4)  # 1→4
    parser.add_argument('--num_blocks', type=int, default=3)  # 4→3
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_mult', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)  # 0.01→0.1
    parser.add_argument('--action_num', type=int, default=1)
    parser.add_argument('--use_time_buckets', action='store_true', default=True)
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false')
    parser.add_argument('--use_rope', action='store_true', default=True)  # False→True
    parser.add_argument('--rope_base', type=float, default=10000.0)

    # BFTS / NLIR / Adaptive SWA
    parser.add_argument('--bfts_l_full', type=int, default=2)
    parser.add_argument('--bfts_swa_window', type=int, default=50)
    parser.add_argument('--use_nlir', action='store_true', default=True)
    parser.add_argument('--no_nlir', dest='use_nlir', action='store_false')
    parser.add_argument('--adaptive_swa_ratio', type=float, default=0.2)

    # Anti-drift & time modeling
    parser.add_argument('--ns_token_dropout', type=float, default=0.1)
    parser.add_argument('--use_time_diff', action='store_true', default=False)
    parser.add_argument('--use_time_decay', action='store_true', default=True)
    parser.add_argument('--time_decay_factor', type=float, default=86400.0)  # 30天→1天
    parser.add_argument('--ns_summary_tokens', type=int, default=4)

    # Multi-task & hard negative
    parser.add_argument('--predict_conversion_time', action='store_true', default=False)
    parser.add_argument('--conversion_time_weight', type=float, default=0.1)
    parser.add_argument('--use_hard_negative', action='store_true', default=False)
    parser.add_argument('--hard_neg_ratio', type=float, default=0.3)
    parser.add_argument('--use_gauc', action='store_true', default=True)
    parser.add_argument('--multi_task', action='store_true', default=True)  # 【新增】默认开启
    parser.add_argument('--ctr_weight', type=float, default=0.3)
    parser.add_argument('--ctcvr_weight', type=float, default=0.2)

    # 类别权重：默认关闭
    parser.add_argument('--use_class_weight', action='store_true', default=False)  # True→False
    parser.add_argument('--pos_weight', type=float, default=None)  # 15.0→None
    parser.add_argument('--pos_weight_update_steps', type=int, default=100)

    # Cross-domain interaction
    parser.add_argument('--use_cross_domain', action='store_true', default=False)

    # Loss 组合
    parser.add_argument('--loss_type', type=str, default='bce',
                    choices=['bce', 'focal'],
                    help='Compatibility parameter: not used when loss_components is specified')
    parser.add_argument('--loss_components', type=str, default='bce,continuous_pairwise')
    parser.add_argument('--loss_weights', type=str, default='0.7,0.3')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Sparse optimizer
    parser.add_argument('--sparse_lr', type=float, default=0.01)  # 0.05→0.01
    parser.add_argument('--sparse_weight_decay', type=float, default=1e-5)  # 0.0→1e-5
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1)
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=10000)

    # Embedding control
    parser.add_argument('--emb_skip_threshold', type=int, default=10000)
    parser.add_argument('--seq_id_threshold', type=int, default=10000)

    # NS tokenizer
    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups)
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'])
    parser.add_argument('--user_ns_tokens', type=int, default=0)
    parser.add_argument('--item_ns_tokens', type=int, default=0)

    # GPU optimization
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--compile', action='store_true', default=False)

    # EMA & global NS
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_ema', action='store_true', default=False)
    parser.add_argument('--compute_global_ns', action='store_true', default=False)
    parser.add_argument('--cache_ns', action='store_true', default=False)

    # 【新增】学习率调度
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)

    # 【新增】评估指标
    parser.add_argument('--primary_metric', type=str, default='auc', choices=['auc', 'gauc', 'logloss'])

    args = parser.parse_args()

    # Environment variables
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    args.ema_enabled = not args.no_ema and args.ema_decay > 0
    args.amp_dtype_obj = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    return args


def main() -> None:
    args = parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using dataset: DEFUSE continuous labels + user-level sampling")
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
        time_decay_factor=args.time_decay_factor,
        use_time_decay=args.use_time_decay,
        neg_downsample_ratio=args.neg_downsample_ratio,
        pos_upsample_ratio=args.pos_upsample_ratio,
        min_neg_per_batch=args.min_neg_per_batch,
        min_batch_size_ratio=args.min_batch_size_ratio,
        label_mode=args.label_mode,
        soft_label_decay_hours=args.soft_label_decay_hours,
    )

    # 打印采样统计
    if hasattr(pcvr_dataset, 'get_sampling_stats'):
        stats = pcvr_dataset.get_sampling_stats()
        logging.info(f"Sampling stats: {stats}")

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        logging.info("No NS groups JSON found, using default")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model = PCVRTokenFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=pcvr_dataset.user_dense_schema.total_dim,
        item_dense_dim=pcvr_dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=pcvr_dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        num_queries=args.num_queries,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        hidden_mult=args.hidden_mult,
        dropout_rate=args.dropout_rate,
        action_num=args.action_num,
        num_time_buckets=NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        emb_skip_threshold=args.emb_skip_threshold,
        seq_id_threshold=args.seq_id_threshold,
        ns_tokenizer_type=args.ns_tokenizer_type,
        user_ns_tokens=args.user_ns_tokens,
        item_ns_tokens=args.item_ns_tokens,
        bfts_l_full=args.bfts_l_full,
        bfts_swa_window=args.bfts_swa_window,
        use_nlir=args.use_nlir,
        ns_token_dropout=args.ns_token_dropout,
        use_time_diff=args.use_time_diff,
        use_time_decay=args.use_time_decay,
        ns_summary_tokens=args.ns_summary_tokens,
        cache_ns_during_training=args.cache_ns,
        seq_max_lens=seq_max_lens,
        use_cross_domain=args.use_cross_domain,
        predict_conversion_time=args.predict_conversion_time,
        multi_task=args.multi_task,  # 【新增】
        num_action_types=3,  # 0=曝光, 1=点击, 2=转化
    ).to(args.device)

    # torch.compile
    if args.compile and hasattr(torch, 'compile'):
        logging.info("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=False)
        logging.info("Model compiled")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"PCVRTokenFormer DEFUSE created:")
    logging.info(f"  - num_ns={model.num_ns}, d_model={args.d_model}, blocks={args.num_blocks}")
    logging.info(f"  - num_queries={args.num_queries} (多任务)")
    logging.info(f"  - static_masks={model._has_static_masks}")
    logging.info(f"  - cross_domain={args.use_cross_domain}")
    logging.info(f"  - time_decay={args.use_time_decay}")
    logging.info(f"  - predict_conversion_time={args.predict_conversion_time}")
    logging.info(f"  - multi_task={args.multi_task}")
    logging.info(f"  - label_mode={args.label_mode}")
    logging.info(f"  - neg_downsample={args.neg_downsample_ratio}")
    logging.info(f"  - pos_upsample={args.pos_upsample_ratio}")
    logging.info(f"  - Total parameters: {total_params:,}")
    logging.info(f"  - Effective batch size: {args.batch_size * args.grad_accumulation_steps}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRTokenFormerRankingTrainer(
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
        train_config={k: v for k, v in vars(args).items() if not isinstance(v, torch.dtype)},
        use_amp=args.amp,
        amp_dtype=args.amp_dtype_obj,
        ema_decay=args.ema_decay,
        ema_enabled=args.ema_enabled,
        compute_global_ns=args.compute_global_ns,
        global_ns_path=os.path.join(args.ckpt_dir, 'global_ns.pt') if args.compute_global_ns else None,
        grad_accumulation_steps=args.grad_accumulation_steps,
        use_hard_negative=args.use_hard_negative,
        hard_neg_ratio=args.hard_neg_ratio,
        use_gauc=args.use_gauc,
        predict_conversion_time=args.predict_conversion_time,
        conversion_time_weight=args.conversion_time_weight,
        pos_weight=args.pos_weight,
        use_class_weight=args.use_class_weight,
        pos_weight_update_steps=args.pos_weight_update_steps,
        # 【新增】Loss 组合
        loss_components=args.loss_components,
        loss_weights=args.loss_weights,
        multi_task=args.multi_task,
        ctr_weight=args.ctr_weight,
        ctcvr_weight=args.ctcvr_weight,
        # 【新增】学习率调度
        use_lr_scheduler=args.use_lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        # 【新增】评估指标
        primary_metric=args.primary_metric,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()