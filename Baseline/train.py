"""PCVRHyFormer training entry point (Scaling + Curriculum + Muon + Delay Feedback)."""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: List[int]) -> List[Tuple[int, int, int]]:
    specs = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training (Delay-Aware Curriculum + Muon)")

    # Paths
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--schema_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5, help='AdamW learning rate (Muon lr defaults to lr*0.1)')
    parser.add_argument('--num_epochs', type=int, default=999)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--buffer_batches', type=int, default=50)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--eval_every_n_steps', type=int, default=0)
    parser.add_argument('--seq_max_lens', type=str, default='seq_a:256,seq_b:256,seq_c:512,seq_d:512')

    # Model (Scaling)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--num_hyformer_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--seq_encoder_type', type=str, default='longer',
                        choices=['swiglu', 'transformer', 'longer'])
    parser.add_argument('--hidden_mult', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)  # FIX: 0.01->0.1 抑制过拟合尖峰
    parser.add_argument('--seq_top_k', type=int, default=100)  # Increased from 50
    parser.add_argument('--seq_causal', action='store_true', default=False)
    parser.add_argument('--action_num', type=int, default=1)
    parser.add_argument('--use_time_buckets', action='store_true', default=True)
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false')
    parser.add_argument('--rank_mixer_mode', type=str, default='ffn_only',  # 保持 ffn_only 避免启动报错；run.sh 中可覆盖为 full
                        choices=['full', 'ffn_only', 'none'])
    parser.add_argument('--use_rope', action='store_true', default=False)
    parser.add_argument('--rope_base', type=float, default=10000.0)
    parser.add_argument('--use_cross_seq_attn', action='store_true', default=True)
    parser.add_argument('--use_continuous_time', action='store_true', default=True)

    # Loss & Curriculum
    parser.add_argument('--stage1_epochs', type=int, default=2,  # FIX: 先短warmup再进Focal
                        help='Number of pure delay-aware BCE epochs (Stage 1)')
    parser.add_argument('--stage2_epochs', type=int, default=5,
                        help='Number of delay-aware BCE+Focal epochs (Stage 2)')
    parser.add_argument('--info_nce_weight_stage3', type=float, default=0.05,
                        help='λ for Focal‑InfoNCE in Stage 3')
    parser.add_argument('--info_nce_temperature_stage3', type=float, default=0.5,
                        help='Temperature for Focal‑InfoNCE in Stage 3')
    parser.add_argument('--stage3_lr_factor', type=float, default=0.5,
                        help='Learning rate decay factor at Stage 3 start')
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Delay-feedback specific
    parser.add_argument('--delay_loss_type', type=str, default='bce',  # FIX: esdfm->bce 先稳定复现
                        choices=['bce', 'esdfm', 'fncw'])
    parser.add_argument('--delay_decay_window', type=float, default=604800.0,
                        help='Fake negative decay window in seconds (default: 7 days)')
    parser.add_argument('--dynamic_neg_sampling', action='store_true', default=False)  # FIX: 默认关闭，避免user_id错位风险
    parser.add_argument('--neg_sample_ratio', type=float, default=3.0)
    parser.add_argument('--split_by_time', action='store_true', default=True,
                        help='Time-ordered train/valid split')

    # Muon specific
    parser.add_argument('--muon_lr', type=float, default=None,
                        help='Muon learning rate (default: lr * 0.1)')
    parser.add_argument('--muon_momentum', type=float, default=0.95)
    parser.add_argument('--muon_weight_decay', type=float, default=0.1)

    # Sparse optimizer
    parser.add_argument('--sparse_lr', type=float, default=0.01)  # FIX: 0.05->0.01 降低embedding抖动
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0)
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1)
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0)

    # Embedding
    parser.add_argument('--emb_skip_threshold', type=int, default=0)
    parser.add_argument('--seq_id_threshold', type=int, default=10000)

    # NS groups
    _default_ns = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns)
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'])
    parser.add_argument('--user_ns_tokens', type=int, default=0)
    parser.add_argument('--item_ns_tokens', type=int, default=0)

    args = parser.parse_args()
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')
    return args


def main():
    args = parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")
    writer = SummaryWriter(args.tf_events_dir)

    schema_path = args.schema_path or os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens: {seq_max_lens}")

    # ===== Delay-feedback aware data loading =====
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
        split_by_time=args.split_by_time,
        delay_decay_window=args.delay_decay_window,
        dynamic_negative_sampling=args.dynamic_neg_sampling,
        neg_sample_ratio=args.neg_sample_ratio,
    )

    # NS groups
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
    else:
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

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
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_cross_seq_attn": args.use_cross_seq_attn,
        "use_continuous_time": args.use_continuous_time,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: total params {total_params:,}")

    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {"layer": args.num_hyformer_blocks, "head": args.num_heads, "hidden": args.d_model}

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type='curriculum',
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
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        info_nce_weight_stage3=args.info_nce_weight_stage3,
        info_nce_temperature_stage3=args.info_nce_temperature_stage3,
        stage3_lr_factor=args.stage3_lr_factor,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        accumulation_steps=args.accumulation_steps,
        # ===== Delay-feedback parameters =====
        delay_loss_type=args.delay_loss_type,
        delay_decay_window=args.delay_decay_window,
        # =====================================
    )

    trainer.train()
    writer.close()
    logging.info("Training complete!")


if __name__ == "__main__":
    main()