"""PCVRHeteroFormer trainer — v7.1 Architecture-Preserving GSE Edition."""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import EarlyStopping
from model import ModelInput
import math


class AdaptiveFocalLoss(nn.Module):
    def __init__(
        self,
        alpha_pos: float = 0.5,
        alpha_neg: float = 0.5,
        max_gamma: float = 4.0,
        gamma_lr: float = 0.05,
        gamma_clip: float = 5.0,
        gamma_warmup_steps: int = 1000,
        gamma_min: float = 0.5,
        gamma_reg_coef: float = 0.05,
    ):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.max_gamma = max_gamma
        self.gamma_lr = gamma_lr
        self.gamma_clip = gamma_clip
        self.gamma_warmup_steps = gamma_warmup_steps
        self.gamma_min = gamma_min
        self.gamma_reg_coef = gamma_reg_coef

        self.gamma_logit = nn.Parameter(torch.tensor(0.0))
        self.gamma_optimizer = torch.optim.Adam([self.gamma_logit], lr=gamma_lr)
        self._gamma_history = []
        self._step_count = 0

    def get_gamma(self) -> torch.Tensor:
        learned = torch.sigmoid(self.gamma_logit) * self.max_gamma
        learned = torch.clamp(learned, min=self.gamma_min)
        if self._step_count < self.gamma_warmup_steps:
            progress = self._step_count / max(self.gamma_warmup_steps, 1)
            warmup_gamma = self.gamma_min + (learned - self.gamma_min) * progress
            return warmup_gamma
        return learned

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gamma = self.get_gamma()
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=1e-7, max=1-1e-7)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_weight = (1 - p_t) ** gamma
        focal_weight = torch.clamp(focal_weight, max=10.0)
        
        alpha_t = targets * self.alpha_pos + (1 - targets) * self.alpha_neg
        focal_loss = (alpha_t * focal_weight * bce_loss).mean()

        gamma_reg = self.gamma_reg_coef * torch.abs(gamma - self.gamma_min) / (self.max_gamma - self.gamma_min + 1e-8)
        loss = focal_loss + gamma_reg

        self._gamma_history.append(gamma.item())
        if len(self._gamma_history) > 1000:
            self._gamma_history = self._gamma_history[-1000:]

        return loss

    def step(self) -> None:
        self._step_count += 1
        if self.gamma_logit.grad is not None:
            torch.nn.utils.clip_grad_value_(self.gamma_logit, self.gamma_clip)
        self.gamma_optimizer.step()
        self.gamma_optimizer.zero_grad()


class GraftedOptimizer:
    """Preserved from v5.2."""
    def __init__(
        self,
        sparse_optimizer: torch.optim.Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        gate_optimizer: Optional[torch.optim.Optimizer] = None,
        target_ratio: float = 1.0,
        adapt_interval: int = 100,
    ):
        self.sparse_opt = sparse_optimizer
        self.dense_opt = dense_optimizer
        self.gate_opt = gate_optimizer
        self.target_ratio = target_ratio
        self.adapt_interval = adapt_interval
        self.step_count = 0
        self._sparse_norm_history = []
        self._dense_norm_history = []

    def zero_grad(self) -> None:
        self.sparse_opt.zero_grad()
        self.dense_opt.zero_grad()
        if self.gate_opt is not None:
            self.gate_opt.zero_grad()

    def step(self, model: nn.Module) -> Dict[str, float]:
        self.step_count += 1

        sparse_norms = []
        dense_norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                if 'embedding' in name or 'emb' in name.lower():
                    sparse_norms.append(p.grad.norm())
                else:
                    dense_norms.append(p.grad.norm())
        
        sparse_norm = torch.stack(sparse_norms).mean().item() if sparse_norms else 0.0
        dense_norm = torch.stack(dense_norms).mean().item() if dense_norms else 0.0

        self._sparse_norm_history.append(sparse_norm)
        self._dense_norm_history.append(dense_norm)

        diag = {
            'sparse_grad_norm': sparse_norm,
            'dense_grad_norm': dense_norm,
            'grad_norm_ratio': sparse_norm / (dense_norm + 1e-8),
        }

        if self.step_count % self.adapt_interval == 0:
            avg_sparse = np.mean(self._sparse_norm_history[-self.adapt_interval:])
            avg_dense = np.mean(self._dense_norm_history[-self.adapt_interval:])
            ratio = avg_sparse / (avg_dense + 1e-8)

            if ratio > self.target_ratio * 5.0:
                for g in self.sparse_opt.param_groups:
                    g['lr'] *= 0.9
                diag['action'] = 'reduced_sparse_lr'
            elif ratio < self.target_ratio * 0.2:
                for g in self.dense_opt.param_groups:
                    g['lr'] *= 1.1
                diag['action'] = 'increased_dense_lr'
            else:
                diag['action'] = 'balanced'

        if len(self._sparse_norm_history) >= self.adapt_interval:
            avg_sparse = np.mean(self._sparse_norm_history[-self.adapt_interval:])
            if sparse_norm > 5.0 * avg_sparse:
                for group in self.sparse_opt.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.mul_(avg_sparse / (sparse_norm + 1e-8))
                diag['sparse_grad_rescaled'] = True

        self.sparse_opt.step()
        self.dense_opt.step()
        if self.gate_opt is not None:
            self.gate_opt.step()

        return diag

    def state_dict(self):
        d = {
            'sparse': self.sparse_opt.state_dict(),
            'dense': self.dense_opt.state_dict(),
            'step_count': self.step_count,
        }
        if self.gate_opt is not None:
            d['gate'] = self.gate_opt.state_dict()
        return d

    def load_state_dict(self, state_dict):
        self.sparse_opt.load_state_dict(state_dict['sparse'])
        self.dense_opt.load_state_dict(state_dict['dense'])
        if 'gate' in state_dict and self.gate_opt is not None:
            self.gate_opt.load_state_dict(state_dict['gate'])
        self.step_count = state_dict.get('step_count', 0)


def calibration_loss(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Soft Expected Calibration Error (ECE)"""
    with torch.no_grad():
        probs = torch.sigmoid(logits.detach())
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        bin_ids = torch.bucketize(probs, bin_boundaries) - 1
        bin_ids = bin_ids.clamp(0, n_bins - 1)
        bin_sums = torch.zeros(n_bins, device=logits.device)
        bin_counts = torch.zeros(n_bins, device=logits.device)
        bin_true = torch.zeros(n_bins, device=logits.device)
        for i in range(n_bins):
            mask = (bin_ids == i)
            bin_counts[i] = mask.sum()
            if bin_counts[i] > 0:
                bin_sums[i] = probs[mask].sum()
                bin_true[i] = targets[mask].sum()
    valid = bin_counts > 10
    if valid.sum() > 0:
        avg_pred = bin_sums[valid] / bin_counts[valid]
        avg_true = bin_true[valid] / bin_counts[valid]
        weights = bin_counts[valid] / bin_counts[valid].sum()
        ece = (torch.abs(avg_pred - avg_true) * weights).sum()
        return ece
    else:
        return torch.tensor(0.0, device=logits.device)


def lambdarank_loss(logits: torch.Tensor, targets: torch.Tensor, n_pairs: int = 100) -> torch.Tensor:
    """Pairwise hinge loss for AUC optimization."""
    pos_mask = targets == 1
    neg_mask = targets == 0
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, device=logits.device)
    n = min(n_pairs, len(pos_idx) * len(neg_idx))
    p_idx = pos_idx[torch.randint(0, len(pos_idx), (n,), device=logits.device)]
    n_idx = neg_idx[torch.randint(0, len(neg_idx), (n,), device=logits.device)]
    diff = logits[p_idx] - logits[n_idx]
    loss = F.softplus(-diff).mean()
    return loss

class PCVRHeteroFormerTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        warmup_steps: int = 0,
        gate_anneal_steps: int = 2000,
        use_grafted_optimizer: bool = True,
        use_adaptive_focal: bool = True,
        enable_progressive_layers: bool = False,
        stochastic_depth_prob: float = 0.0,
        label_smoothing_strategy: str = 'hybrid',
        label_smoothing_max_eps: float = 0.05,
        label_smoothing_min_eps: float = 0.001,
        label_smoothing_anneal_steps: int = 5000,
        gse_aux_weight: float = 0.5,  # NEW: GSE auxiliary loss weight
        # 新增不对称 focal 参数
        focal_alpha_pos: float = 0.5,
        focal_alpha_neg: float = 0.5,
        focal_max_gamma: float = 4.0,
        # 新增先验权重、校准权重、lambda权重
        prior_weight: float = 0.001,
        ece_weight: float = 0.05,
        lambdarank_weight: float = 0.1,
        global_ctr: float = 0.01,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path
        self.train_config = train_config
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.use_grafted_optimizer = use_grafted_optimizer
        self.use_adaptive_focal = use_adaptive_focal
        self.enable_progressive_layers = enable_progressive_layers
        self.gse_aux_weight = gse_aux_weight

        self.label_smoothing_strategy = label_smoothing_strategy
        self.label_smoothing_max_eps = label_smoothing_max_eps
        self.label_smoothing_min_eps = label_smoothing_min_eps
        self.label_smoothing_anneal_steps = label_smoothing_anneal_steps
        self.pos_rate_ema = 0.01
        self.prior_weight = prior_weight
        self.ece_weight = ece_weight
        self.lambdarank_weight = lambdarank_weight
        self.global_ctr = global_ctr

        if hasattr(model, 'get_sparse_params') and use_grafted_optimizer:
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            gate_params = model.get_gate_params() if hasattr(model, 'get_gate_params') else []
            scale_params = model.get_scale_params() if hasattr(model, 'get_scale_params') else []

            sparse_opt = torch.optim.Adagrad(sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            scale_ids = {id(p) for p in scale_params}
            dense_param_groups = [
                {'params': [p for p in dense_params if id(p) not in scale_ids]}, 
                {'params': scale_params, 'lr': 1e-6, 'weight_decay': 0.0}
            ]
            dense_opt = torch.optim.AdamW(dense_param_groups, lr=lr, betas=(0.9, 0.98))
            gate_opt = torch.optim.AdamW(gate_params, lr=lr*2, betas=(0.9, 0.98), weight_decay=1e-3) if gate_params else None
            self.optimizer = GraftedOptimizer(sparse_opt, dense_opt, gate_opt, target_ratio=1.0,adapt_interval=1000000)
            self.sparse_optimizer = None
            self.dense_optimizer = None
        elif hasattr(model, 'get_sparse_params'):
            self.sparse_optimizer = torch.optim.Adagrad(
                model.get_sparse_params(), lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer = torch.optim.AdamW(
                model.get_dense_params(), lr=lr, betas=(0.9, 0.98)
            )
            self.optimizer = None
        else:
            self.dense_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
            self.sparse_optimizer = None
            self.optimizer = None

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.loss_type = loss_type
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps

        self._emb_grad_history: Dict[int, list] = {}
        self._emb_death_threshold = 500

        self.use_lambda_rank = (lambdarank_weight > 0.99)
        self.lambda_n_pairs = 500               # 增加 pair 数
        self.bce_aux_weight = 0.4              # 弱 BCE 辅助权重

        # LambdaRank 模式下不需要 AdaptiveFocalLoss，避免额外参数和干扰
        if use_adaptive_focal and loss_type == 'focal' and not self.use_lambda_rank:
            self.adaptive_focal = AdaptiveFocalLoss(
                alpha_pos=focal_alpha_pos,
                alpha_neg=focal_alpha_neg,
                max_gamma=focal_max_gamma,
                gamma_lr=0.05,
                gamma_warmup_steps=1000,
                gamma_min=0.5,
                gamma_reg_coef=0.05,
            )
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
            logging.info(  
                f"v7.1-fix AdaptiveFocalLoss: max_gamma={focal_max_gamma}, "
                f"alpha_pos={focal_alpha_pos}, alpha_neg={focal_alpha_neg}, gamma_min=0.5"
                )
        else:
            self.adaptive_focal = None
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma

        if warmup_steps > 0:
            logging.info(f"Warmup: {warmup_steps} steps")

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing_strategy == 'none':
            eps = self.label_smoothing_max_eps
            return labels * (1 - 2 * eps) + eps

        if self.use_adaptive_focal and self.adaptive_focal is not None:
            gamma = self.adaptive_focal.get_gamma().item()
            if gamma > 1.0:
                return labels

        with torch.no_grad():
            pos_ratio = labels.float().mean().item()
            self.pos_rate_ema = 0.99 * self.pos_rate_ema + 0.01 * pos_ratio

        if self.label_smoothing_strategy == 'hybrid':
            eps = self.pos_rate_ema * 2.0
            eps = max(self.label_smoothing_min_eps, min(self.label_smoothing_max_eps, eps))
        elif self.label_smoothing_strategy == 'anneal':
            progress = min(self.global_step / max(self.label_smoothing_anneal_steps, 1), 1.0)
            eps = self.label_smoothing_max_eps * (1 - progress) + self.label_smoothing_min_eps * progress
        else:
            eps = self.label_smoothing_max_eps

        if self.use_adaptive_focal and self.adaptive_focal is not None:
            gamma = self.adaptive_focal.get_gamma().item()
            if 0.5 < gamma <= 1.0:
                eps = eps * 0.5

        return labels * (1 - 2 * eps) + eps

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
        if self.train_config:
            import json
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(self.train_config, f, indent=2)

    def _save_step_checkpoint(self, global_step: int, is_best: bool = False, skip_model_file: bool = False) -> str:
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)

    def _handle_validation_result(self, total_step: int, val_auc: float, val_logloss: float) -> None:
        old_best = self.early_stopping.best_score
        is_likely_new_best = old_best is None or val_auc > old_best + self.early_stopping.delta
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {"best_val_AUC": val_auc, "best_val_logloss": val_logloss})
            return

        best_dir = os.path.join(self.save_dir, self._build_step_dir_name(total_step, is_best=True))
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()
        self.early_stopping(val_auc, self.model, {"best_val_AUC": val_auc, "best_val_logloss": val_logloss})

        if self.early_stopping.best_score != old_best and os.path.exists(self.early_stopping.checkpoint_path):
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        seq_domains = device_batch['_seq_domains']
        seq_data, seq_lens, seq_time_buckets, seq_decay_weights = {}, {}, {}, {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B, _, L = device_batch[domain].shape
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device)
            )
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

    def _smart_reinit_embeddings(self) -> None:
        if self.reinit_cardinality_threshold <= 0:
            return
        reinit_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Embedding,)) and module.num_embeddings > self.reinit_cardinality_threshold:
                ptr = module.weight.data_ptr()
                if ptr in self._emb_grad_history:
                    recent_grads = self._emb_grad_history[ptr][-self._emb_death_threshold:]
                    avg_grad = np.mean(recent_grads) if recent_grads else 1.0
                    if avg_grad < 1e-6:
                        nn.init.normal_(module.weight, std=0.01)
                        reinit_count += 1
                else:
                    self._emb_grad_history[ptr] = []
        if reinit_count > 0:
            logging.info(f"Smart reinit: {reinit_count} embeddings")
            if hasattr(self.model, 'get_sparse_params'):
                sparse_params = self.model.get_sparse_params()
                if self.optimizer is not None:
                    self.optimizer.sparse_opt = torch.optim.Adagrad(
                        sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                    )
                elif self.sparse_optimizer is not None:
                    self.sparse_optimizer = torch.optim.Adagrad(
                        sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                    )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        # LambdaRank 使用原始 0/1 标签，不进行平滑
        if not self.use_lambda_rank:
            label = self._smooth_labels(label)

        # 梯度清零
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
        # 仅在 Focal Loss 模式下才需要清除 gamma_optimizer 的梯度
        if not self.use_lambda_rank and self.adaptive_focal is not None:
            self.adaptive_focal.gamma_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input).squeeze(-1)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning(f"NaN/Inf in logits at step {self.global_step}, skipping entire batch")
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.dense_optimizer.zero_grad()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad()
            return float('nan')

        # ================================================================
        # 主损失：LambdaRank (AUC 优化) + 弱 BCE 辅助 或 Focal Loss
        # ================================================================
        if self.use_lambda_rank:
            # 核心排序损失
            loss = lambdarank_loss(logits, label, n_pairs=self.lambda_n_pairs)
            # 微弱的 BCE 辅助，仅用于提供稳定的 logits 基线，防止均值漂移
            bce_aux = F.binary_cross_entropy_with_logits(logits, label, reduction='mean')
            loss = loss + self.bce_aux_weight * bce_aux
        else:
            if self.adaptive_focal is not None:
                loss = self.adaptive_focal(logits, label)
            elif self.loss_type == 'focal':
                p = torch.sigmoid(logits)
                p = torch.clamp(p, min=1e-7, max=1-1e-7)
                bce_loss = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
                p_t = p * label + (1 - p) * (1 - label)
                focal_weight = (1 - p_t) ** self.focal_gamma
                focal_weight = torch.clamp(focal_weight, max=10.0)
                alpha_t = self.focal_alpha * label + (1 - self.focal_alpha) * (1 - label)
                loss = (alpha_t * focal_weight * bce_loss).mean()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        # ================================================================
        # GSE 辅助损失 (大幅降权，仅作微弱正则)
        # ================================================================
        gse_aux_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'get_aux_loss'):
            gse_aux_loss = self.model.get_aux_loss()
        aux_weight = self.gse_aux_weight
        # LambdaRank 模式下进一步降低 GSE 权重，避免干扰排序目标
        if self.use_lambda_rank:
            aux_weight *= 0.1
        loss = loss + aux_weight * gse_aux_loss

        # ================================================================
        # NaN/Inf 防护
        # ================================================================
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss at step {self.global_step}, skipping batch")
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.dense_optimizer.zero_grad()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad()
            return float('nan')

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            main_loss = loss.item()
            gse_val = gse_aux_loss.item() if isinstance(gse_aux_loss, torch.Tensor) else gse_aux_loss
            detail = [f"total={main_loss:.4f}", f"gse_aux={gse_val:.4f}"]
            with torch.no_grad():
                logits_min = logits.min().item()
                logits_max = logits.max().item()
                logits_mean = logits.mean().item()
                logits_std = logits.std().item()
                detail.append(f"logits(min={logits_min:.2f},max={logits_max:.2f},mean={logits_mean:.2f},std={logits_std:.2f})")
            logging.warning(f"NaN/Inf loss detected before backward at step {self.global_step}, skipping. Details: " + " | ".join(detail))
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.dense_optimizer.zero_grad()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad()
            return float('nan')

        loss.backward()

        # === Gradient Collection (batched, preserved pattern) ===
        emb_grad_norms = []
        for name, p in self.model.named_parameters():
            if p.grad is not None and ('embedding' in name or 'emb' in name.lower()):
                ptr = p.data_ptr()
                g_norm = p.grad.norm()
                emb_grad_norms.append(g_norm)
                if ptr not in self._emb_grad_history:
                    self._emb_grad_history[ptr] = []
                self._emb_grad_history[ptr].append(g_norm.detach())
                if len(self._emb_grad_history[ptr]) > self._emb_death_threshold * 2:
                    self._emb_grad_history[ptr] = self._emb_grad_history[ptr][-self._emb_death_threshold:]
        total_emb_grad_norm = torch.stack(emb_grad_norms).norm().item() if emb_grad_norms else 0.0

        # === NaN Gradient Check ===
        has_nan_grad = False
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if grads:
            flat_grads = torch.cat([g.flatten()[:1000] for g in grads])
            has_nan_grad = torch.isnan(flat_grads).any() or torch.isinf(flat_grads).any()
        if has_nan_grad:
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.dense_optimizer.zero_grad()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad()
            return float('nan')

        # === Gradient Clipping ===
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5, foreach=False)

        self.global_step += 1
        if self.warmup_steps > 0 and self.global_step <= self.warmup_steps:
            warmup_factor = self.global_step / self.warmup_steps
            if self.optimizer is not None:
                for g in self.optimizer.dense_opt.param_groups:
                    g['lr'] = self.lr * warmup_factor
                for g in self.optimizer.sparse_opt.param_groups:
                    g['lr'] = self.sparse_lr * warmup_factor
                if self.optimizer.gate_opt is not None:
                    for g in self.optimizer.gate_opt.param_groups:
                        g['lr'] = self.lr * 2 * warmup_factor
            else:
                for g in self.dense_optimizer.param_groups:
                    g['lr'] = self.lr * warmup_factor
                if self.sparse_optimizer is not None:
                    for g in self.sparse_optimizer.param_groups:
                        g['lr'] = self.sparse_lr * warmup_factor

        # === Optimizer Step ===
        opt_diag = {}
        if self.optimizer is not None:
            opt_diag = self.optimizer.step(self.model)
        else:
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        # NaN 参数检查
        has_nan_param = any(
            torch.isnan(p).any() or torch.isinf(p).any()
            for p in self.model.parameters()
        )
        if has_nan_param:
            logging.error(f"NaN/Inf parameter detected after optimizer step {self.global_step}! Training broken.")

        # 仅在非 LambdaRank 模式下更新可学习的 focal gamma
        if not self.use_lambda_rank and self.adaptive_focal is not None:
            self.adaptive_focal.step()

        # === Logging ===
        if self.global_step % 500 == 0:
            log_parts = [f"[Step {self.global_step}] loss={loss.item():.4f}"]
            log_parts.append(f"gse_aux={gse_aux_loss.item():.4f}")
            with torch.no_grad():
                logits_mean = logits.mean()
                logits_std = logits.std()
                log_parts.append(f"logits_mean={logits_mean.item():.3f}")
                log_parts.append(f"logits_std={logits_std.item():.3f}")
                std_target = 0.8
                std_dev = abs(logits_std.item() - std_target)
                log_parts.append(f"std_dev={std_dev:.3f}(target={std_target})")
                try:
                    from sklearn.metrics import roc_auc_score
                    probs = torch.sigmoid(logits).cpu().numpy()
                    labels_np = label.cpu().numpy()
                    train_auc = roc_auc_score(labels_np, probs)
                    log_parts.append(f"train_AUC={train_auc:.4f}")
                except Exception:
                    pass
                if hasattr(self.model, 'output_scale_logit'):
                    scale = torch.sigmoid(self.model.output_scale_logit).item() * 2.0
                    log_parts.append(f"out_scale={scale:.3f}")
                log_parts.append(f"emb_grad_norm={total_emb_grad_norm:.3f}")
                log_parts.append(f"aux_w={aux_weight:.3f}")
            if not self.use_lambda_rank and self.adaptive_focal is not None:
                g = self.adaptive_focal.get_gamma().item()
                log_parts.append(f"gamma={g:.3f}")
            if opt_diag:
                log_parts.append(f"sparse/dense_ratio={opt_diag.get('grad_norm_ratio', 0):.2f}")
            if hasattr(self.model, 'get_diagnostics'):
                d = self.model.get_diagnostics()
                for k, v in d.items():
                    if 'null_gate' in k or 'usage_rate' in k or 'empty_ratio' in k:
                        log_parts.append(f"{k}={v:.3f}")
            logging.info(" | ".join(log_parts))

        # === TensorBoard ===
        if self.writer:
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.writer.add_scalar('Diagnostics/logits_mean', logits.mean().item(), self.global_step)
            self.writer.add_scalar('Diagnostics/logits_std', logits.std().item(), self.global_step)
            self.writer.add_scalar('GSE/aux_loss', gse_aux_loss.item(), self.global_step)
            self.writer.add_scalar('GSE/aux_weight', aux_weight, self.global_step)
            if not self.use_lambda_rank and self.adaptive_focal is not None:
                self.writer.add_scalar('Focal/gamma', self.adaptive_focal.get_gamma().item(), self.global_step)
            if hasattr(self.model, 'get_diagnostics'):
                d = self.model.get_diagnostics()
                for k, v in d.items():
                    if 'temperature' in k:
                        self.writer.add_scalar(f'GSE/{k}', v, self.global_step)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                device_batch = self._batch_to_device(batch)
                model_input = self._make_model_input(device_batch)
                logits, _ = self.model.predict(model_input)
                all_logits.append(logits.squeeze(-1).detach().cpu())
                all_labels.append(device_batch['label'].detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        auc = float(roc_auc_score(labels_np, probs)) if len(np.unique(labels_np)) >= 2 else 0.0

        valid_logits_all = all_logits[~torch.isnan(all_logits)]
        if self.writer and len(valid_logits_all) > 0:
            self.writer.add_scalar('Diagnostics/valid_logits_mean', valid_logits_all.mean().item(), self.global_step)
            self.writer.add_scalar('Diagnostics/valid_logits_std', valid_logits_all.std().item(), self.global_step)

        valid_logits = valid_logits_all
        valid_labels = all_labels[~torch.isnan(all_logits)]
        logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item() if len(valid_logits) > 0 else float('inf')

        return auc, logloss

    def train(self) -> None:
        logging.info("Start training HeteroFormer v7.1 (Architecture-Preserving GSE Edition)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            if self.enable_progressive_layers and hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch - 1)

            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=False)
            loss_sum = 0.0
            valid_steps = 0
            nan_count = 0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1

                if math.isnan(loss):
                    nan_count += 1
                else:
                    loss_sum += loss
                    valid_steps += 1

                gamma_str = f"{self.adaptive_focal.get_gamma().item():.2f}" if self.adaptive_focal else "N/A"
                train_pbar.set_postfix({"loss": f"{loss:.4f}", "gamma": gamma_str, "nan": nan_count})

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()
                    logging.info(f"Step {total_step} | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            avg_loss = loss_sum / max(valid_steps, 1)
            logging.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | NaN batches: {nan_count}")

            emb_norm = sum(p.norm().item() for n, p in self.model.named_parameters()
                           if 'embedding' in n or 'emb' in n.lower())
            dense_norm = sum(p.norm().item() for n, p in self.model.named_parameters()
                             if 'embedding' not in n and 'emb' not in n.lower())
            logging.info(f"Epoch {epoch} | EmbNorm: {emb_norm:.1f} | DenseNorm: {dense_norm:.1f} | Ratio: {emb_norm/(dense_norm+1e-8):.1f}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            if epoch >= self.reinit_sparse_after_epoch:
                self._smart_reinit_embeddings()