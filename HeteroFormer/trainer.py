"""PCVRHeteroFormer trainer — v7.3 Collaborative Multi-Objective Edition (+VQ collapse fix)."""

import os
import glob
import shutil
import logging
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

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
    """v7.3: Supports three-stream optimization with conflict-aware gradient scaling."""
    def __init__(
        self,
        sparse_optimizer: torch.optim.Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        gate_optimizer: Optional[torch.optim.Optimizer] = None,
        rank_optimizer: Optional[torch.optim.Optimizer] = None,
        calib_optimizer: Optional[torch.optim.Optimizer] = None,
        target_ratio: float = 1.0,
        adapt_interval: int = 100,
    ):
        self.sparse_opt = sparse_optimizer
        self.dense_opt = dense_optimizer
        self.gate_opt = gate_optimizer
        self.rank_opt = rank_optimizer
        self.calib_opt = calib_optimizer
        self.target_ratio = target_ratio
        self.adapt_interval = adapt_interval
        self.step_count = 0
        self._sparse_norm_history = []
        self._dense_norm_history = []
        self._rank_norm_history = []
        self._calib_norm_history = []

    def zero_grad(self) -> None:
        self.sparse_opt.zero_grad()
        self.dense_opt.zero_grad()
        if self.gate_opt is not None:
            self.gate_opt.zero_grad()
        if self.rank_opt is not None:
            self.rank_opt.zero_grad()
        if self.calib_opt is not None:
            self.calib_opt.zero_grad()

    def step(self, model: nn.Module) -> Dict[str, float]:
        self.step_count += 1

        sparse_norms, dense_norms, rank_norms, calib_norms = [], [], [], []
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                if 'rank_' in name or 'rank_predictor' in name:
                    rank_norms.append(p.grad.norm())
                elif 'calibrator' in name:
                    calib_norms.append(p.grad.norm())
                elif 'embedding' in name or 'emb' in name.lower():
                    sparse_norms.append(p.grad.norm())
                else:
                    dense_norms.append(p.grad.norm())
        
        sparse_norm = torch.stack(sparse_norms).mean().item() if sparse_norms else 0.0
        dense_norm = torch.stack(dense_norms).mean().item() if dense_norms else 0.0
        rank_norm = torch.stack(rank_norms).mean().item() if rank_norms else 0.0
        calib_norm = torch.stack(calib_norms).mean().item() if calib_norms else 0.0

        self._sparse_norm_history.append(sparse_norm)
        self._dense_norm_history.append(dense_norm)
        self._rank_norm_history.append(rank_norm)
        self._calib_norm_history.append(calib_norm)

        diag = {
            'sparse_grad_norm': sparse_norm,
            'dense_grad_norm': dense_norm,
            'rank_grad_norm': rank_norm,
            'calib_grad_norm': calib_norm,
            'grad_norm_ratio': sparse_norm / (dense_norm + 1e-8),
        }

        if self.step_count % self.adapt_interval == 0:
            avg_sparse = np.mean(self._sparse_norm_history[-self.adapt_interval:])
            avg_dense = np.mean(self._dense_norm_history[-self.adapt_interval:])
            avg_rank = np.mean(self._rank_norm_history[-self.adapt_interval:]) if rank_norms else 0
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

            if avg_rank > 0 and avg_rank < avg_dense * 0.1:
                for g in self.rank_opt.param_groups:
                    g['lr'] *= 1.2
                diag['rank_action'] = 'boosted_rank_lr'

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
        if self.rank_opt is not None:
            self.rank_opt.step()
        if self.calib_opt is not None:
            self.calib_opt.step()

        return diag

    def state_dict(self):
        d = {
            'sparse': self.sparse_opt.state_dict(),
            'dense': self.dense_opt.state_dict(),
            'step_count': self.step_count,
        }
        if self.gate_opt is not None:
            d['gate'] = self.gate_opt.state_dict()
        if self.rank_opt is not None:
            d['rank'] = self.rank_opt.state_dict()
        if self.calib_opt is not None:
            d['calib'] = self.calib_opt.state_dict()
        return d

    def load_state_dict(self, state_dict):
        self.sparse_opt.load_state_dict(state_dict['sparse'])
        self.dense_opt.load_state_dict(state_dict['dense'])
        if 'gate' in state_dict and self.gate_opt is not None:
            self.gate_opt.load_state_dict(state_dict['gate'])
        if 'rank' in state_dict and self.rank_opt is not None:
            self.rank_opt.load_state_dict(state_dict['rank'])
        if 'calib' in state_dict and self.calib_opt is not None:
            self.calib_opt.load_state_dict(state_dict['calib'])
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
        gse_aux_weight: float = 0.5,
        focal_alpha_pos: float = 0.5,
        focal_alpha_neg: float = 0.5,
        focal_max_gamma: float = 4.0,
        prior_weight: float = 0.001,
        ece_weight: float = 0.05,
        lambdarank_weight: float = 0.1,
        global_ctr: float = 0.01,
        zmlc_lambda: float = 0.1,
        zmlc_on_calib_only: bool = True,
        rank_lr_multiplier: float = 2.0,
        calib_lr_multiplier: float = 1.0,
        enable_grad_conflict_check: bool = True,
        loss_conflict_threshold: float = 0.8,
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
        self.pos_rate_ema = global_ctr  # 正样本率指数移动平均，与标签平滑解耦
        self.prior_weight = prior_weight
        self.ece_weight = ece_weight
        self.lambdarank_weight = lambdarank_weight
        self.global_ctr = global_ctr
        self.zmlc_lambda = zmlc_lambda
        
        self.zmlc_on_calib_only = zmlc_on_calib_only
        self.enable_grad_conflict_check = enable_grad_conflict_check
        self.loss_conflict_threshold = loss_conflict_threshold
        self._loss_history = {'focal': [], 'lrank': [], 'zmlc': [], 'prior': [], 'ece': []}
        self._grad_conflict_history = []

        # 优化器构建（与原版完全一致）
        if hasattr(model, 'get_sparse_params') and use_grafted_optimizer:
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            gate_params = model.get_gate_params() if hasattr(model, 'get_gate_params') else []
            scale_params = model.get_scale_params() if hasattr(model, 'get_scale_params') else []
            rank_params = model.get_rank_params() if hasattr(model, 'get_rank_params') else []
            calib_params = model.get_calib_params() if hasattr(model, 'get_calib_params') else []

            sparse_opt = torch.optim.Adagrad(sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            scale_ids = {id(p) for p in scale_params}
            dense_param_groups = [
                {'params': [p for p in dense_params if id(p) not in scale_ids]}, 
                {'params': scale_params, 'lr': 1e-6, 'weight_decay': 0.0}
            ]
            dense_opt = torch.optim.AdamW(dense_param_groups, lr=lr, betas=(0.9, 0.98))
            gate_opt = torch.optim.AdamW(gate_params, lr=lr*2, betas=(0.9, 0.98), weight_decay=1e-3) if gate_params else None
            
            rank_opt = torch.optim.AdamW(
                rank_params, 
                lr=lr * rank_lr_multiplier, 
                betas=(0.9, 0.98),
                weight_decay=1e-3
            ) if rank_params else None
            
            calib_opt = torch.optim.AdamW(
                calib_params,
                lr=lr * calib_lr_multiplier,
                betas=(0.9, 0.98),
                weight_decay=1e-3
            ) if calib_params else None
            
            self.optimizer = GraftedOptimizer(
                sparse_opt, dense_opt, gate_opt, rank_opt, calib_opt,
                target_ratio=1.0, adapt_interval=200
            )
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
        self.lambda_n_pairs = 1000

        if use_adaptive_focal and loss_type == 'focal':
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
                f"v7.2 AdaptiveFocalLoss: max_gamma={focal_max_gamma}, "
                f"alpha_pos={focal_alpha_pos}, alpha_neg={focal_alpha_neg}, gamma_min=0.5"
            )
        else:
            self.adaptive_focal = None
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma

        if warmup_steps > 0:
            logging.info(f"Warmup: {warmup_steps} steps")

        # [v7.3-VQ-FIX] VQ 崩塌监控与重启状态
        self.seq_usage_history = {}          # {domain: deque(maxlen=200)}
        self.seq_collapse_counter = {}      # {domain: int}
        self.seq_in_restart_cooldown = {}   # {domain: bool}
        self.seq_restart_step = {}          # {domain: int}
        self.restart_cooldown_steps = 500
        self.usage_alert_threshold = 0.05
        self.usage_recover_threshold = 0.3
        self.collapse_min_steps = 200

    # ========== 标签平滑（不再更新 pos_rate_ema） ==========
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
            # 注意：pos_rate_ema 已在 _train_step 中更新，此处不再更新

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

    # ========== 其他辅助函数（完全保留） ==========
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
                    self.optimizer.sparse_opt.param_groups[0]['params'] = sparse_params
                elif self.sparse_optimizer is not None:
                    self.sparse_optimizer.param_groups[0]['params'] = sparse_params

    # ========== 动态权重调整（整合预热、VQ保护、原有逻辑） ==========
    def _get_dynamic_weights(self, step: int, loss_values: Dict[str, float]) -> Dict[str, float]:
        # 目标权重
        target_weights = {
            'focal': 1.0,
            'lrank': self.lambdarank_weight,
            'zmlc':  self.zmlc_lambda,
            'prior': self.prior_weight,
            'ece':   self.ece_weight,
            'gse':   self.gse_aux_weight,
        }

        # 平滑预热
        warmup_schedule = {
            'focal': (0, 0),
            'gse':   (0, 0),
            'prior': (500, 1500),
            'ece':   (500, 1500),
            'lrank': (0, 2000),
            'zmlc':  (1000, 3000),
        }

        weights = {}
        for key, target_val in target_weights.items():
            w_start, w_end = warmup_schedule.get(key, (0, 0))
            if step <= w_start:
                w = 0.0 if key != 'focal' else target_val
            elif step >= w_end:
                w = target_val
            else:
                progress = (step - w_start) / max(w_end - w_start, 1)
                w = target_val * progress
            weights[key] = w

        # VQ 崩溃紧急保护
        for domain, history in self.seq_usage_history.items():
            if len(history) < 10:
                continue
            recent_usage = np.mean(list(history)[-10:])
            if recent_usage < self.usage_alert_threshold * 2:
                weights['gse'] = max(weights['gse'], 0.5)
                weights['lrank'] = min(weights['lrank'], 0.02)
                weights['zmlc'] = min(weights['zmlc'], 0.01)
                weights['ece'] = min(weights['ece'], 0.01)
                weights['prior'] = min(weights['prior'], 0.01)
                if recent_usage < self.usage_alert_threshold:
                    weights['lrank'] = 0.0
                    weights['zmlc'] = 0.0
                    weights['ece'] = 0.0
                    weights['prior'] = 0.0
                    weights['gse'] = min(weights['gse'], 1.0)
                    logging.warning(
                        f"[VQ_PROTECT] {domain} usage={recent_usage:.4f}, "
                        "deactivating conflicting losses to rescue codebook."
                    )
                else:
                    logging.info(
                        f"[VQ_PROTECT] {domain} usage={recent_usage:.4f}, "
                        "suppressing conflicting losses."
                    )
            elif recent_usage < self.usage_recover_threshold:
                weights['gse'] = max(weights['gse'], 0.3)

        # 基于正样本率的早期强制关闭（使用解耦后的 EMA）
        if self.pos_rate_ema < 0.015 and step < self.warmup_steps:
            weights['lrank'] = 0.0
            weights['zmlc'] = 0.0
            weights['prior'] = 0.0
            weights['ece'] = 0.0
            return weights

        # 原有动态调整
        if len(self._loss_history['focal']) >= 100 and weights['lrank'] > 0:
            recent_focal = np.mean(self._loss_history['focal'][-50:])
            older_focal = np.mean(self._loss_history['focal'][-100:-50])
            focal_improvement = (older_focal - recent_focal) / (abs(older_focal) + 1e-8)
            if focal_improvement > 0.1:
                weights['lrank'] *= 0.5
                weights['zmlc'] *= 0.3
                weights['prior'] *= 0.5
                weights['ece'] *= 0.3
                logging.info(
                    f"Focal improving rapidly ({focal_improvement:.3f}), "
                    f"reducing auxiliary weights."
                )

        if len(self._loss_history['lrank']) >= 100 and weights['lrank'] > 0:
            recent_lrank = np.mean(self._loss_history['lrank'][-50:])
            older_lrank = np.mean(self._loss_history['lrank'][-100:-50])
            if recent_lrank > older_lrank * 0.95:
                weights['lrank'] *= 1.5
                weights['zmlc'] *= 0.5
                logging.info(
                    f"LambdaRank stalled ({recent_lrank:.4f} vs {older_lrank:.4f}), "
                    "boosting rank weight."
                )

        if len(self._loss_history['focal']) >= 200:
            recent_var = np.var(self._loss_history['focal'][-50:])
            if recent_var < 1e-6:
                weights['gse'] *= 0.5
                logging.info("Focal loss flat, reducing GSE weight to prevent overfitting.")

        # 安全裁剪
        weights['gse']   = max(weights['gse'], 0.05)
        weights['lrank'] = min(weights['lrank'], 1.0)
        weights['zmlc']  = min(weights['zmlc'], 0.5)

        return weights

    # ========== VQ 码本重启 ==========
    def _check_and_restart_vq(self):
        for domain in self.model.seq_domains:
            block = self.model.seq_blocks[domain]
            if not hasattr(block, 'vq'):
                continue

            if self.seq_in_restart_cooldown.get(domain, False):
                if self.global_step - self.seq_restart_step.get(domain, 0) > self.restart_cooldown_steps:
                    self.seq_in_restart_cooldown[domain] = False
                else:
                    continue

            current_usage = block._last_info_usage.item() if hasattr(block, '_last_info_usage') else 1.0
            if current_usage < self.usage_alert_threshold:
                self.seq_collapse_counter[domain] = self.seq_collapse_counter.get(domain, 0) + 1
            else:
                self.seq_collapse_counter[domain] = 0

            if self.seq_collapse_counter.get(domain, 0) >= self.collapse_min_steps:
                nn.init.normal_(block.vq.codebook.weight, std=0.1)
                with torch.no_grad():
                    block.vq.codebook.weight[0].zero_()
                if block.vq.codebook.grad is not None:
                    block.vq.codebook.grad.zero_()
                self.seq_collapse_counter[domain] = 0
                self.seq_in_restart_cooldown[domain] = True
                self.seq_restart_step[domain] = self.global_step
                logging.warning(
                    f"[VQ_RESTART] Codebook for {domain} was collapsed for {self.collapse_min_steps} steps, "
                    f"reinitialized at global step {self.global_step}."
                )

    # ========== 梯度冲突计算（保留） ==========
    def _compute_grad_conflict(self, losses: Dict[str, torch.Tensor], shared_params: List[torch.Tensor]) -> Dict[str, float]:
        if not self.enable_grad_conflict_check or not shared_params:
            return {}
        grads = {}
        for name, loss in losses.items():
            if loss.requires_grad:
                loss.backward(retain_graph=True)
                grads[name] = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                              for p in shared_params]
                for p in shared_params:
                    if p.grad is not None:
                        p.grad.zero_()
        conflicts = {}
        loss_names = list(grads.keys())
        for i in range(len(loss_names)):
            for j in range(i+1, len(loss_names)):
                name_i, name_j = loss_names[i], loss_names[j]
                grad_i = torch.cat([g.flatten() for g in grads[name_i]])
                grad_j = torch.cat([g.flatten() for g in grads[name_j]])
                cos_sim = F.cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0)).item()
                conflicts[f'{name_i}_vs_{name_j}'] = cos_sim
                if cos_sim < -self.loss_conflict_threshold:
                    conflicts[f'{name_i}_vs_{name_j}_conflict'] = True
        return conflicts

    # ========== 训练步骤（完整保留所有日志） ==========
    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label_raw = device_batch['label'].float()

        # ===== 解耦：始终更新正样本率 EMA =====
        with torch.no_grad():
            pos_ratio = label_raw.mean().item()
            self.pos_rate_ema = 0.99 * self.pos_rate_ema + 0.01 * pos_ratio

        # 标签平滑
        if self.loss_type in ('focal', 'bce') and self.label_smoothing_strategy != 'none':
            label_main = self._smooth_labels(label_raw)
        else:
            label_main = label_raw

        # 梯度清零
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
        if self.adaptive_focal is not None:
            self.adaptive_focal.gamma_optimizer.zero_grad()

        # 前向传播
        model_input = self._make_model_input(device_batch)
        logits, logits_main, logits_rank, calib_residual = self.model(
            model_input, return_components=True
        )
        logits = logits.squeeze(-1)
        logits_main = logits_main.squeeze(-1)
        logits_rank = logits_rank.squeeze(-1)
        calib_residual = calib_residual.squeeze(-1)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning(f"NaN/Inf in logits at step {self.global_step}, skipping batch")
            return float('nan')

        # 各损失分量
        loss_components = {}
        
        if self.adaptive_focal is not None:
            loss_focal_or_bce = self.adaptive_focal(logits_main, label_main)
        elif self.loss_type == 'focal':
            p = torch.sigmoid(logits_main)
            p = torch.clamp(p, min=1e-7, max=1-1e-7)
            bce_loss = F.binary_cross_entropy_with_logits(logits_main, label_main, reduction='none')
            p_t = p * label_main + (1 - p) * (1 - label_main)
            focal_weight = (1 - p_t) ** self.focal_gamma
            focal_weight = torch.clamp(focal_weight, max=10.0)
            alpha_t = self.focal_alpha * label_main + (1 - self.focal_alpha) * (1 - label_main)
            loss_focal_or_bce = (alpha_t * focal_weight * bce_loss).mean()
        else:
            loss_focal_or_bce = F.binary_cross_entropy_with_logits(logits_main, label_main)
        
        loss_components['focal'] = loss_focal_or_bce

        lrank_loss_val = torch.tensor(0.0, device=self.device)
        if self.lambdarank_weight > 0:
            lrank_loss_val = lambdarank_loss(logits_rank, label_raw, n_pairs=self.lambda_n_pairs)
            loss_components['lrank'] = lrank_loss_val

        prior_loss_val = torch.tensor(0.0, device=self.device)
        if self.prior_weight > 0:
            prior = torch.sigmoid(logits).mean()
            prior_loss_val = self.prior_weight * F.mse_loss(
                prior, torch.tensor(self.global_ctr, device=self.device)
            )
            loss_components['prior'] = prior_loss_val

        ece_loss_val = torch.tensor(0.0, device=self.device)
        if self.ece_weight > 0:
            ece_loss_val = calibration_loss(logits, label_raw)
            loss_components['ece'] = ece_loss_val

        gse_aux_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'get_aux_loss'):
            gse_aux_loss = self.model.get_aux_loss()
            loss_components['gse'] = gse_aux_loss

        zmlc_loss_val = torch.tensor(0.0, device=self.device)
        if self.zmlc_lambda > 0:
            if self.zmlc_on_calib_only:
                zmlc_loss_val = self.zmlc_lambda * (calib_residual.mean() ** 2)
            else:
                zmlc_loss_val = self.zmlc_lambda * (logits.mean() ** 2)
            loss_components['zmlc'] = zmlc_loss_val

        # 动态权重
        dynamic_weights = self._get_dynamic_weights(self.global_step, {
            k: v.item() for k, v in loss_components.items()
        })
        
        loss = dynamic_weights['focal'] * loss_focal_or_bce
        if 'lrank' in loss_components:
            loss = loss + dynamic_weights['lrank'] * loss_components['lrank']
        if 'prior' in loss_components:
            loss = loss + dynamic_weights['prior'] * loss_components['prior']
        if 'ece' in loss_components:
            loss = loss + dynamic_weights['ece'] * loss_components['ece']
        if 'gse' in loss_components:
            loss = loss + dynamic_weights['gse'] * loss_components['gse']
        if 'zmlc' in loss_components:
            loss = loss + dynamic_weights['zmlc'] * loss_components['zmlc']

        # 梯度冲突检测（原有逻辑）
        if self.enable_grad_conflict_check and self.global_step % 50 == 0:
            shared_params = []
            for name, p in self.model.named_parameters():
                if p.requires_grad and 'predictor' in name and p.grad is not None:
                    shared_params.append(p)
            if shared_params:
                conflicts = self._compute_grad_conflict(loss_components, shared_params)
                if conflicts:
                    self._grad_conflict_history.append(conflicts)
                    for key, val in conflicts.items():
                        if '_conflict' in key:
                            logging.warning(f"Gradient conflict detected: {key}")

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss at step {self.global_step}, skipping batch")
            self.optimizer.zero_grad() if self.optimizer else (
                self.dense_optimizer.zero_grad(),
                self.sparse_optimizer.zero_grad() if self.sparse_optimizer else None
            )
            return float('nan')

        loss.backward()

        # embedding 梯度记录
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

        # NaN 梯度检查
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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.global_step += 1

        # 学习率 warmup
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
                if self.optimizer.rank_opt is not None:
                    for g in self.optimizer.rank_opt.param_groups:
                        g['lr'] = self.lr * 2 * warmup_factor
                if self.optimizer.calib_opt is not None:
                    for g in self.optimizer.calib_opt.param_groups:
                        g['lr'] = self.lr * warmup_factor
            else:
                for g in self.dense_optimizer.param_groups:
                    g['lr'] = self.lr * warmup_factor
                if self.sparse_optimizer is not None:
                    for g in self.sparse_optimizer.param_groups:
                        g['lr'] = self.sparse_lr * warmup_factor

        # 优化器更新
        opt_diag = {}
        if self.optimizer is not None:
            opt_diag = self.optimizer.step(self.model)
        else:
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.model.parameters()):
            logging.error(f"NaN/Inf parameter after step {self.global_step}!")

        if self.adaptive_focal is not None:
            self.adaptive_focal.step()

        # 更新 loss history
        for key, val in loss_components.items():
            if key not in self._loss_history:
                self._loss_history[key] = []
            self._loss_history[key].append(val.item())
            if len(self._loss_history[key]) > 500:
                self._loss_history[key] = self._loss_history[key][-500:]

        # VQ 使用率记录
        try:
            for domain in self.model.seq_domains:
                block = self.model.seq_blocks[domain]
                usage = block._last_info_usage.item() if hasattr(block, '_last_info_usage') else 1.0
                if domain not in self.seq_usage_history:
                    self.seq_usage_history[domain] = deque(maxlen=200)
                self.seq_usage_history[domain].append(usage)
        except Exception:
            pass

        # ---- 完整的 TensorBoard 与日志（与原始完全一致 + 新增指标） ----
        with torch.no_grad():
            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            logits_main_mean = logits_main.mean().item()
            logits_main_std = logits_main.std().item()
            logits_rank_mean = logits_rank.mean().item()
            logits_rank_std = logits_rank.std().item()
            calib_mean = calib_residual.mean().item()
            calib_std = calib_residual.std().item()
            
            try:
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = label_raw.cpu().numpy()
                train_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) >= 2 else 0.0
            except Exception:
                train_auc = 0.0

        if self.writer:
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.writer.add_scalar('Loss/focal_or_bce', loss_focal_or_bce.item(), self.global_step)
            self.writer.add_scalar('Loss/lambdarank', lrank_loss_val.item(), self.global_step)
            self.writer.add_scalar('Loss/prior', prior_loss_val.item(), self.global_step)
            self.writer.add_scalar('Loss/ece', ece_loss_val.item(), self.global_step)
            self.writer.add_scalar('Loss/zmlc', zmlc_loss_val.item(), self.global_step)
            self.writer.add_scalar('GSE/aux_loss', gse_aux_loss.item(), self.global_step)
            
            self.writer.add_scalar('Stream/logits_main_mean', logits_main_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_main_std', logits_main_std, self.global_step)
            self.writer.add_scalar('Stream/logits_rank_mean', logits_rank_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_rank_std', logits_rank_std, self.global_step)
            self.writer.add_scalar('Stream/calib_residual_mean', calib_mean, self.global_step)
            self.writer.add_scalar('Stream/calib_residual_std', calib_std, self.global_step)
            self.writer.add_scalar('Stream/final_logits_mean', logits_mean, self.global_step)
            self.writer.add_scalar('Stream/final_logits_std', logits_std, self.global_step)
            
            self.writer.add_scalar('Diagnostics/train_auc', train_auc, self.global_step)
            self.writer.add_scalar('Diagnostics/pos_rate_ema', self.pos_rate_ema, self.global_step)  # 新增
            
            if self.adaptive_focal is not None:
                self.writer.add_scalar('Focal/gamma', self.adaptive_focal.get_gamma().item(), self.global_step)
            
            self.writer.add_scalar('Grad/emb_grad_norm', total_emb_grad_norm, self.global_step)
            if opt_diag:
                self.writer.add_scalar('Grad/sparse', opt_diag.get('sparse_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/dense', opt_diag.get('dense_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/rank', opt_diag.get('rank_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/calib', opt_diag.get('calib_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/ratio', opt_diag.get('grad_norm_ratio', 0), self.global_step)
            
            if self.optimizer is not None:
                self.writer.add_scalar('LR/dense', self.optimizer.dense_opt.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('LR/sparse', self.optimizer.sparse_opt.param_groups[0]['lr'], self.global_step)
                if self.optimizer.rank_opt is not None:
                    self.writer.add_scalar('LR/rank', self.optimizer.rank_opt.param_groups[0]['lr'], self.global_step)
                if self.optimizer.calib_opt is not None:
                    self.writer.add_scalar('LR/calib', self.optimizer.calib_opt.param_groups[0]['lr'], self.global_step)
            
            for key, val in dynamic_weights.items():
                self.writer.add_scalar(f'Weight/{key}', val, self.global_step)
            
            # 新增：VQ 使用率
            for domain, history in self.seq_usage_history.items():
                if history:
                    self.writer.add_scalar(f'VQ/usage_{domain}', history[-1], self.global_step)

        # 控制台日志（每100步）
        if self.global_step % 100 == 0:
            log_parts = [f"[Step {self.global_step}] loss={loss.item():.4f}"]
            log_parts.append(f"focal={loss_focal_or_bce.item():.4f}")
            log_parts.append(f"lrank={lrank_loss_val.item():.4f}")
            log_parts.append(f"zmlc={zmlc_loss_val.item():.4f}")
            log_parts.append(f"prior={prior_loss_val.item():.4f}")
            log_parts.append(f"ece={ece_loss_val.item():.4f}")
            log_parts.append(f"gse={gse_aux_loss.item():.4f}")
            log_parts.append(f"logits_main={logits_main_mean:.3f}±{logits_main_std:.3f}")
            log_parts.append(f"logits_rank={logits_rank_mean:.3f}±{logits_rank_std:.3f}")
            log_parts.append(f"calib={calib_mean:.3f}±{calib_std:.3f}")
            log_parts.append(f"train_AUC={train_auc:.4f}")
            log_parts.append(f"pos_ema={self.pos_rate_ema:.4f}")  # 新增
            if self.adaptive_focal is not None:
                log_parts.append(f"gamma={self.adaptive_focal.get_gamma().item():.3f}")
            if opt_diag:
                log_parts.append(f"grad_ratio={opt_diag.get('grad_norm_ratio', 0):.2f}")
                if 'rank_action' in opt_diag:
                    log_parts.append(f"rank_action={opt_diag['rank_action']}")
            # 新增：VQ usage
            for domain, history in self.seq_usage_history.items():
                if history:
                    log_parts.append(f"{domain}_usage={history[-1]:.3f}")
            logging.info(" | ".join(log_parts))

        return loss.item()

    # ========== 评估函数（完全保留） ==========
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

    # ========== 训练循环（完全保留 + VQ重启检查） ==========
    def train(self) -> None:
        logging.info("Start training HeteroFormer v7.3 (Collaborative Multi-Objective + VQ collapse protection)")
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

                # VQ 重启检查
                self._check_and_restart_vq()

                gamma_str = f"{self.adaptive_focal.get_gamma().item():.2f}" if self.adaptive_focal else "N/A"
                train_pbar.set_postfix({
                    "loss": f"{loss:.4f}", 
                    "gamma": gamma_str, 
                    "nan": nan_count,
                    "lrank_w": f"{self._get_dynamic_weights(self.global_step, {}).get('lrank', 0):.3f}"
                })

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