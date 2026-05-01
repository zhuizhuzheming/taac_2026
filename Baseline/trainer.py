"""PCVRHyFormer trainer with curriculum learning (BCE → BCE+Focal → BCE+λ·Focal‑InfoNCE)
and Muon + AdamW + Adagrad mixed optimiser.

Delay-feedback enhancements:
- Delay-aware BCE/Focal losses with fake-negative weighting.
- ES-DFM loss for explicit delay modeling.
- User-level InfoNCE for better contrastive learning.
- GAUC and delay-bucketed evaluation.
- Fixed gradient accumulation logic.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


# ══════════════════════════════════════════════════════════════
# PyTorch < 2.6 自定义 Muon 优化器 (修正牛顿‑舒尔茨迭代)
# ══════════════════════════════════════════════════════════════

class Muon(Optimizer):
    """Muon optimizer for 2D parameters (approximate orthogonal updates).

    Newton‑Schulz iteration: X = 1.5*X - 0.5*X @ (X.T @ X)
    """

    def __init__(self, params, lr=0.001, momentum=0.95, weight_decay=0.1,
                 nesterov=True, ns_steps=5, ns_epsilon=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum > 0")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps, ns_epsilon=ns_epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            ns_epsilon = group['ns_epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = state['momentum_buffer']

                # Combine gradient and momentum
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Newton‑Schulz orthogonalization (only for 2D parameters)
                assert update.ndim == 2, "Muon only supports 2D parameters"
                X = update
                norm = X.norm()
                if norm < ns_epsilon:
                    continue
                X = X / norm

                # Newton‑Schulz: X = 1.5*X - 0.5*X @ (X.T @ X)
                for _ in range(ns_steps):
                    A = X.mT @ X
                    X = 1.5 * X - 0.5 * X @ A

                p.add_(X, alpha=-lr)

        return loss


# ══════════════════════════════════════════════════════════════
# Delay-Feedback Aware Loss Functions
# ══════════════════════════════════════════════════════════════

def delay_aware_bce_loss(logits: torch.Tensor, labels: torch.Tensor,
                         fake_negative_weight: torch.Tensor) -> torch.Tensor:
    """Delay-aware BCE: down-weight likely fake negatives."""
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    weights = torch.where(labels == 1,
                          torch.ones_like(labels, dtype=torch.float32),
                          fake_negative_weight)
    return (bce * weights).mean()


def delay_aware_focal_loss(logits: torch.Tensor, labels: torch.Tensor,
                           fake_negative_weight: torch.Tensor,
                           alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Delay-aware Focal Loss with fake-negative weighting."""
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    probs = torch.sigmoid(logits)
    
    p_t = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = (1 - p_t) ** gamma
    
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    adjusted_alpha = alpha_t * torch.where(labels == 1,
                                           torch.ones_like(fake_negative_weight),
                                           fake_negative_weight)
    
    loss = adjusted_alpha * focal_weight * bce
    return loss.mean()


def esdfm_loss(logits: torch.Tensor, labels: torch.Tensor,
               fake_negative_weight: torch.Tensor,
               observed_delay: torch.Tensor,
               temperature: float = 0.1) -> torch.Tensor:
    """ES-DFM (Exponential Survival Delay Feedback Model) loss.
    
    Models the probability of eventual conversion given observed delay.
    Reference: https://arxiv.org/abs/2202.01807
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
    
    # Model output as hazard rate estimate
    lambda_est = torch.sigmoid(logits).clamp(min=1e-6)
    
    # Survival probability: P(not converted after t | will convert)
    # Normalize delay by day (86400 seconds)
    normalized_delay = observed_delay / 86400.0
    survival_prob = torch.exp(-lambda_est * normalized_delay)
    
    # For negative samples: P(will eventually convert | not converted yet)
    # = 1 - survival_prob (higher for recent clicks)
    conversion_prob = 1.0 - survival_prob.squeeze()
    
    adjusted_weight = torch.where(labels == 1,
                                  torch.ones_like(labels, dtype=torch.float32),
                                  fake_negative_weight * (1.0 - conversion_prob))
    
    return (bce * adjusted_weight).mean()


def user_level_info_nce_loss(logits: torch.Tensor, labels: torch.Tensor,
                             user_ids: List[str],
                             temperature: float = 0.2,
                             gamma: float = 2.0,
                             min_pos: int = 2) -> torch.Tensor:
    """User-level Focal-InfoNCE: contrast within same user only."""
    # Group by user
    user_groups = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        user_groups[uid].append(idx)
    
    total_loss = 0.0
    valid_groups = 0
    
    for uid, indices in user_groups.items():
        if len(indices) < 2:
            continue
            
        user_logits = logits[indices]
        user_labels = labels[indices]
        pos_mask = user_labels == 1
        
        if pos_mask.sum() < 1 or (~pos_mask).sum() < 1:
            continue
        
        scaled = user_logits / temperature
        log_prob = torch.log_softmax(scaled, dim=0)
        prob = torch.exp(log_prob)
        
        # Focal weighting for positive samples
        focal_weight = (1 - prob[pos_mask]) ** gamma
        loss = -(focal_weight * log_prob[pos_mask]).mean()
        
        total_loss += loss
        valid_groups += 1
    
    if valid_groups == 0:
        return (logits * 0.0).sum()
    
    return total_loss / valid_groups


# ══════════════════════════════════════════════════════════════
# Evaluation Metrics
# ══════════════════════════════════════════════════════════════

def compute_gauc(labels: np.ndarray, probs: np.ndarray, user_ids: List[str]) -> Tuple[float, float]:
    """Compute Group AUC and mean user AUC."""
    user_preds = defaultdict(list)
    user_labels = defaultdict(list)
    
    for uid, label, prob in zip(user_ids, labels, probs):
        user_preds[uid].append(prob)
        user_labels[uid].append(label)
    
    user_aucs = []
    user_weights = []
    
    for uid in user_preds:
        u_labels = np.array(user_labels[uid])
        u_probs = np.array(user_preds[uid])
        
        if len(np.unique(u_labels)) > 1:
            auc = roc_auc_score(u_labels, u_probs)
            user_aucs.append(auc)
            user_weights.append(len(u_labels))
    
    if not user_aucs:
        return 0.0, 0.0
    
    gauc = np.average(user_aucs, weights=user_weights)
    mean_auc = np.mean(user_aucs)
    
    return gauc, mean_auc


# ══════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════

class PCVRHyFormerRankingTrainer:
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
        loss_type: str = 'curriculum',
        focal_alpha: float = 0.25,  # Changed from 0.1 to more standard 0.25
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
        # 课程学习参数
        stage1_epochs: int = 3,  # Reduced from 5
        stage2_epochs: int = 5,
        info_nce_weight_stage3: float = 0.05,
        info_nce_temperature_stage3: float = 0.5,
        stage3_lr_factor: float = 0.5,
        # Muon 优化器参数
        muon_lr: Optional[float] = None,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.1,
        accumulation_steps: int = 2,
        # ===== Delay-feedback parameters =====
        delay_loss_type: str = 'esdfm',  # 'bce', 'esdfm', 'fncw'
        delay_decay_window: float = 604800.0,
        # =====================================
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config

        # 损失 & 课程
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.info_nce_weight_stage3 = info_nce_weight_stage3
        self.info_nce_temperature_stage3 = info_nce_temperature_stage3
        self.stage3_lr_factor = stage3_lr_factor
        self.accumulation_steps = accumulation_steps
        self._current_epoch = 0
        self.global_step = 0

        # 动态损失参数
        self._cur_info_nce_weight = 0.0
        self._cur_info_nce_temperature = 0.2
        
        # ===== Delay-feedback parameters =====
        self.delay_loss_type = delay_loss_type
        self.delay_decay_window = delay_decay_window
        # =====================================

        # ---------- 混合优化器 ----------
        sparse_params = model.get_sparse_params() if hasattr(model, 'get_sparse_params') else []
        sparse_ptr_set = {p.data_ptr() for p in sparse_params}

        muon_params = []
        adam_params = []
        for p in model.parameters():
            if p.data_ptr() in sparse_ptr_set:
                continue
            if p.ndim == 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        # Adagrad for sparse embeddings
        if sparse_params:
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
        else:
            self.sparse_optimizer = None

        # Muon
        muon_lr_val = muon_lr if muon_lr is not None else lr * 0.1
        try:
            self.optimizer_muon = torch.optim.Muon(
                muon_params,
                lr=muon_lr_val,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                nesterov=True,
            )
            logging.info("Using official torch.optim.Muon")
        except AttributeError:
            self.optimizer_muon = Muon(
                muon_params,
                lr=muon_lr_val,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                nesterov=True,
            )
            logging.info("Using custom Muon implementation (PyTorch < 2.6)")

        self.base_muon_lr = muon_lr_val

        # AdamW
        self.optimizer_adam = torch.optim.AdamW(
            adam_params,
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=1e-2,
        )
        self.base_adam_lr = lr

        # 学习率调度器（标配 warmup + cosine）
        total_steps = len(train_loader) * num_epochs // accumulation_steps
        warmup_steps = max(1, int(0.1 * total_steps))
        self.scheduler_muon = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer_muon,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_muon, start_factor=0.1, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_muon, T_max=max(1, total_steps - warmup_steps)
                ),
            ],
            milestones=[warmup_steps],
        )
        self.scheduler_adam = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer_adam,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_adam, start_factor=0.1, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_adam, T_max=max(1, total_steps - warmup_steps)
                ),
            ],
            milestones=[warmup_steps],
        )

        # 其余成员
        self.num_epochs = num_epochs
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.ckpt_params = ckpt_params or {}

        logging.info(
            f"Trainer: curriculum stages -> Delay-aware BCE {stage1_epochs} epochs, "
            f"BCE+Focal {stage2_epochs} epochs, then weak Focal‑InfoNCE (λ={info_nce_weight_stage3}, T={info_nce_temperature_stage3}). "
            f"Delay loss: {delay_loss_type}. "
            f"Muon lr={muon_lr_val}, AdamW lr={lr}"
        )

    # ========== 根据 epoch 切换课程 ==========
    def _setup_curriculum(self, epoch: int) -> None:
        """Set loss weights and possibly adjust learning rate based on epoch."""
        if epoch <= self.stage1_epochs:
            self.loss_type = 'delay_bce'
            self._cur_info_nce_weight = 0.0
            logging.info(f"Curriculum: Stage 1 (Delay-aware BCE) – epoch {epoch}")
        elif epoch <= self.stage1_epochs + self.stage2_epochs:
            self.loss_type = 'delay_mixed'
            self._cur_info_nce_weight = 0.0
            logging.info(f"Curriculum: Stage 2 (Delay-aware BCE+Focal) – epoch {epoch}")
        else:
            self.loss_type = 'delay_mixed'
            self._cur_info_nce_weight = self.info_nce_weight_stage3
            if epoch == self.stage1_epochs + self.stage2_epochs + 1:
                factor = self.stage3_lr_factor
                for pg in self.optimizer_muon.param_groups:
                    pg['lr'] = self.base_muon_lr * factor
                for pg in self.optimizer_adam.param_groups:
                    pg['lr'] = self.base_adam_lr * factor
                logging.info(
                    f"Curriculum: Stage 3 (Delay-aware BCE+λ·Focal‑InfoNCE) – "
                    f"λ={self._cur_info_nce_weight}, T={self.info_nce_temperature_stage3}, "
                    f"lr reduced by factor {factor}"
                )

    # ========== 工具函数 ==========
    def _build_step_dir_name(self, global_step, is_best=False):
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)
        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True
        if self.train_config:
            import json
            cfg = dict(self.train_config)
            if ns_groups_copied:
                cfg['ns_groups_json'] = os.path.basename(self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg, f, indent=2)

    def _save_step_checkpoint(self, global_step, is_best=False, skip_model_file=False):
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self):
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(self, total_step, val_auc, val_logloss):
        old_best = self.early_stopping.best_score
        likely = old_best is None or val_auc > old_best + self.early_stopping.delta
        if not likely:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return
        best_dir = os.path.join(self.save_dir, self._build_step_dir_name(total_step, is_best=True))
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()
        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _make_model_input(self, device_batch):
        # FIX: Handle _seq_domains as either list or string
        seq_domains_raw = device_batch['_seq_domains']
        logging.info(f"_seq_domains type: {type(seq_domains_raw)}, value: {seq_domains_raw}")
        if isinstance(seq_domains_raw, str):
            seq_domains = [d.strip() for d in seq_domains_raw.split(',')]
        elif isinstance(seq_domains_raw, list):
            seq_domains = seq_domains_raw
        else:
            # Fallback: infer from keys
            seq_domains = [k for k in device_batch.keys() 
                          if k not in ['user_int_feats', 'item_int_feats', 'user_dense_feats', 
                                      'item_dense_feats', 'label', 'label_type', 'label_time',
                                      'observed_delay', 'fake_negative_weight', 'timestamp',
                                      'user_id', '_seq_domains', 'is_clicked_not_converted']
                          and not k.endswith('_len') and not k.endswith('_time_bucket') 
                          and not k.endswith('_timestamps')]
        
        seq_data = {}
        seq_lens = {}
        seq_time_buckets = {}
        seq_timestamps = {}
        for domain in seq_domains:
            if domain not in device_batch:
                logging.warning(f"Domain {domain} not found in batch, skipping")
                continue
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device)
            )
            seq_timestamps[domain] = device_batch.get(
                f'{domain}_timestamps',
                torch.zeros(B, L, dtype=torch.long, device=self.device)
            )
        
        # FIX: Handle timestamp - might be tensor or missing
        timestamp = device_batch.get('timestamp')
        if timestamp is None:
            first_domain = next(iter(seq_data.keys())) if seq_data else None
            if first_domain and first_domain in seq_timestamps:
                timestamp = seq_timestamps[first_domain][:, 0]
            else:
                timestamp = None
        
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_timestamps=seq_timestamps,
            timestamp=timestamp,
        )

    # ========== 训练主循环 ==========
    def train(self) -> None:
        print("Start training (PCVRHyFormer + Delay-Aware Curriculum + Muon)")
        self.model.train()
        self.global_step = 0
        
        # ===== FIX: Ensure clean gradient state at start =====
        self.optimizer_muon.zero_grad()
        self.optimizer_adam.zero_grad()
        if self.sparse_optimizer:
            self.sparse_optimizer.zero_grad()

        for epoch in range(1, self.num_epochs + 1):
            self._current_epoch = epoch
            self._setup_curriculum(epoch)

            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in pbar:
                loss = self._train_step(batch)
                self.global_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, self.global_step)
                pbar.set_postfix({"loss": f"{loss:.4f}", "stage": self.loss_type})

                # ===== FIX: Correct gradient accumulation logic =====
                if self.global_step % self.accumulation_steps == 0:
                    # Gradient clipping for all optimizers
                    torch.nn.utils.clip_grad_norm_(
                        [p for pg in self.optimizer_muon.param_groups for p in pg['params']], 
                        max_norm=1.0
                    )
                    torch.nn.utils.clip_grad_norm_(
                        [p for pg in self.optimizer_adam.param_groups for p in pg['params']], 
                        max_norm=1.0
                    )
                    if self.sparse_optimizer:
                        torch.nn.utils.clip_grad_norm_(
                            [p for pg in self.sparse_optimizer.param_groups for p in pg['params']], 
                            max_norm=0.5
                        )

                    # Optimizer steps
                    self.optimizer_muon.step()
                    self.optimizer_adam.step()
                    if self.sparse_optimizer:
                        self.sparse_optimizer.step()

                    # LR schedulers
                    self.scheduler_muon.step()
                    self.scheduler_adam.step()

                    # Zero gradients (CRITICAL FIX)
                    self.optimizer_muon.zero_grad()
                    self.optimizer_adam.zero_grad()
                    if self.sparse_optimizer:
                        self.sparse_optimizer.zero_grad()

                if self.eval_every_n_steps > 0 and self.global_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {self.global_step}")
                    val_metrics = self.evaluate(epoch=epoch)
                    self.model.train()
                    # torch.cuda.empty_cache()  # Removed: causes memory fragmentation
                    
                    val_auc = val_metrics['global_auc']
                    val_logloss = val_metrics['logloss']
                    logging.info(f"Step {self.global_step} | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")
                    
                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, self.global_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, self.global_step)
                        self.writer.add_scalar('GAUC/valid', val_metrics.get('gauc', 0), self.global_step)
                    
                    self._handle_validation_result(self.global_step, val_auc, val_logloss)
                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {self.global_step}")
                        return

            avg_loss = loss_sum / len(self.train_loader)
            logging.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            val_metrics = self.evaluate(epoch=epoch)
            self.model.train()
            # torch.cuda.empty_cache()  # Removed
            
            val_auc = val_metrics['global_auc']
            val_logloss = val_metrics['logloss']
            gauc = val_metrics.get('gauc', 0)
            
            logging.info(f"Epoch {epoch} | AUC: {val_auc:.4f}, GAUC: {gauc:.4f}, LogLoss: {val_logloss:.4f}")
            
            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, self.global_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, self.global_step)
                self.writer.add_scalar('GAUC/valid', gauc, self.global_step)
            
            self._handle_validation_result(self.global_step, val_auc, val_logloss)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                self._rebuild_sparse_optimizer()

    def _rebuild_sparse_optimizer(self):
        """Rebuild sparse optimizer with state preservation."""
        old_state = {}
        for group in self.sparse_optimizer.param_groups:
            for p in group['params']:
                if p.data_ptr() in self.sparse_optimizer.state:
                    old_state[p.data_ptr()] = self.sparse_optimizer.state[p]
        
        reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
        sparse_params = self.model.get_sparse_params()
        self.sparse_optimizer = torch.optim.Adagrad(
            sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
        )
        
        restored = 0
        for p in sparse_params:
            if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                restored += 1
        logging.info(f"Rebuilt Adagrad after epoch {self._current_epoch}, restored {restored} states")

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        fake_negative_weight = device_batch['fake_negative_weight'].float()
        observed_delay = device_batch.get('observed_delay', torch.zeros_like(label))
        
        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input).squeeze(-1)

        # ===== Delay-aware loss computation =====
        if self.loss_type == 'delay_bce':
            if self.delay_loss_type == 'esdfm':
                loss = esdfm_loss(logits, label, fake_negative_weight, observed_delay)
            else:
                loss = delay_aware_bce_loss(logits, label, fake_negative_weight)
                
        elif self.loss_type == 'delay_mixed':
            if self._cur_info_nce_weight == 0.0:
                # Stage 2: Delay-aware BCE + Focal
                if self.delay_loss_type == 'esdfm':
                    bce = esdfm_loss(logits, label, fake_negative_weight, observed_delay)
                else:
                    bce = delay_aware_bce_loss(logits, label, fake_negative_weight)
                focal = delay_aware_focal_loss(logits, label, fake_negative_weight,
                                               alpha=self.focal_alpha, gamma=self.focal_gamma)
                loss = bce + focal
            else:
                # Stage 3: Add user-level InfoNCE
                if self.delay_loss_type == 'esdfm':
                    bce = esdfm_loss(logits, label, fake_negative_weight, observed_delay)
                else:
                    bce = delay_aware_bce_loss(logits, label, fake_negative_weight)
                
                # User-level InfoNCE
                user_ids = device_batch.get('user_id', None)
                ince = user_level_info_nce_loss(
                    logits, label, user_ids,
                    temperature=self.info_nce_temperature_stage3,
                    gamma=self.focal_gamma
                )
                loss = bce + self._cur_info_nce_weight * ince
                
        elif self.loss_type == 'bce':
            # Fallback to standard BCE
            loss = F.binary_cross_entropy_with_logits(logits, label)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        loss = loss / self.accumulation_steps
        loss.backward()
        return loss.item() * self.accumulation_steps

    def evaluate(self, epoch=None) -> Dict[str, float]:
        """Evaluate with GAUC and delay-bucketed metrics."""
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        all_logits_list, all_labels_list = [], []
        all_user_ids = []
        all_delays = []
        
        with torch.no_grad():
            for step, batch in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader)):
                device_batch = self._batch_to_device(batch)
                model_input = self._make_model_input(device_batch)
                logits, _ = self.model.predict(model_input)
                logits = logits.squeeze(-1)
                
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(device_batch['label'].detach().cpu())
                all_user_ids.extend(device_batch['user_id'])
                if 'observed_delay' in device_batch:
                    all_delays.append(device_batch['observed_delay'].detach().cpu())
        
        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()
        
        # Handle NaN
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            logging.warning(f"NaN predictions, removing {nan_mask.sum()} items")
            probs = probs[~nan_mask]
            labels_np = labels_np[~nan_mask]
            all_user_ids = [uid for i, uid in enumerate(all_user_ids) if not nan_mask[i]]
        
        # Global metrics
        global_auc = float(roc_auc_score(labels_np, probs)) if len(np.unique(labels_np)) > 1 else 0.0
        
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        
        # GAUC
        gauc, mean_user_auc = compute_gauc(labels_np, probs, all_user_ids)
        
        # Delay-bucketed evaluation
        delay_aucs = {}
        if all_delays:
            delays = torch.cat(all_delays).numpy()
            # Remove NaN indices from delays too
            if nan_mask.any():
                delays = delays[~nan_mask]
            
            # Bucket by observed delay: 0 (unconverted), <1d, <3d, <7d, <14d, >14d
            delay_boundaries = [0, 86400, 259200, 604800, 1209600, float('inf')]
            delay_buckets = np.digitize(delays, delay_boundaries)
            
            for b in range(len(delay_boundaries)):
                # Include both converted samples in bucket and unconverted (bucket=0)
                if b == 0:
                    mask = (delay_buckets == 0)
                else:
                    mask = (delay_buckets == b)
                
                if mask.sum() > 0 and len(np.unique(labels_np[mask])) > 1:
                    bucket_auc = roc_auc_score(labels_np[mask], probs[mask])
                    delay_aucs[f'delay_bucket_{b}'] = bucket_auc
        
        metrics = {
            'global_auc': global_auc,
            'gauc': gauc,
            'mean_user_auc': mean_user_auc,
            'logloss': logloss,
        }
        metrics.update(delay_aucs)
        
        logging.info(f"Epoch {epoch} | Global AUC: {global_auc:.4f}, GAUC: {gauc:.4f}, "
                     f"Mean User AUC: {mean_user_auc:.4f}, LogLoss: {logloss:.4f}")
        if delay_aucs:
            logging.info(f"Delay bucket AUCs: {delay_aucs}")
        
        return metrics

    def _evaluate_step(self, batch):
        """Legacy method for backward compatibility."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']
        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)
        logits = logits.squeeze(-1)
        return logits, label