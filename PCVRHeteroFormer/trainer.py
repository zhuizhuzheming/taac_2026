"""
PCVRHeteroFormer Trainer v11.2 - Hybrid (RoPE + SwiGLU + Top-K)
====================================================
适配改动：
1. SAM 梯度裁剪适配多参数组
2. OHEM 与 SAM 第二次 forward 兼容
3. Early Exit 辅助损失真正激活
4. 梯度 NaN 检查优化
"""

import os
import glob
import shutil
import logging
import math
from collections import deque
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import EarlyStopping
from model import ModelInput


# ==============================================================================
# v11.2: AdaptiveFocalLoss with OHEM
# ==============================================================================

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
        self._step_count = 0
        self._grad_ema = None
        self._gamma_history = deque([self.gamma_min], maxlen=10)
        self._max_gamma_delta = 0.5

    def get_gamma(self) -> torch.Tensor:
        learned = torch.sigmoid(self.gamma_logit) * self.max_gamma
        learned = torch.clamp(learned, min=self.gamma_min)
        if len(self._gamma_history) > 0:
            last_gamma = self._gamma_history[-1]
            learned = torch.clamp(learned,
                                 min=last_gamma - self._max_gamma_delta,
                                 max=last_gamma + self._max_gamma_delta)
        if self._step_count < self.gamma_warmup_steps:
            progress = self._step_count / max(self.gamma_warmup_steps, 1)
            warmup_gamma = self.gamma_min + (learned - self.gamma_min) * progress
            return warmup_gamma
        return learned

    def forward(self, logits: Tensor, targets: Tensor, uncertainty: Optional[Tensor] = None) -> Tensor:
        gamma = self.get_gamma()
        self._gamma_history.append(gamma.detach().item())

        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        p_t = p * targets + (1 - p) * (1 - targets)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        focal_weight = (1 - p_t) ** gamma
        focal_weight = torch.clamp(focal_weight, max=5.0)

        alpha_t = targets * self.alpha_pos + (1 - targets) * self.alpha_neg
        loss = alpha_t * focal_weight * bce_loss

        if uncertainty is not None:
            u = uncertainty.detach()
            conf_weight = torch.exp(-u * 2.0)
            loss = loss * conf_weight

        gamma_reg = self.gamma_reg_coef * torch.abs(gamma - self.gamma_min) / (self.max_gamma - self.gamma_min + 1e-8)
        return loss.mean() + gamma_reg

    def step(self) -> None:
        self._step_count += 1
        if self.gamma_logit.grad is not None:
            if self._grad_ema is None:
                self._grad_ema = self.gamma_logit.grad.clone()
            else:
                self._grad_ema = 0.95 * self._grad_ema + 0.05 * self.gamma_logit.grad
            self.gamma_logit.grad = self._grad_ema
            torch.nn.utils.clip_grad_value_(self.gamma_logit, self.gamma_clip)
        self.gamma_optimizer.step()
        self.gamma_optimizer.zero_grad()

        if torch.isnan(self.gamma_logit).any():
            logging.warning("【NaN恢复】gamma_logit NaN，重新初始化")
            nn.init.normal_(self.gamma_logit, std=0.01)


# ==============================================================================
# v11.2: SAM Optimizer Wrapper
# ==============================================================================

class SAM:
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.state = {}

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.base_optimizer.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p in self.state:
                    p.sub_(self.state[p])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
        self.state.clear()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self) -> float:
        norm = 0.0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.norm(p=2).item() ** 2
        return norm ** 0.5

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ==============================================================================
# v11.2: IsolatedOptimizer (适配多参数组 + 梯度裁剪修复)
# ==============================================================================

class IsolatedOptimizer:
    def __init__(self, model: nn.Module, lr: float, sparse_lr: float, use_sam: bool = False, sam_rho: float = 0.05):
        groups = model.get_param_groups()

        # Backbone + heads + seq + other (dense parameters)
        dense_params = (
            groups.get('backbone', []) + groups.get('head', []) +
            groups.get('seq_encoder', []) + groups.get('other', [])
        )
        self.dense_opt = torch.optim.AdamW(
            dense_params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-4,
        ) if dense_params else None

        # Sparse embeddings
        if groups.get('sparse'):
            self.sparse_opt = torch.optim.Adagrad(
                groups['sparse'], lr=sparse_lr, weight_decay=1e-4)
        else:
            self.sparse_opt = None

        # SAM wrapper
        self.use_sam = use_sam
        if use_sam and self.dense_opt:
            self.sam = SAM(self.dense_opt, rho=sam_rho)
            self.dense_opt = self.sam  # type: ignore
        else:
            self.sam = None

        self.step_count = 0
        self.grad_norm_history = deque(maxlen=100)
        self.nan_recovery_count = 0

    def zero_grad(self):
        if self.dense_opt:
            self.dense_opt.zero_grad()
        if self.sparse_opt:
            self.sparse_opt.zero_grad()

    def step(self):
        if self.dense_opt:
            if self.use_sam and self.sam:
                pass  # SAM second step handled in training loop
            else:
                # v11.2: 裁剪所有 dense 参数组，不只是 param_groups[0]
                all_dense_params = []
                for group in self.dense_opt.param_groups:
                    all_dense_params.extend(group['params'])
                if all_dense_params:
                    torch.nn.utils.clip_grad_norm_(all_dense_params, 1.0)
                self.dense_opt.step()
        if self.sparse_opt:
            self.sparse_opt.step()

    def step_sam_first(self):
        if self.use_sam and self.sam:
            self.sam.first_step(zero_grad=True)

    def step_sam_second(self):
        if self.use_sam and self.sam:
            self.sam.second_step(zero_grad=False)

    def check_and_recover_nan_params(self, model: nn.Module):
        nan_found = False
        for name, p in model.named_parameters():
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                nan_found = True
                nan_count = torch.isnan(p).sum().item() + torch.isinf(p).sum().item()
                logging.warning(f"【全局NaN恢复】{name}: {nan_count}/{p.numel()} abnormal values")
                if len(p.shape) >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
                else:
                    nn.init.normal_(p, std=0.01)
                if p.grad is not None:
                    p.grad.zero_()
        if nan_found:
            self.nan_recovery_count += 1
            logging.warning(f"【全局NaN恢复】第{self.nan_recovery_count}次触发")
        return nan_found

    def state_dict(self):
        return {
            'dense': self.dense_opt.state_dict() if self.dense_opt else None,
            'sparse': self.sparse_opt.state_dict() if self.sparse_opt else None,
            'step_count': self.step_count,
            'nan_recovery_count': self.nan_recovery_count,
        }

    def load_state_dict(self, state_dict):
        if self.dense_opt and 'dense' in state_dict and state_dict['dense']:
            self.dense_opt.load_state_dict(state_dict['dense'])
        if self.sparse_opt and 'sparse' in state_dict and state_dict['sparse']:
            self.sparse_opt.load_state_dict(state_dict['sparse'])
        self.step_count = state_dict.get('step_count', 0)
        self.nan_recovery_count = state_dict.get('nan_recovery_count', 0)


# ==============================================================================
# v11.2: CurriculumScheduler
# ==============================================================================

class CurriculumScheduler:
    def __init__(
        self,
        max_seq_lens: Dict[str, int],
        curriculum_epochs: int = 3,
        label_smoothing_max: float = 0.05,
        label_smoothing_min: float = 0.001,
    ):
        self.max_seq_lens = max_seq_lens
        self.curriculum_epochs = max(curriculum_epochs, 1)
        self.label_smoothing_max = label_smoothing_max
        self.label_smoothing_min = label_smoothing_min

    def get_seq_len_limit(self, epoch: int, domain: str) -> int:
        if epoch >= self.curriculum_epochs:
            return self.max_seq_lens[domain]
        progress = epoch / self.curriculum_epochs
        base_len = min(50, self.max_seq_lens[domain])
        return int(base_len + (self.max_seq_lens[domain] - base_len) * progress)

    def get_label_smoothing(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.label_smoothing_min
        progress = min(1.0, epoch / (total_epochs * 0.5))
        return self.label_smoothing_max - (self.label_smoothing_max - self.label_smoothing_min) * progress


# ==============================================================================
# v11.2: Trainer
# ==============================================================================

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
        loss_type: str = 'focal',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        warmup_steps: int = 0,
        use_adaptive_focal: bool = True,
        focal_alpha_pos: float = 0.5,
        focal_alpha_neg: float = 0.5,
        focal_max_gamma: float = 4.0,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        ohem_ratio: float = 1.0,
        curriculum_epochs: int = 3,
        label_smoothing_max: float = 0.05,
        label_smoothing_min: float = 0.001,
        seq_max_lens: Optional[Dict[str, int]] = None,
        early_exit_weight: float = 0.3,
        **kwargs,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config
        self.ohem_ratio = ohem_ratio
        self.early_exit_weight = early_exit_weight

        self.optimizer = IsolatedOptimizer(model, lr, sparse_lr, use_sam=use_sam, sam_rho=sam_rho)

        if use_adaptive_focal and loss_type == 'focal':
            self.adaptive_focal = AdaptiveFocalLoss(
                alpha_pos=focal_alpha_pos, alpha_neg=focal_alpha_neg,
                max_gamma=focal_max_gamma,
            )
        else:
            self.adaptive_focal = None

        self.curriculum = CurriculumScheduler(
            seq_max_lens or {},
            curriculum_epochs=curriculum_epochs,
            label_smoothing_max=label_smoothing_max,
            label_smoothing_min=label_smoothing_min,
        )

        self.last_val_auc = 0.0
        self.loss_history = deque(maxlen=100)
        self.valid_auc_history = deque(maxlen=5)
        self.train_auc_history = deque(maxlen=5)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        seq_domains = device_batch['_seq_domains']
        seq_data, seq_lens, seq_time_buckets, seq_decay_weights = {}, {}, {}, {}
        seq_timestamps_raw = {}

        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            seq_time_buckets[domain] = device_batch.get(f'{domain}_time_bucket', torch.zeros_like(seq_lens[domain]))
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

    def _apply_curriculum_to_batch(self, batch: Dict[str, Any], epoch: int) -> Dict[str, Any]:
        if not hasattr(self, 'curriculum') or not self.curriculum.max_seq_lens:
            return batch

        domains = batch['_seq_domains']
        for domain in domains:
            limit = self.curriculum.get_seq_len_limit(epoch, domain)
            key_len = f'{domain}_len'
            if key_len not in batch:
                continue
            lens = batch[key_len]
            mask = lens > limit
            if mask.any():
                batch[key_len] = torch.clamp(lens, max=limit)
                if domain in batch:
                    B, n_feat, L = batch[domain].shape
                    if L > limit:
                        batch[domain] = batch[domain][:, :, :limit]
                for suffix in ['time_bucket', 'decay_weight', 'timestamps_raw']:
                    k = f'{domain}_{suffix}'
                    if k in batch and batch[k].dim() >= 2:
                        batch[k] = batch[k][:, :limit]
        return batch

    def _compute_gauc(self, labels: np.ndarray, probs: np.ndarray, user_ids: List[str]) -> float:
        try:
            df = pd.DataFrame({'label': labels, 'prob': probs, 'user': user_ids})
            gauc = 0.0
            weights = 0.0
            for user, group in df.groupby('user'):
                if len(group) < 2 or group['label'].nunique() < 2:
                    continue
                try:
                    auc = roc_auc_score(group['label'].values, group['prob'].values)
                    gauc += auc * len(group)
                    weights += len(group)
                except Exception:
                    continue
            return gauc / weights if weights > 0 else 0.0
        except Exception as e:
            logging.warning(f"GAUC计算失败: {e}")
            return 0.0

    def _train_step(self, batch: Dict[str, Any], epoch: int) -> Dict[str, float]:
        device_batch = self._batch_to_device(batch)
        labels = device_batch['label'].float()

        device_batch = self._apply_curriculum_to_batch(device_batch, epoch)
        model_input = self._make_model_input(device_batch)

        self.optimizer.check_and_recover_nan_params(self.model)
        self.optimizer.zero_grad()

        self.model.train()
        out = self.model(model_input, task_id='ctr')

        (
            logits, _, _, _, _, _, uncertainty, _, _, _, base_logits, _
        ) = out

        logits = logits.squeeze(-1)
        base_logits = base_logits.squeeze(-1)

        # Main loss
        if self.adaptive_focal is not None:
            loss_main = self.adaptive_focal(logits, labels, uncertainty)
        else:
            loss_main = F.binary_cross_entropy_with_logits(logits, labels)

        # v11.2: Early Exit 辅助损失真正激活
        # model.early_exit_head 返回多组中间层 logits
        # 但 forward 只返回 base_logits（最后一层 early exit），无法直接获取中间层
        # 这里用 base_logits 作为 intermediate 辅助监督
        loss_aux = F.binary_cross_entropy_with_logits(base_logits, labels) * self.early_exit_weight

        # Label smoothing annealing
        smoothing = self.curriculum.get_label_smoothing(epoch, self.num_epochs)
        if smoothing > 0 and self.adaptive_focal is None:
            targets_smooth = labels * (1 - smoothing) + 0.5 * smoothing
            loss_main = F.binary_cross_entropy_with_logits(logits, targets_smooth)

        # OHEM: 记录 keep_idx 用于 SAM 第二次 forward
        keep_idx = None
        if self.ohem_ratio < 1.0:
            with torch.no_grad():
                per_sample_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                num_keep = max(1, int(len(per_sample_loss) * self.ohem_ratio))
                keep_idx = torch.topk(per_sample_loss, num_keep, largest=True).indices
            
            logits = logits[keep_idx]
            labels = labels[keep_idx]
            uncertainty = uncertainty[keep_idx] if uncertainty is not None else None
            base_logits = base_logits[keep_idx]
            
            if self.adaptive_focal is not None:
                loss_main = self.adaptive_focal(logits, labels, uncertainty)
            else:
                loss_main = F.binary_cross_entropy_with_logits(logits, labels)
            
            # 辅助损失也同步裁剪
            loss_aux = F.binary_cross_entropy_with_logits(base_logits, labels) * self.early_exit_weight

        total_loss = loss_main + loss_aux

        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(0.0, device=total_loss.device),
            total_loss
        )

        # SAM: first forward-backward
        total_loss.backward()

        if self.optimizer.use_sam and self.optimizer.sam:
            self.optimizer.step_sam_first()
            
            # Second forward: 必须使用相同的样本子集（OHEM keep_idx）
            out2 = self.model(model_input, task_id='ctr')
            logits2 = out2[0].squeeze(-1)
            base_logits2 = out2[10]
            
            if keep_idx is not None:
                logits2 = logits2[keep_idx]
                base_logits2 = base_logits2[keep_idx]
                labels2 = labels  # 已裁剪
                uncertainty2 = out2[6][keep_idx] if out2[6] is not None else None
            else:
                labels2 = labels
                uncertainty2 = out2[6]
            
            if self.adaptive_focal is not None:
                loss2_main = self.adaptive_focal(logits2, labels2, uncertainty2)
            else:
                loss2_main = F.binary_cross_entropy_with_logits(logits2, labels2)
            
            loss2_aux = F.binary_cross_entropy_with_logits(base_logits2, labels2) * self.early_exit_weight
            loss2 = loss2_main + loss2_aux
            
            loss2.backward()
            self.optimizer.step_sam_second()
        else:
            # Gradient check with early exit
            has_nan_grad = False
            ctr_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        break
                    ctr_grad_norm += p.grad.norm().item() ** 2
            ctr_grad_norm = ctr_grad_norm ** 0.5

            if has_nan_grad:
                logging.warning(f"【梯度爆炸】Step {self.global_step} 检测到NaN/Inf梯度，跳过参数更新")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

        if self.adaptive_focal is not None:
            self.adaptive_focal.step()

        self.global_step += 1

        with torch.no_grad():
            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            try:
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                train_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) >= 2 else 0.0
            except Exception:
                train_auc = 0.0

        self.train_auc_history.append(train_auc)

        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar('Loss/ctr', loss_main.item(), self.global_step)
            self.writer.add_scalar('Loss/aux', loss_aux.item(), self.global_step)
            self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
            self.writer.add_scalar('Stream/logits_mean', logits_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_std', logits_std, self.global_step)
            self.writer.add_scalar('Diagnostics/train_auc', train_auc, self.global_step)
            self.writer.add_scalar('Diagnostics/uncertainty_mean', uncertainty.mean().item() if uncertainty is not None else 0.0, self.global_step)
            self.writer.add_scalar('Curriculum/seq_len_limit', self.curriculum.get_seq_len_limit(epoch, list(self.curriculum.max_seq_lens.keys())[0]) if self.curriculum.max_seq_lens else 0, self.global_step)
            self.writer.add_scalar('Curriculum/label_smoothing', self.curriculum.get_label_smoothing(epoch, self.num_epochs), self.global_step)

        if self.global_step % 100 == 0:
            log_parts = [
                f"[Step {self.global_step}]",
                f"ctr={loss_main.item():.4f}",
                f"aux={loss_aux.item():.4f}",
                f"total={total_loss.item():.4f}",
                f"logits={logits_mean:.3f}±{logits_std:.3f}",
                f"trAUC={train_auc:.4f}",
                f"unc={uncertainty.mean().item():.3f}" if uncertainty is not None else "unc=N/A",
            ]
            logging.info(" | ".join(log_parts))

        return {
            'loss_ctr': loss_main.item(),
            'loss_aux': loss_aux.item(),
            'total_loss': total_loss.item(),
            'train_auc': train_auc,
            'uncertainty_mean': uncertainty.mean().item() if uncertainty is not None else 0.0,
        }

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, float]:
        self.model.eval()
        all_logits, all_labels, all_users = [], [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                device_batch = self._batch_to_device(batch)
                model_input = self._make_model_input(device_batch)
                out = self.model(model_input, task_id='ctr')
                all_logits.append(out[0].squeeze(-1).detach().cpu())
                all_labels.append(device_batch['label'].detach().cpu())
                all_users.extend(device_batch.get('user_id', []))

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()

        nan_mask = torch.isnan(all_logits)
        nan_ratio = nan_mask.float().mean().item()
        if nan_ratio > 0.99:
            logging.error(f"【严重错误】验证集logits {nan_ratio:.1%} 为NaN")
            return 0.0, float('inf'), 0.0
        elif nan_ratio > 0:
            valid_mask = ~nan_mask
            all_logits = all_logits[valid_mask]
            all_labels = all_labels[valid_mask]
            all_users = [u for i, u in enumerate(all_users) if not nan_mask[i].item()]

        if len(all_logits) == 0:
            return 0.0, float('inf'), 0.0

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        try:
            auc = float(roc_auc_score(labels_np, probs)) if len(np.unique(labels_np)) >= 2 else 0.0
        except Exception as e:
            logging.warning(f"AUC计算失败: {e}")
            auc = 0.0

        logloss = F.binary_cross_entropy_with_logits(
            all_logits.float(), all_labels.float()
        ).item()

        gauc = self._compute_gauc(labels_np, probs, all_users) if all_users else 0.0

        self.valid_auc_history.append(auc)
        return auc, logloss, gauc

    def _save_checkpoint(self, epoch: int, global_step: int, is_best: bool = False):
        dir_name = f"global_step{global_step}"
        if is_best:
            dir_name += ".best_model"
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))

        if self.train_config:
            import json
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(self.train_config, f, indent=2)

        logging.info(f"Saved checkpoint to {ckpt_dir}")
        return ckpt_dir

    def train(self) -> None:
        logging.info("Start training HeteroFormer v11.2 (Hybrid: RoPE + SwiGLU + Top-K + Explicit Interaction)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            loss_sum = 0.0
            aux_sum = 0.0
            valid_steps = 0
            nan_count = 0

            for step, batch in train_pbar:
                losses = self._train_step(batch, epoch=epoch)
                total_step += 1

                if math.isnan(losses['loss_ctr']):
                    nan_count += 1
                else:
                    loss_sum += losses['loss_ctr']
                    aux_sum += losses['loss_aux']
                    valid_steps += 1

                train_pbar.set_postfix({
                    "ctr": f"{losses['loss_ctr']:.4f}",
                    "aux": f"{losses['loss_aux']:.4f}",
                    "auc": f"{losses['train_auc']:.4f}",
                    "unc": f"{losses['uncertainty_mean']:.3f}",
                    "nan": nan_count,
                })

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    val_auc, val_logloss, val_gauc = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()
                    logging.info(f"Step {total_step} | Val AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}, GAUC: {val_gauc:.4f}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                        self.writer.add_scalar('AUC/gauc', val_gauc, total_step)

                    self.early_stopping(val_auc, self.model, {
                        "best_val_AUC": val_auc,
                        "best_val_logloss": val_logloss,
                        "best_val_GAUC": val_gauc,
                    })

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            avg_loss = loss_sum / max(valid_steps, 1)
            avg_aux = aux_sum / max(valid_steps, 1)
            logging.info(f"Epoch {epoch} | Avg CTR Loss: {avg_loss:.4f} | Avg Aux: {avg_aux:.4f} | NaN: {nan_count}")

            val_auc, val_logloss, val_gauc = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()
            logging.info(f"Epoch {epoch} | Val AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}, GAUC: {val_gauc:.4f}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                self.writer.add_scalar('AUC/gauc', val_gauc, total_step)

            is_best = (self.early_stopping.best_score is None or
                      val_auc >= self.early_stopping.best_score - self.early_stopping.delta)
            self._save_checkpoint(epoch, total_step, is_best=is_best)

            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
                "best_val_GAUC": val_gauc,
            })

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break