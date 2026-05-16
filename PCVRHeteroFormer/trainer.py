"""
PCVRHeteroFormer Trainer v9.3 - Overfit-Aware & Generative Fusion
===================================================================
v9.3修改内容：
1. MetaAligner: 传入valid/train AUC，过拟合时自动提升辅助任务权重
2. 生成模块: diff/energy的gen_repr梯度流入共享层，正则化主表示
3. TensorBoard: 新增生成门控监控、过拟合信号监控
4. evaluate: 记录valid_auc历史供MetaAligner使用
"""

import os
import glob
import shutil
import logging
import math
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


# ==============================================================================
# v9.1: AdaptiveFocalLoss (preserved)
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gamma = self.get_gamma()
        self._gamma_history.append(gamma.detach().item())

        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        p_t = p * targets + (1 - p) * (1 - targets)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        focal_weight = (1 - p_t) ** gamma
        focal_weight = torch.clamp(focal_weight, max=5.0)

        alpha_t = targets * self.alpha_pos + (1 - targets) * self.alpha_neg
        focal_loss = (alpha_t * focal_weight * bce_loss).mean()

        gamma_reg = self.gamma_reg_coef * torch.abs(gamma - self.gamma_min) / (self.max_gamma - self.gamma_min + 1e-8)
        return focal_loss + gamma_reg

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
# v9.1: IsolatedOptimizer (preserved)
# ==============================================================================

class IsolatedOptimizer:
    def __init__(self, model: nn.Module, lr: float, sparse_lr: float):
        groups = model.get_param_groups()

        shared_params = (
            groups.get('shared_encoder', []) + groups.get('seq_encoder', []) +
            groups.get('cross_field', []) + groups.get('proto_cond', []) +
            groups.get('ctr_head', []) + groups.get('task_proj_ctr', []) +
            groups.get('fp', [])
        )
        self.shared_opt = torch.optim.AdamW(
            shared_params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-4,
        ) if shared_params else None

        if groups.get('sparse'):
            self.sparse_opt = torch.optim.Adagrad(
                groups['sparse'], lr=sparse_lr, weight_decay=1e-4)
        else:
            self.sparse_opt = None

        diff_params = groups.get('diff_head', []) + groups.get('task_proj_diff', [])
        self.diff_opt = torch.optim.AdamW(
            diff_params, lr=lr * 0.5,
            betas=(0.9, 0.98), weight_decay=1e-4,
        ) if diff_params else None

        energy_params = groups.get('energy_head', []) + groups.get('task_proj_energy', [])
        self.energy_opt = torch.optim.AdamW(
            energy_params, lr=lr * 0.3,
            betas=(0.9, 0.98), weight_decay=1e-4,
        ) if energy_params else None

        self.geo_opt = torch.optim.SGD(
            groups.get('proto_geo', []), lr=0.01, momentum=0.9
        ) if groups.get('proto_geo') else None

        self.meta_opt = torch.optim.Adam(
            groups.get('meta', []), lr=1e-3,
        ) if groups.get('meta') else None

        self.disc_opt = torch.optim.Adam(
            groups.get('disc', []), lr=lr * 0.1,
        ) if groups.get('disc') else None

        self.step_count = 0
        self.grad_norm_history = {
            name: deque(maxlen=100)
            for name in ['ctr', 'diff', 'energy', 'geo', 'fp']
        }

        self.nan_recovery_count = 0

    def zero_grad_shared(self):
        if self.shared_opt: self.shared_opt.zero_grad()
        if self.sparse_opt: self.sparse_opt.zero_grad()

    def zero_grad_diff(self):
        if self.diff_opt: self.diff_opt.zero_grad()

    def zero_grad_energy(self):
        if self.energy_opt: self.energy_opt.zero_grad()

    def zero_grad_geo(self):
        if self.geo_opt: self.geo_opt.zero_grad()

    def zero_grad_meta(self):
        if self.meta_opt: self.meta_opt.zero_grad()

    def step_shared(self):
        if self.shared_opt:
            torch.nn.utils.clip_grad_norm_(self.shared_opt.param_groups[0]['params'], 1.0)
            self.shared_opt.step()
        if self.sparse_opt:
            self.sparse_opt.step()

    def step_diff(self):
        if self.diff_opt:
            torch.nn.utils.clip_grad_norm_(self.diff_opt.param_groups[0]['params'], 1.0)
            self.diff_opt.step()

    def step_energy(self):
        if self.energy_opt:
            torch.nn.utils.clip_grad_norm_(self.energy_opt.param_groups[0]['params'], 1.0)
            self.energy_opt.step()

    def step_geo(self):
        if self.geo_opt:
            torch.nn.utils.clip_grad_norm_(self.geo_opt.param_groups[0]['params'], 2.0)
            self.geo_opt.step()

    def step_meta(self):
        if self.meta_opt:
            self.meta_opt.step()

    def record_grad_norm(self, name: str, model: nn.Module):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        self.grad_norm_history[name].append(total_norm ** 0.5)

    def get_grad_norms(self) -> Dict[str, float]:
        return {k: np.mean(v) if v else 0.0 for k, v in self.grad_norm_history.items()}

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

                for opt in [self.shared_opt, self.sparse_opt, self.diff_opt,
                           self.energy_opt, self.geo_opt, self.meta_opt, self.disc_opt]:
                    if opt is not None:
                        for state_name in list(opt.state.keys()):
                            if state_name is p:
                                opt.state[state_name] = {}

                if p.grad is not None:
                    p.grad.zero_()

        if nan_found:
            self.nan_recovery_count += 1
            logging.warning(f"【全局NaN恢复】第{self.nan_recovery_count}次触发")

        return nan_found

    def state_dict(self):
        return {
            'shared': self.shared_opt.state_dict() if self.shared_opt else None,
            'sparse': self.sparse_opt.state_dict() if self.sparse_opt else None,
            'diff': self.diff_opt.state_dict() if self.diff_opt else None,
            'energy': self.energy_opt.state_dict() if self.energy_opt else None,
            'geo': self.geo_opt.state_dict() if self.geo_opt else None,
            'meta': self.meta_opt.state_dict() if self.meta_opt else None,
            'disc': self.disc_opt.state_dict() if self.disc_opt else None,
            'step_count': self.step_count,
            'nan_recovery_count': self.nan_recovery_count,
        }

    def load_state_dict(self, state_dict):
        for name, opt in [
            ('shared', self.shared_opt), ('sparse', self.sparse_opt),
            ('diff', self.diff_opt), ('energy', self.energy_opt),
            ('geo', self.geo_opt), ('meta', self.meta_opt),
            ('disc', self.disc_opt),
        ]:
            if opt and name in state_dict and state_dict[name]:
                opt.load_state_dict(state_dict[name])
        self.step_count = state_dict.get('step_count', 0)
        self.nan_recovery_count = state_dict.get('nan_recovery_count', 0)


# ==============================================================================
# v9.3: Trainer — 过拟合感知 + 生成融合
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
        use_adaptive_focal: bool = True,
        enable_progressive_layers: bool = False,
        focal_alpha_pos: float = 0.5,
        focal_alpha_neg: float = 0.5,
        focal_max_gamma: float = 4.0,
        global_ctr: float = 0.01,
        use_diffusion: bool = True,
        use_energy: bool = True,
        use_domain_adversarial: bool = False,
        packing_weight: float = 0.01,
        energy_margin: float = 1.0,
        energy_weight: float = 0.1,
        diff_weight: float = 0.05,
        meta_update_interval: int = 100,
        curriculum_warmup: int = 5000,
        diffusion_warmup: int = 1000,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.use_diffusion = use_diffusion
        self.use_energy = use_energy
        self.packing_weight = packing_weight
        self.energy_margin = energy_margin
        self.energy_weight = energy_weight
        self.diff_weight = diff_weight
        self.meta_update_interval = meta_update_interval
        self.curriculum_warmup = curriculum_warmup
        self.diffusion_warmup = diffusion_warmup

        self.optimizer = IsolatedOptimizer(model, lr, sparse_lr)

        if use_adaptive_focal and loss_type == 'focal':
            self.adaptive_focal = AdaptiveFocalLoss(
                alpha_pos=focal_alpha_pos, alpha_neg=focal_alpha_neg,
                max_gamma=focal_max_gamma,
            )
        else:
            self.adaptive_focal = None

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config

        self.last_val_auc = 0.0
        self.meta_alpha = {'ctr': 1.0, 'diff': 0.0, 'energy': 0.0}
        self.loss_history = deque(maxlen=100)

        # 【v9.3】AUC历史供MetaAligner过拟合检测
        self.train_auc_history = deque(maxlen=5)
        self.valid_auc_history = deque(maxlen=5)

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
            B, _, L = device_batch[domain].shape
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device)
            )
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

    def _energy_margin_loss(self, energy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_mask = labels.unsqueeze(1)
        neg_mask = (1.0 - labels).unsqueeze(0)
        valid_pairs = pos_mask * neg_mask
        margin_matrix = F.relu(energy.unsqueeze(1) - energy.unsqueeze(0) + self.energy_margin)
        loss = (margin_matrix * valid_pairs).sum() / (valid_pairs.sum() + 1e-8)
        return loss

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        device_batch = self._batch_to_device(batch)
        labels = device_batch['label'].float()
        model_input = self._make_model_input(device_batch)

        self.optimizer.check_and_recover_nan_params(self.model)

        # ================================================================
        # Phase 1: CTR Forward
        # ================================================================
        self.model.train()
        out = self.model(model_input, task_id='ctr')

        if isinstance(out, tuple) and len(out) == 7 and isinstance(out[-1], dict):
            logits, proto_weights, proto_repr, kappa_mean, assign_entropy, gen_gates, uncertainty_pkg = out
        else:
            logits, proto_weights, proto_repr, kappa_mean, assign_entropy, gen_gates = out
            uncertainty_pkg = {}

        logits = logits.squeeze(-1)

        # ================================================================
        # Phase 2: 提前计算辅助任务 loss（用于 MetaAligner V2 加权）
        # ================================================================
        loss_diff = torch.tensor(0.0, device=self.device)
        loss_energy = torch.tensor(0.0, device=self.device)

        if self.use_diffusion:
            pred_noise, target_noise, t = self.model(model_input, task_id='diff')
            loss_diff_raw = F.mse_loss(pred_noise, target_noise)
            loss_diff = torch.clamp(loss_diff_raw, max=10.0)

        if self.use_energy:
            energy = self.model(model_input, task_id='energy')
            loss_energy_raw = self._energy_margin_loss(energy, labels)
            loss_energy = torch.clamp(loss_energy_raw, max=10.0)

        # ================================================================
        # Phase 3: MetaAligner V2 决策
        # ================================================================
        valid_auc = self.valid_auc_history[-1] if self.valid_auc_history else None
        train_auc_latest = self.train_auc_history[-1] if self.train_auc_history else None

        # 计算当前 grad_norms（用于 MetaAligner）
        grad_norms = self.optimizer.get_grad_norms()

        meta = self.model.meta_aligner(
            losses={'ctr': 0.0, 'diff': loss_diff.item(), 'energy': loss_energy.item()},
            grad_norms=grad_norms,
            valid_auc=valid_auc,
            train_auc=train_auc_latest,
            global_step=self.global_step,
        )

        # V2: 直接使用返回的权重
        aux_weight = meta['aux_weight']
        diff_weight = meta['diff_weight']
        energy_weight = meta['energy_weight']
        ctr_weight = meta['ctr_weight']

        # ================================================================
        # Phase 4: 总 Loss 计算与 Backward
        # ================================================================
        if self.adaptive_focal is not None:
            loss_ctr = self.adaptive_focal(logits, labels)
        else:
            loss_ctr = F.binary_cross_entropy_with_logits(logits, labels)

        # 序列融合熵正则（手术1）
        entropy_loss = getattr(self.model, '_cached_entropy_loss', torch.tensor(0.0, device=self.device))
        if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
            entropy_loss = torch.tensor(0.0, device=self.device)

        # V2: 单一总 loss 加权
        total_loss = (ctr_weight * loss_ctr + 
                      diff_weight * loss_diff + 
                      energy_weight * loss_energy + 
                      0.01 * entropy_loss)

        # NaN 检查
        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(0.0, device=total_loss.device),
            total_loss
        )

        # 梯度检查与 backward
        self.optimizer.zero_grad_shared()
        total_loss.backward(retain_graph=True)

        ctr_grad_norm = 0.0
        has_nan_grad = False
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_nan_grad = True
                    break
                ctr_grad_norm += p.grad.norm().item() ** 2
        ctr_grad_norm = ctr_grad_norm ** 0.5

        if has_nan_grad:
            logging.warning(f"【梯度爆炸】Step {self.global_step} 检测到NaN/Inf梯度，跳过参数更新")
            self.optimizer.zero_grad_shared()
            grad_norm_ctr = 0.0
        else:
            self.optimizer.record_grad_norm('ctr', self.model)
            self.optimizer.step_shared()
            grad_norm_ctr = self.optimizer.grad_norm_history['ctr'][-1] if self.optimizer.grad_norm_history['ctr'] else 0.0

        # ================================================================
        # Phase 5: Geometry 始终激活（packing loss）
        # ================================================================
        loss_packing = self.model.get_packing_loss()
        loss_packing = torch.clamp(loss_packing, max=10.0)
        if not (torch.isnan(loss_packing) or torch.isinf(loss_packing)):
            self.optimizer.zero_grad_geo()
            loss_packing.backward()
            self.optimizer.record_grad_norm('geo', self.model)
            self.optimizer.step_geo()

        # ================================================================
        # Phase 6: 更新历史 & Logging
        # ================================================================
        if self.adaptive_focal is not None:
            self.adaptive_focal.step()

        self.global_step += 1

        # 计算 train_auc
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

        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/ctr', loss_ctr.item(), self.global_step)
            self.writer.add_scalar('Loss/diff', loss_diff.item(), self.global_step)
            self.writer.add_scalar('Loss/energy', loss_energy.item(), self.global_step)
            self.writer.add_scalar('Loss/packing', loss_packing.item(), self.global_step)
            self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
            self.writer.add_scalar('Stream/logits_mean', logits_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_std', logits_std, self.global_step)
            self.writer.add_scalar('Diagnostics/train_auc', train_auc, self.global_step)
            self.writer.add_scalar('Diagnostics/grad_norm_ctr', grad_norm_ctr, self.global_step)
            self.writer.add_scalar('Diagnostics/nan_recovery_count', self.optimizer.nan_recovery_count, self.global_step)
            self.writer.add_scalar('Proto/kappa_mean', kappa_mean.item(), self.global_step)
            self.writer.add_scalar('Proto/assign_entropy', assign_entropy.mean().item(), self.global_step)

            # MetaAligner V2 监控
            mode_map = {'warmup': 0, 'probe_burn_in': 1, 'train_schedule': 2, 
                       'hybrid': 3, 'valid_pid': 4, 'interpolated': 5, 'unknown': -1}
            self.writer.add_scalar('Meta/mode', mode_map.get(meta.get('mode', 'unknown'), -1), self.global_step)
            self.writer.add_scalar('Meta/gap', meta.get('gap', 0.0), self.global_step)
            self.writer.add_scalar('Meta/aux_weight', meta.get('aux_weight', 0.0), self.global_step)
            self.writer.add_scalar('Meta/ctr_weight', meta.get('ctr_weight', 1.0), self.global_step)
            self.writer.add_scalar('Meta/diff_weight', meta.get('diff_weight', 0.0), self.global_step)
            self.writer.add_scalar('Meta/energy_weight', meta.get('energy_weight', 0.0), self.global_step)
            self.writer.add_scalar('Meta/ema_aux', meta.get('ema_aux', 0.0), self.global_step)
            self.writer.add_scalar('Meta/steps_since_valid', meta.get('steps_since_valid', 0), self.global_step)

            # 手术1: 序列融合门控监控
            if hasattr(self.model, '_cached_seq_gate_short') and self.model._cached_seq_gate_short is not None:
                gate_short = self.model._cached_seq_gate_short
                for i, domain in enumerate(self.model.seq_domains):
                    if i < gate_short.size(1):
                        self.writer.add_scalar(f'SeqFusion/short_{domain}', 
                                              gate_short[:, i].mean().item(), self.global_step)

            # 手术2: 显式交叉监控（如果有）
            if hasattr(self.model, 'explicit_cross') and self.model.explicit_cross is not None:
                cross_norm = self.model.explicit_cross.cross_weights.norm().item()
                self.writer.add_scalar('CrossLayer/weight_norm', cross_norm, self.global_step)

            # Gen 门控监控
            if gen_gates is not None:
                self.writer.add_scalar('Gen/gate_proto', gen_gates[:, 0].mean().item(), self.global_step)
                self.writer.add_scalar('Gen/gate_diff', gen_gates[:, 1].mean().item(), self.global_step)
                self.writer.add_scalar('Gen/gate_energy', gen_gates[:, 2].mean().item(), self.global_step)

        # Console log
        if self.global_step % 100 == 0:
            log_parts = [
                f"[Step {self.global_step}]",
                f"mode={meta.get('mode', '?')}",
                f"ctr={loss_ctr.item():.4f}",
                f"diff={loss_diff.item():.4f}",
                f"energy={loss_energy.item():.4f}",
                f"pack={loss_packing.item():.4f}",
                f"total={total_loss.item():.4f}",
                f"logits={logits_mean:.3f}±{logits_std:.3f}",
                f"trAUC={train_auc:.4f}",
                f"k={kappa_mean.item():.2f}",
                f"ent={assign_entropy.mean().item():.2f}",
                f"grad={grad_norm_ctr:.2f}",
                f"nanR={self.optimizer.nan_recovery_count}",
                f"meta=[{meta.get('ctr_weight', 1.0):.2f},{meta.get('diff_weight', 0.0):.2f},{meta.get('energy_weight', 0.0):.2f}]",
                f"aux_w={meta.get('aux_weight', 0.0):.3f}",
                f"gap={meta.get('gap', 0.0):.3f}",
                f"ema={meta.get('ema_aux', 0.0):.3f}",
            ]
            if gen_gates is not None:
                log_parts.append(
                    f"genG=[{gen_gates[:, 0].mean():.2f},{gen_gates[:, 1].mean():.2f},{gen_gates[:, 2].mean():.2f}]"
                )
            logging.info(" | ".join(log_parts))

        return {
            'loss_ctr': loss_ctr.item(),
            'loss_diff': loss_diff.item(),
            'loss_energy': loss_energy.item(),
            'loss_packing': loss_packing.item(),
            'total_loss': total_loss.item(),
            'train_auc': train_auc,
            'meta_mode': meta.get('mode', 'unknown'),
            'meta_aux_weight': meta.get('aux_weight', 0.0),
        }

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        self.model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in self.valid_loader:
                device_batch = self._batch_to_device(batch)
                model_input = self._make_model_input(device_batch)
                out = self.model(model_input, task_id='ctr')
                all_logits.append(out[0].squeeze(-1).detach().cpu())
                all_labels.append(device_batch['label'].detach().cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).long()

        nan_mask = torch.isnan(all_logits)
        nan_ratio = nan_mask.float().mean().item()

        if nan_ratio > 0.99:
            logging.error(f"【严重错误】验证集logits {nan_ratio:.1%} 为NaN，模型已崩溃")
            return 0.0, float('inf')
        elif nan_ratio > 0:
            logging.warning(f"【警告】验证集logits {nan_ratio:.1%} 为NaN，已过滤")
            valid_mask = ~nan_mask
            all_logits = all_logits[valid_mask]
            all_labels = all_labels[valid_mask]

        if len(all_logits) == 0:
            return 0.0, float('inf')

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

        # 记录 valid AUC 历史（关键：供 MetaAligner 使用）
        self.valid_auc_history.append(auc)
        return auc, logloss

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
        logging.info("Start training HeteroFormer v9.3 (Overfit-Aware + Generative Fusion)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            loss_sum = 0.0
            valid_steps = 0
            nan_count = 0

            for step, batch in train_pbar:
                losses = self._train_step(batch)
                total_step += 1

                if math.isnan(losses['loss_ctr']):
                    nan_count += 1
                else:
                    loss_sum += losses['loss_ctr']
                    valid_steps += 1

                train_pbar.set_postfix({
                    "ctr": f"{losses['loss_ctr']:.4f}",
                    "diff": f"{losses['loss_diff']:.4f}",
                    "auc": f"{losses['train_auc']:.4f}",
                    "nan": nan_count,
                    "nanR": self.optimizer.nan_recovery_count,
                })

                # Validation
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()
                    logging.info(f"Step {total_step} | Val AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self.early_stopping(val_auc, self.model, {
                        "best_val_AUC": val_auc,
                        "best_val_logloss": val_logloss,
                    })

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            # Epoch end validation
            avg_loss = loss_sum / max(valid_steps, 1)
            logging.info(f"Epoch {epoch} | Avg CTR Loss: {avg_loss:.4f} | NaN: {nan_count} | nanR: {self.optimizer.nan_recovery_count}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()
            logging.info(f"Epoch {epoch} | Val AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            is_best = (self.early_stopping.best_score is None or 
                      val_auc >= self.early_stopping.best_score - self.early_stopping.delta)
            self._save_checkpoint(epoch, total_step, is_best=is_best)

            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break