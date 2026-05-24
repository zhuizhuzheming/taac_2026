"""
PCVRHeteroFormer Trainer v10 - Generative Semantics & Decoupled Optimization
==============================================================================
v10 修改核心：
1. 生成模块（prototype + diffusion_explainer + energy_calibrator）自监督解耦：
   - 所有生成输出进入 CTR 路径前均已 detach()
   - 生成模块仅通过 ib/recon/ortho/packing/energy_ranking loss 训练
2. 单次 total_loss backward，通过架构 detach 实现梯度隔离（无需多步 backward）
3. MetaAligner 简化：仅输出 aux_weight 控制生成模块总权重
4. TensorBoard 新增生成模块诊断指标（ib/recon/ortho/diff_quality/energy）
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
# v10: AdaptiveFocalLoss (preserved)
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
# v10: IsolatedOptimizer（简化分组：shared / gen / sparse / meta）
# ==============================================================================

class IsolatedOptimizer:
    def __init__(self, model: nn.Module, lr: float, sparse_lr: float):
        groups = model.get_param_groups()

        shared_params = (
            groups.get('shared_encoder', []) + groups.get('seq_encoder', []) +
            groups.get('cross_field', []) + groups.get('ctr_head', [])
        )
        self.shared_opt = torch.optim.AdamW(
            shared_params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-4,
        ) if shared_params else None

        if groups.get('sparse'):
            self.sparse_opt = torch.optim.Adagrad(
                groups['sparse'], lr=sparse_lr, weight_decay=1e-4)
        else:
            self.sparse_opt = None

        gen_params = groups.get('gen_module', [])
        self.gen_opt = torch.optim.AdamW(
            gen_params, lr=lr * 0.3,
            betas=(0.9, 0.98), weight_decay=1e-4,
        ) if gen_params else None

        self.meta_opt = torch.optim.Adam(
            groups.get('meta', []), lr=1e-3,
        ) if groups.get('meta') else None

        self.step_count = 0
        self.grad_norm_history = {
            name: deque(maxlen=100)
            for name in ['ctr', 'gen']
        }
        self.nan_recovery_count = 0

    def zero_grad_shared(self):
        if self.shared_opt: self.shared_opt.zero_grad()
        if self.sparse_opt: self.sparse_opt.zero_grad()

    def zero_grad_gen(self):
        if self.gen_opt: self.gen_opt.zero_grad()

    def zero_grad_meta(self):
        if self.meta_opt: self.meta_opt.zero_grad()

    def step_shared(self):
        if self.shared_opt:
            torch.nn.utils.clip_grad_norm_(self.shared_opt.param_groups[0]['params'], 1.0)
            self.shared_opt.step()
        if self.sparse_opt:
            self.sparse_opt.step()

    def step_gen(self):
        if self.gen_opt:
            torch.nn.utils.clip_grad_norm_(self.gen_opt.param_groups[0]['params'], 1.0)
            self.gen_opt.step()

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

                for opt in [self.shared_opt, self.sparse_opt, self.gen_opt, self.meta_opt]:
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
            'gen': self.gen_opt.state_dict() if self.gen_opt else None,
            'meta': self.meta_opt.state_dict() if self.meta_opt else None,
            'step_count': self.step_count,
            'nan_recovery_count': self.nan_recovery_count,
        }

    def load_state_dict(self, state_dict):
        for name, opt in [
            ('shared', self.shared_opt), ('sparse', self.sparse_opt),
            ('gen', self.gen_opt), ('meta', self.meta_opt),
        ]:
            if opt and name in state_dict and state_dict[name]:
                opt.load_state_dict(state_dict[name])
        self.step_count = state_dict.get('step_count', 0)
        self.nan_recovery_count = state_dict.get('nan_recovery_count', 0)


# ==============================================================================
# v10: Trainer
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
        packing_weight: float = 0.01,
        energy_margin: float = 1.0,
        energy_weight: float = 0.1,
        ib_weight: float = 0.01,
        recon_weight: float = 0.05,
        ortho_weight: float = 0.01,
        **kwargs,  # 吸收旧版兼容参数
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.global_step = 0
        self.packing_weight = packing_weight
        self.energy_margin = energy_margin
        self.energy_weight = energy_weight
        self.ib_weight = ib_weight
        self.recon_weight = recon_weight
        self.ortho_weight = ortho_weight

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
        self.loss_history = deque(maxlen=100)

        # AUC历史供MetaAligner使用
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

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        device_batch = self._batch_to_device(batch)
        labels = device_batch['label'].float()
        model_input = self._make_model_input(device_batch)

        self.optimizer.check_and_recover_nan_params(self.model)

        # Phase 0: 清零所有梯度
        self.optimizer.zero_grad_shared()
        self.optimizer.zero_grad_gen()
        self.optimizer.zero_grad_meta()

        # Phase 1: CTR Forward
        self.model.train()
        out = self.model(model_input, task_id='ctr')

        (
            logits, proto_weights, proto_repr, kappa_mean, assign_entropy,
            diff_explain, uncertainty, gen_align_loss, energy_score,
            packing_loss, base_logits, final_repr
        ) = out

        logits = logits.squeeze(-1)
        base_logits = base_logits.squeeze(-1)

        # Phase 2: CTR loss
        if self.adaptive_focal is not None:
            loss_ctr = self.adaptive_focal(logits, labels)
        else:
            loss_ctr = F.binary_cross_entropy_with_logits(logits, labels)

        loss_ctr_base = F.binary_cross_entropy_with_logits(base_logits, labels)

        # Phase 3: Energy target loss
        energy_target_loss = self.model.energy_calibrator.compute_target(
            energy_score, base_logits.detach(), labels
        )

        # Phase 4: Gen loss
        gen_loss = (
            gen_align_loss +
            energy_target_loss * 0.1 +
            packing_loss * self.packing_weight
        )

        # Phase 5: 【关键】计算 residual_benefit 后传入 MetaAligner
        with torch.no_grad():
            residual_benefit = loss_ctr_base.item() - loss_ctr.item()

        valid_auc = self.valid_auc_history[-1] if self.valid_auc_history else None
        train_auc_latest = self.train_auc_history[-1] if self.train_auc_history else None
        grad_norms = self.optimizer.get_grad_norms()

        meta = self.model.meta_aligner(
            valid_auc=valid_auc,
            train_auc=train_auc_latest,
            global_step=self.global_step,
            ctr_loss=loss_ctr.item(),
            uncertainty_mean=uncertainty.mean().item(),
            residual_benefit=residual_benefit,
        )

        aux_weight = meta['aux_weight']

        # Phase 6: 总 Loss
        total_loss = loss_ctr + aux_weight * gen_loss

        total_loss = torch.where(
            torch.isnan(total_loss) | torch.isinf(total_loss),
            torch.tensor(0.0, device=total_loss.device),
            total_loss
        )

        total_loss.backward()

        # Phase 7: 梯度检查与参数更新
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
            self.optimizer.zero_grad_shared()
            self.optimizer.zero_grad_gen()
            grad_norm_ctr = 0.0
        else:
            self.optimizer.record_grad_norm('ctr', self.model)
            self.optimizer.record_grad_norm('gen', self.model)
            self.optimizer.step_shared()
            if gen_loss.item() > 0:
                self.optimizer.step_gen()
            grad_norm_ctr = self.optimizer.grad_norm_history['ctr'][-1] if self.optimizer.grad_norm_history['ctr'] else 0.0

        # Phase 8: 更新历史 & Logging
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
            self.writer.add_scalar('Loss/ctr', loss_ctr.item(), self.global_step)
            self.writer.add_scalar('Loss/ctr_base', loss_ctr_base.item(), self.global_step)
            self.writer.add_scalar('Loss/gen', gen_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/gen_align', gen_align_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/energy_target', energy_target_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/packing', packing_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/total', total_loss.item(), self.global_step)
            self.writer.add_scalar('Stream/logits_mean', logits_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_std', logits_std, self.global_step)
            self.writer.add_scalar('Diagnostics/train_auc', train_auc, self.global_step)
            self.writer.add_scalar('Diagnostics/grad_norm_ctr', grad_norm_ctr, self.global_step)
            self.writer.add_scalar('Diagnostics/nan_recovery_count', self.optimizer.nan_recovery_count, self.global_step)
            self.writer.add_scalar('Proto/kappa_mean', kappa_mean.item(), self.global_step)
            self.writer.add_scalar('Proto/assign_entropy', assign_entropy.mean().item(), self.global_step)

            self.writer.add_scalar('Meta/mode', 0 if meta.get('mode') == 'valid_pid' else 1, self.global_step)
            self.writer.add_scalar('Meta/gap', meta.get('gap', 0.0), self.global_step)
            self.writer.add_scalar('Meta/aux_weight', meta.get('aux_weight', 0.0), self.global_step)
            self.writer.add_scalar('Meta/ema_aux', meta.get('ema_aux', 0.0), self.global_step)
            self.writer.add_scalar('Meta/residual_benefit', meta.get('residual_benefit', 0.0), self.global_step)

            self.writer.add_scalar('Gen/uncertainty_mean', uncertainty.mean().item(), self.global_step)
            self.writer.add_scalar('Gen/energy_mean', energy_score.mean().item(), self.global_step)
            self.writer.add_scalar('Gen/energy_std', energy_score.std().item(), self.global_step)

            self.writer.add_scalar('Residual/benefit', residual_benefit, self.global_step)
            self.writer.add_scalar('Residual/base_vs_final_logits',
                (base_logits.mean() - logits.mean()).item(), self.global_step)

        # Console log
        if self.global_step % 100 == 0:
            log_parts = [
                f"[Step {self.global_step}]",
                f"mode={meta.get('mode', '?')}",
                f"ctr={loss_ctr.item():.4f}",
                f"ctr_base={loss_ctr_base.item():.4f}",
                f"gen={gen_loss.item():.4f}",
                f"align={gen_align_loss.item():.4f}",
                f"e_target={energy_target_loss.item():.4f}",
                f"total={total_loss.item():.4f}",
                f"logits={logits_mean:.3f}±{logits_std:.3f}",
                f"trAUC={train_auc:.4f}",
                f"k={kappa_mean.item():.2f}",
                f"grad={grad_norm_ctr:.2f}",
                f"nanR={self.optimizer.nan_recovery_count}",
                f"meta=[ctr=1.0,aux={meta.get('aux_weight', 0.0):.3f}]",
                f"gap={meta.get('gap', 0.0):.3f}",
                f"resBen={residual_benefit:+.3f}",
                f"unc={uncertainty.mean().item():.3f}",
                f"energy={energy_score.mean().item():.3f}",
            ]
            logging.info(" | ".join(log_parts))

        return {
            'loss_ctr': loss_ctr.item(),
            'loss_ctr_base': loss_ctr_base.item(),
            'loss_gen': gen_loss.item(),
            'loss_gen_align': gen_align_loss.item(),
            'loss_energy_target': energy_target_loss.item(),
            'loss_packing': packing_loss.item(),
            'total_loss': total_loss.item(),
            'train_auc': train_auc,
            'meta_mode': meta.get('mode', 'unknown'),
            'meta_aux_weight': meta.get('aux_weight', 0.0),
            'residual_benefit': residual_benefit,
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
        logging.info("Start training HeteroFormer v10 (Generative Semantics & Decoupled Optimization)")
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
                    "gen": f"{losses['loss_gen']:.4f}",
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
