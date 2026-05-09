"""
PCVRHeteroFormer Trainer v8.0 - Symplectic Multi-Scale Optimization
====================================================================
基于辛约化的多任务训练动力学。

核心创新:
1. 辛约化梯度流: 通过守恒量消除任务冲突
2. 多尺度分解: 低频大步 + 高频小步，避免振荡
3. 课程学习: 先重构后结构化，平滑损失景观
4. 自然梯度: Fisher信息矩阵作为黎曼度量

Author: v8.0-symplectic
Date: 2026-05-09
"""

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


# ==============================================================================
# Adaptive Focal Loss (preserved, enhanced)
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
        return loss

    def step(self) -> None:
        self._step_count += 1
        if self.gamma_logit.grad is not None:
            torch.nn.utils.clip_grad_value_(self.gamma_logit, self.gamma_clip)
        self.gamma_optimizer.step()
        self.gamma_optimizer.zero_grad()


# ==============================================================================
# Symplectic Multi-Scale Optimizer
# ==============================================================================

class SymplecticMultiScaleOptimizer:
    """
    辛约化多尺度优化器。

    理论框架:
    - 将损失分解为低频(平滑)和高频(粗糙)成分
    - 低频成分定义慢流形，用较大学习率优化
    - 高频成分在正交补空间中优化，用较小学习率
    - 通过守恒量约束消除任务冲突

    任务频率分类:
    - low: focal, recon (大尺度结构)
    - high: div, spec, empty, align (精细调整)
    """
    def __init__(
        self,
        sparse_optimizer: torch.optim.Optimizer,
        dense_optimizer: torch.optim.Optimizer,
        gate_optimizer: Optional[torch.optim.Optimizer] = None,
        proto_optimizer: Optional[torch.optim.Optimizer] = None,
        target_ratio: float = 1.0,
        adapt_interval: int = 100,
        low_lr_scale: float = 1.0,
        high_lr_scale: float = 0.1,
    ):
        self.sparse_opt = sparse_optimizer
        self.dense_opt = dense_optimizer
        self.gate_opt = gate_optimizer
        self.proto_opt = proto_optimizer
        self.target_ratio = target_ratio
        self.adapt_interval = adapt_interval
        self.low_lr_scale = low_lr_scale
        self.high_lr_scale = high_lr_scale

        self.step_count = 0
        self._sparse_norm_history = deque(maxlen=adapt_interval)
        self._dense_norm_history = deque(maxlen=adapt_interval)
        self._proto_norm_history = deque(maxlen=adapt_interval)

        # 任务频率映射
        self.task_frequency = {
            'focal': 'low',
            'recon': 'low', 
            'div': 'high',
            'spec': 'high',
            'empty': 'high',
            'align': 'high',
        }

    def zero_grad(self) -> None:
        self.sparse_opt.zero_grad()
        self.dense_opt.zero_grad()
        if self.gate_opt is not None:
            self.gate_opt.zero_grad()
        if self.proto_opt is not None:
            self.proto_opt.zero_grad()

    def _compute_task_gradients(self, model: nn.Module, task_losses: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        分别计算每个任务的梯度。

        Returns:
            task_grads: {task_name: [grad_for_param1, grad_for_param2, ...]}
        """
        task_grads = {}

        for task_name, loss in task_losses.items():
            # 清空梯度
            model.zero_grad()

            # 反向传播
            loss.backward(retain_graph=True)

            # 收集梯度
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.clone().flatten())
                else:
                    grads.append(torch.zeros_like(p.flatten()))

            task_grads[task_name] = grads

        return task_grads

    def _symplectic_reduction(self, task_grads: Dict[str, List[torch.Tensor]], 
                              task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        辛约化: 将任务梯度分解为低频和高频成分。

        数学:
        1. 构建梯度矩阵 G = [g_1, ..., g_K]
        2. SVD分解: G = U S V^T
        3. 低频成分 = U[:, :r] @ U[:, :r]^T @ g_total (大奇异值)
        4. 高频成分 = g_total - g_low (小奇异值)
        """
        # 展平所有梯度
        task_names = list(task_grads.keys())
        num_params = len(task_grads[task_names[0]])

        # 构建梯度矩阵 [num_params_total, K]
        grad_matrix = []
        for task_name in task_names:
            flat_grad = torch.cat(task_grads[task_name])
            grad_matrix.append(flat_grad)

        G = torch.stack(grad_matrix, dim=1)  # [d, K]

        # 总梯度
        g_total = G.sum(dim=1)  # [d]

        # SVD分解 (低秩近似)
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)

            # 自适应截断: 保留累积奇异值90%的成分
            cumsum = torch.cumsum(S, dim=0)
            total = cumsum[-1] + 1e-8
            r = (cumsum / total < 0.9).sum().item() + 1
            r = min(r, len(S) - 1)
            r = max(r, 1)

            # 低频投影
            U_low = U[:, :r]  # [d, r]
            g_low = U_low @ (U_low.T @ g_total)  # [d]

        except:
            # SVD失败时回退到简单平均
            g_low = g_total.clone()

        # 高频成分 = 总梯度 - 低频投影
        g_high = g_total - g_low

        return g_low, g_high

    def _apply_reduced_gradients(self, model: nn.Module, g_low: torch.Tensor, g_high: torch.Tensor):
        """
        应用约化后的梯度，使用不同学习率。
        """
        # 组合梯度: 低频大步 + 高频小步
        g_combined = g_low + self.high_lr_scale * g_high

        # 重新塑形并设置梯度
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.grad = g_combined[offset:offset+numel].view_as(p)
            offset += numel

    def step(self, model: nn.Module, task_losses: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        辛约化优化步骤。

        Args:
            model: 模型
            task_losses: 各任务的损失字典 {task_name: loss_tensor}
        """
        self.step_count += 1

        if task_losses is not None and len(task_losses) > 1:
            # 多任务辛约化
            task_grads = self._compute_task_gradients(model, task_losses)
            g_low, g_high = self._symplectic_reduction(task_grads, task_losses)
            self._apply_reduced_gradients(model, g_low, g_high)

        # 标准优化器步骤
        sparse_norms, dense_norms, proto_norms = [], [], []

        for name, p in model.named_parameters():
            if p.grad is not None:
                if 'prototype' in name or 'proto_' in name or 'empty_prior' in name or 'match_vectors' in name or 'eta' in name:
                    proto_norms.append(p.grad.norm())
                elif 'embedding' in name or 'emb' in name.lower():
                    sparse_norms.append(p.grad.norm())
                else:
                    dense_norms.append(p.grad.norm())

        sparse_norm = torch.stack(sparse_norms).mean().item() if sparse_norms else 0.0
        dense_norm = torch.stack(dense_norms).mean().item() if dense_norms else 0.0
        proto_norm = torch.stack(proto_norms).mean().item() if proto_norms else 0.0

        self._sparse_norm_history.append(sparse_norm)
        self._dense_norm_history.append(dense_norm)
        self._proto_norm_history.append(proto_norm)

        diag = {
            'sparse_grad_norm': sparse_norm,
            'dense_grad_norm': dense_norm,
            'proto_grad_norm': proto_norm,
            'grad_norm_ratio': sparse_norm / (dense_norm + 1e-8),
        }

        # 自适应学习率调整
        if self.step_count % self.adapt_interval == 0:
            avg_sparse = np.mean(self._sparse_norm_history)
            avg_dense = np.mean(self._dense_norm_history)
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

            if proto_norms and proto_norm < dense_norm * 0.05:
                if self.proto_opt is not None:
                    for g in self.proto_opt.param_groups:
                        g['lr'] *= 1.2
                diag['proto_action'] = 'boosted_proto_lr'

        # 梯度裁剪
        if len(self._sparse_norm_history) >= self.adapt_interval:
            avg_sparse = np.mean(self._sparse_norm_history)
            if sparse_norm > 5.0 * avg_sparse:
                for group in self.sparse_opt.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.mul_(avg_sparse / (sparse_norm + 1e-8))
                diag['sparse_grad_rescaled'] = True

        # 优化器步骤
        self.sparse_opt.step()
        self.dense_opt.step()
        if self.gate_opt is not None:
            self.gate_opt.step()
        if self.proto_opt is not None:
            self.proto_opt.step()

        return diag

    def state_dict(self):
        d = {
            'sparse': self.sparse_opt.state_dict(),
            'dense': self.dense_opt.state_dict(),
            'step_count': self.step_count,
        }
        if self.gate_opt is not None:
            d['gate'] = self.gate_opt.state_dict()
        if self.proto_opt is not None:
            d['proto'] = self.proto_opt.state_dict()
        return d

    def load_state_dict(self, state_dict):
        self.sparse_opt.load_state_dict(state_dict['sparse'])
        self.dense_opt.load_state_dict(state_dict['dense'])
        if 'gate' in state_dict and self.gate_opt is not None:
            self.gate_opt.load_state_dict(state_dict['gate'])
        if 'proto' in state_dict and self.proto_opt is not None:
            self.proto_opt.load_state_dict(state_dict['proto'])
        self.step_count = state_dict.get('step_count', 0)


# ==============================================================================
# Trainer v8.0
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
        gate_anneal_steps: int = 2000,
        use_grafted_optimizer: bool = True,
        use_adaptive_focal: bool = True,
        enable_progressive_layers: bool = False,
        stochastic_depth_prob: float = 0.0,
        label_smoothing_strategy: str = 'hybrid',
        label_smoothing_max_eps: float = 0.05,
        label_smoothing_min_eps: float = 0.001,
        label_smoothing_anneal_steps: int = 5000,
        focal_alpha_pos: float = 0.5,
        focal_alpha_neg: float = 0.5,
        focal_max_gamma: float = 4.0,
        global_ctr: float = 0.01,
        # v8.0: Symplectic Loss weights
        recon_weight: float = 0.1,
        div_weight: float = 0.15,
        empty_weight: float = 0.1,
        spec_weight: float = 0.01,
        align_weight: float = 0.05,
        mp_weight: float = 0.005,
        # v8.0: Curriculum learning
        curriculum_warmup: int = 5000,
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
        self.curriculum_warmup = curriculum_warmup

        self.label_smoothing_strategy = label_smoothing_strategy
        self.label_smoothing_max_eps = label_smoothing_max_eps
        self.label_smoothing_min_eps = label_smoothing_min_eps
        self.label_smoothing_anneal_steps = label_smoothing_anneal_steps
        self.pos_rate_ema = global_ctr
        self.global_ctr = global_ctr

        # v8.0: Loss weights
        self.recon_weight = recon_weight
        self.div_weight = div_weight
        self.empty_weight = empty_weight
        self.spec_weight = spec_weight
        self.align_weight = align_weight
        self.mp_weight = mp_weight

        self._loss_history = {'focal': [], 'recon': [], 'div': [], 'empty': [], 'spec': [], 'align': [], 'mp': []}

        # 优化器构建
        if hasattr(model, 'get_sparse_params') and use_grafted_optimizer:
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            gate_params = model.get_gate_params() if hasattr(model, 'get_gate_params') else []
            scale_params = model.get_scale_params() if hasattr(model, 'get_scale_params') else []
            proto_params = model.get_proto_params() if hasattr(model, 'get_proto_params') else []

            sparse_opt = torch.optim.Adagrad(sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            scale_ids = {id(p) for p in scale_params}
            dense_param_groups = [
                {'params': [p for p in dense_params if id(p) not in scale_ids]}, 
                {'params': scale_params, 'lr': 1e-6, 'weight_decay': 0.0}
            ]
            dense_opt = torch.optim.AdamW(dense_param_groups, lr=lr, betas=(0.9, 0.98))
            gate_opt = torch.optim.AdamW(gate_params, lr=lr*2, betas=(0.9, 0.98), weight_decay=1e-3) if gate_params else None

            proto_opt = torch.optim.AdamW(
                proto_params,
                lr=lr * 2.0,
                betas=(0.9, 0.98),
                weight_decay=1e-3
            ) if proto_params else None

            self.optimizer = SymplecticMultiScaleOptimizer(
                sparse_opt, dense_opt, gate_opt, proto_opt,
                target_ratio=1.0, adapt_interval=200,
                low_lr_scale=1.0, high_lr_scale=0.1,
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
                f"v8.0 AdaptiveFocalLoss: max_gamma={focal_max_gamma}, "
                f"alpha_pos={focal_alpha_pos}, alpha_neg={focal_alpha_neg}"
            )
        else:
            self.adaptive_focal = None
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma

        if warmup_steps > 0:
            logging.info(f"Warmup: {warmup_steps} steps")

        logging.info(
            f"v8.0 Symplectic Loss: recon={recon_weight}, div={div_weight}, "
            f"empty={empty_weight}, spec={spec_weight}, align={align_weight}, mp={mp_weight}"
        )
        logging.info(f"Curriculum warmup: {curriculum_warmup} steps")

    def _get_dynamic_weights(self, global_step: int) -> Dict[str, float]:
        """
        课程学习: 动态调整损失权重。

        前curriculum_warmup步:
        - recon权重保持，div/spec逐渐引入
        之后:
        - 所有权重达到目标值
        """
        if global_step < self.curriculum_warmup:
            alpha = global_step / max(self.curriculum_warmup, 1)
            alpha_sq = alpha * alpha  # 二次缓增

            return {
                'recon': self.recon_weight,
                'div': self.div_weight * alpha_sq,
                'spec': self.spec_weight * alpha,
                'empty': self.empty_weight * alpha,
                'align': self.align_weight * alpha,
                'mp': self.mp_weight * alpha,
            }

        return {
            'recon': self.recon_weight,
            'div': self.div_weight,
            'spec': self.spec_weight,
            'empty': self.empty_weight,
            'align': self.align_weight,
            'mp': self.mp_weight,
        }

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
                    self.optimizer.sparse_opt.param_groups[0]['params'] = sparse_params
                elif self.sparse_optimizer is not None:
                    self.sparse_optimizer.param_groups[0]['params'] = sparse_params

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        labels = device_batch['label'].float()

        with torch.no_grad():
            pos_ratio = labels.mean().item()
            self.pos_rate_ema = 0.99 * self.pos_rate_ema + 0.01 * pos_ratio

        if self.loss_type in ('focal', 'bce') and self.label_smoothing_strategy != 'none':
            labels_smooth = self._smooth_labels(labels)
        else:
            labels_smooth = labels

        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
        if self.adaptive_focal is not None:
            self.adaptive_focal.gamma_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        # v8.0: 新的forward返回6个值
        logits, seq_feat, proto_weights, proto_repr, align_loss, log_det = self.model(model_input)
        logits = logits.squeeze(-1)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logging.warning(f"NaN/Inf in logits at step {self.global_step}, skipping batch")
            return float('nan')

        # ===== 1. Focal Loss (low frequency) =====
        if self.adaptive_focal is not None:
            loss_focal = self.adaptive_focal(logits, labels_smooth)
        elif self.loss_type == 'focal':
            p = torch.sigmoid(logits)
            p = torch.clamp(p, min=1e-7, max=1-1e-7)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels_smooth, reduction='none')
            p_t = p * labels_smooth + (1 - p) * (1 - labels_smooth)
            focal_weight = (1 - p_t) ** self.focal_gamma
            focal_weight = torch.clamp(focal_weight, max=10.0)
            alpha_t = self.focal_alpha * labels_smooth + (1 - self.focal_alpha) * (1 - labels_smooth)
            loss_focal = (alpha_t * focal_weight * bce_loss).mean()
        else:
            loss_focal = F.binary_cross_entropy_with_logits(logits, labels_smooth)

        # ===== 2. Prototype Reconstruction (low frequency) =====
        loss_recon = F.mse_loss(proto_repr, seq_feat)

        # ===== 3. Prototype Diversity (high frequency) =====
        weighted_pos = (proto_weights * labels.unsqueeze(1)).sum(dim=0)
        weighted_total = proto_weights.sum(dim=0)
        proto_ctr_est = weighted_pos / (weighted_total + 1e-8)

        active_mask = (weighted_total > 1e-3).float()
        active_count = active_mask.sum()

        if active_count > 1:
            active_ctr = proto_ctr_est * active_mask
            mean_ctr = active_ctr.sum() / (active_count + 1e-8)
            var_active = ((active_ctr - mean_ctr) ** 2 * active_mask).sum() / (active_count + 1e-8)
        else:
            var_active = torch.tensor(0.0, device=self.device)

        death_ratio = 1.0 - active_mask.mean()
        loss_div = -var_active + 0.2 * death_ratio

        # ===== 4. Empty Calibration (high frequency) =====
        first_domain = list(device_batch['_seq_domains'])[0] if len(device_batch['_seq_domains']) > 0 else None
        if first_domain is not None:
            seq_lens = device_batch[f'{first_domain}_len']
            empty_mask = (seq_lens == 0).float()
            empty_target = torch.full_like(logits, self.global_ctr)
            loss_empty = (empty_mask * F.binary_cross_entropy_with_logits(
                logits, empty_target, reduction='none'
            )).sum() / (empty_mask.sum() + 1e-8)
        else:
            loss_empty = torch.tensor(0.0, device=self.device)

        # ===== 5. vMF Volume (high frequency) =====
        loss_spec = -log_det  # 最大化体积

        # ===== 6. MP Spectral Regularization (high frequency) =====
        if hasattr(self.model, 'mp_regularizer'):
            first_proto_vq = list(self.model.prototype_vqs.values())[0] if len(self.model.prototype_vqs) > 0 else None
            if first_proto_vq is not None:
                mu, _ = first_proto_vq.get_prototypes()
                loss_mp = self.model.mp_regularizer(mu)
            else:
                loss_mp = torch.tensor(0.0, device=self.device)
        else:
            loss_mp = torch.tensor(0.0, device=self.device)

        # ===== 7. Fusion Alignment (high frequency) =====
        loss_align = align_loss

        # ===== 动态权重 =====
        weights = self._get_dynamic_weights(self.global_step)

        # ===== Total Loss =====
        loss = (loss_focal + 
                weights['recon'] * loss_recon + 
                weights['div'] * loss_div + 
                weights['empty'] * loss_empty + 
                weights['spec'] * loss_spec + 
                weights['mp'] * loss_mp +
                weights['align'] * loss_align)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss at step {self.global_step}, skipping batch")
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                self.dense_optimizer.zero_grad()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.zero_grad()
            return float('nan')

        # v8.0: 辛约化优化
        if self.optimizer is not None and hasattr(self.optimizer, '_symplectic_reduction'):
            # 分别计算各任务损失
            task_losses = {
                'focal': loss_focal,
                'recon': loss_recon,
                'div': weights['div'] * loss_div,
                'spec': weights['spec'] * loss_spec,
                'empty': weights['empty'] * loss_empty,
                'align': weights['align'] * loss_align,
                'mp': weights['mp'] * loss_mp,
            }

            # 使用辛约化优化器
            opt_diag = self.optimizer.step(self.model, task_losses)
        else:
            # 回退到标准反向传播
            loss.backward()
            opt_diag = {}
            if self.optimizer is not None:
                opt_diag = self.optimizer.step(self.model)
            else:
                self.dense_optimizer.step()
                if self.sparse_optimizer is not None:
                    self.sparse_optimizer.step()

        if self.adaptive_focal is not None:
            self.adaptive_focal.step()

        # 更新loss history
        for key, val in [('focal', loss_focal), ('recon', loss_recon), ('div', loss_div), 
                          ('empty', loss_empty), ('spec', loss_spec), ('align', loss_align), ('mp', loss_mp)]:
            self._loss_history[key].append(val.item())
            if len(self._loss_history[key]) > 500:
                self._loss_history[key] = self._loss_history[key][-500:]

        self.global_step += 1

        # 学习率warmup
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
                if self.optimizer.proto_opt is not None:
                    for g in self.optimizer.proto_opt.param_groups:
                        g['lr'] = self.lr * 2 * warmup_factor
            else:
                for g in self.dense_optimizer.param_groups:
                    g['lr'] = self.lr * warmup_factor
                if self.sparse_optimizer is not None:
                    for g in self.sparse_optimizer.param_groups:
                        g['lr'] = self.sparse_lr * warmup_factor

        # ===== TensorBoard 日志 =====
        with torch.no_grad():
            logits_mean = logits.mean().item()
            logits_std = logits.std().item()

            try:
                probs = torch.sigmoid(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                train_auc = roc_auc_score(labels_np, probs) if len(np.unique(labels_np)) >= 2 else 0.0
            except Exception:
                train_auc = 0.0

            # 原型诊断
            proto_usage = (proto_weights.argmax(dim=-1) != 0).float().mean().item()
            proto_entropy = -(proto_weights * torch.log(proto_weights + 1e-10)).sum(dim=-1).mean().item()
            proto_max_freq = (proto_weights.argmax(dim=-1) == proto_weights.argmax(dim=-1).mode().values).float().mean().item()

            # Sinkhorn列和均匀度
            col_sums = proto_weights.sum(dim=0)
            col_uniformity = 1.0 - (col_sums.std() / (col_sums.mean() + 1e-8))

            # 死亡原型统计
            dead_count = (weighted_total < 1e-3).sum().item()
            alive_count = proto_weights.size(1) - dead_count

            # v8.0: vMF集中度
            if hasattr(self.model, 'prototype_vqs') and len(self.model.prototype_vqs) > 0:
                first_vq = list(self.model.prototype_vqs.values())[0]
                _, kappa = first_vq.get_prototypes()
                kappa_mean = kappa.mean().item()
                kappa_std = kappa.std().item()
            else:
                kappa_mean = kappa_std = 0.0

        if self.writer:
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.writer.add_scalar('Loss/focal', loss_focal.item(), self.global_step)
            self.writer.add_scalar('Loss/recon', loss_recon.item(), self.global_step)
            self.writer.add_scalar('Loss/div', loss_div.item(), self.global_step)
            self.writer.add_scalar('Loss/empty', loss_empty.item(), self.global_step)
            self.writer.add_scalar('Loss/spec', loss_spec.item(), self.global_step)
            self.writer.add_scalar('Loss/align', loss_align.item(), self.global_step)
            self.writer.add_scalar('Loss/mp', loss_mp.item(), self.global_step)

            self.writer.add_scalar('Stream/logits_mean', logits_mean, self.global_step)
            self.writer.add_scalar('Stream/logits_std', logits_std, self.global_step)
            self.writer.add_scalar('Diagnostics/train_auc', train_auc, self.global_step)
            self.writer.add_scalar('Diagnostics/pos_rate_ema', self.pos_rate_ema, self.global_step)

            self.writer.add_scalar('Proto/usage', proto_usage, self.global_step)
            self.writer.add_scalar('Proto/entropy', proto_entropy, self.global_step)
            self.writer.add_scalar('Proto/max_freq', proto_max_freq, self.global_step)
            self.writer.add_scalar('Proto/dead_count', dead_count, self.global_step)
            self.writer.add_scalar('Proto/alive_count', alive_count, self.global_step)
            self.writer.add_scalar('Proto/col_uniformity', col_uniformity, self.global_step)
            self.writer.add_scalar('Proto/kappa_mean', kappa_mean, self.global_step)
            self.writer.add_scalar('Proto/kappa_std', kappa_std, self.global_step)

            for k in range(proto_weights.size(1)):
                freq = (proto_weights.argmax(dim=-1) == k).float().mean().item()
                self.writer.add_scalar(f'Proto/freq_{k}', freq, self.global_step)
                if k < len(proto_ctr_est):
                    self.writer.add_scalar(f'Proto/ctr_est_{k}', proto_ctr_est[k].item(), self.global_step)

            if self.adaptive_focal is not None:
                self.writer.add_scalar('Focal/gamma', self.adaptive_focal.get_gamma().item(), self.global_step)

            if opt_diag:
                self.writer.add_scalar('Grad/sparse', opt_diag.get('sparse_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/dense', opt_diag.get('dense_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/proto', opt_diag.get('proto_grad_norm', 0), self.global_step)
                self.writer.add_scalar('Grad/ratio', opt_diag.get('grad_norm_ratio', 0), self.global_step)

            if self.optimizer is not None:
                self.writer.add_scalar('LR/dense', self.optimizer.dense_opt.param_groups[0]['lr'], self.global_step)
                self.writer.add_scalar('LR/sparse', self.optimizer.sparse_opt.param_groups[0]['lr'], self.global_step)
                if self.optimizer.proto_opt is not None:
                    self.writer.add_scalar('LR/proto', self.optimizer.proto_opt.param_groups[0]['lr'], self.global_step)

            # v8.0: 课程学习进度
            if self.global_step < self.curriculum_warmup:
                self.writer.add_scalar('Curriculum/alpha', self.global_step / self.curriculum_warmup, self.global_step)
                for key, w in weights.items():
                    self.writer.add_scalar(f'Curriculum/weight_{key}', w, self.global_step)

        # 控制台日志
        if self.global_step % 100 == 0:
            log_parts = [f"[Step {self.global_step}] loss={loss.item():.4f}"]
            log_parts.append(f"focal={loss_focal.item():.4f}")
            log_parts.append(f"recon={loss_recon.item():.4f}")
            log_parts.append(f"div={loss_div.item():.4f}")
            log_parts.append(f"empty={loss_empty.item():.4f}")
            log_parts.append(f"spec={loss_spec.item():.4f}")
            log_parts.append(f"align={loss_align.item():.4f}")
            log_parts.append(f"mp={loss_mp.item():.4f}")
            log_parts.append(f"logits={logits_mean:.3f}±{logits_std:.3f}")
            log_parts.append(f"train_AUC={train_auc:.4f}")
            log_parts.append(f"pos_ema={self.pos_rate_ema:.4f}")
            log_parts.append(f"proto_usage={proto_usage:.3f}")
            log_parts.append(f"proto_entropy={proto_entropy:.3f}")
            log_parts.append(f"kappa={kappa_mean:.2f}±{kappa_std:.2f}")
            log_parts.append(f"col_uniformity={col_uniformity:.3f}")
            log_parts.append(f"dead_protos={dead_count}/{proto_weights.size(1)}")
            if self.adaptive_focal is not None:
                log_parts.append(f"gamma={self.adaptive_focal.get_gamma().item():.3f}")
            if opt_diag:
                log_parts.append(f"grad_ratio={opt_diag.get('grad_norm_ratio', 0):.2f}")
                if 'proto_action' in opt_diag:
                    log_parts.append(f"proto_action={opt_diag['proto_action']}")
            logging.info(" | ".join(log_parts))

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
        logging.info("Start training HeteroFormer v8.0 (Symplectic Multi-Scale)")
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
                train_pbar.set_postfix({
                    "loss": f"{loss:.4f}", 
                    "gamma": gamma_str, 
                    "nan": nan_count,
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
            proto_norm = sum(p.norm().item() for n, p in self.model.named_parameters()
                             if 'prototype' in n or 'proto_' in n or 'empty_prior' in n or 'match_vectors' in n or 'eta' in n)
            logging.info(
                f"Epoch {epoch} | EmbNorm: {emb_norm:.1f} | DenseNorm: {dense_norm:.1f} | "
                f"ProtoNorm: {proto_norm:.1f} | Ratio: {emb_norm/(dense_norm+1e-8):.1f}"
            )

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