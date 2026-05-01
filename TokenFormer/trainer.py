# ==================== trainer.py (TokenFormer 最强版) ====================

"""PCVRTokenFormer Trainer: DEFUSE 连续标签 + 用户内 Pairwise + 多任务"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import EarlyStopping
from model import ModelInput


def sigmoid_focal_loss_stable(logits, targets, alpha=0.25, gamma=2.0, eps=1e-7):
    p = torch.sigmoid(logits)
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = alpha * (1 - p_t) ** gamma * ce_loss
    return loss.mean()


def compute_gauc(labels: np.ndarray, preds: np.ndarray, groups: np.ndarray) -> float:
    """Compute Group AUC (GAUC) weighted by group size."""
    unique_groups = np.unique(groups)
    aucs = []
    weights = []
    for g in unique_groups:
        mask = groups == g
        if len(np.unique(labels[mask])) < 2:
            continue
        try:
            g_auc = roc_auc_score(labels[mask], preds[mask])
            aucs.append(g_auc)
            weights.append(mask.sum())
        except ValueError:
            continue
    
    if len(aucs) == 0:
        return 0.0
    return float(np.average(aucs, weights=weights))


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class PCVRTokenFormerRankingTrainer:
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
        focal_alpha: float = 0.25,
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
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        ema_decay: float = 0.999,
        ema_enabled: bool = True,
        compute_global_ns: bool = False,
        global_ns_path: Optional[str] = None,
        grad_accumulation_steps: int = 1,
        use_hard_negative: bool = False,
        hard_neg_ratio: float = 0.3,
        use_gauc: bool = True,
        gauc_group_col: str = 'user_id',
        predict_conversion_time: bool = False,
        conversion_time_weight: float = 0.1,
        pos_weight: Optional[float] = None,
        use_class_weight: bool = True,
        pos_weight_update_steps: int = 100,
        # 【新增】Loss 组合参数
        loss_components: str = 'bce,continuous_pairwise',
        loss_weights: str = '0.7,0.3',
        multi_task: bool = False,
        ctr_weight: float = 0.3,
        ctcvr_weight: float = 0.2,
        # 【新增】学习率调度
        use_lr_scheduler: bool = True,
        warmup_ratio: float = 0.1,
        primary_metric: str = 'auc',
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.writer = writer
        self.schema_path = schema_path
        self.ns_groups_path = ns_groups_path
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = GradScaler(
            enabled=use_amp,
            init_scale=256.0 if amp_dtype == torch.float16 else 512.0
        )
        self.ema_enabled = ema_enabled
        if ema_enabled:
            self.ema = EMAModel(model, decay=ema_decay)
        else:
            self.ema = None

        # Dual optimizers
        self.sparse_optimizer: Optional[torch.optim.Optimizer] = None
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            logging.info(f"Sparse params: {len(sparse_params)}, Dense params: {len(dense_params)}")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay)
            self.dense_optimizer = torch.optim.AdamW(dense_params, lr=lr, betas=(0.9, 0.999))  # 【修改】beta2=0.999
        else:
            self.dense_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))

        # 【新增】学习率调度
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            total_steps = len(train_loader) * num_epochs
            warmup_steps = int(warmup_ratio * total_steps)
            self.dense_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.dense_optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.dense_optimizer, start_factor=0.1, total_iters=warmup_steps
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.dense_optimizer, T_max=total_steps - warmup_steps
                    ),
                ],
                milestones=[warmup_steps],
            )

        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reinit_sparse_after_epoch = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold = reinit_cardinality_threshold
        self.sparse_lr = sparse_lr
        self.sparse_weight_decay = sparse_weight_decay
        self.ckpt_params = ckpt_params or {}
        self.eval_every_n_steps = eval_every_n_steps
        self.train_config = train_config
        self.compute_global_ns_flag = compute_global_ns
        self.global_ns_path = global_ns_path
        self.grad_accumulation_steps = grad_accumulation_steps
        self._accum_step = 0
        
        self.use_hard_negative = use_hard_negative
        
        # P0: GAUC evaluation
        self.use_gauc = use_gauc
        self.gauc_group_col = gauc_group_col
        
        # P0: Multi-task
        self.predict_conversion_time = predict_conversion_time
        self.conversion_time_weight = conversion_time_weight
        
        # 【新增】Loss 组合
        self.loss_components = loss_components.split(',')
        self.loss_weights = [float(w) for w in loss_weights.split(',')]
        self.multi_task = multi_task
        self.ctr_weight = ctr_weight
        self.ctcvr_weight = ctcvr_weight
        
        # 全局统计变量
        self._global_pos_count = 0
        self._global_neg_count = 0
        self._debug_step = 0

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)
        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True
        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(self, global_step, is_best=False, skip_model_file=False):
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            if self.ema_enabled:
                self.ema.apply(self.model)
                torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model_ema.pt"))
                self.ema.restore(self.model)
        self._write_sidecar_files(ckpt_dir)
        if self.compute_global_ns_flag and self.global_ns_path and os.path.exists(self.global_ns_path):
            shutil.copy(self.global_ns_path, os.path.join(ckpt_dir, 'global_ns.pt'))
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self):
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)

    def _handle_validation_result(self, total_step, val_auc, val_logloss, val_gauc=None):
        # 【核心修改】用 AUC 主导 early stopping
        primary_metric = val_auc  # 直接用 AUC，不用 GAUC
        
        old_best = self.early_stopping.best_score
        is_likely_new_best = (old_best is None or primary_metric > old_best + self.early_stopping.delta)
        
        if not is_likely_new_best:
            self.early_stopping(primary_metric, self.model, {
                "best_val_AUC": val_auc,
                "best_val_GAUC": val_gauc,
                "best_val_logloss": val_logloss
            })
            return
        
        best_dir = os.path.join(self.save_dir, self._build_step_dir_name(total_step, is_best=True))
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()
        self.early_stopping(primary_metric, self.model, {
            "best_val_AUC": val_auc,
            "best_val_GAUC": val_gauc,
            "best_val_logloss": val_logloss
        })
        
        if self.early_stopping.best_score != old_best and os.path.exists(self.early_stopping.checkpoint_path):
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def compute_global_ns(self):
        self.model.eval()
        from torch.utils.data import DataLoader
        temp_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=None,          # 保持数据集提供的批次
            num_workers=2,            # 降低 worker 数，稳定性优先
            pin_memory=False
        )
        ns_sum = None
        nb_samples = 0
        for batch in tqdm(self.train_loader, desc="Computing global NS"):
            device_batch = self._batch_to_device(batch)
            with torch.no_grad():
                user_ns = self.model.user_ns_tokenizer(device_batch['user_int_feats'])
                item_ns = self.model.item_ns_tokenizer(device_batch['item_int_feats'])
                ns_parts = [user_ns]
                if self.model.has_user_dense:
                    ns_parts.append(F.silu(self.model.user_dense_proj(device_batch['user_dense_feats'])).unsqueeze(1))
                ns_parts.append(item_ns)
                if self.model.has_item_dense:
                    ns_parts.append(F.silu(self.model.item_dense_proj(device_batch['item_dense_feats'])).unsqueeze(1))
                ns_tokens = torch.cat(ns_parts, dim=1)
                if ns_sum is None:
                    ns_sum = ns_tokens.sum(dim=0)
                else:
                    ns_sum += ns_tokens.sum(dim=0)
            nb_samples += ns_tokens.shape[0]
        global_ns = ns_sum / nb_samples
        torch.save(global_ns, self.global_ns_path)
        logging.info(f"Computed global NS mean (shape={global_ns.shape}) saved to {self.global_ns_path}")
        return global_ns

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        seq_domains = device_batch['_seq_domains']
        seq_data = {}
        seq_lens = {}
        seq_time_buckets = {}
        seq_time_decay = {}
        # 【新增】action_type 传入序列嵌入
        action_types = device_batch.get('action_type', None)
        
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B, _, L = device_batch[domain].shape
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            if f'{domain}_time_decay' in device_batch:
                seq_time_decay[domain] = device_batch[f'{domain}_time_decay']
        
        user_id = device_batch.get('user_id', None)
        
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_time_decay=seq_time_decay if seq_time_decay else None,
            user_id=user_id,
            action_type=action_types,  # 【新增】
        )

    # ───【核心新增】连续 Pairwise Loss ───
    def _continuous_pairwise_loss(
        self, 
        logits: torch.Tensor, 
        soft_labels: torch.Tensor, 
        user_ids: torch.Tensor,
        margin: float = 1.0
    ) -> torch.Tensor:
        """用户内连续 Pairwise Loss：标签差异越大，期望分数差异越大"""
        if isinstance(user_ids, list):
          user_ids = torch.tensor(user_ids, device=logits.device, dtype=torch.long)
        elif isinstance(user_ids, np.ndarray):
          user_ids = torch.from_numpy(user_ids).to(logits.device)
        unique_users = torch.unique(user_ids)
        loss_sum = torch.tensor(0.0, device=logits.device)
        n_pairs = 0
        
        for user in unique_users:
            mask = (user_ids == user)
            user_logits = logits[mask]
            user_labels = soft_labels[mask]
            
            if len(user_labels) < 2:
                continue
            
            # 标签差异矩阵
            label_diff = user_labels.unsqueeze(1) - user_labels.unsqueeze(0)  # [N, N]
            logits_diff = user_logits.unsqueeze(1) - user_logits.unsqueeze(0)  # [N, N]
            
            # 只考虑标签差异 > 0.2 的正对（i 比 j 更像正样本）
            valid_mask = (label_diff > 0.2)
            if valid_mask.any():
                loss_sum += F.relu(margin * label_diff[valid_mask] - logits_diff[valid_mask]).sum()
                n_pairs += valid_mask.sum().item()
        
        return loss_sum / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=logits.device)

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[float, Optional[float]]:
        """Returns (main_loss, aux_loss)."""
        device_batch = self._batch_to_device(batch)
        
        # 【修改】DEFUSE 连续标签
        label = device_batch['label'].float()  # [0, 1] 连续值
        sample_weights = device_batch.get('sample_weights', None)
        if sample_weights is not None:
            sample_weights = sample_weights.to(self.device).float()
        
        action_type = device_batch.get('action_type', None)
        soft_label = device_batch.get('soft_label', None)
        
        # 数据监控
        batch_size = label.shape[0]
        n_pos = (label > 0.5).sum().item()  # 软标签 > 0.5 视为正
        n_neg = (label <= 0.5).sum().item()
        pos_ratio = n_pos / batch_size if batch_size > 0 else 0
        
        self._debug_step += 1
        if self._debug_step % 20 == 0:
            logging.info(f"📊 DEBUG: Batch Size={batch_size}, Pos={n_pos}, Neg={n_neg}, Pos Ratio={pos_ratio:.4f}")
            logging.info(f"📊 Label range: [{label.min():.3f}, {label.max():.3f}]")
            if sample_weights is not None:
                logging.info(f"⚖️ Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

        if self._accum_step == 0:
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            model_input = self._make_model_input(device_batch)
            
            # 前向
            if self.multi_task:
                cvr_logits, ctr_logits, ctcvr_logits = self.model(model_input)
                cvr_logits = cvr_logits.squeeze(-1)
                ctr_logits = ctr_logits.squeeze(-1)
                ctcvr_logits = ctcvr_logits.squeeze(-1)
                logits = cvr_logits  # 主任务用 CVR
            else:
                logits = self.model(model_input).squeeze(-1)
            
            # === Loss 计算 ===
            total_loss = torch.tensor(0.0, device=self.device)
            
            for comp, weight in zip(self.loss_components, self.loss_weights):
                if comp == 'bce':
                    if sample_weights is not None:
                        loss = F.binary_cross_entropy_with_logits(logits, label, weight=sample_weights)
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits, label)
                
                elif comp == 'focal':
                    loss = sigmoid_focal_loss_stable(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
                
                elif comp == 'pairwise':
                    # 硬标签 pairwise
                    hard_label = (label > 0.5).float()
                    loss = self._pairwise_auc_loss(logits, hard_label, device_batch['user_id'])
                
                elif comp == 'continuous_pairwise':
                    # 连续标签 pairwise
                    loss = self._continuous_pairwise_loss(logits, label, device_batch['user_id'])
                
                total_loss += weight * loss
            
            # 多任务辅助 Loss
            aux_loss = None
            if self.multi_task and action_type is not None:
                # CTR loss
                ctr_label = (action_type >= 1).float()
                loss_ctr = F.binary_cross_entropy_with_logits(ctr_logits, ctr_label)
                total_loss += self.ctr_weight * loss_ctr
                aux_loss = loss_ctr.item()
                
                # CTCVR loss（只在点击样本上算）
                click_mask = (action_type == 1)
                if click_mask.sum() > 0 and soft_label is not None:
                    ctcvr_label = soft_label[click_mask].to(self.device)
                    loss_ctcvr = F.binary_cross_entropy_with_logits(
                        ctcvr_logits[click_mask], ctcvr_label
                    )
                    total_loss += self.ctcvr_weight * loss_ctcvr
                    aux_loss += loss_ctcvr.item()
            
            total_loss = total_loss / self.grad_accumulation_steps

        self.scaler.scale(total_loss).backward()
        self._accum_step += 1
        
        if self._accum_step >= self.grad_accumulation_steps:
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
            
            # 【修改】分层梯度裁剪
            if hasattr(self.model, 'get_sparse_params'):
                sparse_params = self.model.get_sparse_params()
                dense_params = self.model.get_dense_params()
                torch.nn.utils.clip_grad_norm_(sparse_params, max_norm=0.5, foreach=False)
                torch.nn.utils.clip_grad_norm_(dense_params, max_norm=1.0, foreach=False)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()

            if self.ema_enabled:
                self.ema.update(self.model)
            
            # 【新增】学习率调度
            if self.use_lr_scheduler:
                self.dense_scheduler.step()
            
            self._accum_step = 0

        return total_loss.item() * self.grad_accumulation_steps, aux_loss

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float, Optional[float]]:
        print("Start Evaluation (TokenFormer DEFUSE)")
        self.model.eval()
        if self.ema_enabled:
            self.ema.apply(self.model)

        all_logits, all_labels, all_groups = [], [], []
        for batch in tqdm(self.valid_loader, total=len(self.valid_loader), desc="Validating"):
            with torch.no_grad():
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    logits, labels, groups = self._evaluate_step(batch)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                if groups is not None:
                    all_groups.append(groups.cpu() if isinstance(groups, torch.Tensor) else groups)

        if self.ema_enabled:
            self.ema.restore(self.model)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0).float()  # 连续标签
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            logging.warning(f"Filtering {nan_mask.sum()} NaN predictions")
            probs = probs[~nan_mask]
            labels_np = labels_np[~nan_mask]

        # 【修改】验证时用硬标签算 AUC（0.5 阈值）
        hard_labels = (labels_np > 0.5).astype(np.int64)
        
        # AUC
        if len(probs) == 0 or len(np.unique(hard_labels)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(hard_labels, probs))

        # GAUC
        gauc = None
        if self.use_gauc and len(all_groups) > 0:
            all_groups_np = np.concatenate([g.numpy() if isinstance(g, torch.Tensor) else g for g in all_groups])
            if not nan_mask.all():
                all_groups_np = all_groups_np[~nan_mask]
            gauc = compute_gauc(hard_labels, probs, all_groups_np)
            logging.info(f"GAUC: {gauc:.4f}")

        # LogLoss（用连续标签计算，反映概率校准）
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels).item() if len(valid_logits) > 0 else float('inf')
        
        return auc, logloss, gauc

    def _evaluate_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']  # 连续标签
    
        # 提取 GAUC 分组信息（用户 ID）
        groups = None
        if self.use_gauc and self.gauc_group_col in device_batch:
            g = device_batch[self.gauc_group_col]
            # 转移到 CPU，支持 tensor 和 list 两种形式
            if isinstance(g, torch.Tensor):
                g = g.cpu()
            groups = np.array(g)
    
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            model_input = self._make_model_input(device_batch)
            outputs = self.model.predict(model_input)
            if self.multi_task:
                # 多任务模式：predict 返回 (cvr_logits, ctr_logits, ctcvr_logits, repr)
                logits = outputs[0].squeeze(-1)
            else:
                # 单任务模式：predict 返回 (logits, repr)
                logits = outputs[0].squeeze(-1)
    
        return logits, label, groups

    def train(self):
        print("Start training (TokenFormer DEFUSE)")
        self._global_pos_count = 0
        self._global_neg_count = 0
        self._debug_step = 0

        if self.compute_global_ns_flag:
            self.compute_global_ns()
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), dynamic_ncols=True)
            loss_sum = 0.0
            aux_loss_sum = 0.0
            n_steps = 0
            
            for step, batch in train_pbar:
                batch_size = batch['label'].shape[0]
                if batch_size < 50:
                    logging.warning(f"Small batch detected: size={batch_size}, step={step}")
                
                loss, aux_loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                if aux_loss is not None:
                    aux_loss_sum += aux_loss
                n_steps += 1
                
                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    if aux_loss is not None:
                        self.writer.add_scalar('Loss/aux', aux_loss, total_step)
                    if self.use_lr_scheduler:
                        self.writer.add_scalar('LR/dense', self.dense_optimizer.param_groups[0]['lr'], total_step)
                
                postfix = {"loss": f"{loss:.4f}", "bs": batch_size}
                if aux_loss is not None:
                    postfix["aux"] = f"{aux_loss:.4f}"
                train_pbar.set_postfix(postfix)

                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss, val_gauc = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()
                    
                    metrics_str = f"Step {total_step} Validation | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}"
                    if val_gauc is not None:
                        metrics_str += f", GAUC: {val_gauc:.4f}"
                    logging.info(metrics_str)
                    
                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                        if val_gauc is not None:
                            self.writer.add_scalar('GAUC/valid', val_gauc, total_step)
                    
                    self._handle_validation_result(total_step, val_auc, val_logloss, val_gauc)
                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            avg_loss = loss_sum / n_steps if n_steps > 0 else 0
            avg_aux = aux_loss_sum / n_steps if n_steps > 0 and aux_loss_sum > 0 else 0
            logging.info(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}" + 
                        (f", Avg Aux: {avg_aux:.4f}" if avg_aux > 0 else ""))
            
            if hasattr(self.train_loader.dataset, 'get_sampling_stats'):
                stats = self.train_loader.dataset.get_sampling_stats()
                logging.info(f"Sampling stats: {stats}")
            
            val_auc, val_logloss, val_gauc = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()
            
            metrics_str = f"Epoch {epoch} Validation | AUC: {val_auc:.4f}, LogLoss: {val_logloss:.4f}"
            if val_gauc is not None:
                metrics_str += f", GAUC: {val_gauc:.4f}"
            logging.info(metrics_str)
            
            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)
                if val_gauc is not None:
                    self.writer.add_scalar('GAUC/valid', val_gauc, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss, val_gauc)
            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # Sparse reinit
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay)
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad after epoch {epoch}, restored {restored} low‑card params")