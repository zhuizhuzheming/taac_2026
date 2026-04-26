"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

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

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss + Pairwise AUC Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

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
        sparse_lr: float = 0.01,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        self.schema_path: Optional[str] = schema_path
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay, lr_decay=1e-4
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.999)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.999)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config

        # Warmup + CosineAnnealing for dense optimizer
        total_steps = len(train_loader) * num_epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        self.dense_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.dense_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.dense_optimizer, start_factor=0.1, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.dense_optimizer, T_max=max(1, total_steps - warmup_steps)
                ),
            ],
            milestones=[warmup_steps],
        )

        self.global_step: int = 0
        self.accumulation_steps: int = 4

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"accumulation_steps={self.accumulation_steps}, "
                     f"warmup_steps={warmup_steps}, total_steps={total_steps}")

    @staticmethod
    def _pairwise_auc_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        pos_mask = labels == 1
        neg_mask = labels == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        pos_expanded = pos_logits.unsqueeze(1)   # (N_pos, 1)
        neg_expanded = neg_logits.unsqueeze(0)   # (1, N_neg)
        pairwise_loss = F.relu(margin - (pos_expanded - neg_expanded))
        return pairwise_loss.mean()

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

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
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
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        print("Start training (PCVRHyFormer)")
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                self.global_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, self.global_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self.eval_every_n_steps > 0 and self.global_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {self.global_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {self.global_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, self.global_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, self.global_step)

                    self._handle_validation_result(self.global_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {self.global_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, self.global_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, self.global_step)

            self._handle_validation_result(self.global_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state: Dict[int, Any] = {}
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
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input).squeeze(-1)  # (B,)

        # ========== Hard Negative Mining ==========
        # 1. 分离正负样本
        pos_mask = label == 1
        neg_mask = label == 0
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        # 2. 如果正负样本都存在，进行 Hard Negative 筛选
        if n_pos > 0 and n_neg > 0:
            with torch.no_grad():
                # 计算每个负样本的 BCE loss（不 reduction）
                neg_logits = logits[neg_mask]
                neg_labels = label[neg_mask]
                neg_losses = F.binary_cross_entropy_with_logits(
                    neg_logits, neg_labels, reduction='none'
                )  # (N_neg,)

                # 选 loss 最大的 k 个负样本 = "模型最困惑的" = Hard Negative
                k = max(int(n_pos), int(0.15 * n_neg))  # 至少保留 n_pos 个，或 15% 负样本
                k = min(k, n_neg)  # 不超过实际负样本数

                # 获取负样本在原始 batch 中的索引
                neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)  # (N_neg,)
                
                # topk 返回的是负样本子集内的索引
                _, topk_local_idx = torch.topk(neg_losses, k, largest=True, sorted=False)
                hard_neg_global_idx = neg_indices[topk_local_idx]  # 映射回 batch 全局索引

            # 3. 构建筛选后的样本 mask：正样本全留 + Hard Negative
            keep_mask = torch.zeros_like(label, dtype=torch.bool)
            keep_mask[pos_mask] = True          # 所有正样本
            keep_mask[hard_neg_global_idx] = True  # 精选的 Hard Negative

            # 4. 只在筛选后的样本上计算 loss（无加权，全量梯度）
            kept_logits = logits[keep_mask]
            kept_labels = label[keep_mask]
            bce = F.binary_cross_entropy_with_logits(kept_logits, kept_labels)
        else:
            # 极端情况：只有一类样本，退化为普通 BCE
            bce = F.binary_cross_entropy_with_logits(logits, label)

        # ========== Pairwise Loss（可选，建议先关闭验证） ==========
        # pairwise = self._pairwise_auc_loss(logits, label, margin=1.0)
        # loss = 0.9 * bce + 0.1 * pairwise  # 降低 Pairwise 权重，BCE 主导

        loss = bce  # 先纯 BCE，确认 Hard Negative 有效后再加 Pairwise

        loss = loss / self.accumulation_steps
        loss.backward()

        # ========== 梯度累积 + 裁剪（保持不变） ==========
        if (self.global_step + 1) % self.accumulation_steps == 0:
            if hasattr(self.model, 'get_sparse_params'):
                sparse_params = self.model.get_sparse_params()
                dense_params = self.model.get_dense_params()
                torch.nn.utils.clip_grad_norm_(sparse_params, max_norm=0.5, foreach=False)
                torch.nn.utils.clip_grad_norm_(dense_params, max_norm=1.0, foreach=False)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

            self.dense_optimizer.step()
            self.dense_scheduler.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()

        return loss.item() * self.accumulation_steps

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)
        logits = logits.squeeze(-1)

        return logits, label