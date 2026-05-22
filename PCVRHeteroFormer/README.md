# PCVRHeteroFormer v10 — Generative Semantics & Decoupled Optimization

> **版本**: v10.2  
> **主题**: 生成式语义原型 + 解耦优化框架  
> **适用任务**: 点击转化率（CVR）预估 / 用户行为序列建模

---

## 1. 概述

PCVRHeteroFormer v10 是一套面向**异构特征空间**与**长程行为序列**的深度 CTR/CVR 预估框架。本版本的核心创新在于引入**生成式语义层（Generative Semantics）**，通过**动态原型流形（Dynamic Prototype Manifold）**将用户行为序列编码为可解释的语义原型分布，并采用**解耦优化（Decoupled Optimization）**策略实现生成模块与判别模块的梯度隔离，从而在保持主任务（CTR）稳定训练的同时，利用自监督信号提升序列表征质量。

---

## 2. 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入层 (Input Layer)                       │
│  ├─ 用户离散特征 (User Int)                                      │
│  ├─ 物品离散特征 (Item Int)                                      │
│  ├─ 用户稠密特征 (User Dense)                                    │
│  ├─ 物品稠密特征 (Item Dense)                                    │
│  └─ 多域行为序列 (Multi-Domain Sequences: seq_a/b/c/d)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    基础编码层 (Base Encoding)                      │
│  ├─ MultiViewEncoder ──→ 用户/物品/上下文表征 (z_shared)          │
│  └─ ContinuousSequenceEncoder ──→ SSM-based 序列编码             │
│       (short / long / static / full_seq)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  生成式语义层 (Generative Semantics)                │
│  ├─ DynamicPrototypeManifold                                    │
│  │   ├─ CayleyRotation ──→ 用户条件化原型旋转                     │
│  │   ├─ LangevinSinkhorn ──→ 最优传输分配                        │
│  │   └─ 输出: proto_weights, proto_repr, kappa                  │
│  ├─ DiffusionExplainer ──→ 信息瓶颈残差编码 (diff_explain)        │
│  └─ EnergyCalibrator ──→ 预测误差能量估计 (energy_score)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  特征交互层 (Feature Interaction)                   │
│  ├─ BilinearCrossLayer ──→ 显式二阶域间交互                     │
│  └─ ProtoConditionedCrossFieldNet                               │
│       ├─ Multi-Head Self-Attention                               │
│       ├─ Proto-bias Attention (温和注入原型权重)                   │
│       ├─ SeqFiLM 条件化调制                                      │
│       └─ Diff-gate FFN (扩散解释门控)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    预测层 (Prediction Head)                       │
│  └─ CalibratedCTRHead                                           │
│       ├─ Static / Proto / Pos-boost 三路融合                     │
│       ├─ Uncertainty-gated Residual (diff_explain)               │
│       └─ 温度缩放 + 自适应偏置 (logit_temperature/bias)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   元控制层 (Meta Control)                         │
│  └─ MetaAligner ──→ 过拟合感知辅助权重调度 (aux_weight)            │
│       ├─ 残差收益驱动 (residual_benefit)                         │
│       ├─ 过拟合硬上限 (train-valid AUC gap)                     │
│       └─ Valid-AUC 缺失时保守插值                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键创新点

### 3.1 生成式语义原型 (DynamicPrototypeManifold)

传统序列编码直接输出固定向量，难以处理行为稀疏、意图漂移等问题。v10 引入**语义原型空间**：

- **全局原型 `η`**: 可学习的 `num_codes × code_dim` 矩阵，构成语义字典。
- **CayleyRotation**: 根据用户条件特征对原型进行保距旋转，实现个性化语义空间。
- **LangevinSinkhorn**: 通过熵正则化最优传输（Sinkhorn）+ Langevin 噪声，将序列表征 `z_seq` 软分配到原型上，输出 `proto_weights`（分配概率）。
- **κ 动态锐化**: 基于用户特征和时间跨度自适应调节分配锐度，避免退化的均匀分配。

**自监督目标**：
- **Packing Loss**: 最大化原型间余弦距离（避免坍缩）。
- **正交正则**: Cayley 旋转矩阵的正交约束。
- **κ 正则**: 防止锐度过高或过低。

### 3.2 解耦优化 (Decoupled Optimization)

为避免生成模块的复杂梯度干扰主 CTR 任务，v10 采用**架构级梯度隔离**：

| 模块 | 梯度归属 | 说明 |
|------|---------|------|
| `shared_encoder` / `seq_encoder` / `cross_field` / `ctr_head` | **CTR 路径** | 主判别任务 |
| `prototype` / `diffusion_explainer` / `energy_calibrator` | **生成模块 (gen_module)** | 自监督任务 |
| `meta_aligner` | **Meta** | 辅助权重控制 |
| `embedding` | **Sparse** | 高维稀疏特征，独立 Adagrad |

**关键机制**：
- 生成模块输出（`proto_repr`, `diff_explain`, `energy_score`）在进入 CTR 路径前均已 **detach()**。
- 生成模块仅通过 `ib/recon/ortho/packing/energy_target` 等自监督损失训练。
- **单次 `total_loss.backward()`**，通过 PyTorch 计算图自动实现梯度隔离，无需多步 backward。
- `IsolatedOptimizer` 管理四组优化器，分别 step，实现参数更新隔离。

### 3.3 扩散解释器 (DiffusionExplainer)

作为生成模块的核心组件，DiffusionExplainer 学习 CTR 路径的**残差解释**：

- **输入**: 序列表征 `z_seq`、原型分配 `proto_weights`、原型残差 `proto_res`、CTR 表征 `ctr_repr`、CTR logits。
- **瓶颈编码**: 将高维交互信息压缩到 `bottleneck_dim`（默认 64），强制学习紧凑的残差模式。
- **输出**: `diff_explain`（残差修正向量，维度 `d_model×4`）和 `uncertainty`（不确定性估计）。
- **对齐约束**: 
  - `align_loss`: diff_explain 在 CTR 表征方向上的投影（学习有效修正）。
  - `diversity_bonus`: 正交于 CTR 表征的分量（鼓励探索新信息）。
  - `residual_target_loss`: 在决策边界附近（p∈[0.3,0.7]）施加更大残差目标。

### 3.4 能量校准器 (EnergyCalibrator)

EnergyCalibrator 不直接参与训练，而是作为**预测质量评估器**：

- **功能**: 根据 `proto_weights`、`user_feat`、`item_feat` 和 `|ctr_logits|` 预测当前样本的**预测误差能量**。
- **训练目标**: MSE 拟合当前 batch 的 BCE 误差。
- **CTR Head 应用**: 在推理时，高 energy 样本会降低 diff_explain 残差权重，实现**不确定性感知的保守预测**。

### 3.5 过拟合感知的元对齐 (MetaAligner)

MetaAligner 替代传统固定权重的多任务加权，实现动态辅助权重调度：

- **残差收益驱动 (`residual_benefit`)**: 当 `loss_base - loss_final > 0`（生成模块帮助了 CTR），提升 `aux_weight`；反之则降低。
- **过拟合硬上限**: 当 `train_AUC - valid_AUC > 0.03` 时，强制压缩 `aux_weight`。
- **Valid-AUC 缺失保护**: 两次验证之间采用指数衰减插值，避免离线盲目探索。
- **输出**: 仅 `aux_weight`（生成模块总权重），不干预内部各损失配比。

---

## 4. 数据加载与防泄露 (Anti-Leak)

### 4.1 统一 CVR 标签逻辑

- **过滤**: 仅保留 `label_type > 0` 的点击样本（CVR 任务天然条件于点击）。
- **标签**: `label = (label_type == 2)`，即点击且转化=1，点击未转化=0。
- **训练/验证统一**: 训练和验证集采用完全一致的过滤逻辑，避免 valid_AUC=0 的陷阱。

### 4.2 防泄露模式 (`ANTI_LEAK_MODE`)

| 模式 | 机制 | 适用场景 |
|------|------|---------|
| `timestamp` (默认) | 按时间戳排序 RowGroup，后 10% 作为验证 | 时序数据，严格因果 |
| `user_id` | 按用户划分，不同用户不跨集 | 用户级独立同分布 |
| `none` | 原始随机划分 | 快速实验 |

### 4.3 词汇表防泄露

- 验证集自动限制 vocab 为训练集观测到的最大值（`train_vocab`），避免 unseen ID 泄露信息。
- 序列截断：验证集序列仅保留 `timestamp <= split_timestamp` 的历史行为。

---

## 5. 训练流程

```python
# Phase 0: 清零梯度
optimizer.zero_grad_shared()
optimizer.zero_grad_gen()
optimizer.zero_grad_meta()

# Phase 1: CTR Forward
out = model(input, task_id='ctr')
(logits, proto_weights, proto_repr, kappa_mean, assign_entropy,
 diff_explain, uncertainty, gen_align_loss, energy_score,
 packing_loss, base_logits, final_repr) = out

# Phase 2: CTR Loss
loss_ctr = adaptive_focal(logits, labels)  # 或 BCE
loss_ctr_base = bce(base_logits, labels)

# Phase 3: Energy Target
energy_target_loss = energy_calibrator.compute_target(energy_score, base_logits.detach(), labels)

# Phase 4: Gen Loss (自监督)
gen_loss = gen_align_loss + energy_target_loss * 0.1 + packing_loss * packing_weight

# Phase 5: MetaAligner 调度
residual_benefit = loss_ctr_base.item() - loss_ctr.item()
meta = model.meta_aligner(valid_auc=..., train_auc=..., residual_benefit=...)
aux_weight = meta['aux_weight']

# Phase 6: 总 Loss
total_loss = loss_ctr + aux_weight * gen_loss
total_loss.backward()  # 单次 backward，梯度自动隔离

# Phase 7: 参数更新
optimizer.step_shared()  # CTR 路径
if gen_loss > 0:
    optimizer.step_gen()   # 生成模块
optimizer.step_meta()      # MetaAligner
```

---

## 6. 模块详解

### 6.1 MultiViewEncoder
- 用户/物品离散特征通过 `ConstrainedEmbedding`（L2 归一化约束）编码。
- 稠密特征通过 `IntraLinear` 投影到 `d_model`。
- 输出多任务视角：`shared`、`user`、`item`、`context`。

### 6.2 ContinuousSequenceEncoder
- 基于 **SSM (State Space Model)** 的序列编码器，替代传统 Transformer，线性复杂度。
- `SSMCell`: 连续时间离散化状态空间，支持非均匀时间戳（`time_deltas`）。
- 输出四路序列摘要：`short`（近期加权）、`long`（长期平均）、`static`（统计量）、`full_seq`。
- 空序列处理：通过 `empty_mu + empty_sigma * noise` 生成随机空状态，避免全零退化。

### 6.3 ProtoConditionedCrossFieldNet
- 4 层交叉场网络，场定义：`['user', 'item', 'context', 'seq']`。
- 每层包含：
  1. **MHA**: 场间自注意力。
  2. **Proto-bias**: `proto_weights` 温和注入（scale=0.05），避免主导。
  3. **SeqFiLM**: 序列条件化特征调制。
  4. **Diff-gate FFN**: `diff_explain` 通过门控调制 FFN 输出。

### 6.4 CalibratedCTRHead
- 三路融合：`static`（基础表征）、`proto`（原型语义）、`pos_boost`（原型激活增强）。
- **Uncertainty-gated Residual**: `uncertainty` 高时降低 `diff_explain` 权重，防止噪声修正。
- **温度缩放**: `logit_temperature` 自适应调节输出锐度。
- **训练时偏置校正**: 若 logits 均值 <-3.0，自动提升偏置防止过度悲观。

### 6.5 AdaptiveFocalLoss
- `gamma` 为可学习参数（`gamma_logit`），通过独立 Adam 优化器更新。
- 支持 `gamma_min`（默认 0.5）硬下限，防止早期过度聚焦。
- 变化率限制：每次更新不超过 `±0.5`，防止震荡。
- Warmup：前 `gamma_warmup_steps` 步线性升温。

---

## 7. 全链路 NaN 防护

v10 在多个层级实施 NaN/Inf 检测与自动恢复：

| 层级 | 机制 |
|------|------|
| **数据层** | `check_nan_tensor` 检测输入/输出异常 |
| **模型层** | `DynamicPrototypeManifold._check_and_recover_nan()` 参数重初始化 |
| **梯度层** | `IsolatedOptimizer.check_and_recover_nan_params()` 全局参数恢复 |
| **损失层** | `total_loss` NaN 时替换为 0.0，跳过有害 batch |
| **Logits** | 硬裁剪到 `[-20, 20]`，防止极端值 |
| **SSM 状态** | `h_states`  clamp 到 `[-100, 100]`，指数项 clamp 到 `[-80, 80]` |

---

## 8. 快速开始

### 8.1 环境要求

```bash
python >= 3.9
pytorch >= 2.1
pyarrow >= 12.0
numpy, scikit-learn, tqdm
```

### 8.2 数据准备

数据目录需包含：
```
data/
  ├── *.parquet          # 训练数据（多文件）
  └── schema.json        # 特征模式定义
```

`schema.json` 格式：
```json
{
  "user_int": [[fid, vocab_size, dim], ...],
  "item_int": [[fid, vocab_size, dim], ...],
  "user_dense": [[fid, dim], ...],
  "seq": {
    "seq_a": {"prefix": "seq_a", "ts_fid": 100, "features": [[fid, vs], ...]},
    ...
  }
}
```

### 8.3 启动训练

```bash
export TRAIN_DATA_PATH=/path/to/data
export TRAIN_CKPT_PATH=/path/to/ckpt
export TRAIN_LOG_PATH=/path/to/log
export TRAIN_TF_EVENTS_PATH=/path/to/tf_events
export ANTI_LEAK_MODE=timestamp  # 或 user_id / none

bash run.sh
```

### 8.4 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d_model` | 128 | 模型主维度 |
| `--num_layers` | 4 | 交叉场层数 |
| `--num_codes` | 128 | 原型数量 |
| `--kappa_base` | 2.0 | 原型分配基础锐度 |
| `--sinkhorn_epsilon` | 0.05 | 最优传输熵正则 |
| `--packing_weight` | 0.01 | 原型打包损失权重 |
| `--lr` | 1e-4 | 稠密参数学习率 |
| `--sparse_lr` | 0.001 | 稀疏 Embedding 学习率 |
| `--focal_alpha_pos` | 0.6 | Focal 正样本权重 |
| `--focal_alpha_neg` | 0.4 | Focal 负样本权重 |
| `--dropout_rate` | 0.2 | 全局 Dropout |
| `--eval_every_n_steps` | 600 | 验证频率 |

---

## 9. 监控与诊断

训练过程中通过 TensorBoard 记录以下指标：

### 9.1 损失指标
- `Loss/ctr`: 主 CTR 损失
- `Loss/ctr_base`: 无残差基线损失
- `Loss/gen`: 生成模块总损失
- `Loss/gen_align`: 扩散解释对齐损失
- `Loss/energy_target`: 能量校准目标损失
- `Loss/packing`: 原型打包损失

### 9.2 诊断指标
- `Stream/logits_mean`, `Stream/logits_std`: Logits 分布监控
- `Diagnostics/train_auc`: 训练集 AUC
- `Diagnostics/grad_norm_ctr`: CTR 路径梯度范数
- `Diagnostics/nan_recovery_count`: NaN 恢复次数

### 9.3 原型指标
- `Proto/kappa_mean`: 平均分配锐度
- `Proto/assign_entropy`: 分配熵（越低越尖锐）

### 9.4 Meta 指标
- `Meta/aux_weight`: 当前辅助权重
- `Meta/residual_benefit`: 残差收益
- `Meta/gap`: 训练-验证 AUC 差距
- `Meta/mode`: 调度模式（valid_pid / interpolated）

### 9.5 生成指标
- `Gen/uncertainty_mean`: 平均不确定性
- `Gen/energy_mean`, `Gen/energy_std`: 能量分布

---

## 10. 文件结构

```
.
├── train.py           # 训练入口，参数解析与流程编排
├── trainer.py         # Trainer 核心：解耦优化、训练循环、评估
├── model.py           # 模型定义：生成式语义层 + 交叉场网络 + 预测头
├── dataset.py         # Parquet 数据加载：防泄露、统一 CVR 标签、动态缓冲池
├── run.sh             # 启动脚本，含完整默认参数
└── utils.py           # 工具函数（需自行提供：EarlyStopping, set_seed 等）
```

---

## 11. 版本演进

| 版本 | 核心改进 |
|------|---------|
| v9.x | Anti-Leak 数据加载、SSM 序列编码、动态缓冲池 |
| v10.0 | 引入 DynamicPrototypeManifold + DiffusionExplainer + EnergyCalibrator |
| v10.1 | 解耦优化架构（单次 backward + 架构 detach） |
| v10.2 | MetaAligner 残差收益驱动、EnergyCalibrator 误差预测、全链路 NaN 防护 |

---

## 12. 引用与致谢

本框架基于以下技术构建：
- **State Space Models (SSM)**: 连续时间序列建模
- **Optimal Transport (Sinkhorn)**: 熵正则化分配
- **Information Bottleneck**: 瓶颈残差编码
- **Cayley Transform**: 保距流形旋转

---

> **维护提示**: 若遇到 `valid_AUC` 随 epoch 下降但 `LogLoss` 也下降的情况，请检查 `MetaAligner` 的 `residual_benefit` 符号是否正常（应为正表示生成模块有帮助），并确认 `ANTI_LEAK_MODE` 设置是否符合数据时序特性。
