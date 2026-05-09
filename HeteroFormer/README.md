# PCVRHeteroFormer v8.0 - Symplectic Multi-Scale Prototype Learning

## 概述

v8.0是基于**信息几何**、**辛几何**与**随机矩阵理论**的推荐系统架构，专为2026腾讯广告算法大赛设计。核心解决v7.4中存在的谱损失恒定、任务冲突、梯度竞争等结构性缺陷。

## 核心创新

### 1. von Mises-Fisher统计流形原型 (`VonMisesFisherPrototype`)

**问题**: v7.4的欧氏空间原型导致范数漂移和SVD梯度灾难。

**解决方案**: 
- 将原型编码为vMF分布的自然参数 `η = κ·μ`
- 样本和原型都在单位球面上，距离天然有界
- Sinkhorn分配在概率单纯形上进行，与Bregman散度兼容

**数学保证**:
```
p(x|μ, κ) = C(κ) exp(κ μ^T x),  ||μ||=1, κ>0
```

**torch.compile优化**:
- 所有操作都是固定形状张量运算
- 使用`torch.where`替代数据依赖条件分支
- 支持`fullgraph=True`编译

### 2. 线性注意力序列编码 (`LinearAttentionLayer`)

**问题**: Transformer的O(n²)复杂度限制序列长度。

**解决方案**:
- 通过ReLU特征映射 `φ(x) = ReLU(x) + ε` 将注意力降到O(n)
- 保留512序列长度（实践验证的最优值）

**数学**:
```
Softmax(QK^T)V ≈ φ(Q)(φ(K)^T V) / (φ(Q) · sum(φ(K)))
```

### 3. 辛约化多尺度优化 (`SymplecticMultiScaleOptimizer`)

**问题**: v7.4中`focal_loss`与`spec_loss`等任务梯度冲突。

**解决方案**:
- 将损失分解为**低频**（focal, recon）和**高频**（div, spec, align）
- 通过SVD将梯度投影到正交子空间
- 低频大步优化 + 高频小步修正

**数学保证**:
```
G = [g_1, ..., g_K] = U S V^T  (SVD分解)
g_low = U[:, :r] U[:, :r]^T g_total    (大奇异值 = 共识方向)
g_high = g_total - g_low                (小奇异值 = 差异方向)
g_combined = g_low + ε·g_high           (ε=0.1 高频衰减)
```

### 4. MP谱正则化 (`MPSpectralRegularizer`)

**问题**: v7.4的SVD谱多样性导致梯度不稳定。

**解决方案**:
- 基于Marchenko-Pastur定律约束原型谱分布
- 使用可微分直方图替代直接SVD优化
- 惩罚条件数防止主成分垄断

**理论依据**:
- 随机矩阵的奇异值分布收敛于MP分布
- 临界相（谱分布接近MP）对应最优表达能力

### 5. Hamiltonian特征交互 (`HamiltonianInteraction`)

**问题**: 标准MLP交互导致能量漂移和logits不稳定。

**解决方案**:
- 将特征交互建模为能量守恒的哈密顿系统
- 使用Leapfrog辛积分器保证相空间体积守恒
- 可学习时间步长限制在稳定区域

### 6. 融合门控与对比对齐 (`FusionGate`)

**问题**: v7.4中静态分支相加导致梯度竞争。

**解决方案**:
- 自适应门控权重决定静态/原型分支贡献
- InfoNCE对比损失对齐两个分支的语义空间
- 防止任一分支主导训练

## 文件结构

```
.
├── model_v8.py          # 模型架构 (torch.compile优化)
├── trainer_v8.py        # 辛约化训练器
├── README.md            # 本文档
└── run_v8.sh            # 启动脚本 (需自行配置)
```

## 关键超参数

| 参数 | 默认值 | 说明 | 理论依据 |
|------|--------|------|----------|
| `num_codes` | 64 | 原型数量 | `√N` (N=样本数) |
| `sinkhorn_epsilon` | 0.05 | 熵正则化 | `2/K` (与码本大小成反比) |
| `min_mass_ratio` | 0.016 | 最小列配额 | `1/K` (均匀分布临界值) |
| `spec_weight` | 0.01 | 谱正则权重 | `1/D` (与维度成反比) |
| `curriculum_warmup` | 5000 | 课程学习步数 | 先重构后结构化 |
| `high_lr_scale` | 0.1 | 高频学习率缩放 | 辛约化理论 |

## torch.compile配置

```python
# 推荐配置
compile_kwargs = {
    "backend": "inductor",
    "fullgraph": True,      # v8.0支持全图编译
    "dynamic": False,       # 固定形状
    "mode": "max-autotune",
}
model = torch.compile(model, **compile_kwargs)
```

**编译安全保证**:
- 无数据依赖的Python控制流（如`if x > 0:`）
- 使用`torch.where`替代条件分支
- 固定步数的循环（如`for _ in range(num_steps)`）
- 无动态形状变化

## 训练动力学

### 课程学习调度

```
Step 0-5000:   recon=0.1, div=0→0.15, spec=0→0.01  (先重构)
Step 5000+:    所有权重达到目标值                  (后结构化)
```

### 损失频率分类

| 损失 | 频率 | 说明 |
|------|------|------|
| `focal` | low | 分类主任务，大尺度结构 |
| `recon` | low | 原型重构，基础表示 |
| `div` | high | 多样性约束，精细调整 |
| `spec` | high | 谱正则，防止坍缩 |
| `align` | high | 分支对齐，语义协调 |
| `mp` | high | MP分布约束，临界相 |

### 梯度监控指标

TensorBoard新增监控:
- `Proto/kappa_mean`: vMF集中度均值
- `Proto/kappa_std`: vMF集中度标准差
- `Curriculum/alpha`: 课程学习进度
- `Curriculum/weight_*`: 动态损失权重

## 使用示例

```python
from model_v8 import PCVRHeteroFormer, ModelInput
from trainer_v8 import PCVRHeteroFormerTrainer

# 构建模型
model = PCVRHeteroFormer(
    user_int_feature_specs=user_specs,
    item_int_feature_specs=item_specs,
    user_dense_dim=user_dense_dim,
    item_dense_dim=item_dense_dim,
    seq_vocab_sizes=seq_vocab_sizes,
    user_ns_groups=user_ns_groups,
    item_ns_groups=item_ns_groups,
    d_model=128,
    num_codes=64,
    sinkhorn_epsilon=0.05,
    sinkhorn_iter=20,
    min_mass_ratio=0.016,
)

# torch.compile
model = torch.compile(model, backend="inductor", fullgraph=True)

# 训练
trainer = PCVRHeteroFormerTrainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    lr=1e-4,
    num_epochs=999,
    recon_weight=0.1,
    div_weight=0.15,
    spec_weight=0.01,
    align_weight=0.05,
    mp_weight=0.005,
    curriculum_warmup=5000,
)
trainer.train()
```

## 理论参考

1. **信息几何**: Amari, S. (2016). *Information Geometry and Its Applications*
2. **辛几何**: Marsden, J. & Ratiu, T. (1999). *Introduction to Mechanics and Symmetry*
3. **随机矩阵**: Tao, T. (2012). *Topics in Random Matrix Theory*
4. **最优传输**: Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport*
5. **线性注意力**: Katharopoulos et al. (2020). "Transformers are RNNs"

## 版本信息

- **Version**: 8.0-symplectic
- **Date**: 2026-05-09
- **PyTorch**: >= 2.0 (torch.compile支持)
- **CUDA**: >= 11.7
