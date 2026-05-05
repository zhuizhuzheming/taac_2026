# PCVRHeteroFormer v7.1-Academic

**Heterogeneous Feature Interaction Network with Generative Sequence Encoding**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## 目录

- [概述](#概述)
- [核心设计哲学](#核心设计哲学)
- [架构详解](#架构详解)
  - [三层抽象范式](#三层抽象范式)
  - [HeteroBlock 统一组合单元](#heteroblock-统一组合单元)
  - [GenerativeSequenceEncoder](#generativesequenceencoder)
  - [VectorQuantizer](#vectorquantizer)
  - [预测头与 Lipschitz 约束](#预测头与-lipschitz-约束)
- [数学基础](#数学基础)
  - [残差门控](#残差门控)
  - [VQ 损失函数](#vq-损失函数)
  - [温度退火](#温度退火)
- [快速开始](#快速开始)
  - [安装依赖](#安装依赖)
  - [模型初始化](#模型初始化)
  - [前向传播示例](#前向传播示例)
  - [辅助损失提取](#辅助损失提取)
- [配置参数参考](#配置参数参考)
- [诊断与监控](#诊断与监控)
- [设计决策记录](#设计决策记录)
- [引用](#引用)

---

## 概述

PCVRHeteroFormer 是一个为**点击率/转化率预测**任务设计的深度异构特征交互网络。其核心创新在于：

1. **生成式序列编码（Generative Sequence Encoding, GSE）**：通过双向 Transformer 将变长序列编码为离散码本（Vector Quantization）表示，再经注意力聚合为固定维度向量，彻底摆脱自回归暴露偏差。

2. **三层抽象范式（Intra → Inter → Pool）**：将特征交互分解为域内（Intra-field）、域间（Inter-field）和池化（Pool-field）三个层次，每个层次由独立的算子实现，通过 `HeteroBlock` 统一组合。

3. **Lipschitz 约束预测头**：预测网络的所有线性层均施加谱归一化（Spectral Normalization），保证输出对输入扰动的稳定性，提升泛化性能。

4. **可学习的残差门控**：深度交互栈中的每一层均配备可学习的 sigmoid 门控残差连接，初始化为半开状态（gate=0.5），允许网络自适应决定信息流。

---

## 核心设计哲学

### 1. 特征交互的三层分解

| 层次 | 作用域 | 典型算子 | 计算复杂度 |
|------|--------|----------|-----------|
| **Intra** | 单个特征域内部 | Embedding, Linear, Transformer, VQ | O(L·d²) |
| **Inter** | 多个特征域之间 | Self-Attention, Cross-Attention, Bilinear, Hadamard | O(n²·d) |
| **Pool** | 全局聚合 | Concat+Linear, Subspace Router | O(d²) |

这种分解使得模型能够**在不同粒度上捕捉特征关系**：域内捕捉局部语义（如用户历史序列的时序模式），域间捕捉全局关联（如用户画像与商品属性的交互），池化层则进行最终的容量自适应分配。

### 2. HeteroBlock 作为统一组合单元

```
Input Fields ──→ Intra Ops ──→ Inter Op ──→ Pool Op ──→ Residual ──→ Output Fields
                    ↑                                              ↑
                    └──────────── 可选残差门控 ──────────────────────┘
```

`HeteroBlock` 不感知内部算子的具体实现，仅通过统一的字典接口（`Dict[str, Tensor]`）进行数据传递。这种**架构透明性**使得新算子（如将 GSE 替换为其他序列编码器）可以零成本接入。

### 3. 生成式范式 vs. 判别式范式

传统序列编码器（如 DIN、DIEN）采用**判别式注意力**机制，直接对序列元素加权求和。PCVRHeteroFormer 的 GSE 模块采用**生成式 VQ 压缩**：

- **双向 Transformer** 编码完整上下文（无因果掩码）
- **Vector Quantizer** 将连续表示压缩到离散码本
- **注意力聚合** 从量化后的表示中提取全局语义

这一设计消除了自回归模型的暴露偏差，同时通过离散化引入信息瓶颈，迫使模型学习更具泛化性的序列表示。

---

## 架构详解

### 三层抽象范式

#### Intra-Field Operations（域内操作）

域内操作处理单个特征域的原始输入，将其映射到统一的 `d_model` 维度空间。

**`IntraEmbedding`**：支持多槽位（multi-slot）嵌入，对高基数特征自动应用增强 Dropout。使用 `ConstrainedEmbedding`（L2 最大范数约束）防止嵌入空间坍缩。

**`IntraLinear`**：简单的线性投影 + RMSNorm + Dropout，用于稠密特征的维度对齐。

#### Inter-Field Operations（域间操作）

域间操作在多个特征域之间建立交互。

**`InterSelfAttention`**：域间自注意力，每个域作为 token，计算域间的注意力权重。

**`InterCrossAttention`**：非对称交叉注意力，支持指定 query 域和 key-value 域。例如，序列域作为 query，用户-商品交互域作为 key-value。

**`InterBilinear`**：低秩双线性交互，通过 `num_fields` 个投影矩阵将各域映射到共享的低秩空间，进行逐元素相乘后投影回高维。参数复杂度为 `O(num_fields · d · r + r · d)`，远低于全双线性 `O(d²)`。

**`InterHadamard`**：Hadamard 乘积交互网络，通过残差连接的逐元素乘法实现高效的多阶特征交叉。

#### Pool-Field Operations（池化操作）

池化操作将多个域的表示聚合为单个向量。

**`PoolRouter`**：子空间路由机制。维护 `num_banks` 个正交初始化的低秩子空间矩阵，通过 softmax 路由权重进行软选择，实现自适应容量分配。

**`PoolConcatLinear`**：简单的拼接 + 线性投影。

---

### HeteroBlock 统一组合单元

```python
class HeteroBlock(nn.Module):
    def forward(self, fields, residuals=None):
        # 1. Intra-field: 每个域独立变换
        intra_out = {k: self.intra_ops[k](v) if k in self.intra_ops else v 
                     for k, v in fields.items()}

        # 2. Inter-field: 域间交互
        inter_out = self.inter_op(intra_out) if self.inter_op else intra_out

        # 3. Pool-field: 全局聚合
        pool_out = self.pool_op(inter_out) if self.pool_op else inter_out

        # 4. 残差连接（可选门控）
        if residuals and self.residual_gate:
            gate = sigmoid(self.residual_gate)  # 标量门控
            output = {k: residuals[k] + gate * dropout(pool_out[k] - residuals[k])
                      for k in pool_out}

        # 5. 随机深度（Stochastic Depth）
        if self.training and self.stochastic_depth_prob > 0:
            output = apply_stochastic_depth(output)

        return output
```

**残差门控**（Residual Gate）是一个可学习的标量参数，通过 sigmoid 激活控制残差流的强度：

$$
g = \sigma(\gamma), \quad \gamma \in \mathbb{R}
$$

初始化为 `gate_init=0.0`，对应 $g \approx 0.5$（半开状态）。这允许网络在训练初期同时探索变换路径和恒等路径，随着训练进行自适应调整。

**随机深度**（Stochastic Depth）以概率 $p$ 随机跳过整个 block，作为深度网络的正则化手段，其期望输出通过缩放因子 $1/(1-p)$ 保持不变。

---

### GenerativeSequenceEncoder

GSE 是模型的核心创新，负责将变长序列编码为固定维度的语义向量。

#### 架构流程

```
seq_ids [B, n_feats, L]
    │
    ▼
Multi-feature Embedding ──→ Concat ──→ Projection ──→ LayerNorm
    │
    ▼
+ Time Embedding (optional)
    │
    ▼
Bidirectional TransformerEncoder (num_layers, full self-attention)
    │
    ▼
Linear Projection to code_dim
    │
    ▼
VectorQuantizer (Gumbel-Softmax, K codes + 1 null)
    │
    ▼
Linear Projection back to d_model
    │
    ▼
Attention Aggregation (learnable query token)
    │
    ▼
Empty Sequence Gate (sigmoid-blended with learnable empty_state)
    │
    ▼
Output Projection (SiLU + LayerNorm)
    │
    ▼
output [B, d_model]
```

#### 关键设计决策

**1. 双向 Transformer（无因果掩码）**

与自回归序列模型不同，GSE 使用标准的 `nn.TransformerEncoder`，不施加因果掩码。这意味着每个位置都可以 attend 到序列中的**所有**其他位置（受 padding mask 限制）。这消除了暴露偏差，允许模型基于完整上下文进行编码。

**2. 空序列门控（Empty Sequence Gate）**

对于空序列（`seq_len == 0`）或极短序列，直接应用注意力聚合可能产生噪声。空序列门控通过一个轻量 MLP 计算门控值：

$$
g_{\text{null}} = \sigma(W_2 \cdot \text{SiLU}(W_1 \cdot h_{\text{agg}}) + b)
$$

输出为门控混合：

$$
h_{\text{out}} = g_{\text{null}} \cdot h_{\text{empty}} + (1 - g_{\text{null}}) \cdot h_{\text{agg}}
$$

其中 $h_{\text{empty}}$ 是可学习的空状态向量。门控偏置初始化为 `-2.0`，使得初始时 $g_{\text{null}} \approx 0.12$，倾向于使用真实聚合表示。

**3. 辅助损失内部化**

GSE 的 VQ 损失（commitment、codebook、entropy、null regularization）在 `forward()` 内部计算并存储，不通过返回值泄漏。外部通过 `get_aux_loss()` 显式提取。这保持了 `forward()` 的纯函数契约。

---

### VectorQuantizer

基于 Gumbel-Softmax 的可微向量量化器，支持空码（null code）处理变长序列。

#### 码本结构

- **Code 0**: Null code，初始化为零向量，用于表示 padding 位置或空序列
- **Codes 1..K**: Active codes，均匀初始化在 $[-1/K, 1/K]$ 范围内

#### 前向传播

对于输入 $z \in \mathbb{R}^{B \times L \times D}$：

**1. 距离计算**

$$
d_{b,l,k} = \|z_{b,l} - e_k\|_2^2, \quad k \in \{0, 1, \dots, K\}
$$

**2. Null 掩码**

$$
\text{null}_{b,l} = \mathbb{1}[l \geq \text{seq\_len}_b] \lor \mathbb{1}[\text{seq\_len}_b = 0]
$$

**3. Logits 修正**

$$
\text{logits}_{b,l,k} = \begin{cases}
-10^4 & \text{if } \text{null}_{b,l} \text{ and } k \neq 0 \\
-10^4 & \text{if } \neg\text{null}_{b,l} \text{ and } k = 0 \\
-d_{b,l,k} & \text{otherwise}
\end{cases}
$$

这强制：padding 位置只能选中 null code，有效位置只能选中 active codes。

**4. Gumbel-Softmax 采样**

$$
\nu \sim \text{Uniform}(0,1), \quad g = -\log(-\log(\nu)) \\
p_k = \text{softmax}\left(\frac{\text{logits}_k + g_k}{\tau}\right)_k
$$

其中 $\tau$ 为温度参数，随训练退火。

**5. 量化**

$$
\hat{z} = \sum_{k=0}^{K} p_k \cdot e_k
$$

**6. Straight-Through Estimator**

$$
z_{\text{ST}} = z + (\hat{z} - z)_{\text{detach}}
$$

前向传播使用 $\hat{z}$，反向传播梯度直接流过 $z$。

#### 损失函数

$$
\mathcal{L}_{\text{VQ}} = \lambda_{\text{commit}} \|z - \text{sg}(\hat{z})\|_2^2 + \lambda_{\text{codebook}} \|\text{sg}(z) - \hat{z}\|_2^2 + \lambda_{\text{entropy}} \cdot (-H(\bar{p})) + \lambda_{\text{null}} \|e_0\|_2^2
$$

其中：
- **Commitment loss**: 迫使编码器输出靠近码本向量（stop-gradient 在码本侧）
- **Codebook loss**: 迫使码本向量靠近编码器输出（stop-gradient 在编码器侧）
- **Entropy loss**: $-H(\bar{p})$，最大化码本使用率的熵，防止索引坍缩（index collapse）
- **Null regularization**: 约束 null code 保持接近零向量

默认系数：$\lambda_{\text{commit}} = 0.25$, $\lambda_{\text{codebook}} = 0.25$, $\lambda_{\text{entropy}} = 0.1$, $\lambda_{\text{null}} = 0.01$

#### 温度退火

温度 $\tau$ 随训练步数指数退火：

$$
\tau(t) = \tau_{\min} + (\tau_0 - \tau_{\min}) \cdot \exp\left(-5 \cdot \frac{t}{T_{\text{anneal}}}\right)
$$

其中 $T_{\text{anneal}} = 10000$ 步，$\tau_0 = 1.0$，$\tau_{\min} = 0.1$。退火确保早期探索（高温使分布均匀），后期收敛（低温使分布尖锐）。

---

### 预测头与 Lipschitz 约束

预测头是一个两层 MLP，所有线性层均施加谱归一化（Spectral Normalization）：

```python
predictor = Sequential(
    SN(Linear(d_model*4, d_model*2)),   # 谱归一化
    SwiGLU(d_model*2, expand_ratio=1.0),
    Dropout(dropout),
    SN(Linear(d_model*2, action_num)),  # 谱归一化
)
```

**谱归一化**约束权重矩阵的谱范数：

$$
\|W\|_2 = \sigma_{\max}(W) \leq 1
$$

通过幂迭代（power iteration）在每次前向传播时估计最大奇异值，并将权重除以此估计值。这保证了预测头作为函数的 Lipschitz 常数不超过 1，提升模型对输入扰动的鲁棒性。

**输出温度缩放**：

$$
\text{logits}_{\text{out}} = \frac{\text{MLP}(h) \cdot 2\sigma(s)}{\exp(t)}
$$

其中 $s$ 是可学习的输出尺度 logit（初始化为 0.0，对应尺度 $\approx 1.0$），$t$ 是可学习的温度 logit（初始化为 0.0，对应温度 $\approx 1.0$），并通过 clamp 限制在 $[0.5, 5.0]$ 范围内。

---

## 数学基础

### 残差门控

对于第 $l$ 层的 HeteroBlock，残差连接定义为：

$$
h^{(l)} = h^{(l-1)} + g_l \cdot \mathcal{F}_l(h^{(l-1)})
$$

其中 $g_l = \sigma(\gamma_l)$，$\mathcal{F}_l$ 是 block 的变换函数（Intra → Inter → Pool → Dropout）。

当 $g_l \to 0$ 时，block 退化为恒等映射；当 $g_l \to 1$ 时，block 完全应用变换。初始 $g_l \approx 0.5$ 允许网络在早期训练阶段平滑地探索两种极端之间的平衡。

### VQ 损失函数

完整的 VQ 损失可以看作是对编码器 $E$、码本 $C = \{e_k\}_{k=0}^K$ 和量化操作 $Q$ 的联合优化：

$$
\mathcal{L}_{\text{VQ}} = \underbrace{\|E(x) - \text{sg}[Q(E(x))]\|_2^2}_{\text{commitment}} + \underbrace{\|\text{sg}[E(x)] - Q(E(x))\|_2^2}_{\text{codebook}} + \underbrace{\lambda_E (-H(\bar{p}))}_{\text{entropy}} + \underbrace{\lambda_N \|e_0\|_2^2}_{\text{null reg}}
$$

其中 $\bar{p} = \frac{1}{|\mathcal{V}|} \sum_{(b,l) \in \mathcal{V}} p_{b,l}$ 是有效位置上的平均码本使用分布，$\mathcal{V} = \{(b,l) : \text{null}_{b,l} = \text{False}\}$。

### 温度退火

Gumbel-Softmax 的松弛质量取决于温度 $\tau$。高温时，$p_k \approx 1/(K+1)$（均匀分布），梯度估计方差低但偏差大；低温时，$p_k \approx \mathbb{1}[k = \arg\max]$（one-hot），偏差小但梯度估计方差高。

指数退火策略在训练早期提供稳定的梯度信号，在后期逼近离散的 argmax 行为。

---

## 快速开始

### 安装依赖

```bash
pip install torch>=2.0.0
```

### 模型初始化

```python
from model import PCVRHeteroFormer, ModelInput

# 定义特征规格
user_int_specs = [
    (1000, 0, 1),   # (vocab_size, offset, num_slots)
    (500, 1, 1),
]
item_int_specs = [
    (2000, 0, 1),
    (300, 1, 1),
]

seq_vocab_sizes = {
    'click': [10000, 100],      # [item_id_vocab, category_vocab]
    'cart': [10000, 100],
}

model = PCVRHeteroFormer(
    user_int_feature_specs=user_int_specs,
    item_int_feature_specs=item_int_specs,
    user_dense_dim=10,
    item_dense_dim=10,
    seq_vocab_sizes=seq_vocab_sizes,
    user_ns_groups=[[0, 1]],
    item_ns_groups=[[0, 1]],
    d_model=128,
    emb_dim=16,
    num_layers=4,
    dropout=0.1,
    gse_num_codes=64,
    gse_code_dim=64,
    gse_num_layers=4,
)
```

### 前向传播示例

```python
import torch

B = 32  # batch size

model_input = ModelInput(
    user_int_feats=torch.randint(0, 1000, (B, 2)),
    item_int_feats=torch.randint(0, 2000, (B, 2)),
    user_dense_feats=torch.randn(B, 10),
    item_dense_feats=torch.randn(B, 10),
    seq_data={
        'click': torch.randint(0, 10000, (B, 2, 50)),  # [B, n_feats, max_len]
        'cart': torch.randint(0, 10000, (B, 2, 30)),
    },
    seq_lens={
        'click': torch.randint(1, 50, (B,)),
        'cart': torch.randint(1, 30, (B,)),
    },
    seq_time_buckets={
        'click': torch.randint(0, 10, (B, 50)),
        'cart': torch.randint(0, 10, (B, 30)),
    },
    seq_decay_weights=None,
)

# 前向传播
logits = model(model_input)  # [B, action_num]
print(f"Logits shape: {logits.shape}")
```

### 辅助损失提取

```python
# 在训练循环中
logits = model(model_input)
main_loss = criterion(logits, targets)

# 提取 GSE 的 VQ 辅助损失
aux_loss = model.get_aux_loss()

# 总损失
total_loss = main_loss + aux_loss
total_loss.backward()
```

### 诊断信息提取

```python
# 获取各 block 的诊断指标
diagnostics = model.get_diagnostics()
print(diagnostics)
# {
#   'gate_value': 0.5123,           # 残差门控值
#   'output_norm': 15.2341,         # 输出范数
#   'click_temperature': 0.2341,    # VQ 温度
#   'click_usage_rate': 0.8912,     # 非 null 码使用率
#   'click_perplexity': 45.23,      # 码本困惑度
#   'click_null_gate_mean': 0.1234, # 空序列门控均值
# }
```

---

## 配置参数参考

### 模型架构参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `d_model` | int | 128 | 模型隐藏维度，所有域的统一表示维度 |
| `emb_dim` | int | 16 | 稀疏特征嵌入维度 |
| `num_layers` | int | 4 | 深度 NS 交互栈的层数 |
| `base_rank` | int | d_model/2 | 双线性交互的基础低秩维度 |
| `rank_schedule` | str | 'bottleneck' | 秩调度策略：`constant`/`gentle`/`bottleneck` |
| `num_heads` | int | d_model/32 | 注意力头数（默认自动计算） |
| `num_banks` | int | d_model/16 | PoolRouter 的子空间银行数 |
| `dropout` | float | 0.1 | Dropout 概率 |
| `action_num` | int | 1 | 输出动作/类别数 |

### GSE 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gse_num_codes` | int | 64 | VQ 码本大小（不含 null code） |
| `gse_code_dim` | int | 64 | 码本向量维度 |
| `gse_num_layers` | int | 4 | TransformerEncoder 层数 |
| `gse_temp_anneal_steps` | int | 10000 | 温度退火总步数 |
| `gse_min_temp` | float | 0.1 | 最低温度 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `gate_init` | float | 0.0 | 残差门控初始化值（sigmoid(0)=0.5） |
| `stochastic_depth_prob` | float | 0.0 | 随机深度丢弃概率 |
| `progressive_layer_training` | bool | False | 是否启用逐层解锁训练 |
| `id_dropout_rate` | float | 0.15 | 稀疏特征 ID Dropout 率 |
| `cross_network_layers` | int | 2 | Cross-Network 的 Hadamard 层数 |

---

## 诊断与监控

### 关键监控指标

#### 1. VQ 码本健康度

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `usage_rate` | 0.7 ~ 0.95 | < 0.5（码本利用率不足） |
| `perplexity` | 20 ~ 60 | < 5（索引坍缩）或 > num_codes（分布过散） |
| `temperature` | 0.1 ~ 1.0 | stuck at 1.0（退火未生效） |

#### 2. 残差门控动态

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `gate_value` | 0.3 ~ 0.8 | ≈ 0.0（层被跳过）或 ≈ 1.0（无残差作用） |
| `output_norm` | 5 ~ 30 | 持续下降（梯度消失）或爆炸（>100） |

#### 3. 空序列门控

| 指标 | 正常范围 | 异常信号 |
|------|----------|----------|
| `null_gate_mean` | 0.05 ~ 0.3 | > 0.5（过度依赖空状态） |
| `empty_ratio` | 数据集相关 | 与数据分布一致 |

### TensorBoard 日志建议

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for step, batch in enumerate(dataloader):
    logits = model(batch)
    loss = criterion(logits, batch.targets) + model.get_aux_loss()

    # 每 100 步记录诊断
    if step % 100 == 0:
        diag = model.get_diagnostics()
        for k, v in diag.items():
            writer.add_scalar(f'diagnostics/{k}', v, step)

        # 记录 VQ 码本使用分布
        for domain in model.seq_domains:
            gse = model.seq_blocks[domain]
            if hasattr(gse.vq, '_last_info'):
                hist = gse.vq._last_info.get('code_usage_histogram')
                if hist is not None:
                    writer.add_histogram(f'vq/{domain}_usage', hist, step)
```

---

## 设计决策记录

### 为什么使用 RMSNorm 而不是 LayerNorm？

RMSNorm 去除了 LayerNorm 中的均值中心化操作，仅保留根均方缩放：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

在 Transformer 架构中，研究表明 RMSNorm 与 LayerNorm 性能相当，但计算更简单，且对异常值更鲁棒。

### 为什么预测头使用 SwiGLU 而不是 ReLU/GeLU？

SwiGLU 是 Swish 激活与门控线性单元的组合：

$$
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \otimes xV
$$

它在 LLaMA、PaLM 等大模型中被验证优于传统激活函数，提供更好的梯度流动和非线性表达能力。

### 为什么 VQ 使用 Gumbel-Softmax 而不是直通估计器（STE）+ argmax？

Gumbel-Softmax 提供了可微的松弛，在训练早期提供稳定的梯度信号。虽然 STE 在反向传播时更直接，但 Gumbel-Softmax 的前向传播更加平滑，有助于码本的初期组织。温度退火确保最终行为逼近硬量化。

### 为什么 entropy loss 是 $-H(p)$ 而不是 $H(p)$？

我们要**最大化**码本使用分布的熵，即鼓励所有码被均匀使用。由于优化器执行梯度下降，最小化 $-H(p)$ 等价于最大化 $H(p)$。如果写成 $H(p)$，则梯度下降会最小化熵，导致索引坍缩。

---

## 引用

如果您在研究中使用了 PCVRHeteroFormer，请引用：

```bibtex
@article{pcvrheteroformer2024,
  title={PCVRHeteroFormer: Heterogeneous Feature Interaction with Generative Sequence Encoding},
  author={[Authors]},
  journal={[Venue]},
  year={2024}
}
```

---

## 许可

MIT License
