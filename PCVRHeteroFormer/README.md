# PCVRHeteroFormer v9.3-RSOU

## 技术架构总览

PCVRHeteroFormer 是一个面向**点击后转化率（Post-Click Conversion Rate, PCVR）**预测的深度推荐模型。其核心设计哲学是：将推荐系统中的"序列建模"与"特征交互"从传统的"拼接-压缩"范式，升级为**"流形嵌入-场域动力学-表示手术"**的三层协同范式。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PCVRHeteroFormer v9.3-RSOU                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 3: 表示手术层 (Representation Surgery)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GenerativeFusion (残差融合)                                        │    │
│  │  MetaAligner (双模式调度器: Train-schedule + Valid-PID)             │    │
│  │  CTRHead / DiffusionHead / EnergyHead (隔离任务头)                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  Layer 2: 场域动力学层 (Field Dynamics)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CrossFieldNet (跨场域注意力网络, FiLM条件化)                        │    │
│  │  LangevinForceField (朗之万力场, 不确定性建模)                        │    │
│  │  DynamicPrototypeManifold (动态原型流形, Sinkhorn分配)              │    │
│  │  CayleyRotation (凯莱旋转, 用户条件化几何)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                              │
│  Layer 1: 序列嵌入层 (Sequential Embedding)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MultiViewEncoder (多视角编码器, 用户/物品/上下文分离)                │    │
│  │  ContinuousSequenceEncoder (连续序列编码器, SSM状态空间模型)        │    │
│  │  SSMCell (结构化状态空间单元, 时间感知)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 一、序列建模：从"离散索引"到"连续动力学"

### 1.1 问题背景

传统推荐模型的序列建模通常采用：
- **Transformer 式自注意力**：将序列视为离散 token 的集合，通过位置编码捕捉时序
- **GRU/LSTM 式循环**：逐步压缩历史，但难以建模长程依赖与连续时间间隔

这两种范式的共同缺陷是：**将时间视为"位置索引"而非"物理量"**。在 PCVR 场景中，用户行为序列的时间间隔（上次点击距今 1 小时 vs. 1 周）具有直接的语义意义，不应被平等对待。

### 1.2 SSMCell：结构化状态空间单元

SSMCell 的设计灵感来源于**连续时间信号处理中的状态空间模型（State Space Model）**，将序列建模重新框架化为**线性时不变系统的离散化**：

```
连续形式:    ḣ(t) = A·h(t) + B·x(t)     (状态演化)
            y(t) = C·h(t) + D·x(t)     (观测输出)

离散形式:    h_k = Ā·h_{k-1} + B̄·x_k   (其中 Ā = exp(A·Δt))
```

**关键创新点**：

| 组件 | 传统离散 RNN | SSMCell | 物理意义 |
|------|------------|---------|---------|
| 状态矩阵 A | 随机初始化/正交约束 | `A_real = -exp(linspace(0, 3, state_dim))` | 不同维度具有**指数衰减的时间常数**，模拟多尺度记忆 |
| 离散化步长 Δt | 固定为 1 | `delta_proj(time_deltas)` → Softplus 裁剪 | 将原始时间差（秒）映射为**自适应离散化步长** |
| 输入投影 B | 线性层 | `B_proj(x_seq)` + 硬裁剪 [-10, 10] | 防止极端输入破坏状态演化 |
| 输出投影 C | 线性层 | `C_proj(h_states)` | 从隐藏状态重构观测 |
| 跳跃连接 D | 常数或参数 | `D_skip`（可学习，初始 0.5） | 保留原始输入的"高频成分" |

**累积求和的高效实现**：

传统 RNN 的逐步递归 `h_k = Ā_k · h_{k-1} + B̄_k · x_k` 在 GPU 上难以并行。SSMCell 通过**对数空间累积和（log-space cumsum）**实现向量化：

```python
logA = torch.log(A_discrete.clamp(min=1e-10))
cumsum_logA = torch.cumsum(logA, dim=1)       # 并行前缀和
prefix_A = torch.exp(cumsum_logA)              # 累积衰减因子
weighted_Bx = Bx / prefix_A_safe               # 归一化输入
cumsum_weighted = torch.cumsum(weighted_Bx, dim=1)  # 并行累积
h_states = prefix_A * cumsum_weighted          # 最终状态
```

这等价于**并行扫描（parallel scan）**算法，将 O(L) 的串行依赖转化为 O(log L) 的并行树规约，在保持连续时间语义的同时实现 GPU 友好。

### 1.3 ContinuousSequenceEncoder：序列的多尺度池化

SSMCell 输出的是**完整序列的隐藏状态** `h ∈ [B, L, D]`，但 CTR 预测需要**压缩为固定长度的用户向量**。ContinuousSequenceEncoder 设计了三尺度池化机制：

```
┌────────────────────────────────────────┐
│  ContinuousSequenceEncoder 输出结构      │
├────────────────────────────────────────┤
│  'short'  : 近程注意力池化               │
│            时间权重: exp(-t/10s)        │
│            物理意义: 短期兴趣（最近几次行为）│
├────────────────────────────────────────┤
│  'long'   : 均匀平均池化                 │
│            时间权重: 1/L_valid            │
│            物理意义: 长期偏好（历史全貌）  │
├────────────────────────────────────────┤
│  'static' : 统计特征投影                 │
│            输入: [mean, std, last]       │
│            物理意义: 序列的宏观统计指纹    │
├────────────────────────────────────────┤
│  'full_seq': 完整隐藏序列 (保留用于原型)  │
└────────────────────────────────────────┘
```

**空序列处理**：对于无历史行为的新用户，不使用零填充（导致梯度消失），而是学习一个**参数化的空状态分布**：

```python
empty_mu = nn.Parameter(torch.randn(d_model) * 0.1)
empty_log_sigma = nn.Parameter(torch.randn(d_model) * 0.01 - 2.0)
empty_state = empty_mu + exp(empty_log_sigma) * noise  # 重参数化采样
```

这使得模型能区分"无历史"与"历史被掩码"两种语义，并为冷启动用户提供有意义的随机初始化表示。

---

## 二、特征交互：从"内积/拼接"到"流形-场域动力学"

### 2.1 问题背景

传统特征交互方法（FM、DeepFM、DCN）假设：
- 特征嵌入存在于**固定欧氏空间**
- 交互强度由**静态内积**度量
- 用户/物品表示是**独立学习**的

PCVRHeteroFormer 的核心洞见是：**用户兴趣与物品属性应在"条件化黎曼流形"上交互**，而非平坦的向量空间。

### 2.2 DynamicPrototypeManifold：动态原型流形

这是整个模型的**几何心脏**。它将用户的历史序列压缩嵌入到一个**可学习的原型流形**上：

```
┌─────────────────────────────────────────────────────────────────┐
│              DynamicPrototypeManifold 工作流程                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 全局原型 η ∈ R^(num_codes × code_dim)                      │
│     → 可学习参数，代表"理想化兴趣原子"                          │
│                                                                 │
│  2. 用户条件化旋转: CayleyRotation                              │
│     → 输入: user_feat (用户画像)                                │
│     → 输出: μ_u = Rotate(η; user_feat)                         │
│     → 物理意义: 同一原型，不同用户看到不同"视角"                │
│                                                                 │
│  3. 序列-原型匹配: von Mises-Fisher 分布                        │
│     → cosine_sim = z_norm · μ_u^T  (归一化余弦相似度)           │
│     → log_probs = κ · tanh(cosine_sim)                           │
│     → κ: 浓度参数（用户条件化，控制"分配锐度"）                │
│                                                                 │
│  4. 最优传输分配: LangevinSinkhorn                              │
│     → 将 log_probs 视为成本矩阵 C = -log_probs                  │
│     → 通过熵正则化 Sinkhorn 迭代求解最优传输计划 Π               │
│     → 添加 Langevin 噪声防止坍缩到退化分配                      │
│                                                                 │
│  5. 原型加权聚合:                                               │
│     → proto_repr = Σ_k Π_k · μ_u,k   (软分配加权平均)           │
│     → 空序列回退: empty_prior (学习到的均匀/偏置分布)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**von Mises-Fisher (vMF) 分布的语义**：

在欧氏空间中，softmax 内积 `exp(z·μ)` 假设向量模长任意。vMF 通过**显式归一化** `z_norm = z/||z||` 和 **浓度参数 κ**，将匹配度转化为**球面上的概率密度**：

```
p(z | μ, κ) ∝ exp(κ · μ^T · z)    where ||z|| = ||μ|| = 1
```

- κ → 0: 均匀分布（对所有原型无偏好）
- κ → ∞: 退化到最近原型（硬聚类）

κ 被设计为**用户条件化 + 时间衰减**：活跃用户（近期行为多）获得高 κ（锐分配），沉默用户获得低 κ（探索性分配）。

**LangevinSinkhorn 的作用**：

标准 Sinkhorn 迭代 `u = -logsumexp(log(K) + v)` 是确定性的。LangevinSinkhorn 在最后几步注入噪声：

```python
for _ in range(langevin_steps):
    u = -torch.logsumexp(...)  # 标准 Sinkhorn 步
    v = -torch.logsumexp(...)
    u = u + randn_like(u) * noise_scale  # Langevin 扰动
    v = v + randn_like(v) * noise_scale
```

这等价于在**传输计划的空间中进行随机梯度下降**，防止所有样本坍缩到同一原型（模式崩溃），同时保持边际约束。

### 2.3 CrossFieldNet：跨场域注意力网络

原型流形提供了**序列侧的压缩表示**，但 PCVR 还需要建模**用户画像、物品属性、上下文、序列**四者之间的交互。CrossFieldNet 将这四者视为**物理场（field）**，通过场间动力学实现交互：

```
┌─────────────────────────────────────────────────────────────┐
│                  CrossFieldNet 场域结构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入场:  {user, item, context, seq}                         │
│                                                             │
│  场嵌入:  field_emb ∈ R^(4 × d_model)                       │
│           → 每个场有一个可学习的"类型标识向量"              │
│                                                             │
│  层操作:  CrossFieldLayer × num_layers                      │
│           ├─ MultiheadSelfAttention: 场间信息交换           │
│           ├─ SeqFiLM: 序列场作为条件调制其他场              │
│           └─ FFN: 场内非线性变换                            │
│                                                             │
│  输出场:  {user', item', context', seq'}                    │
│           → 每个场都吸收了其他场的信息                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**SeqFiLM（序列条件化特征线性调制）**：

传统 FiLM 用全局向量生成 scale/shift。SeqFiLM 用**序列压缩表示** `s_short` 作为条件：

```python
gamma = sigmoid(W_γ · s_short)   # [B, d_model]
beta  = W_β · s_short             # [B, d_model]
output_field = input_field * gamma.unsqueeze(1) + beta.unsqueeze(1)
```

这意味着：**序列场不直接参与交互，而是作为"控制器"调制其他场的表达**。物理类比：序列是"磁场"，其他场是"磁化材料"——磁场不改变自身，但改变材料的响应方式。

### 2.4 LangevinForceField：朗之万力场（不确定性建模）

CrossFieldNet 建模的是**确定性交互**，但真实推荐中存在**固有不确定性**（用户行为随机性、数据噪声）。LangevinForceField 将四场表示视为**统计力学中的粒子**，通过朗之万方程建模其随机演化：

```
朗之万方程:
    dq/dt = p/M                     (位置演化)
    dp/dt = -γ·p + F(q) + √(2γT)·ξ  (动量演化，含噪声)

离散化 (Verlet-like):
    q_half = q + 0.5·dt·p/M
    F_det = ForceNet(q_half)         # 确定性力（神经网络）
    F = F_det + noise_scale·randn    # 添加热噪声
    p = p·(1-γ·dt) + F·dt            # 阻尼更新
    q = q_half + 0.5·dt·p/M          # 位置更新
```

**可解释性**：
- `M = exp(mass_log)`: 各场各维度的"惯性质量"——质量大的场更难被外力改变（用户画像通常比上下文质量大）
- `γ = softplus(gamma_log)`: 阻尼系数——高阻尼意味着系统快速达到稳态（短序列场景）
- `T = softplus(temperature_log)`: 温度——高温增加随机性，低温趋向确定性
- `uncertainty_head`: 从最终状态预测**各场各维度的不确定性**，用于后续门控决策

---

## 三、表示手术：Train/Eval 一致性修正

### 3.1 问题诊断

v9.3 原版的设计存在一个**结构性缺陷**：

```python
# 原版
if self.use_generative_fusion and self.training:   # ← eval 时跳过
    fused = gate0*proto + gate1*diff + gate2*energy
else:
    fused = proto   # ← eval 回退
```

这导致：
- **Train**：CTR head 学习 `fused_repr` 的分布（含 diff/energy 修正）
- **Eval**：CTR head 突然看到 `proto_repr`（完全不同的分布）
- **后果**：valid_auc 可能被人为压低，MetaAligner 误判过拟合，错误地提升辅助权重

### 3.2 残差融合（Representation Surgery）

v9.3-RS 将融合重新框架化为**残差修正**：

```python
# 新版
delta_diff = gen_diff_repr - proto_repr
delta_energy = gen_energy_repr - proto_repr
fused = proto_repr + gate1*delta_diff + gate2*delta_energy
```

**语义转变**：

| 维度 | 旧版（竞争性） | 新版（增量性） |
|------|--------------|--------------|
| gate1=0, gate2=0 | `fused = proto`（通过 g0=1 实现） | `fused = proto`（恒等映射） |
| gate1>0 | diff "抢夺" proto 的份额 | diff 提供**修正方向**，proto 作为基线保留 |
| Eval 行为 | 完全丢弃 diff/energy | 保留融合结构，仅将随机采样改为确定性中值 |
| 训练稳定性 | softmax 竞争可能导致梯度冲突 | 残差项天然有梯度衰减（接近 0 时影响小） |

**确定性 Eval 路径**：

```python
# DiffusionHead.get_gen_repr(deterministic=True)
t = num_steps // 2          # 固定为中间步（而非随机采样）
noise = zeros_like(proto)    # 零噪声（而非高斯噪声）
```

这使得 eval 时的 `gen_diff_repr` 是 `proto_repr` 的**确定性函数**，而非随机变量。CTR head 在 train/eval 看到的是**同一流形上的不同点**（train 是流形上的随机游走，eval 是固定锚点），而非**两个不同的流形**。

---

## 四、MetaAligner：双模式训练调度器

### 4.1 设计动机

传统多任务学习的权重调度通常依赖**验证集指标**（如 GradNorm、DWA、PCGrad），这在以下场景失效：
- 验证频率低（每 epoch 一次）
- 验证集与训练集分布偏移（时序数据）
- 早期训练阶段验证指标噪声大

MetaAligner 设计为**双模式状态机**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        MetaAligner 状态机                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 0~300:    WARMUP                                          │
│                 → alpha = [1.0, 0.0, 0.0]                       │
│                 → 纯 CTR 预热，但 Probe 已启动记录                │
│                                                                 │
│  Step 300~500:  PROBE_BURN_IN                                   │
│                 → alpha = [1.0, 0.0, 0.0]                       │
│                 → 积累 diff/energy 的 uncertainty 信号           │
│                                                                 │
│  Step 500+:     ┌─────────────────────────────────────────┐      │
│                 │ 分支 A: VALID_PID (valid_auc 可用且历史≥2)│      │
│                 │  → 经典 PID 控制: gap = train_auc - valid_auc │
│                 │                                              │      │
│                 │ 分支 B: TRAIN_SCHEDULE (无 valid_auc)      │      │
│                 │  → Online Uncertainty 驱动:                   │      │
│                 │     health = f(plateau, grad_fatigue,          │      │
│                 │              confidence, proto_chaos,           │      │
│                 │              diff_residual)                   │      │
│                 └─────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Online Uncertainty 信号

当 valid_auc 不可用时，MetaAligner 从训练前向的**副产品**中提取 5 维健康信号：

| 信号 | 计算来源 | 物理意义 | 响应策略 |
|------|---------|---------|---------|
| **Plateau** | `train_auc_history` 的滑动范围 | 模型是否陷入局部最优 | `aux_weight ↑` |
| **Grad Fatigue** | `grad_norm_ctr` 早期/近期比值 | 优化器是否"用力但无效" | `aux_weight ↑` |
| **Confidence Deficit** | `logits_std` 低于目标阈值 | 模型输出过于保守 | `energy_weight ↑` |
| **Proto Chaos** | `assign_entropy` 占最大熵比例 | 原型分配模糊，缺乏结构 | `diff_weight ↑` |
| **Diff Residual** | `MSE(pred_noise, noise)` | 表示空间去噪难度大 | `diff_weight ↑` |

这些信号全部来自 **no_grad 前向传播**，不增加 backward 开销，但使 MetaAligner 能在**每个 step** 更新权重，而非等待 validation。

### 4.3 PID 控制（Valid-aware 模式）

当 valid_auc 可用时，MetaAligner 切换为经典 PID：

```
error (gap) = train_auc - valid_auc          # 过拟合程度
P = kp * gap                                  # 当前误差响应
I = ki * ∫ gap dt                             # 历史误差累积（限幅 ±5）
D = kd * (-d(gap)/dt)                         # 抑制震荡
aux_weight = clamp(P + I + D, 0, 0.5)         # 辅助任务最多 50%
```

**积分限幅的重要性**：防止 valid_auc 暂时波动导致积分饱和（windup），确保系统可恢复。

---

## 五、序列建模 × 特征交互的协同机制

### 5.1 信息流动全景

```
用户行为序列 [seq_a, seq_b, ...]
        ↓
┌─────────────────────────────────────┐
│ ContinuousSequenceEncoder            │
│ ├─ SSMCell: 时间感知的连续状态演化   │
│ └─ 多尺度池化: short / long / static │
└─────────────────────────────────────┘
        ↓
   s_short (序列压缩表示)
        ↓
┌─────────────────────────────────────┐
│ DynamicPrototypeManifold             │
│ ├─ CayleyRotation: 用户条件化旋转    │
│ ├─ vMF 匹配: 序列-原型相似度         │
│ ├─ LangevinSinkhorn: 最优传输分配    │
│ └─ 任务投影: ctr/diff/energy 分离    │
└─────────────────────────────────────┘
        ↓
   proto_repr (用户兴趣的原子分解)
        ↓
┌─────────────────────────────────────┐
│ CrossFieldNet                        │
│ ├─ 四场输入: user/item/context/seq   │
│ ├─ SeqFiLM: 序列调制其他场           │
│ └─ 场间注意力: 信息交换与聚合          │
└─────────────────────────────────────┘
        ↓
   final_repr (交互后的联合表示)
        ↓
┌─────────────────────────────────────┐
│ GenerativeFusion (RS)                │
│ ├─ DiffusionHead: 去噪鲁棒表示       │
│ ├─ EnergyHead: 判别边界表示          │
│ └─ 残差融合: proto + Δdiff + Δenergy │
└─────────────────────────────────────┘
        ↓
   fused_repr → CTRHead → logit → sigmoid → PCVR
```

### 5.2 关键协同点

**协同点 1：序列 → 原型（时间几何化）**

SSMCell 将离散点击序列转化为**连续时间动力学**，其输出 `s_short` 不是"最近 K 个物品的嵌入平均"，而是"带有时间衰减核的卷积结果"。这使得：
- 1 小时前的点击与 1 周前的点击**自动获得不同权重**（通过 `exp(-t/10s)` 和 `exp(-t/86400s)` 双衰减）
- 时间间隔不均匀的数据无需填充或截断

**协同点 2：原型 → 场域（原子组合）**

`proto_repr` 是用户兴趣的**原子分解**（128 个原型的加权组合）。CrossFieldNet 不是直接用这个向量做预测，而是将其作为**场间交互的催化剂**：
- `seq` 场携带 `proto_repr`
- `user` 场携带用户画像
- `item` 场携带物品属性
- `context` 场携带上下文

四者在 CrossFieldLayer 中通过注意力交换信息，实现**"序列-informed 的特征交互"**——不是"序列 + 特征"的拼接，而是"序列调制下的特征重组"。

**协同点 3：场域 → 表示手术（不确定性消化）**

CrossFieldNet 的输出是确定性的，但推荐系统需要**认知不确定性建模**（"这个用户我真的了解吗？"）。DiffusionHead 和 EnergyHead 提供两种不确定性消化机制：
- **Diffusion**：通过去噪过程学习"表示空间的鲁棒邻域"——如果 proto_repr 附近的小扰动能被正确去噪，说明表示空间平滑
- **Energy**：通过能量函数学习"判别边界的置信度"——能量差大的样本对更容易分类

GenerativeFusion 不是简单加权三者，而是让 CTR head 学习**"何时信任原始表示、何时需要鲁棒修正、何时需要判别增强"**。

**协同点 4：MetaAligner → 全系统（动态平衡）**

MetaAligner 不是外部插件，而是**系统的"自主神经系统"**：
- 检测训练健康度（过拟合/停滞/正常）
- 调节辅助任务的介入强度
- 确保主任务（CTR）始终获得 ≥50% 的注意力

这使得系统能在**无人工调参**的情况下，自动适应不同数据阶段：早期快速收敛时专注 CTR，平台期时引入 diff/energy 打破僵局，过拟合时加强正则化。

---

## 六、设计原则总结

| 原则 | 具体体现 |
|------|---------|
| **时间即物理量** | SSMCell 将时间差作为离散化步长，而非位置索引 |
| **交互即场动力学** | CrossFieldNet 用 FiLM 和注意力建模场间耦合，而非静态内积 |
| **表示即流形嵌入** | DynamicPrototypeManifold 将用户兴趣嵌入到可学习的原型流形上 |
| **不确定性即信号** | Diffusion/Energy head 将噪声和能量 landscape 转化为有用特征 |
| **手术即最小侵入** | 残差融合保证 eval 一致性，不破坏主任务学习 |
| **调度即自主感知** | MetaAligner 从训练副产品提取信号，无需外部验证 |

---

*Version: v9.3-RSOU (Representation Surgery + Online Uncertainty)*
*Last Updated: 2026-05-14*
