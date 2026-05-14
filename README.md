# "序列建模 × 特征交互" 领域难题：九篇论文深度分析报告

## 一、领域背景与核心难题

工业级推荐系统面临一个根本性的架构困境：**用户行为序列建模**与**异构特征交互**长期以来是两个独立发展的技术分支。传统架构采用"编码-再交互"（encode-then-interaction）的两阶段流水线：先用序列编码器（如DIN、SIM、LONGER）压缩长行为历史，再将压缩后的序列表征与静态特征一起输入特征交互模块（如DCN、Wukong、RankMixer）。这种设计存在三大核心难题：

| 难题 | 具体表现 |
|------|---------|
| **信息流动单向化** | 序列压缩后才与特征交互，静态特征无法反向塑造序列表征 |
| **参数分配次优化** | 序列模块与交互模块独立参数化，在有限算力预算下互相竞争 |
| **Scaling效率瓶颈** | 分离架构导致计算资源无法全局最优配置，Scaling收益递减 |

2024年底至2026年初，字节跳动（ByteDance）、腾讯（Tencent）、快手（Kuaishou）等工业界团队密集发表了一系列突破性工作，试图从根本上解决这一难题。以下是对九篇关键论文的系统分析。

---

## 二、各论文核心贡献详解

### 1. InterFormer (2024.11)  — 双向交互的先行者

**机构**: 字节跳动 (ByteDance)  
**核心范式**: **交错式双向异构交互学习 (Interleaving Bidirectional Heterogeneous Interaction)**

InterFormer是这一领域最早的系统性尝试之一，其核心贡献在于识别并解决了传统架构中的两个根本缺陷：

- **单向信息流问题**：传统方法中序列到特征的流动是单向的，InterFormer通过"交错式"（interleaving）设计实现了双向信息交换
- **过早聚合导致信息损失**：传统方法在序列压缩阶段就进行激进的信息聚合，InterFormer通过保留各数据模式的完整信息，使用独立的桥接架构进行信息选择和摘要

**架构创新**：
- 引入可学习的交互令牌（interaction tokens）作为序列编码器与特征交互网络之间的双向桥梁
- 每个数据模式保留完整信息，通过独立的桥接架构实现有效的信息选择和摘要

**Scaling视角**：InterFormer主要聚焦于架构层面的交互深度优化，尚未系统探索模型规模的Scaling规律，参数规模相对较小。

---

### 2. LONGER (2025.05)  — 工业级长序列建模的标杆

**机构**: 字节跳动 (ByteDance) — RecSys'25  
**核心范式**: **端到端超长序列GPU高效Transformer (Long-sequence Optimized traNsformer for GPU-Efficient Recommenders)**

LONGER专注于解决序列建模侧的Scaling问题，将工业级序列长度从数百扩展到**10,000+**：

**三大核心创新**：

| 创新点 | 技术细节 | 效果 |
|--------|---------|------|
| **Global Token机制** | 引入目标物品表征、可学习CLS、UID嵌入等全局令牌作为注意力锚点 | 稳定长上下文注意力分布，缓解"注意力汇聚"现象 |
| **Token Merge模块** | 将相邻令牌分组，通过轻量级InnerTransformer压缩序列长度K倍 | FLOPs降低约**42.8%**（K=4时），几乎无损性能 |
| **混合注意力策略** | 第一层Cross-Causal Attention + 后续N层Self-Causal Attention | 聚焦关键局部行为同时捕获全局上下文 |

**工程优化体系**：
- **全同步训练框架**：Dense和Sparse参数统一在GPU上存储和更新，消除外部Parameter Server
- **混合精度训练 + 激活重计算**：BF16/FP16混合精度，+18%吞吐量，-16%训练时间，-28%内存
- **KV Cache Serving**：用户序列KV预计算缓存，跨候选复用，在线吞吐量损失从-40%降至可忽略

**Scaling成果**：在抖音广告系统CVR任务上，AUC达到0.85290，相对基线提升1.57%。已在字节跳动数十个场景全量部署，服务数十亿用户。

---

### 3. RankMixer (2025.07)  — 硬件感知的特征交互Scaling

**机构**: 字节跳动 (ByteDance)  
**核心范式**: **多头部令牌混合 + 逐令牌FFN + 稀疏MoE (Multi-Head Token Mixing + Per-Token FFN + Sparse-MoE)**

RankMixer是特征交互侧的Scaling标杆，核心洞察是：**传统CPU时代设计的特征交叉模块在现代GPU上MFU极低（仅4.5%）**：

**架构创新**：

```
RankMixer Block = TokenMixing → Per-Token FFN (PFFN)
                 ↓
              Sparse-MoE variant
```

1. **Multi-Head Token Mixing**：将每个令牌均分为H个头，跨令牌重组实现信息交互
   - 完全**无参数**的混洗操作，避免自注意力中异构语义空间内积相似度计算的不可靠性
   - 计算复杂度从O(T²D)降至O(TD)，且高度并行化

2. **Per-Token FFN (PFFN)**：为每个特征令牌分配独立的FFN参数
   - 解决"高频特征主导、低频信号淹没"问题
   - 参数量随令牌数线性扩展，但计算量保持不变

3. **Sparse-MoE扩展**：将PFFN升级为稀疏专家混合
   - **ReLU Routing**：替代Top-k+Softmax，通过自适应L1惩罚控制稀疏度
   - **Dense-Train/Sparse-Infer (DTSI)**：训练时密集更新，推理时稀疏激活

**Scaling突破**：
- **MFU从4.5%提升至45%**（10倍提升）
- 模型参数扩展**100倍**（至1B Dense参数），推理延迟基本持平
- 在万亿级生产数据集上验证Scaling Law：AUC增益 vs. 参数/Flops呈对数线性关系

**在线成果**：抖音推荐全流量部署，用户活跃天数+0.3%，应用内使用时长+1.08%。

---

### 4. OneTrans (2025.10)  — 统一Transformer Stack的开创者

**机构**: 字节跳动 (ByteDance) + 南洋理工大学 — WWW'26  
**核心范式**: **统一Tokenizer + 混合参数化因果Transformer (Unified Tokenizer + Mixed Parameterization Causal Transformer)**

OneTrans是**首个**将序列建模和特征交互完全统一在单一Transformer Stack中的工业级架构：

**核心架构**：

```
输入 → [Sequential Tokenizer | Non-Seq Tokenizer] → 统一Token序列
       → OneTrans Pyramid Stack (L层Mixed Causal Attention + Mixed FFN)
       → 任务头
```

**四大创新**：

| 创新 | 说明 |
|------|------|
| **统一Tokenizer** | 序列特征→S-tokens，非序列特征→NS-tokens，统一为单一序列 |
| **混合参数化 (Mixed Parameterization)** | S-tokens共享QKV/FFN参数；NS-tokens各自拥有独立参数 |
| **金字塔策略 (Pyramid)** | 逐层裁剪S-tokens查询数量（如从1500→16），渐进蒸馏 |
| **跨请求KV缓存** | S-tokens一次计算跨候选复用，复杂度从O(C)降至O(1) |

**关键洞察**：通过因果掩码设计，NS-tokens可以 attending 整个S-tokens历史，实现"目标注意力"式的序列聚合，同时NS-tokens之间也能相互交互。

**Scaling验证**：
- OneTransS (6层, d=256, ~100M参数) vs. OneTransL (8层, d=384)
- 展示近对数线性Scaling规律：TFLOPs增加→ΔUAUC可预测提升
- 在线A/B测试：每用户GMV提升**5.68%**

---

### 5. HyFormer (2026.01)  — 混合Transformer的深度融合

**机构**: 字节跳动 (ByteDance)  
**核心范式**: **Query Decoding + Query Boosting 交替优化 (Hybrid Transformer with Alternating Optimization)**

HyFormer深刻反思了LONGER+RankMixer分离架构的局限，提出**统一混合Transformer**：

**核心问题诊断**：
- 分离架构中查询令牌过于简化（仅基于候选物品或全局特征）
- 序列-特征交互仅在模型后期发生，延迟融合限制表达能力
- 增加容量主要改善孤立组件而非联合表征，Scaling效率递减

**双核心机制**：

```
HyFormer Block = Query Decoding → Query Boosting
                    ↓                ↓
              (Cross Attention)  (MLP-Mixer Token Mixing)
              全局查询→序列KV      跨查询/跨序列异构交互
```

1. **Query Decoding**：将非序列特征扩展为多个语义全局查询令牌，通过Cross Attention解码长序列的层-wise KV表征
   - 支持三种序列编码策略：Full Transformer / LONGER-style / Decoder-style (SwiGLU)
   - 深层复用前层查询， progressively richer interrogation

2. **Query Boosting**：通过MLP-Mixer风格轻量级令牌混合增强查询表征
   - 将解码后的查询与NS-tokens拼接，进行跨令牌信息混合
   - Per-Token FFN进行子空间特定精化

**多序列处理**：每个行为序列独立处理（独立查询令牌），避免MTGR/OneTrans中序列合并导致的语义混淆。

**Scaling表现**：在十亿级工业数据集上，相同参数和FLOPs预算下一致优于LONGER和RankMixer基线，展示更优的Scaling行为。已在字节跳动高流量生产系统全量部署。

---

### 6. TokenMixer-Large (2026.02)  — 极端规模特征交互

**机构**: 字节跳动 (ByteDance)  
**核心范式**: **Mixing-Reverting + 区间残差 + 稀疏逐令牌MoE (Systematically Evolved Extreme-Scale TokenMixer)**

TokenMixer-Large是RankMixer的系统化演进，解决深度TokenMixer的关键瓶颈：

**五大瓶颈与解决方案**：

| 瓶颈 | 解决方案 | 技术细节 |
|------|---------|---------|
| 残差路径次优化 | **Mixing-Reverting操作** | 两层对称结构：第一层Mixing，第二层Revert，确保维度一致 |
| 梯度消失 | **区间残差 + 辅助损失** | 每2-3层添加跳跃连接；结合低层logits与高层联合损失 |
| 架构碎片化 | **纯模型哲学** | 移除LHUC、DCNv2等历史遗留低MFU算子 |
| MoE稀疏化不充分 | **Sparse-Per-token MoE** | "先扩展再稀疏"策略；Top-k + Shared Expert + Gate Value Scaling |
| 规模受限 | **系统性Scaling探索** | 深度、宽度、令牌数、专家数四维度正交扩展 |

**关键创新详解**：

- **Mixing-Reverting**：
  ```
  Input (T tokens) → Split into H heads → Mix across tokens → H mixed tokens
                   → Split back → Revert to T tokens → Per-token SwiGLU
  ```
  解决RankMixer中输入T与输出H维度不匹配导致的残差传播断裂。

- **Down-Matrix Small Init**：受ReZero启发，将SwiGLU最终层初始化方差从1降至0.01，使F(x)+x近似恒等映射，提升训练稳定性。

- **FP8量化 + Token并行**：推理FP8 E4M3量化，1.7x加速；Token并行解决多设备扩展瓶颈。

**Scaling成果**：
- **离线实验**：15B参数（抖音广告）、7B参数（电商）
- **在线流量**：7B参数（广告）、4B参数（电商）
- **在线收益**：电商订单+1.66%，人均预览支付GMV+2.98%；广告ADSS+2.0%；直播收入+1.4%

---

### 7. MixFormer (2026.02)  — 联合Scaling的统一架构

**机构**: 字节跳动 (ByteDance) — 抖音/抖音Lite  
**核心范式**: **统一参数化 + 用户-物品解耦 (Fully Unified Transformer with Co-Scaling)**

MixFormer提出一个根本性问题：**分离架构下，序列模块和密集交互模块在有限计算预算下存在参数分配的结构性矛盾**。

**核心架构**：

```
MixFormer Block = Query Mixer → Cross Attention → Output Fusion
                      ↓              ↓              ↓
                 (HeadMixing)   (MultiHeadAttn)  (Per-head SwiGLU FFN)
                 + Per-head FFN  (序列聚合)      (深度融合)
```

1. **Query Mixer**：替代Self-Attention
   - HeadMixing：无参数跨头信息交换（reshape + transpose + flatten）
   - Per-head SwiGLU FFN：每个头独立处理，保留异构语义

2. **Cross Attention**：将Query Mixer输出的N个头直接作为N个查询头
   - 每个头作为语义专业化的子查询，分别关注行为序列的不同子空间
   - 序列动作通过Per-Layer SwiGLU FFN逐层精化

3. **Output Fusion**：Per-head SwiGLU FFN深度融合序列与非序列信号

**用户-物品解耦 (UI-MixFormer)**：
- 将非序列特征解耦为用户侧和物品侧
- HeadMixing引入Mask矩阵，确保用户侧头不接收物品侧信号
- 实现Request-Level Batching (RLB)：用户侧计算跨候选复用，**FLOPs降低36%**

**Scaling分析**：
- **Dense Scaling**：固定序列长度512，增加模型容量，MixFormer斜率优于所有基线
- **Sequence Scaling**：固定密集参数，扩展序列长度{512, 2048, 8192, 10000}，MixFormer斜率与STCA相当
- 验证：**统一参数化同时实现最优的Dense Scaling和Sequence Scaling**

**在线成果**：抖音和抖音Lite大规模A/B测试，用户活跃天数和应用内使用时长显著提升。

---

### 8. UniMixer (2026.04)  — 统一理论框架

**机构**: 快手 (Kuaishou)  
**核心范式**: **参数化TokenMixer + 统一Scaling模块设计框架 (Unified Theoretical Framework for Scaling Laws)**

UniMixer的核心贡献是**理论统一**：首次建立注意力机制、TokenMixer、因子分解机三大Scaling范式的统一数学框架。

**核心洞察**：

TokenMixer的规则-based混洗操作可等价表示为置换矩阵：
```
TokenMixer(X) = reshape(W_perm · flatten(X))
```
其中 W_perm ∈ R^(TD×TD) 是大置换矩阵。

**UniMixer通过Kronecker分解实现参数化**：
- **可压缩性**：W_perm = G ⊗ I，其中 G ∈ R^(T²×T²)，I ∈ R^(D/T×D/T)
- **双随机性**：每行每列和为1
- **稀疏性**：每行/列仅一个非零元
- **对称性**：T=H时对称

**UniMixing模块**：
```
UniMixing(X) = reshape((W_G ⊗ {W_B^i}) · flatten(X))
```
- W_G：控制全局交互模式（类比注意力权重）
- W_B^i：控制局部块内交互（类比Value投影）

**统一视角**：

| 方法 | 全局混合模式 G(X,W_G) | 局部混合模式 |
|------|----------------------|-------------|
| Self-Attention | QK^T内积相似度 | V投影 |
| Heterogeneous Attention | 令牌特定QK内积 | 令牌特定V投影 |
| TokenMixer | 输入无关的置换 | 恒等映射 |
| FM (Wukong) | XX^T (退化注意力) | Y投影 |
| **UniMixer** | **可学习的参数化全局模式** | **可学习的局部块交互** |

**UniMixing-Lite**：
- 局部交互：基矩阵动态组合（减少冗余）
- 全局交互：低秩近似（W_G ≈ A_G · B_G）
- 同时保留TokenMixer的低参数优势和注意力的异构建模能力

**Scaling曲线**：在AUC vs. 参数/FLOPs图上，UniMixer和UniMixer-Lite均展示优于RankMixer的Scaling效率。

---

### 9. TokenFormer (2026.04)  — 序列坍塌传播的终结者

**机构**: 腾讯 (Tencent)  
**核心范式**: **Bottom-Full-Top-Sliding注意力 + 非线性交互表征 (SCP-Aware Unified Decoder-Only Architecture)**

TokenFormer识别了一个被前人忽视的**根本性失败模式**：**序列坍塌传播 (Sequential Collapse Propagation, SCP)**。

**SCP现象**：
- 非序列特征（如低基数人口统计字段）由于信息贫乏，嵌入容易占据低维子空间
- 在统一架构中，这些坍塌倾向的静态令牌通过共享算子与序列行为令牌直接交互
- 导致**高维序列表征被低维静态特征"污染"坍塌**

实验验证（KuaiRand-27k）：
- 纯序列Transformer：有效秩(eRank)=71.2，但互信息(MI)低
- 朴素联合Transformer：MI提升，但eRank降至55.1（显著坍塌）
- **TokenFormer**：eRank恢复至62.7，同时MI最高

**三大创新**：

| 创新 | 机制 | 作用 |
|------|------|------|
| **BFTS注意力** | 底层Full Causal Attention + 上层Shrinking SWA | 浅层建立全局跨域交互，深层聚焦局部时序精化 |
| **NLIR** | σ(G) ⊙ A 的逐元素乘法门控 | 注入高阶非线性，恢复表征秩，增强判别性 |
| **非序列令牌丢弃** | 全注意力层后丢弃静态字段令牌 | 强制深层仅关注行为演化 |

**统一令牌流**：
```
X^(0) = [x^F_1, ..., x^F_M, ⟨sep⟩, x^T_s1, x^T_a1, ..., x^T_sT, x^T_aT, ⟨sep⟩, x^V_c1, ...]
```
- 统一RoPE位置编码，类型感知索引方案
- 支持User-Centric (NTP损失) 和 New Impression Only (BCE损失) 两种范式

**工业验证**：腾讯广告平台大规模在线部署，验证Scaling Law和维度鲁棒性。

---

## 三、综合对比分析

### 3.1 架构演进脉络

```
Phase 1: 分离架构 (2024及以前)
    DIN/SIM + DCN/Wide&Deep
    └── 序列编码 → 压缩向量 → 特征交互
    
Phase 2: 双向桥接 (2024.11)
    InterFormer
    └── 序列编码 ↔ 交互令牌 ↔ 特征交互 (双向但分离)
    
Phase 3: 单模块专攻 (2025)
    LONGER ──→ 超长序列端到端建模 (序列侧Scaling)
    RankMixer → 硬件感知特征交互Scaling (交互侧Scaling)
    
Phase 4: 统一架构 (2025.10 - 2026.02)
    OneTrans ──→ 统一Transformer Stack + 混合参数化
    HyFormer ──→ Query Decoding/Boosting交替优化
    MixFormer ─→ 完全统一参数化 + 联合Scaling
    TokenMixer-Large → 极端规模TokenMixer (7B-15B)
    
Phase 5: 理论深化与问题修复 (2026.04)
    UniMixer ──→ 三大范式的统一理论框架
    TokenFormer → SCP问题识别与修复 + BFTS/NLIR
```

### 3.2 关键设计选择对比

| 论文 | 序列-特征关系 | 核心算子 | 参数策略 | 最大规模 | 工业部署 |
|------|-------------|---------|---------|---------|---------|
| InterFormer | 双向桥接 | 交错交互令牌 | 分离 | 小 | 未公开 |
| LONGER | 序列主导 | Global Token + Token Merge | 独立 | 长序列10K | ✅ 字节跳动 |
| RankMixer | 交互主导 | Token Mixing + PFFN | 分离 | 1B Dense | ✅ 字节跳动 |
| OneTrans | **统一Stack** | Mixed Causal Attention | 混合 | ~100M | ✅ 字节跳动 |
| HyFormer | **统一Hybrid** | Query Decoding + Boosting | 混合 | 未公开 | ✅ 字节跳动 |
| TokenMixer-Large | 交互主导 | Mixing-Reverting + S-P MoE | 分离 | **15B** | ✅ 字节跳动 |
| MixFormer | **完全统一** | Query Mixer + Cross Attn | **统一** | 未公开 | ✅ 抖音 |
| UniMixer | 理论统一 | 参数化UniMixing | 统一 | 未公开 | ✅ 快手 |
| TokenFormer | **统一Stream** | BFTS + NLIR | 统一 | 未公开 | ✅ 腾讯 |

### 3.3 Scaling视角总结

**Scaling Law验证**：
- 所有论文均在工业级数据集上验证了**近对数线性Scaling规律**：模型性能随参数/FLOPs增加可预测提升
- RankMixer/TokenMixer-Large在万亿级数据上验证；OneTrans/HyFormer在十亿级样本上验证

**Scaling效率指标 (MFU)**：
- RankMixer: 4.5% → **45%**
- TokenMixer-Large: **60%** (广告骨干网络)
- 关键手段：移除内存受限算子、纯模型哲学、FP8量化、Token并行

**Scaling维度**：
- RankMixer/TokenMixer-Large：四维度正交扩展（Token数T、宽度D、深度L、专家数E）
- MixFormer：统一参数化实现Dense Capacity和Sequence Length的**联合最优Scaling**
- UniMixer：通过理论统一，在相同FLOPs下实现更陡的Scaling曲线

---

## 四、领域趋势与未来方向

### 4.1 已确立的共识

1. **统一架构是方向**：分离的encode-then-interaction流水线已被证明存在根本性效率瓶颈，OneTrans/HyFormer/MixFormer/TokenFormer/UniMixer五种不同路径均指向统一建模
2. **Tokenization是关键**：如何将异构特征（ID、数值、序列、交叉特征）统一为令牌序列，决定了后续架构的上限
3. **硬件协同设计不可或缺**：MFU从个位数提升至60%，证明推荐系统Scaling必须走LLM式的硬件-软件协同优化路径
4. **Scaling Law在推荐领域成立**：从1B到15B参数，性能随规模可预测增长，但架构设计决定Scaling斜率

### 4.2 仍待解决的核心问题

| 问题 | 当前状态 | 潜在方向 |
|------|---------|---------|
| **SCP (序列坍塌)** | TokenFormer识别并提出BFTS/NLIR缓解 | 更系统的表征秩保持机制；理论保证 |
| **多序列异构性** | HyFormer采用独立查询；OneTrans用[SEP]分隔 | 自动学习序列间关系；动态序列权重 |
| **深度模型训练** | TokenMixer-Large用区间残差+辅助损失 | 更深网络（>10层）的稳定训练；自适应深度 |
| **推理效率** | KV Cache、RLB、FP8等 | 投机解码；序列增量更新；模型蒸馏 |
| **多任务统一** | 各论文聚焦CTR/CVR | 生成式推荐统一框架；跨域迁移 |

### 4.3 架构选择建议

- **若序列长度 > 5000且为主要信号**：LONGER + HyFormer/MixFormer混合架构
- **若特征交互复杂且需极端Scaling**：TokenMixer-Large (7B-15B) + Sparse MoE
- **若追求理论优雅与统一**：UniMixer框架，根据场景选择具体变体
- **若担心SCP问题**：TokenFormer的BFTS+NLIR设计
- **若需快速部署且LLM基础设施成熟**：OneTrans（直接复用FlashAttention、KV Cache等）

---

## 五、结论

这九篇论文共同勾勒出"序列建模 × 特征交互"领域从**分离**到**统一**、从**手工设计**到**Scaling驱动**的清晰演进轨迹。字节跳动、腾讯、快手三家公司的工业实践验证了一个核心结论：**在推荐系统中，统一架构不仅能提升建模表达能力，更能通过消除模块间的参数竞争，实现更高效的Scaling**。

特别值得注意的是，这一领域的发展呈现出与LLM领域相似的模式：早期是各种专用架构的百花齐放（DIN/DCN/Wide&Deep），随后是Transformer的引入（AutoInt/HiFormer），接着是Scaling Law的发现与验证（Wukong/RankMixer/LONGER），最终走向统一Backbone（OneTrans/HyFormer/MixFormer/TokenFormer/UniMixer）。TokenFormer对SCP问题的揭示，以及UniMixer对三大范式的理论统一，标志着该领域正从工程实践走向更深层的理论理解。

<img width="1787" height="1387" alt="image" src="https://github.com/user-attachments/assets/14f8722b-4e31-42a1-993e-0c21b248abc1" />

