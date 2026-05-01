在Baseline的基础上的工作：

---

## 一、数据工程层优化

### 1. 全局时间基准的延迟反馈权重
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **Parquet 统计预扫描** | `_get_global_max_timestamp()` 扫描所有文件的 `timestamp` 列统计信息 | 避免 batch-local max 导致权重塌陷 |
| **三类样本差异化权重** | `label_type=0` 权重=1.0；`label_type=1` sigmoid 降权；`label_type=2` 权重=1.0 | 点击未转化样本获得合理梯度，未点击不干扰 |
| **时间衰减平滑** | `sigmoid((time - window) / (window/4))` | 近期点击权重~0.1（假阴性），远期趋近0.9（真阴性） |

### 2. 动态负采样策略
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **保留全部点击未转化** | `click_not_conv_indices` 全部进入训练 | 最难负样本不丢失 |
| **未点击按时间采样** | `torch.topk(timestamps, n_target)` 选最近的 | 近期未点击更难区分，信息量更高 |
| **正样本全保留** | 不采样，全部训练 | 极度不平衡下确保正样本曝光 |

### 3. 数据加载性能
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **预分配 numpy buffer** | `__init__` 阶段分配 `_buf_user_int` 等 | 消除 per-batch `np.zeros` 开销 |
| **file_system 共享策略** | `torch.multiprocessing.set_sharing_strategy('file_system')` | 规避 `/dev/shm` 耗尽 |
| **persistent_workers** | `DataLoader(..., persistent_workers=True)` | 避免 worker 进程反复创建 |
| **Row Group 顺序读取** | 按物理顺序切分 train/valid | 防止未来信息泄露 |

---

## 二、特征工程层优化

### 4. 列名驱动的特征检测
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **正则匹配列名** | `COLUMN_PATTERNS` 识别 `user_int_feats_123` 等 | 不依赖 schema.json 的 fid 映射，防数据泄露 |
| **元数据列隔离** | `METADATA_COLUMN_NAMES` 黑名单 | `label_time` 等绝不可能进入特征 |
| **item_dense 补全** | 新增 `_item_dense_plan` 和转换逻辑 | 商品连续特征（价格、CTR）不再丢失 |

### 5. 时间特征编码
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **离散 Time Bucket** | 64 个非均匀边界（5秒~1年）→ 65 个桶 | 近期行为高分辨率，远期低分辨率 |
| **连续 Time Encoding** | `ContinuousTimeEncoding` 对数尺度正弦编码 | 捕捉 bucket 无法表达的精细时间差 |
| **全局 timestamp 传播** | `ModelInput.timestamp` 传入 `_embed_seq_domain` | 序列内每个位置计算 `current_time - event_time` |

---

## 三、训练流程层优化

### 6. 课程学习（Curriculum Learning）
| 阶段 | 损失函数 | 目的 |
|:---|:---|:---|
| **Stage 1** (epoch ≤ 2) | 纯 delay-aware BCE | 稳定学习基础排序 |
| **Stage 2** (epoch ≤ 7) | BCE + Focal Loss | 聚焦难样本，提升 AUC |
| **Stage 3** (epoch > 7) | BCE + λ·InfoNCE | 引入对比学习，优化用户内排序 |

### 7. 三优化器混合
| 优化器 | 参数 | 学习率 | 特性 |
|:---|:---|:---|:---|
| **Muon** | 2D 参数（Linear 权重） | `lr * 0.1` 或显式设置 | 正交更新，适合深层网络 |
| **AdamW** | 非 2D 稠密参数 | `lr` | 动量稳定，权重衰减 |
| **Adagrad** | Embedding 权重 | `sparse_lr` | 逐坐标累积，适合稀疏更新 |

### 8. 学习率调度
| 组件 | 实现 | 效果 |
|:---|:---|:---|
| **Warmup** | `LinearLR(start_factor=0.1, total_iters=10% steps)` | 防止初期梯度爆炸 |
| **Cosine Annealing** | `CosineAnnealingLR(T_max=90% steps)` | 精细收敛，避免末端震荡 |
| **Stage 3 衰减** | `lr *= stage3_lr_factor` | 对比学习需要更低 lr |

### 9. 梯度与稳定性
| 优化点 | 实现 | 效果 |
|:---|:---|:---|
| **梯度裁剪** | `clip_grad_norm_` Muon/Adam 1.0，Sparse 0.5 | 抑制极端样本的梯度爆炸 |
| **梯度累积** | `accumulation_steps` | 等效大 batch，平滑梯度 |
| **高基数 Embedding 冷重启** | `reinit_high_cardinality_params()` | 每 epoch 重置高频 ID embedding，防过拟合 |
| **NaN 处理** | `torch.nan_to_num(out, nan=0.0)` | SDPA 输出异常值清零 |

---

## 四、验证与评估优化

### 10. 延迟感知评估
| 指标 | 实现 | 效果 |
|:---|:---|:---|
| **GAUC** | 按 `user_id` 分组计算 AUC 后加权平均 | 消除用户间差异，更稳定 |
| **Delay Bucket AUC** | 按 `observed_delay` 分桶（<1d, <3d, <7d 等） | 监控模型对不同延迟样本的排序能力 |
| **LogLoss 校准** | `binary_cross_entropy_with_logits` 直接计算 | 概率校准性可量化 |

---
