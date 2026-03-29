# RQ-KMeans 算法文档

## 1. 算法概述

RQ-KMeans 是一种基于 K-Means 聚类的残差量化（Residual Quantization）方法，用于将物品的连续特征向量离散化为**多层 Semantic ID**。

与 RQ-VAE 的核心区别：用 K-Means 聚类替代 EMA（指数移动平均）码本更新，码本利用率天然可达 100%。

## 2. 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RQ-KMeans 三阶段训练                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段1: 预训练 Encoder+Decoder (无量化)                               │
│  ┌──────────┐    ┌──────────┐                                       │
│  │ 物品特征  │───▶│ Encoder  │───▶ z ───▶ Decoder ───▶ x̂            │
│  │ x [402d] │    │ [402→128]│                                     │
│  └──────────┘    └──────────┘        Loss = MSE(x̂, x)              │
│                                                                     │
│  阶段2: K-Means 初始化码本                                           │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐                   │
│  │ 所有物品  │───▶│ Encoder  │───▶│ z₁...z_N    │───▶ K-Means L0   │
│  │ 特征矩阵  │    │ (冻结)   │    │ [N, 128]    │    → 32个聚类中心 │
│  └──────────┘    └──────────┘    └──────┬──────┘                   │
│                                         │ residual = z - q₀(z)     │
│                                         ▼                           │
│                                   K-Means L1 → 32个聚类中心          │
│                                         │ residual = r₁ - q₁(r₁)   │
│                                         ▼                           │
│                                   K-Means L2 → 32个聚类中心          │
│                                                                     │
│  阶段3: 微调 Decoder (含量化)                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ 物品特征  │───▶│ Encoder  │───▶│ RVQ量化  │───▶│ Decoder  │───▶ x̂│
│  │ x [402d] │    │ (冻结)   │    │ (码本冻结)│    │ (可训练)  │     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                              Loss = MSE(x̂,x) + λ·Commit │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. 数据流详解

### 3.1 输入数据

```
物品特征矩阵: [3416, 402]
  ├── 标题 Sentence Embedding: [3416, 384]  ← all-MiniLM-L6-v2
  └── 类别 One-Hot 编码:       [3416, 18]   ← 18种电影类型
```

### 3.2 编码器（Encoder）

```python
Linear(402 → 512) → BN → ReLU → Dropout
Linear(512 → 512) → BN → ReLU → Dropout
Linear(512 → 128)

# 输入: [B, 402]
# 输出: z [B, 128]  潜在向量
```

### 3.3 残差 K-Means 量化（核心）

```
输入: z [B, 128]

Level 0:
  ┌──────────────────────────────────────┐
  │  K-Means Codebook L0: [32, 128]      │
  │                                      │
  │  对 z 做最近邻查找:                    │
  │  code₀ = argmin_k ||z - c_k||²       │
  │  quantized₀ = C₀[code₀]              │
  │  residual₁ = z - quantized₀          │
  └──────────────────────────────────────┘

Level 1:
  ┌──────────────────────────────────────┐
  │  K-Means Codebook L1: [32, 128]      │
  │                                      │
  │  对 residual₁ 做最近邻查找:            │
  │  code₁ = argmin_k ||r₁ - c_k||²     │
  │  quantized₁ = C₁[code₁]              │
  │  residual₂ = r₁ - quantized₁         │
  └──────────────────────────────────────┘

Level 2:
  ┌──────────────────────────────────────┐
  │  K-Means Codebook L2: [32, 128]      │
  │                                      │
  │  code₂ = argmin_k ||r₂ - c_k||²     │
  │  quantized₂ = C₂[code₂]              │
  └──────────────────────────────────────┘

输出:
  量化向量 z_q = q₀ + q₁ + q₂  [B, 128]
  Semantic ID = [code₀, code₁, code₂]  [3, B]
```

**残差量化的意义**：每一层只编码上一层未捕获的信息，逐层细化。

### 3.4 解码器（Decoder）

```python
Linear(128 → 512) → BN → ReLU → Dropout
Linear(512 → 512) → BN → ReLU → Dropout
Linear(512 → 402)

# 输入: z_q [B, 128]  量化后的向量
# 输出: x̂ [B, 402]    重建的物品特征
```

### 3.5 损失函数

```
总损失 = 重建损失 + λ × Commitment损失

重建损失 = MSE(x̂, x)
  - 解码器重建的特征与原始特征的均方误差

Commitment损失 = MSE(z, z_q.detach())
  - 鼓励编码器输出靠近量化后的向量（直通估计器）
  - 梯度不回传到码本（码本由K-Means固定）
```

## 4. Semantic ID 生成

训练完成后，对每个物品生成 Semantic ID：

```
物品i: x_i [402]
  │
  ▼ Encoder
z_i [128]
  │
  ▼ RVQ (最近邻查找)
  Level 0: code₀ = argmin_k ||z_i - C₀[k]||     → 7
  Level 1: code₁ = argmin_k ||r₁ - C₁[k]||       → 22
  Level 2: code₂ = argmin_k ||r₂ - C₂[k]||       → 15

Semantic ID = [7, 22, 15]
```

**语义空间大小**：32 × 32 × 32 = 32,768 种可能的 ID

## 5. 码本利用率分析

| 方法 | 码本利用率 | 原因 |
|------|-----------|------|
| RQ-VAE (EMA) | ~72% | EMA更新有累积偏差，部分码本被冷落 |
| RQ-KMeans | **100%** | K-Means 保证每个 cluster 都有数据点分配 |

K-Means 的 Lloyd 算法保证每个 cluster 至少包含一个最近的点，因此码本利用率天然 100%。

## 6. 完整数据流（从原始数据到推荐）

```
Step 1-2: 数据预处理
  movies.dat → titles + genres
  ratings.dat → filter (≥5) → train/val/test split

Step 3: 特征提取
  titles → Sentence Embedding [3416, 384]
  genres → One-Hot [3416, 18]
  拼接 → item_features.npy [3416, 402]

Step 4: RQ-KMeans 训练
  item_features → Encoder → z → RVQ → z_q → Decoder → x̂
  输出: rq_kmeans_best.pt (模型 + 码本)

Step 5: Semantic ID 生成
  item_features → Encoder → RVQ (最近邻查找)
  输出: semantic_ids.npy [3416, 3]

Step 6: Decoder 训练 (序列推荐)
  用户历史: [itemA_sid, itemB_sid, itemC_sid] → Transformer → 预测 next_sid
  输出: decoder_best.pt

Step 7: 推荐推理
  用户输入 → Decoder → 预测 SID → 查找匹配物品 → 推荐列表
```

## 7. 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `input_dim` | 402 | 特征维度 (384 + 18) |
| `hidden_dim` | 512 | 编码器/解码器隐藏层维度 |
| `latent_dim` | 128 | 潜在空间维度 |
| `num_codebooks` | 3 | 残差量化层数 (Semantic ID 位数) |
| `codebook_size` | 32 | 每层码本大小 |
| `encoder_pretrain_epochs` | 30 | 阶段1预训练轮数 |
| `decoder_finetune_epochs` | 20 | 阶段3微调轮数 |
