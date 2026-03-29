# RQ-KMeans 算法全解析
## 1. 算法核心定义
RQ-KMeans（Residual Quantization with K-Means）是结合**残差量化**和**K-Means聚类**的向量离散化方法，核心目标是将高维连续特征向量转化为多层离散语义ID（Semantic ID），同时保证码本利用率100%（解决传统RQ-VAE的EMA码本冷落问题）。

## 2. 核心公式
### 2.1 基础符号定义
| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | 原始物品特征向量 | $[D_{in}] = [402]$ |
| $E(\cdot)$ | 编码器 | $E: \mathbb{R}^{402} \to \mathbb{R}^{128}$ |
| $z$ | 编码器输出的潜在向量 | $[D_{latent}] = [128]$ |
| $C_l$ | 第$l$层K-Means码本（聚类中心） | $[K, D_{latent}] = [32, 128]$ |
| $q_l(\cdot)$ | 第$l$层量化函数（最近邻查找） | - |
| $r_l$ | 第$l$层残差向量 | $[128]$ |
| $z_q$ | 最终量化向量 | $[128]$ |
| $D(\cdot)$ | 解码器 | $D: \mathbb{R}^{128} \to \mathbb{R}^{402}$ |
| $\hat{x}$ | 解码器重建向量 | $[402]$ |
| $\lambda$ | Commitment损失权重 | 超参数（默认0.25） |

### 2.2 核心公式体系
#### (1) 编码器公式
编码器为三层全连接网络，数学表达：
$$
z = \text{Linear}_3(\text{Dropout}(\text{ReLU}(\text{BN}(\text{Linear}_2(\text{Dropout}(\text{ReLU}(\text{BN}(\text{Linear}_1(x))))))))
$$
其中：
- $\text{Linear}_1: 402 \to 512$
- $\text{Linear}_2: 512 \to 512$
- $\text{Linear}_3: 512 \to 128$
- $\text{BN}$：批量归一化，$\text{ReLU}$：激活函数，$\text{Dropout}$：正则化（默认概率0.1）

#### (2) 残差K-Means量化公式
**Level 0 量化**：
$$
\text{code}_0 = \arg\min_{k \in \{0,1,...,31\}} \| z - C_0[k] \|_2^2
$$
$$
q_0(z) = C_0[\text{code}_0]
$$
$$
r_1 = z - q_0(z)
$$

**Level 1 量化**（输入为上一层残差）：
$$
\text{code}_1 = \arg\min_{k \in \{0,1,...,31\}} \| r_1 - C_1[k] \|_2^2
$$
$$
q_1(r_1) = C_1[\text{code}_1]
$$
$$
r_2 = r_1 - q_1(r_1)
$$

**Level 2 量化**：
$$
\text{code}_2 = \arg\min_{k \in \{0,1,...,31\}} \| r_2 - C_2[k] \|_2^2
$$
$$
q_2(r_2) = C_2[\text{code}_2]
$$

**最终量化向量**：
$$
z_q = q_0(z) + q_1(r_1) + q_2(r_2)
$$

**Semantic ID生成**：
$$
\text{SID} = [\text{code}_0, \text{code}_1, \text{code}_2]
$$

#### (3) 解码器公式
解码器为编码器的逆结构，数学表达：
$$
\hat{x} = \text{Linear}_6(\text{Dropout}(\text{ReLU}(\text{BN}(\text{Linear}_5(\text{Dropout}(\text{ReLU}(\text{BN}(\text{Linear}_4(z_q))))))))
$$
其中：
- $\text{Linear}_4: 128 \to 512$
- $\text{Linear}_5: 512 \to 512$
- $\text{Linear}_6: 512 \to 402$

#### (4) 损失函数公式
**阶段1（预训练Encoder+Decoder）**：仅重建损失
$$
\mathcal{L}_{\text{pretrain}} = \text{MSE}(\hat{x}, x) = \frac{1}{402} \sum_{i=1}^{402} (\hat{x}_i - x_i)^2
$$

**阶段3（微调解码器）**：重建损失 + Commitment损失
$$
\mathcal{L}_{\text{finetune}} = \text{MSE}(\hat{x}, x) + \lambda \cdot \text{MSE}(z, z_q.detach())
$$
- $z_q.detach()$：阻断码本梯度回传（码本由K-Means固定）
- Commitment损失：约束编码器输出靠近量化向量，提升离散化效果

## 3. 完整算法流程
### 3.1 三阶段训练流程
```mermaid
flowchart TD
    subgraph 阶段1：预训练Encoder+Decoder（无量化）
        A[物品特征x [402]] --> B[Encoder（可训练）]
        B --> C[潜在向量z [128]]
        C --> D[Decoder（可训练）]
        D --> E[重建向量x̂ [402]]
        E --> F[计算MSE(x̂,x)，反向传播更新Encoder+Decoder]
    end
    
    subgraph 阶段2：K-Means初始化码本
        G[全量物品特征矩阵 [3416,402]] --> H[Encoder（冻结）]
        H --> I[所有物品的z矩阵 [3416,128]]
        I --> J[K-Means L0：聚类为32类 → 码本C0 [32,128]]
        J --> K[计算残差r1 = z - q0(z)]
        K --> L[K-Means L1：聚类为32类 → 码本C1 [32,128]]
        L --> M[计算残差r2 = r1 - q1(r1)]
        M --> N[K-Means L2：聚类为32类 → 码本C2 [32,128]]
    end
    
    subgraph 阶段3：微调解码器（含量化）
        O[物品特征x [402]] --> P[Encoder（冻结）]
        P --> Q[潜在向量z [128]]
        Q --> R[RVQ量化（C0/C1/C2冻结）]
        R --> S[量化向量z_q [128]]
        S --> T[Decoder（可训练）]
        T --> U[重建向量x̂ [402]]
        U --> V[计算总损失，反向传播更新Decoder]
    end
    
    阶段1 --> 阶段2 --> 阶段3