"""
RQ-KMeans 模型：基于K-Means的残差量化

核心思想：
- 与RQ-VAE类似，但量化部分使用K-Means聚类而非EMA码本更新
- 编码器将物品特征编码到潜在空间
- 对潜在向量逐层进行K-Means聚类，每层对上一层的残差做聚类
- 最终Semantic ID = [cluster_id_level0, cluster_id_level1, ...]

优势：K-Means直接优化聚类目标，码本利用率天然较高
劣势：需要预先运行K-Means（非端到端），推理时编码器需重新训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class KMeansQuantizer(nn.Module):
    """
    K-Means向量量化层

    使用sklearn的K-Means预训练码本，之后冻结使用
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # 码本向量（通过K-Means初始化后冻结）
        self.register_buffer("embedding", torch.zeros(codebook_size, codebook_dim))

    def forward(self, z: torch.Tensor) -> tuple:
        """
        输入: z [B, D] - 连续潜在向量
        返回:
            quantized [B, D] - 量化后的向量
            indices [B] - 码本索引
        """
        # 计算距离
        distances = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.t()
            + self.embedding.pow(2).sum(1, keepdim=True).t()
        )  # [B, K]

        indices = distances.argmin(dim=1)  # [B]
        quantized = self.embedding[indices]  # [B, D]

        # 直通估计器
        quantized = z + (quantized - z).detach()

        # Commitment loss: 鼓励编码器输出靠近聚类中心
        commitment_loss = F.mse_loss(z, quantized.detach())

        return quantized, indices, commitment_loss

    @torch.no_grad()
    def init_from_data(self, data: np.ndarray):
        """
        用K-Means从数据中初始化码本
        输入: data [N, D] numpy数组
        """
        kmeans = KMeans(n_clusters=self.codebook_size, n_init=10, random_state=42)
        kmeans.fit(data)
        self.embedding.copy_(torch.from_numpy(kmeans.cluster_centers_).float())

    def get_codebook_usage(self, all_indices: torch.Tensor) -> int:
        """返回使用的码本向量数量"""
        return len(torch.unique(all_indices))


class ResidualKMeansQuantizer(nn.Module):
    """
    残差K-Means量化器
    多层级联，每层对上一层的残差做K-Means聚类
    """

    def __init__(self, num_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.quantizers = nn.ModuleList(
            [KMeansQuantizer(codebook_size, codebook_dim) for _ in range(num_codebooks)]
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        输入: z [B, D]
        返回: quantized_total [B, D], all_indices [num_codebooks, B], total_commitment_loss
        """
        quantized_total = torch.zeros_like(z)
        all_indices = []
        total_commitment_loss = 0.0
        residual = z

        for quantizer in self.quantizers:
            quantized, indices, commitment_loss = quantizer(residual)
            quantized_total = quantized_total + quantized
            residual = residual - quantized
            all_indices.append(indices)
            total_commitment_loss += commitment_loss

        all_indices = torch.stack(all_indices, dim=0)  # [num_codebooks, B]
        return quantized_total, all_indices, total_commitment_loss

    @torch.no_grad()
    def init_from_data(self, data: np.ndarray):
        """
        逐层用K-Means初始化所有码本
        输入: data [N, D] - 编码器输出
        """
        residual = data.copy()

        for i, quantizer in enumerate(self.quantizers):
            quantizer.init_from_data(residual)
            # 计算量化结果，更新残差
            embedding_np = quantizer.embedding.cpu().numpy()
            dists = np.sum(
                (residual[:, None, :] - embedding_np[None, :, :]) ** 2,
                axis=-1,
            )
            indices = np.argmin(dists, axis=1)
            quantized = embedding_np[indices]
            residual = residual - quantized
            print(f"    Level {i}: K-Means聚类完成, 码本大小={quantizer.codebook_size}")


class RQKMeans(nn.Module):
    """
    RQ-KMeans 完整模型

    结构: Encoder → K-Means Residual Quantizer → Decoder
    训练流程：
      1. 先训练编码器+解码器（不使用量化）
      2. 冻结编码器，用其输出初始化K-Means码本
      3. 微调解码器（使用量化后的向量）
    """

    def __init__(self, config: dict):
        super().__init__()
        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        latent_dim = config["latent_dim"]
        num_codebooks = config["num_codebooks"]
        codebook_size = config["codebook_size"]
        codebook_dim = config["codebook_dim"]

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 残差K-Means量化器
        self.rkq = ResidualKMeansQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

        self.commitment_weight = config.get("commitment_weight", 1.0)
        self._codebooks_initialized = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码: 特征 → 潜在向量"""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> tuple:
        """量化: 潜在向量 → 量化向量 + Semantic ID"""
        return self.rkq(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """解码: 量化向量 → 重建特征"""
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> dict:
        """完整前向传播"""
        z = self.encode(x)

        if self._codebooks_initialized:
            z_q, semantic_ids, commitment_loss = self.rkq(z)
        else:
            z_q = z
            semantic_ids = torch.zeros(
                self.rkq.num_codebooks, z.size(0), dtype=torch.long, device=z.device
            )
            commitment_loss = torch.tensor(0.0, device=z.device)

        x_recon = self.decode(z_q)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + self.commitment_weight * commitment_loss

        return {
            "recon": x_recon,
            "semantic_ids": semantic_ids,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "total_loss": total_loss,
            "z": z,
            "z_q": z_q,
        }

    def init_codebooks(self, features_tensor: torch.Tensor):
        """用编码器输出初始化K-Means码本"""
        print("[RQ-KMeans] 正在初始化K-Means码本...")
        self.encoder.eval()
        with torch.no_grad():
            z = self.encode(features_tensor).cpu().numpy()
        self.rkq.init_from_data(z)
        self._codebooks_initialized = True
        print("[RQ-KMeans] K-Means码本初始化完成!")

    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """推理时使用：只获取Semantic ID"""
        with torch.no_grad():
            z = self.encode(x)
            _, semantic_ids, _ = self.rkq(z)
        return semantic_ids

    def cosine_anti_collapse_loss(
        self, z: torch.Tensor, semantic_ids: torch.Tensor, level_weights: list = None
    ) -> torch.Tensor:
        """
        余弦反塌缩正则损失

        目标：同一SID内的物品在潜在空间中保持差异化，避免塌缩

        参数:
            z: [B, D] 编码器输出的潜在向量
            semantic_ids: [num_codebooks, B] 每个物品的Semantic ID
            level_weights: 每层的权重，默认 [0.3, 0.5, 1.0]
        """
        num_levels = semantic_ids.size(0)
        if level_weights is None:
            # 默认权重：粗粒度权重小，细粒度权重大
            level_weights = [0.3, 0.5, 1.0][:num_levels]

        # L2归一化，便于计算余弦相似度
        z_norm = F.normalize(z, dim=1)  # [B, D]

        total_loss = torch.tensor(0.0, device=z.device)
        total_weight = 0.0

        for level in range(num_levels):
            codes = semantic_ids[level]  # [B]
            unique_codes = torch.unique(codes)

            level_loss = torch.tensor(0.0, device=z.device)
            num_groups = 0

            for code in unique_codes:
                mask = codes == code
                group_size = mask.sum().item()

                # 组内至少2个样本才有意义
                if group_size < 2:
                    continue

                group_z = z_norm[mask]  # [G, D]

                # 计算组内两两余弦相似度（上三角，不含对角线）
                sim_matrix = group_z @ group_z.t()  # [G, G]
                # 取上三角（不含对角线）
                triu_indices = torch.triu_indices(group_size, group_size, offset=1)
                pair_sims = sim_matrix[triu_indices[0], triu_indices[1]]

                # 组内平均余弦相似度（越小越好，说明越多样化）
                # 按组大小自适应：小组贡献小，大组贡献大
                effective_weight = min(group_size, 10) / 10.0
                level_loss = level_loss + pair_sims.mean() * effective_weight
                num_groups += 1

            if num_groups > 0:
                level_loss = level_loss / num_groups
                total_loss = total_loss + level_weights[level] * level_loss
                total_weight += level_weights[level]

        if total_weight > 0:
            total_loss = total_loss / total_weight

        return total_loss
