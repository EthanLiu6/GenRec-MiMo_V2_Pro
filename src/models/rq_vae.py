"""
RQ-VAE 模型：残差量化变分自编码器

核心思想：
- VAE 的编码器将物品特征编码到潜在空间
- 使用多层残差量化（Residual Quantization）将连续潜在向量离散化为多个code
- 多个code的序列即为该物品的 Semantic ID
- 解码器从量化后的向量重建原始特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    向量量化层（VQ）

    将连续向量映射到码本中最近的离散向量
    使用EMA（指数移动平均）更新码本
    包含码本重启机制，防止码本崩塌
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon

        # 初始化码本
        embed = torch.randn(codebook_size, codebook_dim)
        nn.init.xavier_uniform_(embed)
        self.register_buffer("embedding", embed)

        # EMA统计量
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple:
        """
        输入: z [B, D] - 连续潜在向量
        返回:
            quantized [B, D] - 量化后的向量
            indices [B] - 码本索引
            commitment_loss - 编码器的commitment损失
        """
        # 计算z与所有码本向量的距离
        distances = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.t()
            + self.embedding.pow(2).sum(1, keepdim=True).t()
        )  # [B, K]

        # 找到最近的码本向量
        indices = distances.argmin(dim=1)  # [B]
        quantized = self.embedding[indices]  # [B, D]

        # 训练时使用EMA更新码本 + 码本重启
        if self.training:
            self._ema_update(z, indices)

        # Commitment loss
        commitment_loss = F.mse_loss(z.detach(), quantized)

        # 直通估计器（Straight-Through Estimator）
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """使用EMA更新码本，包含码本重启机制"""
        one_hot = F.one_hot(indices, self.codebook_size).float()  # [B, K]

        # 更新聚类大小
        cluster_size = one_hot.sum(0)  # [K]
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        # 更新嵌入均值
        embed_sum = z.t() @ one_hot  # [D, K]
        self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

        # 归一化更新embedding
        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.copy_(embed_normalized)

        # 码本重启：将未使用的码本向量替换为随机编码器输出
        # 未使用定义：使用次数 < 总次数的 1/(3*codebook_size)
        usage_threshold = n / (3.0 * self.codebook_size)
        unused_mask = self.cluster_size < usage_threshold  # [K]

        if unused_mask.any():
            num_unused = unused_mask.sum().item()
            # 从当前batch中随机选取向量替换未使用的码本
            rand_indices = torch.randint(0, z.size(0), (num_unused,), device=z.device)
            self.embedding[unused_mask] = z[rand_indices]
            # 重置对应统计量
            self.cluster_size[unused_mask] = 0.0
            self.embed_avg[unused_mask] = z[rand_indices]

    def get_codebook_usage(self) -> int:
        """返回使用的码本向量数量"""
        return (self.cluster_size > 0).sum().item()


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化（RVQ）
    多层级联的向量量化，每一级对上一级的残差进行量化
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, codebook_dim, decay)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        输入: z [B, D] - 潜在向量
        返回:
            quantized_total [B, D] - 所有层级量化后累加的结果
            all_indices [num_codebooks, B] - 每一层的码本索引
            total_commitment_loss
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

    def get_usage(self) -> list:
        """返回每一层的码本使用情况"""
        return [q.get_codebook_usage() for q in self.quantizers]


class RQVAE(nn.Module):
    """
    RQ-VAE 完整模型
    结构: Encoder → Residual Quantizer → Decoder
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

        # 残差向量量化器
        self.rvq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            decay=config.get("decay", 0.99),
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码: 特征 → 潜在向量"""
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> tuple:
        """量化: 潜在向量 → 量化向量 + Semantic ID"""
        return self.rvq(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """解码: 量化向量 → 重建特征"""
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> dict:
        """完整前向传播"""
        z = self.encode(x)
        z_q, semantic_ids, commitment_loss = self.rvq(z)
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

    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """推理时使用：只获取Semantic ID"""
        with torch.no_grad():
            z = self.encode(x)
            _, semantic_ids, _ = self.rvq(z)
        return semantic_ids
