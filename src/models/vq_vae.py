"""
VQ-VAE 模型：标准向量量化变分自编码器（单层量化）

核心思想：
- 编码器将物品特征编码到潜在空间
- 使用单层向量量化（非残差级联）将连续向量离散化
- 一个codebook_size大小的码本，每个物品分配一个code
- Semantic ID = [single_code]（单个code）

与RQ-VAE的区别：
- RQ-VAE使用多层残差级联，VQ-VAE只用单层
- VQ-VAE的Semantic ID更短，语义空间更小
- 适合对比实验，展示残差量化的优势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EMAVectorQuantizer(nn.Module):
    """
    EMA更新的向量量化层

    使用指数移动平均更新码本，包含码本重启机制
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
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple:
        """
        输入: z [B, D]
        返回: quantized [B, D], indices [B], commitment_loss
        """
        distances = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.t()
            + self.embedding.pow(2).sum(1, keepdim=True).t()
        )

        indices = distances.argmin(dim=1)  # [B]
        quantized = self.embedding[indices]  # [B, D]

        if self.training:
            self._ema_update(z, indices)

        commitment_loss = F.mse_loss(z.detach(), quantized)
        quantized = z + (quantized - z).detach()

        return quantized, indices, commitment_loss

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """EMA更新码本 + 码本重启"""
        one_hot = F.one_hot(indices, self.codebook_size).float()

        cluster_size = one_hot.sum(0)
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        embed_sum = z.t() @ one_hot
        self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.copy_(embed_normalized)

        # 码本重启
        usage_threshold = n / (3.0 * self.codebook_size)
        unused_mask = self.cluster_size < usage_threshold

        if unused_mask.any():
            num_unused = unused_mask.sum().item()
            rand_indices = torch.randint(0, z.size(0), (num_unused,), device=z.device)
            self.embedding[unused_mask] = z[rand_indices]
            self.cluster_size[unused_mask] = 0.0
            self.embed_avg[unused_mask] = z[rand_indices]

    def get_codebook_usage(self) -> int:
        return (self.cluster_size > 0).sum().item()


class VQVAE(nn.Module):
    """
    VQ-VAE 完整模型（单层量化）

    结构: Encoder → Single VQ → Decoder
    Semantic ID只有一个code
    """

    def __init__(self, config: dict):
        super().__init__()
        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        latent_dim = config["latent_dim"]
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

        # 单层向量量化器
        self.vq = EMAVectorQuantizer(
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
        self.num_codebooks = 1  # VQ-VAE固定为单层

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantize(self, z: torch.Tensor) -> tuple:
        return self.vq(z)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encode(x)
        z_q, indices, commitment_loss = self.vq(z)
        x_recon = self.decode(z_q)

        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + self.commitment_weight * commitment_loss

        # 输出格式与RQ-VAE保持一致 [1, B]（单层）
        semantic_ids = indices.unsqueeze(0)

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
        """推理时使用：返回 [1, B] 格式"""
        with torch.no_grad():
            z = self.encode(x)
            _, indices, _ = self.vq(z)
        return indices.unsqueeze(0)  # [1, B]
