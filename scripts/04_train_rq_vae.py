"""
Step 4: 训练 RQ-VAE

- 加载物品特征矩阵
- 训练 RQ-VAE 模型
- 保存最优模型

运行方式:
    .venv/bin/python scripts/04_train_rq_vae.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path
from src.models.rq_vae import RQVAE


def train_rq_vae(cfg: dict):
    """训练RQ-VAE模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    # 1. 加载物品特征
    features_path = get_abs_path(
        os.path.join(cfg["data"]["processed_dir"], "item_features.npy")
    )
    features = np.load(features_path)  # [num_items, feature_dim]
    print(f"[数据] 物品特征矩阵: {features.shape}")

    # 转为Tensor
    features_tensor = torch.from_numpy(features).float()
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["rq_vae"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 2. 初始化模型
    vae_config = {
        "input_dim": cfg["rq_vae"]["input_dim"],
        "hidden_dim": cfg["rq_vae"]["hidden_dim"],
        "latent_dim": cfg["rq_vae"]["latent_dim"],
        "num_codebooks": cfg["rq_vae"]["num_codebooks"],
        "codebook_size": cfg["rq_vae"]["codebook_size"],
        "codebook_dim": cfg["rq_vae"]["codebook_dim"],
        "commitment_weight": cfg["rq_vae"]["commitment_weight"],
        "decay": cfg["rq_vae"]["decay"],
    }
    model = RQVAE(vae_config).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

    # 3. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["rq_vae"]["lr"])

    # 4. 训练循环
    epochs = cfg["rq_vae"]["epochs"]
    best_loss = float("inf")
    avg_loss = float("inf")  # 初始化，防止未绑定
    save_path = get_abs_path(cfg["rq_vae"]["model_save_path"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n[训练] 开始训练 RQ-VAE, 共 {epochs} 个epoch...")
    print(
        f"[训练] 批次大小: {cfg['rq_vae']['batch_size']}, 学习率: {cfg['rq_vae']['lr']}"
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_commit = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)

            # 前向传播
            output = model(batch_x)
            loss = output["total_loss"]

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            epoch_recon += output["recon_loss"].item()
            epoch_commit += output["commitment_loss"].item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{output['recon_loss'].item():.4f}",
                    "commit": f"{output['commitment_loss'].item():.4f}",
                }
            )

        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_commit = epoch_commit / num_batches

        # 计算码本利用率（使用了多少个不同的code）
        model.eval()
        with torch.no_grad():
            sample_output = model(features_tensor.to(device))
            all_codes = sample_output["semantic_ids"]  # [num_codebooks, num_items]
            utilization = []
            for level in range(all_codes.shape[0]):
                unique_codes = len(torch.unique(all_codes[level]))
                utilization.append(unique_codes)
            utilization_str = ", ".join(
                [
                    f"L{i}:{u}/{cfg['rq_vae']['codebook_size']}"
                    for i, u in enumerate(utilization)
                ]
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}) | "
            f"码本利用率: [{utilization_str}]"
        )

        # 保存最优模型（综合考虑loss和码本利用率）
        min_util = min(utilization)
        codebook_size = cfg["rq_vae"]["codebook_size"]
        # 码本利用率超过50%才考虑保存（避免保存崩塌的模型）
        if avg_loss < best_loss and min_util > codebook_size * 0.5:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": vae_config,
                    "best_loss": best_loss,
                },
                save_path,
            )
            print(f"  ✅ 保存最优模型 (loss={best_loss:.4f}) 至: {save_path}")

        # 总是保存最后一个模型（确保有可用模型）
        last_save_path = save_path.replace("_best.pt", "_last.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": vae_config,
                "best_loss": avg_loss,
            },
            last_save_path,
        )

    # 总是保存最后一个模型（确保有可用模型）
    last_save_path = save_path.replace("_best.pt", "_last.pt")
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": vae_config,
            "best_loss": avg_loss if "avg_loss" in dir() else float("inf"),
        },
        last_save_path,
    )

    # 如果best model从未保存（全都是崩塌的），使用last model
    if not os.path.exists(save_path):
        import shutil

        shutil.copy(last_save_path, save_path)
        print(f"  ⚠️ 使用最终模型作为最优模型")

    print(f"\n[完成] RQ-VAE训练完成! 最优loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    cfg = load_config()
    train_rq_vae(cfg)
    print(f"\n✅ 请运行下一步: .venv/bin/python scripts/05_generate_semantic_ids.py")
