"""
Step 4: 训练 SID 量化模型

支持三种量化方式（通过 config.yaml 的 sid_method 配置）：
  - rq_vae:  RQ-VAE（残差量化 + EMA码本 + 码本重启）
  - rq_kmeans: RQ-KMeans（残差量化 + K-Means聚类）
  - vq_vae:  VQ-VAE（单层量化 + EMA码本）

运行方式:
    .venv/bin/python scripts/04_train_sid.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


# ============================================================
# 三种量化方式的训练函数
# ============================================================


def train_rq_vae(
    cfg: dict, device: torch.device, features_tensor: torch.Tensor, save_path: str
):
    """训练RQ-VAE"""
    from src.models.rq_vae import RQVAE

    sid_cfg = cfg["sid"]
    rvq_cfg = cfg["rq_vae"]

    vae_config = {
        "input_dim": sid_cfg["input_dim"],
        "hidden_dim": sid_cfg["hidden_dim"],
        "latent_dim": sid_cfg["latent_dim"],
        "num_codebooks": rvq_cfg["num_codebooks"],
        "codebook_size": sid_cfg["codebook_size"],
        "codebook_dim": sid_cfg["codebook_dim"],
        "commitment_weight": sid_cfg["commitment_weight"],
        "decay": sid_cfg["decay"],
    }

    model = RQVAE(vae_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sid_cfg["lr"])

    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(
        dataset, batch_size=sid_cfg["batch_size"], shuffle=True, num_workers=0
    )

    num_codebooks = rvq_cfg["num_codebooks"]
    epochs = sid_cfg["epochs"]
    best_loss = float("inf")
    avg_loss = float("inf")
    last_save_path = save_path.replace("_best.pt", "_last.pt")

    print(f"[RQ-VAE] 开始训练, 共 {epochs} 个epoch...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"RQ-VAE Epoch {epoch + 1}/{epochs}")
        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = output["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches

        # 检查码本利用率
        model.eval()
        with torch.no_grad():
            sample = model(features_tensor.to(device))
            all_codes = sample["semantic_ids"]
            utilization = [
                len(torch.unique(all_codes[l])) for l in range(all_codes.shape[0])
            ]
            util_str = ", ".join(
                [
                    f"L{i}:{u}/{sid_cfg['codebook_size']}"
                    for i, u in enumerate(utilization)
                ]
            )

        min_util = min(utilization)
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | 码本利用率: [{util_str}]"
        )

        codebook_size = sid_cfg["codebook_size"]
        if avg_loss < best_loss and min_util > codebook_size * 0.5:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": vae_config,
                    "num_codebooks": num_codebooks,
                },
                save_path,
            )
            print(f"  ✅ 保存最优模型 (loss={best_loss:.4f})")

    # 保存last model作为备选
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": vae_config,
            "num_codebooks": num_codebooks,
        },
        last_save_path,
    )
    if not os.path.exists(save_path):
        import shutil

        shutil.copy(last_save_path, save_path)

    return model


def train_rq_kmeans(
    cfg: dict, device: torch.device, features_tensor: torch.Tensor, save_path: str
):
    """训练RQ-KMeans（先预训练编码器，再K-Means聚类，最后微调解码器）"""
    from src.models.rq_kmeans import RQKMeans

    sid_cfg = cfg["sid"]
    rk_cfg = cfg["rq_kmeans"]

    model_config = {
        "input_dim": sid_cfg["input_dim"],
        "hidden_dim": sid_cfg["hidden_dim"],
        "latent_dim": sid_cfg["latent_dim"],
        "num_codebooks": rk_cfg["num_codebooks"],
        "codebook_size": sid_cfg["codebook_size"],
        "codebook_dim": sid_cfg["codebook_dim"],
        "commitment_weight": sid_cfg["commitment_weight"],
    }

    model = RQKMeans(model_config).to(device)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(
        dataset, batch_size=sid_cfg["batch_size"], shuffle=True, num_workers=0
    )

    # === 阶段1: 预训练编码器+解码器（不使用量化）===
    print(
        f"\n[RQ-KMeans] 阶段1: 预训练编码器+解码器 ({rk_cfg['encoder_pretrain_epochs']} epochs)"
    )
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=sid_cfg["lr"],
    )

    for epoch in range(rk_cfg["encoder_pretrain_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = output["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"  Pretrain Epoch {epoch + 1} | Loss: {epoch_loss / num_batches:.4f}"
            )

    # === 阶段2: K-Means初始化码本 ===
    print(f"\n[RQ-KMeans] 阶段2: K-Means初始化码本")
    model.init_codebooks(features_tensor)

    # === 阶段3: 微调解码器（含余弦反塌缩正则）===
    cos_weight = rk_cfg.get("cosine_anti_collapse_weight", 0.0)
    cos_level_weights = rk_cfg.get("cosine_level_weights", [0.3, 0.5, 1.0])
    print(
        f"\n[RQ-KMeans] 阶段3: 微调解码器 ({rk_cfg['decoder_finetune_epochs']} epochs)"
    )
    print(f"  余弦反塌缩权重: {cos_weight}, 层级权重: {cos_level_weights}")
    optimizer = torch.optim.Adam(model.parameters(), lr=sid_cfg["lr"] * 0.1)
    best_loss = float("inf")
    avg_loss = float("inf")
    last_save_path = save_path.replace("_best.pt", "_last.pt")

    for epoch in range(rk_cfg["decoder_finetune_epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_cos = 0.0
        num_batches = 0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = output["total_loss"]

            # 余弦反塌缩正则
            cos_loss_val = 0.0
            if cos_weight > 0:
                cos_loss = model.cosine_anti_collapse_loss(
                    output["z"], output["semantic_ids"], level_weights=cos_level_weights
                )
                loss = loss + cos_weight * cos_loss
                cos_loss_val = cos_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += output["total_loss"].item()
            epoch_cos += cos_loss_val
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_cos = epoch_cos / num_batches

        model.eval()
        with torch.no_grad():
            sample = model(features_tensor.to(device))
            all_codes = sample["semantic_ids"]
            utilization = [
                len(torch.unique(all_codes[l])) for l in range(all_codes.shape[0])
            ]
            util_str = ", ".join(
                [
                    f"L{i}:{u}/{sid_cfg['codebook_size']}"
                    for i, u in enumerate(utilization)
                ]
            )

        cos_str = f", Cos: {avg_cos:.4f}" if cos_weight > 0 else ""
        print(
            f"  Finetune Epoch {epoch + 1} | Loss: {avg_loss:.4f}{cos_str} | 码本利用率: [{util_str}]"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "num_codebooks": rk_cfg["num_codebooks"],
                },
                save_path,
            )

    torch.save(
        {
            "epoch": rk_cfg["decoder_finetune_epochs"],
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "num_codebooks": rk_cfg["num_codebooks"],
        },
        last_save_path,
    )
    if not os.path.exists(save_path):
        import shutil

        shutil.copy(last_save_path, save_path)

    return model


def train_vq_vae(
    cfg: dict, device: torch.device, features_tensor: torch.Tensor, save_path: str
):
    """训练VQ-VAE（单层量化）"""
    from src.models.vq_vae import VQVAE

    sid_cfg = cfg["sid"]
    vq_cfg = cfg.get("vq_vae", {})

    # VQ-VAE可以用更大的码本（单层）
    codebook_size = vq_cfg.get("codebook_size_override", sid_cfg["codebook_size"])

    model_config = {
        "input_dim": sid_cfg["input_dim"],
        "hidden_dim": sid_cfg["hidden_dim"],
        "latent_dim": sid_cfg["latent_dim"],
        "codebook_size": codebook_size,
        "codebook_dim": sid_cfg["codebook_dim"],
        "commitment_weight": sid_cfg["commitment_weight"],
        "decay": sid_cfg["decay"],
    }

    model = VQVAE(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sid_cfg["lr"])

    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(
        dataset, batch_size=sid_cfg["batch_size"], shuffle=True, num_workers=0
    )

    epochs = sid_cfg["epochs"]
    best_loss = float("inf")
    avg_loss = float("inf")
    last_save_path = save_path.replace("_best.pt", "_last.pt")

    print(f"[VQ-VAE] 开始训练 (codebook_size={codebook_size}), 共 {epochs} 个epoch...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"VQ-VAE Epoch {epoch + 1}/{epochs}")
        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = output["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches

        model.eval()
        with torch.no_grad():
            sample = model(features_tensor.to(device))
            usage = model.vq.get_codebook_usage()

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | 码本利用率: {usage}/{codebook_size}"
        )

        if avg_loss < best_loss and usage > codebook_size * 0.5:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "num_codebooks": 1,
                },
                save_path,
            )
            print(f"  ✅ 保存最优模型 (loss={best_loss:.4f})")

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "num_codebooks": 1,
        },
        last_save_path,
    )
    if not os.path.exists(save_path):
        import shutil

        shutil.copy(last_save_path, save_path)

    return model


# ============================================================
# 主入口
# ============================================================

METHOD_TRAINERS = {
    "rq_vae": train_rq_vae,
    "rq_kmeans": train_rq_kmeans,
    "vq_vae": train_vq_vae,
}

METHOD_SAVE_NAMES = {
    "rq_vae": "rq_vae_best.pt",
    "rq_kmeans": "rq_kmeans_best.pt",
    "vq_vae": "vq_vae_best.pt",
}


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = cfg["sid_method"]

    print(f"[设备] 使用: {device}")
    print(f"[方法] 量化方式: {method}")

    if method not in METHOD_TRAINERS:
        raise ValueError(
            f"不支持的sid_method: {method}，可选: {list(METHOD_TRAINERS.keys())}"
        )

    # 加载特征数据
    features_path = get_abs_path(
        os.path.join(cfg["data"]["processed_dir"], "item_features.npy")
    )
    features = np.load(features_path)
    features_tensor = torch.from_numpy(features).float()
    print(f"[数据] 物品特征矩阵: {features.shape}")

    # 训练
    save_name = METHOD_SAVE_NAMES[method]
    save_path = get_abs_path(os.path.join("outputs", save_name))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainer = METHOD_TRAINERS[method]
    model = trainer(cfg, device, features_tensor, save_path)

    # 保存方法名到文件，供后续步骤读取
    method_file = get_abs_path("outputs/current_sid_method.txt")
    with open(method_file, "w") as f:
        f.write(method)

    print(f"\n{'=' * 50}")
    print(f"✅ {method} 训练完成!")
    print(f"  模型保存: {save_path}")
    print(f"  方法标记: {method_file}")
    print(f"\n请运行下一步: .venv/bin/python scripts/05_generate_sid.py")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
