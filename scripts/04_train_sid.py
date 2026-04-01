"""
Step 4: 训练 SID 量化模型

支持三种量化方式（通过 config.yaml 的 sid_method 配置）：
  - rq_vae:  RQ-VAE（残差量化 + EMA码本 + 码本重启）
  - rq_kmeans: RQ-KMeans（残差量化 + K-Means聚类）
  - vq_vae:  VQ-VAE（单层量化 + EMA码本）

运行方式:
    conda run -n py10 python scripts/04_train_sid.py
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/04_train_sid.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


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
    best_util = 0
    avg_loss = float("inf")
    last_save_path = save_path.replace("_best.pt", "_last.pt")

    logger.info("=" * 60)
    logger.info("RQ-VAE 训练配置:")
    logger.info(f"  输入维度: {sid_cfg['input_dim']}")
    logger.info(f"  隐藏维度: {sid_cfg['hidden_dim']}")
    logger.info(f"  潜在维度: {sid_cfg['latent_dim']}")
    logger.info(f"  码本数量: {num_codebooks}")
    logger.info(f"  码本大小: {sid_cfg['codebook_size']}")
    logger.info(f"  训练轮数: {epochs}")
    logger.info(f"  批次大小: {sid_cfg['batch_size']}")
    logger.info(f"  学习率: {sid_cfg['lr']}")
    logger.info(f"  Commitment权重: {sid_cfg['commitment_weight']}")
    logger.info(f"  EMA衰减: {sid_cfg['decay']}")
    logger.info("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_commit = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"RQ-VAE Epoch {epoch + 1}/{epochs}")
        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = output["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_commit = epoch_commit / num_batches

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
        total_util = sum(utilization)

        # 计算每个code的使用频率分布
        code_dist_str = []
        for level in range(num_codebooks):
            codes, counts = torch.unique(all_codes[level], return_counts=True)
            top3_idx = torch.argsort(counts, descending=True)[:3]
            top3_codes = codes[top3_idx].cpu().numpy()
            top3_counts = counts[top3_idx].cpu().numpy()
            code_dist_str.append(f"L{level}: top3={list(zip(top3_codes, top3_counts))}")

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f}) | "
            f"码本利用率: [{util_str}] (min={min_util})"
        )

        # 每10轮打印详细分布
        if (epoch + 1) % 10 == 0 or epoch == 0:
            for dist_info in code_dist_str:
                logger.info(f"  {dist_info}")

        # 保存最优模型（综合考虑loss和码本利用率）
        codebook_size = sid_cfg["codebook_size"]
        # 降低保存阈值，允许更多模型被保存
        if avg_loss < best_loss:
            if min_util >= codebook_size * 0.3:  # 降低到30%
                best_loss = avg_loss
                best_util = min_util
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "config": vae_config,
                        "num_codebooks": num_codebooks,
                        "best_loss": best_loss,
                        "utilization": utilization,
                    },
                    save_path,
                )
                logger.info(
                    f"  ✅ 保存最优模型 (loss={best_loss:.4f}, min_util={min_util})"
                )

    # 保存last model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "config": vae_config,
            "num_codebooks": num_codebooks,
            "best_loss": avg_loss,
        },
        last_save_path,
    )

    # 如果best model从未保存（全都是塌塌的），使用last model
    if not os.path.exists(save_path):
        import shutil

        shutil.copy(last_save_path, save_path)
        logger.warning("  ⚠️ 使用最终模型作为最优模型（之前的模型码本利用率都低于阈值）")

    logger.info("=" * 60)
    logger.info(f"RQ-VAE 训练完成!")
    logger.info(f"  最优loss: {best_loss:.4f}")
    logger.info(f"  最优min_util: {best_util}")
    logger.info(f"  模型保存: {save_path}")
    logger.info("=" * 60)

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

    logger.info("=" * 60)
    logger.info("RQ-KMeans 训练配置:")
    logger.info(f"  输入维度: {sid_cfg['input_dim']}")
    logger.info(f"  隐藏维度: {sid_cfg['hidden_dim']}")
    logger.info(f"  潜在维度: {sid_cfg['latent_dim']}")
    logger.info(f"  码本数量: {rk_cfg['num_codebooks']}")
    logger.info(f"  码本大小: {sid_cfg['codebook_size']}")
    logger.info(f"  编码器预训练轮数: {rk_cfg['encoder_pretrain_epochs']}")
    logger.info(f"  解码器微调轮数: {rk_cfg['decoder_finetune_epochs']}")
    logger.info(f"  余弦反塌缩权重: {rk_cfg['cosine_anti_collapse_weight']}")
    logger.info("=" * 60)

    # === 阶段1: 预训练编码器+解码器（不使用量化）===
    logger.info(
        f"\n阶段1: 预训练编码器+解码器 ({rk_cfg['encoder_pretrain_epochs']} epochs)"
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Pretrain Epoch {epoch + 1} | Loss: {epoch_loss / num_batches:.4f}"
            )

    # === 阶段2: K-Means初始化码本 ===
    logger.info(f"\n阶段2: K-Means初始化码本")
    model.init_codebooks(features_tensor.to(device))

    # 检查初始化后的码本利用率
    with torch.no_grad():
        sample = model(features_tensor.to(device))
        all_codes = sample["semantic_ids"]
        init_util = [len(torch.unique(all_codes[l])) for l in range(all_codes.shape[0])]
        logger.info(f"  初始化后码本利用率: {init_util}")

    # === 阶段3: 微调解码器（含余弦反塌缩正则）===
    cos_weight = rk_cfg.get("cosine_anti_collapse_weight", 0.0)
    cos_level_weights = rk_cfg.get("cosine_level_weights", [0.3, 0.5, 1.0])
    logger.info(f"\n阶段3: 微调解码器 ({rk_cfg['decoder_finetune_epochs']} epochs)")
    logger.info(f"  余弦反塌缩权重: {cos_weight}, 层级权重: {cos_level_weights}")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
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

    logger.info("=" * 60)
    logger.info(f"RQ-KMeans 训练完成!")
    logger.info(f"  最优loss: {best_loss:.4f}")
    logger.info(f"  最终码本利用率: {utilization}")
    logger.info(f"  模型保存: {save_path}")
    logger.info("=" * 60)

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

    logger.info("=" * 60)
    logger.info("VQ-VAE 训练配置:")
    logger.info(f"  输入维度: {sid_cfg['input_dim']}")
    logger.info(f"  码本大小: {codebook_size}")
    logger.info(f"  训练轮数: {epochs}")
    logger.info("=" * 60)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches

        model.eval()
        with torch.no_grad():
            sample = model(features_tensor.to(device))
            usage = model.vq.get_codebook_usage()

        logger.info(
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
            logger.info(f"  ✅ 保存最优模型 (loss={best_loss:.4f})")

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

    logger.info("=" * 60)
    logger.info(f"VQ-VAE 训练完成!")
    logger.info(f"  最优loss: {best_loss:.4f}")
    logger.info(f"  模型保存: {save_path}")
    logger.info("=" * 60)

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

    logger.info("=" * 60)
    logger.info("SID 模型训练")
    logger.info("=" * 60)
    logger.info(f"设备: {device}")
    logger.info(f"量化方式: {method}")

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
    logger.info(f"物品特征矩阵: {features.shape}")
    logger.info(f"  值范围: [{features.min():.4f}, {features.max():.4f}]")
    logger.info(f"  均值: {features.mean():.4f}, 标准差: {features.std():.4f}")

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

    logger.info(f"\n方法标记已保存: {method_file}")
    logger.info(f"请运行下一步: conda run -n py10 python scripts/05_generate_sid.py")


if __name__ == "__main__":
    main()
