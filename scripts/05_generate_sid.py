"""
Step 5: 生成 Semantic ID

支持三种量化方式，自动根据 outputs/current_sid_method.txt 识别

运行方式:
    conda run -n py10 python scripts/05_generate_sid.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path

# 配置日志
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/05_generate_sid.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


METHOD_MODELS = {
    "rq_vae": "src.models.rq_vae",
    "rq_kmeans": "src.models.rq_kmeans",
    "vq_vae": "src.models.vq_vae",
}

METHOD_SAVE_NAMES = {
    "rq_vae": "rq_vae_best.pt",
    "rq_kmeans": "rq_kmeans_best.pt",
    "vq_vae": "vq_vae_best.pt",
}


def load_model(method: str, cfg: dict, device: torch.device):
    """根据方法加载对应的模型"""
    save_name = METHOD_SAVE_NAMES[method]
    model_path = get_abs_path(os.path.join("outputs", save_name))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if method == "rq_vae":
        from src.models.rq_vae import RQVAE

        model = RQVAE(checkpoint["config"]).to(device)
    elif method == "rq_kmeans":
        from src.models.rq_kmeans import RQKMeans

        model = RQKMeans(checkpoint["config"]).to(device)
        model._codebooks_initialized = True
    elif method == "vq_vae":
        from src.models.vq_vae import VQVAE

        model = VQVAE(checkpoint["config"]).to(device)
    else:
        raise ValueError(f"不支持的方法: {method}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    num_codebooks = checkpoint["num_codebooks"]

    logger.info(
        f"加载 {method} 模型 (epoch={checkpoint['epoch']}, num_codebooks={num_codebooks})"
    )
    if "best_loss" in checkpoint:
        logger.info(f"  最优loss: {checkpoint['best_loss']:.4f}")
    if "utilization" in checkpoint:
        logger.info(f"  训练时码本利用率: {checkpoint['utilization']}")

    return model, num_codebooks


def analyze_sid_distribution(semantic_ids: np.ndarray, codebook_size: int):
    """详细分析SID分布"""
    num_items, num_codebooks = semantic_ids.shape

    logger.info("=" * 60)
    logger.info("Semantic ID 分布分析")
    logger.info("=" * 60)

    # 1. 每层的码本利用率
    logger.info("\n1. 码本利用率:")
    total_unique = 1
    for level in range(num_codebooks):
        unique_codes = len(np.unique(semantic_ids[:, level]))
        total_unique *= unique_codes
        logger.info(
            f"  Level {level}: {unique_codes}/{codebook_size} codes used ({unique_codes / codebook_size * 100:.1f}%)"
        )

    # 2. 唯一SID数
    semantic_tuples = [tuple(row) for row in semantic_ids]
    unique_sids = len(set(semantic_tuples))
    logger.info(
        f"\n2. 唯一Semantic ID数: {unique_sids}/{num_items} ({unique_sids / num_items * 100:.1f}%)"
    )
    logger.info(f"  理论最大SID数: {total_unique}")

    # 3. 每层code分布（Top 5）
    logger.info(f"\n3. 每层Code分布 (Top 5):")
    for level in range(num_codebooks):
        codes, counts = np.unique(semantic_ids[:, level], return_counts=True)
        sorted_idx = np.argsort(-counts)
        logger.info(f"  Level {level}:")
        for i in range(min(5, len(codes))):
            idx = sorted_idx[i]
            logger.info(
                f"    Code {codes[idx]}: {counts[idx]} items ({counts[idx] / num_items * 100:.1f}%)"
            )

    # 4. 最常见的SID组合（Top 10）
    logger.info(f"\n4. 最常见的SID组合 (Top 10):")
    counter = Counter(semantic_tuples)
    for i, (sid, count) in enumerate(counter.most_common(10)):
        logger.info(
            f"  #{i + 1}: {sid} -> {count} items ({count / num_items * 100:.1f}%)"
        )

    # 5. 雷同度分析
    logger.info(f"\n5. 雷同度分析:")
    max_count = counter.most_common(1)[0][1]
    logger.info(
        f"  最常见SID的物品数: {max_count} ({max_count / num_items * 100:.1f}%)"
    )
    if max_count / num_items > 0.1:
        logger.warning(f"  ⚠️ 警告: 超过10%的物品具有相同的SID，可能存在码本塌缩!")

    # 6. 熵分析（衡量分布均匀度）
    logger.info(f"\n6. 熵分析 (衡量分布均匀度):")
    for level in range(num_codebooks):
        codes, counts = np.unique(semantic_ids[:, level], return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(codebook_size)
        logger.info(
            f"  Level {level}: 熵={entropy:.3f}/{max_entropy:.3f} (均匀度={entropy / max_entropy * 100:.1f}%)"
        )

    return unique_sids, counter


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    logger.info("=" * 60)
    logger.info("Semantic ID 生成")
    logger.info("=" * 60)
    logger.info(f"设备: {device}")

    # 读取当前使用的方法
    method_file = get_abs_path("outputs/current_sid_method.txt")
    if os.path.exists(method_file):
        with open(method_file, "r") as f:
            method = f.read().strip()
    else:
        method = cfg["sid_method"]

    logger.info(f"量化方式: {method}")

    # 加载数据
    features = np.load(os.path.join(processed_dir, "item_features.npy"))
    features_tensor = torch.from_numpy(features).float().to(device)
    logger.info(f"物品特征: {features.shape}")

    # 加载模型
    model, num_codebooks = load_model(method, cfg, device)

    # 生成Semantic ID
    # TODO: 码表分布情况，训练的分布和最终的结果分布
    logger.info("\n开始生成Semantic ID...")
    with torch.no_grad():
        semantic_ids = model.get_semantic_ids(features_tensor)
    semantic_ids = semantic_ids.cpu().numpy().T  # [num_items, num_codebooks]
    logger.info(f"Semantic ID 矩阵形状: {semantic_ids.shape}")

    # 详细分析
    codebook_size = cfg["sid"]["codebook_size"]
    unique_sids, counter = analyze_sid_distribution(semantic_ids, codebook_size)

    # 保存
    sid_path = os.path.join(processed_dir, "semantic_ids.npy")
    np.save(sid_path, semantic_ids)
    logger.info(f"\n保存Semantic ID矩阵: {sid_path}")

    sid_df = pd.DataFrame(
        semantic_ids, columns=[f"code_level_{i}" for i in range(num_codebooks)]
    )
    sid_df.index.name = "movie_id"
    sid_csv_path = os.path.join(processed_dir, "semantic_ids.csv")
    sid_df.to_csv(sid_csv_path)
    logger.info(f"保存Semantic ID CSV: {sid_csv_path}")

    # 保存num_codebooks到文件，供后续步骤使用
    ncb_path = get_abs_path("outputs/num_codebooks.txt")
    with open(ncb_path, "w") as f:
        f.write(str(num_codebooks))

    # 展示示例
    movies_path = os.path.join(processed_dir, "movies.csv")
    movies_df = pd.read_csv(movies_path)
    logger.info(f"\n看一些电影的Semantic ID:")
    for i in range(0, len(movies_df), 120):
        row = movies_df[movies_df["movie_id"] == i].iloc[0]
        sid = semantic_ids[i]
        logger.info(f'  movie_id={i}: "{row["title"]}" [{row["genres"]}]')
        logger.info(f"    Semantic ID = {sid.tolist()}")

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("生成完成!")
    logger.info(
        f"  唯一SID数: {unique_sids}/{len(semantic_ids)} ({unique_sids / len(semantic_ids) * 100:.1f}%)"
    )
    logger.info(f"  量化方式: {method}")
    logger.info("=" * 60)
    logger.info(f"\n请运行下一步: conda run -n py10 python scripts/06_train_decoder.py")


if __name__ == "__main__":
    main()
