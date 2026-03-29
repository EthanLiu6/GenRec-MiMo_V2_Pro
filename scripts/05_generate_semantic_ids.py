"""
Step 5: 生成 Semantic ID

- 加载训练好的 RQ-VAE 模型
- 对所有物品生成 Semantic ID
- 保存 Semantic ID 映射表

运行方式:
    .venv/bin/python scripts/05_generate_semantic_ids.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path
from src.models.rq_vae import RQVAE


def generate_semantic_ids(cfg: dict):
    """为所有物品生成Semantic ID"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载物品特征
    features_path = get_abs_path(
        os.path.join(cfg["data"]["processed_dir"], "item_features.npy")
    )
    features = np.load(features_path)
    features_tensor = torch.from_numpy(features).float().to(device)
    print(f"[数据] 物品特征: {features.shape}")

    # 2. 加载训练好的模型
    model_path = get_abs_path(cfg["rq_vae"]["model_save_path"])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = RQVAE(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"[模型] 加载RQ-VAE (epoch={checkpoint['epoch']}, loss={checkpoint['best_loss']:.4f})"
    )

    # 3. 生成Semantic ID
    with torch.no_grad():
        semantic_ids = model.get_semantic_ids(
            features_tensor
        )  # [num_codebooks, num_items]

    semantic_ids = semantic_ids.cpu().numpy().T  # [num_items, num_codebooks]
    print(f"[生成] Semantic ID 矩阵: {semantic_ids.shape}")

    # 4. 统计分析
    num_codebooks = cfg["rq_vae"]["num_codebooks"]
    codebook_size = cfg["rq_vae"]["codebook_size"]

    print(f"\n[统计] Semantic ID 分析:")
    for level in range(num_codebooks):
        unique_codes = len(np.unique(semantic_ids[:, level]))
        print(
            f"  Level {level}: 使用了 {unique_codes}/{codebook_size} 个code "
            f"({unique_codes / codebook_size:.1%} 利用率)"
        )

    # 5. 检查唯一性（不同物品是否被编码为不同的Semantic ID）
    semantic_tuples = [tuple(row) for row in semantic_ids]
    unique_sids = len(set(semantic_tuples))
    print(
        f"\n  唯一Semantic ID数: {unique_sids}/{len(semantic_ids)} "
        f"({unique_sids / len(semantic_ids):.1%} 唯一率)"
    )

    # 6. 保存
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 保存为numpy
    sid_path = os.path.join(processed_dir, "semantic_ids.npy")
    np.save(sid_path, semantic_ids)
    print(f"\n[保存] Semantic ID矩阵: {sid_path}")

    # 保存为CSV（便于查看）
    sid_df = pd.DataFrame(
        semantic_ids, columns=[f"code_level_{i}" for i in range(num_codebooks)]
    )
    sid_df.index.name = "movie_id"
    sid_csv_path = os.path.join(processed_dir, "semantic_ids.csv")
    sid_df.to_csv(sid_csv_path)
    print(f"[保存] Semantic ID CSV: {sid_csv_path}")

    # 7. 展示几个例子
    movies_path = os.path.join(processed_dir, "movies.csv")
    movies_df = pd.read_csv(movies_path)
    print(f"\n[示例] 前10部电影的Semantic ID:")
    for i in range(min(10, len(movies_df))):
        row = movies_df[movies_df["movie_id"] == i].iloc[0]
        sid = semantic_ids[i]
        print(f'  movie_id={i}: "{row["title"]}" [{row["genres"]}]')
        print(f"    Semantic ID = {sid.tolist()}")

    print(f"\n✅ 请运行下一步: .venv/bin/python scripts/06_train_decoder.py")


if __name__ == "__main__":
    cfg = load_config()
    generate_semantic_ids(cfg)
