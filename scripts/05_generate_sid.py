"""
Step 5: 生成 Semantic ID

支持三种量化方式，自动根据 outputs/current_sid_method.txt 识别

运行方式:
    .venv/bin/python scripts/05_generate_sid.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


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

    print(
        f"[模型] 加载 {method} (epoch={checkpoint['epoch']}, num_codebooks={num_codebooks})"
    )
    return model, num_codebooks


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 读取当前使用的方法
    method_file = get_abs_path("outputs/current_sid_method.txt")
    if os.path.exists(method_file):
        with open(method_file, "r") as f:
            method = f.read().strip()
    else:
        method = cfg["sid_method"]

    print(f"[方法] 当前量化方式: {method}")

    # 加载数据
    features = np.load(os.path.join(processed_dir, "item_features.npy"))
    features_tensor = torch.from_numpy(features).float().to(device)
    print(f"[数据] 物品特征: {features.shape}")

    # 加载模型
    model, num_codebooks = load_model(method, cfg, device)

    # 生成Semantic ID
    with torch.no_grad():
        semantic_ids = model.get_semantic_ids(features_tensor)
    semantic_ids = semantic_ids.cpu().numpy().T  # [num_items, num_codebooks]
    print(f"[生成] Semantic ID 矩阵: {semantic_ids.shape}")

    # 统计分析
    codebook_size = cfg["sid"]["codebook_size"]
    print(f"\n[统计] Semantic ID 分析:")
    for level in range(num_codebooks):
        unique_codes = len(np.unique(semantic_ids[:, level]))
        print(f"  Level {level}: 使用了 {unique_codes}/{codebook_size} 个code")

    semantic_tuples = [tuple(row) for row in semantic_ids]
    unique_sids = len(set(semantic_tuples))
    print(
        f"\n  唯一Semantic ID数: {unique_sids}/{len(semantic_ids)} ({unique_sids / len(semantic_ids):.1%})"
    )

    # 保存（统一文件名，方便后续步骤使用）
    sid_path = os.path.join(processed_dir, "semantic_ids.npy")
    np.save(sid_path, semantic_ids)
    print(f"\n[保存] Semantic ID矩阵: {sid_path}")

    sid_df = pd.DataFrame(
        semantic_ids, columns=[f"code_level_{i}" for i in range(num_codebooks)]
    )
    sid_df.index.name = "movie_id"
    sid_csv_path = os.path.join(processed_dir, "semantic_ids.csv")
    sid_df.to_csv(sid_csv_path)
    print(f"[保存] Semantic ID CSV: {sid_csv_path}")

    # 保存num_codebooks到文件，供后续步骤使用
    ncb_path = get_abs_path("outputs/num_codebooks.txt")
    with open(ncb_path, "w") as f:
        f.write(str(num_codebooks))

    # 展示示例
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
    main()
