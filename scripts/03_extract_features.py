"""
Step 3: 特征提取

- 电影标题 → Sentence Embedding（使用 sentence-transformers）
- 电影类别 → One-Hot 编码
- 拼接得到每个电影的特征向量

运行方式:
    .venv/bin/python scripts/03_extract_features.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


def clean_title(title: str) -> str:
    """
    清理电影标题：移除年份标记，如 "Toy Story (1995)" → "Toy Story"
    """
    # 移除末尾的 (年份)
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return cleaned.strip()


def extract_genre_onehot(movies_df: pd.DataFrame) -> tuple:
    """
    提取所有电影类别并进行One-Hot编码
    返回: (genre_names列表, onehot矩阵 [num_movies, num_genres])
    """
    # 收集所有类别
    all_genres = set()
    for genres_str in movies_df["genres"]:
        for g in genres_str.split("|"):
            all_genres.add(g)

    genre_names = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(genre_names)}

    print(f"[类别] 共 {len(genre_names)} 种类别: {genre_names}")

    # One-Hot编码（一部电影可能有多种类别，所以是多热编码）
    num_movies = len(movies_df)
    num_genres = len(genre_names)
    onehot = np.zeros((num_movies, num_genres), dtype=np.float32)

    for idx, row in movies_df.iterrows():
        movie_idx = row["movie_id"]
        for genre in row["genres"].split("|"):
            if genre in genre_to_idx:
                onehot[movie_idx, genre_to_idx[genre]] = 1.0

    return genre_names, onehot


def encode_titles(
    movies_df: pd.DataFrame, model_name: str, batch_size: int = 256
) -> np.ndarray:
    """
    使用Sentence Transformer对电影标题进行编码
    返回: [num_movies, embedding_dim] 的numpy数组
    """
    print(f"\n[标题编码] 加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 清理标题
    titles = [clean_title(t) for t in movies_df["title"].tolist()]

    print(f"[标题编码] 正在编码 {len(titles)} 个电影标题...")
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2归一化
    )

    print(f"[标题编码] 完成! 嵌入维度: {embeddings.shape}")
    return embeddings.astype(np.float32)


def main():
    cfg = load_config()
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 加载电影数据
    movies_path = os.path.join(processed_dir, "movies.csv")
    movies_df = pd.read_csv(movies_path)
    print(f"[加载] 电影数据: {len(movies_df)} 部电影")

    # 1. 类别One-Hot编码
    genre_names, genre_onehot = extract_genre_onehot(movies_df)

    # 2. 标题Sentence Embedding
    title_embeddings = encode_titles(
        movies_df, model_name=cfg["features"]["embedding_model"], batch_size=256
    )

    # 3. 拼接特征: [title_embedding | genre_onehot]
    # 确保 movie_id 是从0开始的连续整数，对齐矩阵
    num_movies = movies_df["movie_id"].max() + 1
    feature_dim = title_embeddings.shape[1] + genre_onehot.shape[1]

    # 检查movie_id是否连续
    assert num_movies == len(movies_df), (
        f"movie_id不连续: max_id={num_movies - 1}, 但只有{len(movies_df)}部电影"
    )

    item_features = np.zeros((num_movies, feature_dim), dtype=np.float32)
    for idx, row in movies_df.iterrows():
        mid = row["movie_id"]
        item_features[mid, : title_embeddings.shape[1]] = title_embeddings[mid]
        item_features[mid, title_embeddings.shape[1] :] = genre_onehot[mid]

    print(f"\n[特征拼接] 最终特征矩阵: {item_features.shape}")
    print(f"  标题嵌入维度: {title_embeddings.shape[1]}")
    print(f"  类别One-Hot维度: {genre_onehot.shape[1]}")
    print(f"  总特征维度: {feature_dim}")

    # 4. 保存
    features_path = os.path.join(processed_dir, "item_features.npy")
    np.save(features_path, item_features)
    print(f"\n[保存] 特征矩阵: {features_path}")

    # 保存类别名映射
    genre_path = os.path.join(processed_dir, "genre_names.txt")
    with open(genre_path, "w") as f:
        f.write("\n".join(genre_names))
    print(f"[保存] 类别名: {genre_path}")

    # 验证
    print(f"\n{'=' * 50}")
    print(f"特征提取完成！验证:")
    print(f"  特征矩阵形状: {item_features.shape}")
    print(f"  特征值范围: [{item_features.min():.4f}, {item_features.max():.4f}]")
    print(f"  特征均值: {item_features.mean():.4f}")
    print(f"  特征标准差: {item_features.std():.4f}")

    # 抽样验证几部电影
    print(f"\n  抽样验证:")
    for i in [0, 100, 500]:
        if i < len(movies_df):
            row = movies_df.iloc[i]
            title = clean_title(row["title"])
            genres = row["genres"]
            print(f'    movie_id={row["movie_id"]}: "{title}" [{genres}]')
            print(
                f"      特征范数: {np.linalg.norm(item_features[row['movie_id']]):.4f}"
            )

    print(f"{'=' * 50}")
    print(f"\n✅ 请运行下一步: .venv/bin/python scripts/04_train_rq_vae.py")


if __name__ == "__main__":
    main()
