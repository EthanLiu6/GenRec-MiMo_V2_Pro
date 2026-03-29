"""
Step 2: 数据预处理

- 读取 MovieLens-1M 原始数据
- 过滤：移除评分少于 min_user_interactions 的用户 和 评分少于 min_item_interactions 的电影
- 按时间排序，划分训练/验证/测试集
- 保存处理后的数据

运行方式:
    .venv/bin/python scripts/02_preprocess_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


def load_raw_data(raw_dir: str) -> tuple:
    """
    加载 MovieLens-1M 原始数据
    返回: (ratings_df, movies_df, users_df)
    """
    ml_dir = os.path.join(raw_dir, "ml-1m")

    # 加载评分数据
    ratings_path = os.path.join(ml_dir, "ratings.dat")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    print(
        f"[加载] 评分数据: {len(ratings_df)} 条, 用户数: {ratings_df['user_id'].nunique()}, 电影数: {ratings_df['movie_id'].nunique()}"
    )

    # 加载电影数据
    movies_path = os.path.join(ml_dir, "movies.dat")
    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        header=None,
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    print(f"[加载] 电影数据: {len(movies_df)} 部电影")

    # 加载用户数据
    users_path = os.path.join(ml_dir, "users.dat")
    users_df = pd.read_csv(
        users_path,
        sep="::",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1",
    )
    print(f"[加载] 用户数据: {len(users_df)} 个用户")

    return ratings_df, movies_df, users_df


def filter_data(ratings_df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """
    迭代过滤：移除交互次数不足的用户和物品
    反复过滤直到不再有需要移除的数据（因为移除用户可能导致物品交互不足，反之亦然）
    """
    print(f"\n[过滤] 开始过滤: 用户最少{min_user}条交互, 物品最少{min_item}条交互")
    prev_len = 0
    iteration = 0

    while len(ratings_df) != prev_len:
        iteration += 1
        prev_len = len(ratings_df)

        # 过滤用户
        user_counts = ratings_df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        ratings_df = ratings_df[ratings_df["user_id"].isin(valid_users)]

        # 过滤物品
        item_counts = ratings_df["movie_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        ratings_df = ratings_df[ratings_df["movie_id"].isin(valid_items)]

        print(
            f"  第{iteration}轮: {len(ratings_df)} 条交互, "
            f"{ratings_df['user_id'].nunique()} 用户, "
            f"{ratings_df['movie_id'].nunique()} 电影"
        )

    return ratings_df


def split_by_time(
    ratings_df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> dict:
    """
    按时间戳划分数据集
    每个用户的数据按时间排序，前 train_ratio 作为训练，接下来 val_ratio 作为验证，其余为测试
    """
    print(
        f"\n[划分] 按时间划分数据集: 训练{train_ratio:.0%}, 验证{val_ratio:.0%}, 测试{1 - train_ratio - val_ratio:.0%}"
    )

    train_list, val_list, test_list = [], [], []

    for user_id, group in ratings_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")

    return {"train": train_df, "val": val_df, "test": test_df}


def reindex_ids(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> tuple:
    """
    将 user_id 和 movie_id 重新映射为从0开始的连续整数
    """
    # 用户ID重映射
    unique_users = sorted(ratings_df["user_id"].unique())
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}

    # 物品ID重映射（只保留在过滤后评分数据中出现的电影）
    valid_movie_ids = set(ratings_df["movie_id"].unique())
    movies_df = movies_df[movies_df["movie_id"].isin(valid_movie_ids)].copy()
    unique_movies = sorted(movies_df["movie_id"].unique())
    movie_map = {old_id: new_id for new_id, old_id in enumerate(unique_movies)}

    # 应用重映射
    ratings_df = ratings_df.copy()
    ratings_df["user_id"] = ratings_df["user_id"].map(user_map)
    ratings_df["movie_id"] = ratings_df["movie_id"].map(movie_map)

    movies_df = movies_df.copy()
    movies_df["movie_id"] = movies_df["movie_id"].map(movie_map)
    movies_df = movies_df.sort_values("movie_id").reset_index(drop=True)

    print(
        f"\n[重映射] 用户ID: 0 ~ {max(user_map.values())}, 物品ID: 0 ~ {max(movie_map.values())}"
    )

    return ratings_df, movies_df, user_map, movie_map


def main():
    cfg = load_config()
    raw_dir = get_abs_path(cfg["data"]["raw_dir"])
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])
    os.makedirs(processed_dir, exist_ok=True)

    # 1. 加载原始数据
    ratings_df, movies_df, users_df = load_raw_data(raw_dir)

    # 2. 过滤数据
    ratings_df = filter_data(
        ratings_df,
        min_user=cfg["data"]["min_user_interactions"],
        min_item=cfg["data"]["min_item_interactions"],
    )

    # 3. 重映射ID
    ratings_df, movies_df, user_map, movie_map = reindex_ids(ratings_df, movies_df)

    # 4. 按时间划分数据集
    splits = split_by_time(
        ratings_df, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"]
    )

    # 5. 保存处理后的数据
    ratings_path = os.path.join(processed_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    print(f"\n[保存] 完整评分数据: {ratings_path}")

    for split_name, split_df in splits.items():
        split_path = os.path.join(processed_dir, f"{split_name}.csv")
        split_df.to_csv(split_path, index=False)
        print(f"[保存] {split_name}集: {split_path}")

    movies_path = os.path.join(processed_dir, "movies.csv")
    movies_df.to_csv(movies_path, index=False)
    print(f"[保存] 电影数据: {movies_path}")

    # 保存ID映射表
    user_map_df = pd.DataFrame(
        list(user_map.items()), columns=["original_user_id", "user_id"]
    )
    movie_map_df = pd.DataFrame(
        list(movie_map.items()), columns=["original_movie_id", "movie_id"]
    )
    user_map_df.to_csv(os.path.join(processed_dir, "user_id_map.csv"), index=False)
    movie_map_df.to_csv(os.path.join(processed_dir, "movie_id_map.csv"), index=False)
    print(f"[保存] ID映射表")

    # 打印统计信息
    print(f"\n{'=' * 50}")
    print(f"数据预处理完成！统计信息:")
    print(f"  过滤后用户数: {ratings_df['user_id'].nunique()}")
    print(f"  过滤后电影数: {ratings_df['movie_id'].nunique()}")
    print(f"  总交互数: {len(ratings_df)}")
    print(
        f"  稀疏度: {1 - len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()):.6f}"
    )
    print(f"{'=' * 50}")
    print(f"\n✅ 请运行下一步: .venv/bin/python scripts/03_extract_features.py")


if __name__ == "__main__":
    main()
