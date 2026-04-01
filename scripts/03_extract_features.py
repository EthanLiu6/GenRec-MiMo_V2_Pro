"""
Step 3: 特征提取

- 电影标题+类别 → Sentence Embedding（使用 sentence-transformers）
- 将标题和类别拼接后整体编码

运行方式:
    conda run -n py10 python scripts/03_extract_features.py
"""

import os
import sys
import re
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/03_extract_features.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def clean_title(title: str) -> str:
    """
    清理电影标题：移除年份标记，如 "Toy Story (1995)" → "Toy Story"
    """
    # 移除末尾的 (年份)
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return cleaned.strip()


def format_genres(genres_str: str) -> str:
    """
    格式化类别字符串，使其更适合语义编码
    "Animation|Children's|Comedy" → "Animation, Children's, Comedy"
    """
    return genres_str.replace("|", ", ")


def encode_titles_with_genres(
    movies_df: pd.DataFrame, model_name: str, batch_size: int = 256
) -> np.ndarray:
    """
    使用Sentence Transformer对电影标题+类别进行编码
    将标题和类别拼接后整体编码
    返回: [num_movies, embedding_dim] 的numpy数组
    """
    logger.info(f"加载Sentence Transformer模型: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"模型加载完成，嵌入维度: {model.get_sentence_embedding_dimension()}")

    # 拼接标题和类别，使用更自然的格式
    texts = []
    for idx, row in movies_df.iterrows():
        title = clean_title(str(row["title"]))
        genres = format_genres(str(row["genres"]))
        # 使用更自然的文本格式，增强语义信息
        # TODO: 调整拼接方案
        # text = f"{title}. Genres: {genres}"
        text = f"{title}. {genres}"
        texts.append(text)

    # 打印前几个样本用于验证
    logger.info(f"文本编码示例 (前3个):")
    for i in range(min(3, len(texts))):
        logger.info(f"  [{i}] {texts[i]}")

    logger.info(f"开始编码 {len(texts)} 个电影...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        # normalize_embeddings=True,  # L2归一化
    )

    logger.info(f"编码完成! 嵌入维度: {embeddings.shape}")
    return embeddings.astype(np.float32)


def main():
    cfg = load_config()
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 加载电影数据
    movies_path = os.path.join(processed_dir, "movies.csv")
    movies_df = pd.read_csv(movies_path)
    logger.info(f"加载电影数据: {len(movies_df)} 部电影")

    # 检查movie_id连续性
    max_id = movies_df["movie_id"].max()
    logger.info(f"movie_id范围: 0 ~ {max_id}, 总数: {len(movies_df)}")
    if max_id + 1 != len(movies_df):
        logger.warning(f"movie_id不连续: max_id={max_id}, 但只有{len(movies_df)}部电影")

    # 1. 标题+类别整体Sentence Embedding
    logger.info("=" * 60)
    logger.info("开始特征提取...")
    logger.info("=" * 60)

    item_embeddings = encode_titles_with_genres(
        movies_df, model_name=cfg["features"]["embedding_model"], batch_size=256
    )

    # 2. 构建特征矩阵
    num_movies = max_id + 1
    feature_dim = item_embeddings.shape[1]

    logger.info(f"构建特征矩阵: ({num_movies}, {feature_dim})")
    item_features = np.zeros((num_movies, feature_dim), dtype=np.float32)

    # 使用movie_id作为索引填充
    for idx, row in movies_df.iterrows():
        mid = row["movie_id"]
        item_features[mid] = item_embeddings[idx]

    # 3. 特征质量验证
    logger.info("=" * 60)
    logger.info("特征矩阵验证:")
    logger.info(f"  形状: {item_features.shape}")
    logger.info(f"  值范围: [{item_features.min():.6f}, {item_features.max():.6f}]")
    logger.info(f"  均值: {item_features.mean():.6f}")
    logger.info(f"  标准差: {item_features.std():.6f}")

    # 计算特征间的余弦相似度分布（采样）
    logger.info(f"  特征范数统计:")
    norms = np.linalg.norm(item_features, axis=1)
    logger.info(f"    均值: {norms.mean():.6f}, 标准差: {norms.std():.6f}")
    logger.info(f"    范围: [{norms.min():.6f}, {norms.max():.6f}]")

    # 检查特征是否全为0或异常
    zero_features = np.sum(np.all(item_features == 0, axis=1))
    if zero_features > 0:
        logger.warning(f"  发现 {zero_features} 个全零特征!")

    # 4. 抽样验证几部电影
    logger.info(f"\n抽样验证:")
    for i in [0, 100, 500, 1000]:
        if i < len(movies_df):
            row = movies_df.iloc[i]
            title = clean_title(row["title"])
            genres = row["genres"]
            mid = row["movie_id"]
            feature_norm = np.linalg.norm(item_features[mid])
            logger.info(f'  movie_id={mid}: "{title}" [{genres}]')
            logger.info(f"    特征范数: {feature_norm:.6f}")

    # 5. 保存
    features_path = os.path.join(processed_dir, "item_features.npy")
    np.save(features_path, item_features)
    logger.info(f"\n保存特征矩阵: {features_path}")

    # 6. 特征统计摘要
    logger.info("=" * 60)
    logger.info("特征提取完成!")
    logger.info(f"  特征维度: {feature_dim}")
    logger.info(f"  物品数量: {num_movies}")
    logger.info(f"  特征矩阵: {features_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 确保logs目录存在
    os.makedirs("logs", exist_ok=True)
    main()
