"""
Step 7: 评估与推理

用户特征是否启用，从模型checkpoint中读取（训练时保存的配置）

运行方式:
    .venv/bin/python scripts/07_evaluate.py           # 评估
    .venv/bin/python scripts/07_evaluate.py --demo    # 演示推理
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path
from src.logger import get_logger
from src.models.decoder import SemanticIDDecoder

logger = get_logger(__name__)

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3


def build_sid_to_items(semantic_ids):
    sid_to_items = defaultdict(list)
    for item_id in range(len(semantic_ids)):
        sid = tuple(semantic_ids[item_id].tolist())
        sid_to_items[sid].append(item_id)
    return dict(sid_to_items)


def build_item_to_sid(semantic_ids):
    return {i: tuple(semantic_ids[i].tolist()) for i in range(len(semantic_ids))}


def generate_next_sid(
    model, input_tokens, num_codebooks, device, user_features=None, max_gen_len=10
):
    """自回归生成下一个物品的Semantic ID"""
    model.eval()
    generated = list(input_tokens)

    with torch.no_grad():
        for _ in range(max_gen_len):
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            uf = None
            if user_features is not None:
                uf = torch.tensor([user_features], dtype=torch.long, device=device)
            logits = model.predict_next_token(input_ids, user_features=uf)
            next_token = logits.argmax(dim=-1).item()

            if next_token == EOS_TOKEN:
                break
            if next_token == SEP_TOKEN:
                generated.append(next_token)
                break
            generated.append(next_token)

    gen_tokens = generated[len(input_tokens) :]
    gen_codes = [t - 4 for t in gen_tokens if t >= 4]
    while len(gen_codes) < num_codebooks:
        gen_codes.append(0)
    return gen_codes[:num_codebooks]


def load_decoder(cfg, device):
    """加载Decoder，所有配置从checkpoint中读取"""
    # 确定SID方法
    method_file = get_abs_path("outputs/current_sid_method.txt")
    method = cfg["sid_method"]
    if os.path.exists(method_file):
        with open(method_file, "r") as f:
            method = f.read().strip()

    # 查找模型文件
    model_path = get_abs_path(os.path.join("outputs", f"decoder_{method}_best.pt"))
    if not os.path.exists(model_path):
        model_path = get_abs_path("outputs/decoder_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到Decoder模型文件，请先运行 Step 6 训练")

    # 加载checkpoint，配置全部从checkpoint读取
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    decoder_config = checkpoint["config"]
    model = SemanticIDDecoder(decoder_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_codebooks = decoder_config["num_codebooks"]
    vocab_size = checkpoint["vocab_size"]
    use_user_features = decoder_config.get("use_user_features", False)
    num_user_tokens = 3 if use_user_features else 0

    logger.info(
        f"加载Decoder: method={method}, epoch={checkpoint['epoch']}, "
        f"val_loss={checkpoint['best_val_loss']:.4f}, "
        f"num_codebooks={num_codebooks}, user_feat={use_user_features}"
    )

    return model, num_codebooks, vocab_size, method, use_user_features


def evaluate(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    semantic_ids = np.load(os.path.join(processed_dir, "semantic_ids.npy"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    logger.info(f"Semantic IDs: {semantic_ids.shape}, 测试集: {len(test_df)} 条")

    # 加载模型（use_user_features从checkpoint读取）
    model, num_codebooks, vocab_size, method, use_user_features = load_decoder(
        cfg, device
    )

    # 加载用户特征（仅当模型训练时使用了用户特征）
    user_features_all = None
    if use_user_features:
        user_feat_path = os.path.join(processed_dir, "user_features.npy")
        if os.path.exists(user_feat_path):
            user_features_all = np.load(user_feat_path)
            logger.info(f"用户特征: {user_features_all.shape}")
        else:
            logger.warning(f"模型使用了用户特征但文件不存在: {user_feat_path}")

    sid_to_items = build_sid_to_items(semantic_ids)
    item_to_sid = build_item_to_sid(semantic_ids)
    num_items = len(semantic_ids)

    # 构建用户历史
    all_data = pd.concat([train_df, val_df], ignore_index=True)
    all_data = all_data.sort_values(["user_id", "timestamp"])
    user_histories = {}
    for user_id, group in all_data.groupby("user_id"):
        user_histories[user_id] = group["movie_id"].tolist()

    # 测试标签
    test_labels = {}
    for user_id, group in test_df.groupby("user_id"):
        test_labels[user_id] = group["movie_id"].tolist()

    top_k_values = cfg["eval"]["top_k"]
    recall_sum = {k: 0.0 for k in top_k_values}
    num_eval_users = 0

    logger.info(f"开始评估, {len(test_labels)} 个测试用户...")

    for user_id, true_items in tqdm(test_labels.items(), desc="Evaluating"):
        if user_id not in user_histories or len(user_histories[user_id]) == 0:
            continue

        history = user_histories[user_id]

        # 获取用户特征
        uf = None
        if user_features_all is not None:
            uf = user_features_all[int(user_id)].tolist()

        # 构建输入token
        input_tokens = [BOS_TOKEN]
        for item_id in history:
            codes = semantic_ids[item_id].tolist()
            input_tokens.extend([c + 4 for c in codes])
            input_tokens.append(SEP_TOKEN)

        max_input_len = cfg["data"]["max_seq_len"] - 5
        if len(input_tokens) > max_input_len:
            input_tokens = [BOS_TOKEN] + input_tokens[-max_input_len:]

        pred_codes = generate_next_sid(
            model, input_tokens, num_codebooks, device, user_features=uf
        )
        pred_sid = tuple(pred_codes)

        # 匹配物品
        candidates = set(sid_to_items.get(pred_sid, []))
        if len(candidates) == 0:
            for level in range(num_codebooks - 1, 0, -1):
                partial_sid = pred_sid[:level]
                for iid, sid in item_to_sid.items():
                    if sid[:level] == partial_sid:
                        candidates.add(iid)
                if len(candidates) >= max(top_k_values):
                    break

        if len(candidates) < max(top_k_values):
            remaining = list(set(range(num_items)) - candidates - set(history))
            np.random.shuffle(remaining)
            candidates = (
                list(candidates) + remaining[: max(top_k_values) - len(candidates)]
            )
        else:
            candidates = list(candidates)

        ranked_items = candidates[: max(top_k_values)]
        true_set = set(true_items)
        num_eval_users += 1

        for k in top_k_values:
            hits = len(set(ranked_items[:k]) & true_set)
            recall_sum[k] += hits / min(len(true_set), k)

    # 输出结果
    logger.info("=" * 50)
    logger.info(
        f"评估结果 ({method}, user_feat={use_user_features}, {num_eval_users} 用户):"
    )
    results = {}
    for k in top_k_values:
        avg_recall = recall_sum[k] / num_eval_users if num_eval_users > 0 else 0
        results[f"Recall@{k}"] = avg_recall
        logger.info(f"  Recall@{k}: {avg_recall:.4f}")
    logger.info("=" * 50)

    suffix = "_uf" if use_user_features else ""
    results_path = get_abs_path(f"outputs/eval_results_{method}{suffix}.txt")
    with open(results_path, "w") as f:
        f.write(f"Method: {method}\n")
        f.write(f"User features: {use_user_features}\n")
        f.write(f"Num test users: {num_eval_users}\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    logger.info(f"结果已保存: {results_path}")


def demo_inference(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    semantic_ids = np.load(os.path.join(processed_dir, "semantic_ids.npy"))
    movies_df = pd.read_csv(os.path.join(processed_dir, "movies.csv"))

    model, num_codebooks, vocab_size, method, use_user_features = load_decoder(
        cfg, device
    )

    user_features_all = None
    if use_user_features:
        user_feat_path = os.path.join(processed_dir, "user_features.npy")
        if os.path.exists(user_feat_path):
            user_features_all = np.load(user_feat_path)

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    train_df = train_df.sort_values(["user_id", "timestamp"])

    sid_to_items = build_sid_to_items(semantic_ids)

    demo_user = 0
    user_history = train_df[train_df["user_id"] == demo_user]["movie_id"].tolist()

    uf = None
    if user_features_all is not None:
        uf = user_features_all[demo_user].tolist()
        logger.info(f"用户 {demo_user} 特征: {uf}")

    logger.info(f"用户 {demo_user} 历史 (共 {len(user_history)} 部):")
    for item_id in user_history[-10:]:
        if item_id < len(movies_df):
            row = movies_df[movies_df["movie_id"] == item_id]
            if len(row) > 0:
                logger.info(f"  {row.iloc[0]['title']} [{row.iloc[0]['genres']}]")

    input_tokens = [BOS_TOKEN]
    for item_id in user_history:
        codes = semantic_ids[item_id].tolist()
        input_tokens.extend([c + 4 for c in codes])
        input_tokens.append(SEP_TOKEN)

    pred_codes = generate_next_sid(
        model, input_tokens, num_codebooks, device, user_features=uf
    )
    logger.info(f"预测的Semantic ID: {pred_codes}")

    pred_sid = tuple(pred_codes)
    matched_items = sid_to_items.get(pred_sid, [])

    if matched_items:
        logger.info(f"匹配到的物品:")
        for item_id in matched_items[:5]:
            if item_id < len(movies_df):
                row = movies_df[movies_df["movie_id"] == item_id]
                if len(row) > 0:
                    logger.info(f"  {row.iloc[0]['title']} [{row.iloc[0]['genres']}]")
    else:
        logger.info("无完全匹配")


if __name__ == "__main__":
    cfg = load_config()
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_inference(cfg)
    else:
        evaluate(cfg)
        logger.info("如需演示: .venv/bin/python scripts/07_evaluate.py --demo")
