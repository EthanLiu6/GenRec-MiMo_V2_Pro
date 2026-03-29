"""
Step 6: 训练 Decoder（序列推荐模型）

支持用户特征前缀（gender, age, occupation → 3个token前缀）
是否启用由 config.yaml 的 decoder.use_user_features 控制

运行方式:
    .venv/bin/python scripts/06_train_decoder.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path
from src.models.decoder import SemanticIDDecoder

# 特殊Token定义
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3


class InteractionDataset(Dataset):
    """交互序列数据集，支持可选的用户特征"""

    def __init__(
        self,
        ratings_df,
        semantic_ids,
        num_codebooks,
        user_features=None,
        max_seq_len=50,
    ):
        self.max_seq_len = max_seq_len
        self.semantic_ids = semantic_ids
        self.user_features = user_features
        self.samples = []

        for user_id, group in ratings_df.groupby("user_id"):
            group = group.sort_values("timestamp")
            item_ids = group["movie_id"].tolist()
            if len(item_ids) < 2:
                continue

            # 构建token序列（不含BOS，BOS在__getitem__中加）
            tokens = []
            for item_id in item_ids:
                codes = semantic_ids[item_id].tolist()
                tokens.extend([c + 4 for c in codes])
                tokens.append(SEP_TOKEN)
            tokens.append(EOS_TOKEN)

            # 截断：保留max_seq_len个token（不含BOS）
            if len(tokens) > max_seq_len:
                tokens = tokens[: max_seq_len - 1] + [EOS_TOKEN]

            self.samples.append({"user_id": int(user_id), "tokens": tokens})

        print(f"[数据集] 构建完成: {len(self.samples)} 条序列, max_len={max_seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]

        # input: [BOS] + tokens[:-1], target: tokens
        input_ids = [BOS_TOKEN] + tokens[:-1]
        target_ids = tokens

        # 填充到固定长度
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [PAD_TOKEN] * pad_len
            target_ids = target_ids + [PAD_TOKEN] * pad_len

        # padding mask: True表示填充位置
        padding_mask = [False] * len(tokens) + [True] * max(0, pad_len)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        }

        if self.user_features is not None:
            result["user_features"] = torch.tensor(
                self.user_features[sample["user_id"]], dtype=torch.long
            )

        return result


def collate_fn(batch):
    """自定义collate，支持可选的user_features"""
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch]),
        "padding_mask": torch.stack([b["padding_mask"] for b in batch]),
    }
    if "user_features" in batch[0]:
        result["user_features"] = torch.stack([b["user_features"] for b in batch])
    return result


def train_decoder(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用: {device}")

    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 1. 加载SID数据
    semantic_ids = np.load(os.path.join(processed_dir, "semantic_ids.npy"))
    num_codebooks = semantic_ids.shape[1]
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    print(f"[数据] Semantic IDs: {semantic_ids.shape}, num_codebooks={num_codebooks}")

    # 读取SID方法
    method_file = get_abs_path("outputs/current_sid_method.txt")
    method = cfg["sid_method"]
    if os.path.exists(method_file):
        with open(method_file, "r") as f:
            method = f.read().strip()
    print(f"[方法] 量化方式: {method}")

    # 2. 加载用户特征（由config控制是否使用）
    use_user_features = cfg["decoder"].get("use_user_features", False)
    user_features = None
    user_feat_info = {}

    if use_user_features:
        user_feat_path = os.path.join(processed_dir, "user_features.npy")
        if os.path.exists(user_feat_path):
            user_features = np.load(user_feat_path)
            info_path = os.path.join(processed_dir, "user_feature_info.npy")
            if os.path.exists(info_path):
                user_feat_info = np.load(info_path, allow_pickle=True).item()
            print(
                f"[用户特征] 已加载: {user_features.shape}, "
                f"gender={user_feat_info.get('num_genders', 2)}, "
                f"age={user_feat_info.get('num_ages', 7)}, "
                f"occ={user_feat_info.get('num_occupations', 21)}"
            )
        else:
            print(f"[用户特征] 配置启用但文件不存在: {user_feat_path}")
            print(
                f"[用户特征] 自动关闭，请先运行: .venv/bin/python scripts/02b_preprocess_users.py"
            )
            use_user_features = False
    else:
        print(f"[用户特征] 配置为关闭，不使用用户特征")

    max_seq_len = cfg["data"]["max_seq_len"]
    codebook_size = cfg["sid"]["codebook_size"]
    if method == "vq_vae":
        codebook_size = cfg.get("vq_vae", {}).get(
            "codebook_size_override", codebook_size
        )

    # 3. 构建数据集
    train_dataset = InteractionDataset(
        train_df, semantic_ids, num_codebooks, user_features, max_seq_len
    )
    val_dataset = InteractionDataset(
        val_df, semantic_ids, num_codebooks, user_features, max_seq_len
    )

    bs = cfg["decoder"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn
    )

    # 4. 初始化模型
    decoder_config = {
        "d_model": cfg["decoder"]["d_model"],
        "nhead": cfg["decoder"]["nhead"],
        "num_layers": cfg["decoder"]["num_layers"],
        "dim_feedforward": cfg["decoder"]["dim_feedforward"],
        "dropout": cfg["decoder"]["dropout"],
        "num_codebooks": num_codebooks,
        "codebook_size": codebook_size,
        "special_token_size": cfg["decoder"]["special_token_size"],
        "use_user_features": use_user_features,
        "num_genders": user_feat_info.get("num_genders", 2),
        "num_ages": user_feat_info.get("num_ages", 7),
        "num_occupations": user_feat_info.get("num_occupations", 21),
    }
    model = SemanticIDDecoder(decoder_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 总参数: {total_params:,}")

    # 5. 优化器和损失
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["decoder"]["lr"],
        weight_decay=cfg["decoder"]["weight_decay"],
    )
    vocab_size = cfg["decoder"]["special_token_size"] + codebook_size
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # 用户前缀token数（训练/评估时从模型输出中跳过）
    num_user_tokens = 3 if use_user_features else 0

    # 多样性惩罚参数
    div_penalty = cfg["decoder"].get("diversity_penalty", 0.0)
    div_recent_n = cfg["decoder"].get("diversity_recent_n", 0)
    special_token_size = cfg["decoder"]["special_token_size"]

    if div_penalty > 0:
        print(f"[多样性] 惩罚权重={div_penalty}, 最近N个物品={div_recent_n}(0=全部)")

    def apply_diversity_penalty(logits, input_ids, penalty, recent_n, sp_size):
        """
        对logits中历史出现过的code施加负向偏移

        参数:
            logits: [B, seq_len, vocab_size] 模型输出
            input_ids: [B, seq_len] 输入token序列
            penalty: 惩罚权重
            recent_n: 只看最近N个物品的历史code（0=全部）
            sp_size: 特殊token数量（code值 >= sp_size）
        """
        B, S, V = logits.shape

        # 从input_ids中提取历史code值（排除特殊token、填充和用户前缀）
        # code值范围: [sp_size, sp_size + codebook_size)
        history_mask = (input_ids >= sp_size) & (input_ids != PAD_TOKEN)

        if recent_n > 0:
            # 只取最近N个物品对应的token
            # 找到每个样本中最后一个SEP的位置，往前数recent_n个物品
            sep_mask = input_ids == SEP_TOKEN
            for b in range(B):
                sep_positions = sep_mask[b].nonzero(as_tuple=True)[0]
                if len(sep_positions) > recent_n:
                    # 保留最近recent_n个SEP之前的所有token
                    cutoff = sep_positions[-(recent_n)].item()
                    history_mask[b, :cutoff] = False

        # 构建惩罚矩阵 [B, V]
        penalty_matrix = torch.zeros(B, V, device=logits.device)
        for b in range(B):
            codes = input_ids[b][history_mask[b]].unique()
            valid_codes = codes[codes >= sp_size]
            if len(valid_codes) > 0:
                penalty_matrix[b, valid_codes] = penalty

        # 对所有时间步施加惩罚（每个位置都要考虑历史）
        logits = logits - penalty_matrix.unsqueeze(1)

        return logits

    # 6. 训练循环
    epochs = cfg["decoder"]["epochs"]
    best_val_loss = float("inf")

    save_name = f"decoder_{method}_best.pt"
    save_path = get_abs_path(os.path.join("outputs", save_name))
    generic_save_path = get_abs_path("outputs/decoder_best.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(
        f"\n[训练] Decoder ({method}), 用户特征={'是' if use_user_features else '否'}, "
        f"{epochs} 个epoch..."
    )

    for epoch in range(epochs):
        # --- 训练 ---
        model.train()
        train_loss, train_correct, train_total, nb = 0.0, 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            uf = batch.get("user_features")
            if uf is not None:
                uf = uf.to(device)

            logits = model(input_ids, padding_mask, user_features=uf)

            # 跳过用户前缀位置的输出
            if num_user_tokens > 0:
                logits = logits[:, num_user_tokens:, :]

            # 多样性惩罚：抑制对历史code的重复预测
            if div_penalty > 0:
                # 需要用原始input_ids（不含前缀偏移）来提取历史code
                raw_input = input_ids if num_user_tokens == 0 else input_ids
                logits = apply_diversity_penalty(
                    logits, raw_input, div_penalty, div_recent_n, special_token_size
                )

            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = logits.argmax(dim=-1)
            mask = target_ids != PAD_TOKEN
            train_correct += ((pred == target_ids) & mask).sum().item()
            train_total += mask.sum().item()
            train_loss += loss.item()
            nb += 1

        avg_train_loss = train_loss / nb
        train_acc = train_correct / train_total if train_total > 0 else 0

        # --- 验证 ---
        model.eval()
        val_loss, val_correct, val_total, vb = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                uf = batch.get("user_features")
                if uf is not None:
                    uf = uf.to(device)

                logits = model(input_ids, padding_mask, user_features=uf)

                if num_user_tokens > 0:
                    logits = logits[:, num_user_tokens:, :]

                loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

                pred = logits.argmax(dim=-1)
                mask = target_ids != PAD_TOKEN
                val_correct += ((pred == target_ids) & mask).sum().item()
                val_total += mask.sum().item()
                val_loss += loss.item()
                vb += 1

        avg_val_loss = val_loss / vb if vb > 0 else float("inf")
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": decoder_config,
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                    "sid_method": method,
                },
                save_path,
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": decoder_config,
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                    "sid_method": method,
                },
                generic_save_path,
            )
            print(f"  ✅ 保存最优模型 (val_loss={best_val_loss:.4f})")

    print(f"\n[完成] 最优验证loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    cfg = load_config()
    train_decoder(cfg)
    print(f"\n✅ 下一步: .venv/bin/python scripts/07_evaluate.py")
