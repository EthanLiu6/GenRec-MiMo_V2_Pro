"""
Step 2.5: 用户特征预处理

- 性别(2类)、年龄(7类)、职业(21类) 分别做One-Hot + argmax → 整数索引
- 最终每个用户得到3维向量 [gender_idx, age_idx, occupation_idx]

运行方式:
    .venv/bin/python scripts/02b_preprocess_users.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


def main():
    cfg = load_config()
    raw_dir = get_abs_path(cfg["data"]["raw_dir"])
    processed_dir = get_abs_path(cfg["data"]["processed_dir"])

    # 1. 读取原始用户数据
    users_path = os.path.join(raw_dir, "ml-1m", "users.dat")
    users_df = pd.read_csv(
        users_path,
        sep="::",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1",
    )
    print(f"[加载] 用户数据: {len(users_df)} 个用户")

    # 2. 读取用户ID映射（过滤后的）
    user_map_path = os.path.join(processed_dir, "user_id_map.csv")
    user_map_df = pd.read_csv(user_map_path)
    original_to_new = dict(zip(user_map_df["original_user_id"], user_map_df["user_id"]))
    valid_original_ids = set(user_map_df["original_user_id"])

    users_df = users_df[users_df["user_id"].isin(valid_original_ids)].copy()
    users_df["new_user_id"] = users_df["user_id"].map(original_to_new)
    users_df = users_df.sort_values("new_user_id").reset_index(drop=True)
    print(f"[过滤] 过滤后用户数: {len(users_df)}")

    # 3. 构建类别到索引的映射
    gender_map = {"F": 0, "M": 1}  # 2类
    unique_ages = sorted(users_df["age"].unique())
    age_map = {a: i for i, a in enumerate(unique_ages)}  # 7类
    unique_occs = sorted(users_df["occupation"].unique())
    occ_map = {o: i for i, o in enumerate(unique_occs)}  # 21类

    print(f"[映射] 性别: {gender_map}")
    print(f"[映射] 年龄: {age_map}")
    print(f"[映射] 职业: {len(occ_map)} 类")

    # 4. 构建特征矩阵 [num_users, 3]
    num_users = int(users_df["new_user_id"].max()) + 1
    user_features = np.zeros((num_users, 3), dtype=np.int64)

    for _, row in users_df.iterrows():
        uid = int(row["new_user_id"])
        user_features[uid, 0] = gender_map[row["gender"]]
        user_features[uid, 1] = age_map[row["age"]]
        user_features[uid, 2] = occ_map[row["occupation"]]

    print(f"[矩阵] 用户特征矩阵: {user_features.shape}")

    # 5. 保存
    save_path = os.path.join(processed_dir, "user_features.npy")
    np.save(save_path, user_features)
    print(f"[保存] {save_path}")

    # 保存类别数信息
    info = {
        "num_genders": len(gender_map),
        "num_ages": len(age_map),
        "num_occupations": len(occ_map),
    }
    info_path = os.path.join(processed_dir, "user_feature_info.npy")
    np.save(info_path, info)
    print(f"[保存] {info_path}")

    # 打印几条示例
    print(f"\n[示例] 前5个用户:")
    for i in range(min(5, num_users)):
        row = users_df[users_df["new_user_id"] == i]
        if len(row) > 0:
            row = row.iloc[0]
            print(
                f"  user_id={i}: gender={row['gender']}({user_features[i, 0]}), "
                f"age={row['age']}({user_features[i, 1]}), "
                f"occ={row['occupation']}({user_features[i, 2]})"
            )
            print(f"    特征向量: {user_features[i].tolist()}")

    print(f"\n✅ 用户特征预处理完成!")
    print(f"  特征: 3维 [gender_idx, age_idx, occ_idx]")
    print(
        f"  类别数: gender={len(gender_map)}, age={len(age_map)}, occupation={len(occ_map)}"
    )


if __name__ == "__main__":
    main()
