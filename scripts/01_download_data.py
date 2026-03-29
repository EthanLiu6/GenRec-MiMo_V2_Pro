"""
Step 1: 下载 MovieLens-1M 数据集

运行方式:
    .venv/bin/python scripts/01_download_data.py
"""

import os
import sys
import zipfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_abs_path


def download_movielens_1m(target_dir: str):
    """
    下载并解压 MovieLens-1M 数据集
    数据集来源: https://grouplens.org/datasets/movielens/1m/
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(target_dir, "ml-1m.zip")
    extract_dir = target_dir

    os.makedirs(target_dir, exist_ok=True)

    # 如果已经解压过了，跳过
    if os.path.isdir(os.path.join(extract_dir, "ml-1m")):
        print(f"[跳过] 数据集已存在于: {os.path.join(extract_dir, 'ml-1m')}")
        return

    # 下载
    print(f"[下载] 正在从 {url} 下载数据集...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[完成] 下载至: {zip_path}")

    # 解压
    print("[解压] 正在解压...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[完成] 解压至: {extract_dir}")

    # 清理zip文件
    os.remove(zip_path)
    print("[清理] 已删除zip文件")

    # 打印文件列表
    ml_dir = os.path.join(extract_dir, "ml-1m")
    print(f"\n数据集文件:")
    for f in os.listdir(ml_dir):
        fpath = os.path.join(ml_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")


if __name__ == "__main__":
    cfg = load_config()
    target_dir = get_abs_path(cfg["data"]["raw_dir"])
    download_movielens_1m(target_dir)
    print(
        "\n✅ 数据下载完成！请运行下一步: .venv/bin/python scripts/02_preprocess_data.py"
    )
