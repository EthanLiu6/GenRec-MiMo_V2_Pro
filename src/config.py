"""
配置加载工具模块
"""

import os
import yaml


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path=None) -> dict:
    """加载YAML配置文件，返回字典"""
    if config_path is None:
        config_path = os.path.join(_PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_project_root() -> str:
    """获取项目根目录"""
    return _PROJECT_ROOT


def get_abs_path(relative_path: str) -> str:
    """将相对于项目根目录的路径转为绝对路径"""
    return os.path.join(_PROJECT_ROOT, relative_path)
