"""
日志工具模块

功能：
- 统一的日志格式（时间戳 + 级别 + 模块名 + 消息）
- 同时输出到控制台和日志文件
- 每次运行自动创建带时间戳的日志文件

使用方式：
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
"""

import os
import sys
import logging
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOGGERS = {}  # 缓存已创建的logger


def _setup_root_logger():
    """配置根logger，只调用一次"""
    if "root" in _LOGGERS:
        return _LOGGERS["root"]

    os.makedirs(_LOG_DIR, exist_ok=True)

    # 日志文件名带时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(_LOG_DIR, f"run_{timestamp}.log")

    # 格式：[时间] [级别] [模块名] 消息
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("genrec")
    root_logger.setLevel(logging.DEBUG)

    # 控制台处理器（INFO及以上）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # 文件处理器（DEBUG及以上）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    _LOGGERS["root"] = root_logger
    root_logger.info(f"日志文件: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取一个命名logger

    参数:
        name: 模块名，通常用 __name__

    返回:
        logging.Logger 实例
    """
    _setup_root_logger()
    # 使用简短的模块名
    short_name = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"genrec.{short_name}")
