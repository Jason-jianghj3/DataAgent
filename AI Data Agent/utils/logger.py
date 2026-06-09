"""
日志工具模块
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "report_summary", log_dir: str = "logs") -> logging.Logger:
    """
    创建并配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件存放目录
        
    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 控制台输出格式
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 文件输出格式
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        log_path / f"{name}_{today}.log",
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-5s [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
