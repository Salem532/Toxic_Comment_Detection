import logging
import sys
import os

def setup_logger(save_dir, log_filename="train.log"):
    """
    初始化日志系统
    Args:
        save_dir: 日志保存目录
        log_filename: 日志文件名
    Returns:
        配置好的 logger 对象
    """
    # 确保目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_filepath = os.path.join(save_dir, log_filename)

    # 获取 logger 对象
    logger = logging.getLogger("Toxic_Detection") # 给个名字
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler (Jupyter Notebook 或 多次调用时常见问题)
    if logger.handlers:
        return logger

    # 设置格式 (时间 - 级别 - 消息)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler 1: 输出到文件
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Handler 2: 输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # 添加 Handler
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger