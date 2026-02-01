import torch
import numpy as np
import random
import yaml
import os

def seed_everything(seed=42):
    """固定所有随机种子，保证实验复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 牺牲一点点速度换取确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        raise ValueError("Config file is empty!")
    return config

def ensure_dir(dir_path):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def count_parameters(model):
    """计算模型可训练参数量 (展示在 Log 里显得很专业)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, epoch, save_path):
    """保存检查点 (比单纯 save_dict 更全面)"""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, save_path)