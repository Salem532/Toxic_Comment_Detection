import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    """
    恶意评论检测专用Dataset：将文本ID序列和标签封装为PyTorch可加载的格式
    Args:
        texts: 文本列表（list[str]）
        labels: 标签列表（list[list[float]]），shape=(n_samples, 6)
        tokenizer: TextTokenizer实例（已构建好词表）
    """
    def __init__(self, texts, labels, tokenizer):
        """
        Args:
            texts: 文本列表（list[str]）
            labels: 标签列表（list[list[float]]），shape=(n_samples, 6)
            tokenizer: TextTokenizer实例（已构建好词表）
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.texts)

    def __getitem__(self, idx):
        """按索引获取单样本：文本→ID序列，标签→Tensor"""
        # 获取单条文本和标签
        text = self.texts[idx]
        label = self.labels[idx]

        # 文本→ID序列（调用Tokenizer的转换方法）
        input_ids = self.tokenizer.convert_tokens_to_ids(text)

        # 转换为PyTorch Tensor（匹配模型输入格式）
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 文本ID序列：(max_len,)
            "label": torch.tensor(label, dtype=torch.float32)       # 标签：(6,)
        }
