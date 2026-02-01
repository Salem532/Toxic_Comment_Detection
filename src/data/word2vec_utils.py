import numpy as np
import torch
from gensim.models import Word2Vec
from .data_clean import clean_text  # 复用文本清洗函数

def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    """
    训练Word2Vec词向量模型（基于清洗后的文本）
    Args:
        texts: 原始文本列表（list[str]）
        vector_size: 词向量维度（默认100，NLP基础任务黄金值）
        window: 上下文窗口大小（默认5，即考虑前后5个词）
        min_count: 最小词频（默认2，过滤低频词）
    Returns:
        训练好的Word2Vec模型实例
    """
    print("Starting Word2Vec training...")
    # 1. 对所有文本进行清洗+分词（复用data_clean的清洗逻辑，保证一致性）
    tokenized_texts = [clean_text(text) for text in texts]
    # 2. 训练Word2Vec（Skip-gram模式，适合中小规模数据集）
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4 
    )
    print(f"Word2Vec training completed! Vocab size in W2V: {len(w2v_model.wv)}")
    return w2v_model

def get_embedding_matrix(word2idx, w2v_model, embedding_dim):
    """
    构建嵌入矩阵：用于初始化PyTorch的Embedding层
    Args:
        word2idx: Tokenizer的词→ID映射字典（dict[str, int]）
        w2v_model: 训练好的Word2Vec模型实例
        embedding_dim: 词向量维度（需与Word2Vec的vector_size一致）
    Returns:
        嵌入矩阵（torch.Tensor），shape=(vocab_size, embedding_dim)
    """
    vocab_size = len(word2idx)
    # 初始化嵌入矩阵：全0（PAD词向量默认全0）
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # 遍历词表，用Word2Vec词向量填充矩阵
    for word, idx in word2idx.items():
        if word in w2v_model.wv:
            # 词表中的词在Word2Vec中存在：用训练好的词向量
            embedding_matrix[idx] = w2v_model.wv[word]
        else:
            # 词表中的词在Word2Vec中不存在：
            if word == "<PAD>":
                # PAD符号：保持全0
                embedding_matrix[idx] = np.zeros(embedding_dim)
            else:
                # 其他词（如UNK、低频词）：随机初始化（服从正态分布）
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    # 转换为PyTorch Tensor（适配Embedding层输入）
    return torch.tensor(embedding_matrix, dtype=torch.float32)
