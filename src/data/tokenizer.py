import torch
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
import pickle
import os
from .data_clean import clean_text

class TextTokenizer:
    """
    分词器核心类：负责词表构建、文本→ID序列转换、Tokenizer保存与加载
    Args:
        max_len: 文本最大长度（超过截断，未满填充）
        min_freq: 词表最小词频（低于忽略）
    """
    def __init__(self, max_len=200, min_freq=2):
        self.max_len = max_len
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, texts):
        """
        基于输入文本构建词表
        Args:
            texts: 文本列表（list[str]），如 [ "I hate you", "You are stupid" ]
        """
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            # 调用外部的清洗函数
            words = clean_text(text)
            counter.update(words)
        
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        print(f"Vocab size: {self.vocab_size}")

    def convert_tokens_to_ids(self, text):
        """
        文本→ID序列：清洗→分词→ID转换→截断/填充
        Args:
            text: 单条原始文本（str）
        Returns:
            固定长度的ID序列（list[int]）
        """
        words = clean_text(text)
        ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words] 
        
        # 截断或填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.word2idx["<PAD>"]] * (self.max_len - len(ids))
        return ids
    
    def save(self, file_path):
        """
        保存Tokenizer到本地（pickle格式，方便后续复用词表）
        Args:
            file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Tokenizer saved to {file_path}")

    @staticmethod
    def load(file_path):
        """
        从本地加载Tokenizer（无需重新构建词表）
        Args:
            load_path: 加载路径
        Returns:
            加载后的TextTokenizer实例
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

