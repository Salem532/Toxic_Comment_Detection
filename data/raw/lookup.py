import pandas as pd
import numpy as np

# 加载数据
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_labels_df = pd.read_csv("test_labels.csv")

# 注意：检查缺失值，直接删除或填充
train_df = train_df.dropna(subset=['comment_text'])
test_df = test_df.dropna(subset=['comment_text'])

# 查看标签分布（了解样本不均衡情况）
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print(train_df[label_cols].sum())


