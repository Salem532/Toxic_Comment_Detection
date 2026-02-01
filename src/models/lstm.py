import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM文本分类模型
    Args:
        vocab_size: 词表大小
        embed_dim: 词向量维度
        hidden_dim: LSTM隐藏层维度
        num_layers: LSTM层数
        bidirectional: 是否双向LSTM
        num_classes: 分类类别数
        dropout: Dropout比例
        embedding_matrix: 预训练的词向量矩阵（可选）
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, 
                 bidirectional, num_classes, dropout, embedding_matrix=None):
        super(LSTMClassifier, self).__init__() 
        
        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # 初始化 Embedding 层
        if embedding_matrix is not None:
            print("Loading pretrained embedding matrix...")
            self.embedding.weight.data.copy_(embedding_matrix) # 加载预训练的 Word2Vec 权重
            self.embedding.weight.requires_grad = False # 冻结 Word2Vec 权重
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim # 双向 LSTM 输出维度翻倍
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, num_classes) 
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, max_len]
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
         
        # 取最后一个时间步 (双向则拼接最后两个隐状态)
        if self.lstm.bidirectional:
            final_feature = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            final_feature = hidden[-1,:,:]
            
        logits = self.fc(final_feature)
        return self.sigmoid(logits)
    