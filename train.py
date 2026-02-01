import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os 
from src.utils import load_config, seed_everything, ensure_dir, setup_logger
from src.data import TextTokenizer, train_word2vec, get_embedding_matrix, ToxicDataset
from src.models.lstm import LSTMClassifier
from src.trainer import Trainer

def main():
    # 加载配置
    config = load_config("configs/config.yaml")
    seed_everything(42)
    
    # 建立目录
    ensure_dir(config['train']['model_save_dir'])
    ensure_dir(config['train']['log_save_dir'])
    ensure_dir(config['train']['result_save_dir'])

    logger = setup_logger(config['train']['log_save_dir'])
    logger.info("="*30)
    logger.info("🚀 Project: Toxic Comment Detection Started")
    logger.info("="*30)
    
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"✅ Using device: {device}")

    # 加载数据
    logger.info("📂 Loading data...")
    try:
        df = pd.read_csv(config['data']['train_path'])
    except FileNotFoundError:
        logger.error(f"❌ Data not found at {config['data']['train_path']}") # 使用 error 级别
        return
    
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    texts = df['comment_text'].fillna('').values
    labels = df[label_cols].values 

    # 构建 Tokenizer
    logger.info("🔨 Building Tokenizer...")
    tokenizer = TextTokenizer(max_len=config['data']['max_len'], min_freq=config['data']['min_freq'])
    tokenizer.build_vocab(texts)
    
    # 保存 Tokenizer
    tokenizer_path = os.path.join(config['train']['model_save_dir'], "tokenizer.pkl")
    tokenizer.save(tokenizer_path)
    logger.info(f"💾 Tokenizer saved to {tokenizer_path}")
    
    # 训练 Word2Vec
    logger.info("🧠 Training Word2Vec embeddings...")
    w2v_model = train_word2vec(texts, vector_size=config['model']['embed_dim'])
    embedding_matrix = get_embedding_matrix(tokenizer.word2idx, w2v_model, config['model']['embed_dim'])

    # 数据划分
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config['data']['test_size'], random_state=42
    )

    train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # 模型构建
    logger.info("🏗️ Building LSTM Model...")
    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['model']['embed_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        bidirectional=config['model']['bidirectional'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        embedding_matrix=embedding_matrix
    )
    model.to(device)

    # 训练准备
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    criterion = nn.BCELoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['train']['model_save_dir']
    )

    # 7. 开始训练
    logger.info("🔥 Start Training...")
    for epoch in range(1, config['train']['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.evaluate(epoch)
        
        # 记录每轮的核心指标
        logger.info(f"Epoch {epoch}/{config['train']['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    logger.info("✨ Training Finished!")

if __name__ == "__main__":
    main()