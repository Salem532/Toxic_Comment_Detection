import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils.model_utils import load_config, seed_everything
from src.data.tokenizer import TextTokenizer
from src.models.lstm import LSTMClassifier
import os
from adjustText import adjust_text

def visualize_embeddings(config_path="configs/config.yaml", top_k=200):
    """
    可视化词嵌入的t-SNE降维结果
    1. 加载配置和设置随机种子
    2. 重建Tokenizer（基于训练集，保证词ID与训练时一致）
    3. 加载训练好的模型
    4. 提取词嵌入并选择前top_k个词
    5. 使用t-SNE进行降维
    6. 绘制散点图并保存结果图片
    """
    print("=== 开始运行可视化脚本 ===")
    
    # 1. 初始化配置
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return
        
    config = load_config(config_path)
    seed_everything(42)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    
    # 2. 重建 Tokenizer
    data_path = config['data']['train_path']
    print(f"-> 正在读取数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到训练数据 {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        # 兼容不同的列名
        text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
        if text_col not in df.columns:
             print(f"❌ 错误: CSV中找不到文本列。")
             return
        texts = df[text_col].fillna("").values
        print(f"-> 成功读取 {len(texts)} 条文本数据")
    except Exception as e:
        print(f"❌ 读取CSV失败: {e}")
        return

    # 重建 Tokenizer
    print("-> 正在构建词表...")
    tokenizer = TextTokenizer(max_len=config['data']['max_len'], min_freq=config['data']['min_freq'])
    tokenizer.build_vocab(texts)
    
    if tokenizer.vocab_size <= 2:
        print("❌ 警告: 词表太小，无法可视化。")
        return
    
    # 3. 加载模型
    model_path = os.path.join(config['train']['model_save_dir'], "best_model.pth")
    print(f"-> 正在加载模型权重: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件。")
        return

    try:
        model = LSTMClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=config['model']['embed_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            bidirectional=config['model']['bidirectional'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 提取 Embedding
    embeddings = model.embedding.weight.data.cpu().numpy()
    
    # 筛选单词 (跳过 PAD=0, UNK=1)
    start_idx = 2
    end_idx = min(start_idx + top_k, tokenizer.vocab_size)
    
    selected_embeddings = embeddings[start_idx:end_idx]
    selected_words = [tokenizer.idx2word[i] for i in range(start_idx, end_idx)]
    
    print(f"-> 准备对 {len(selected_words)} 个单词进行 t-SNE 降维...")
    
    # 5. t-SNE 降维
    try:
        perp = min(30, len(selected_words) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(selected_embeddings)
    except Exception as e:
        print(f"❌ t-SNE 计算出错: {e}")
        return
    
    # 6. 绘图
    print("-> 正在绘图...")
    plt.figure(figsize=(16, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='steelblue', alpha=0.6, s=40)
    
    # 准备文字对象列表
    texts_to_plot = []
    for i, word in enumerate(selected_words):
        # 这里只创建文本对象，不直接画死
        t = plt.text(
            embeddings_2d[i, 0], 
            embeddings_2d[i, 1], 
            word, 
            fontsize=9,
            color='darkslategray'
        )
        texts_to_plot.append(t)
    
    # arrowprops 用于在文字被推得很远时画一条线指回来
    adjust_text(texts_to_plot, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        
    plt.title(f"Word Embeddings t-SNE (Top {len(selected_words)} words)", fontsize=16)
    plt.grid(True, alpha=0.2)
    
    output_path = os.path.join(config['train']['result_save_dir'], "embedding_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存到: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()