import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# 尝试导入 adjustText
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

def plot_embeddings(embeddings, words, save_dir, filename="embedding_tsne.png"):
    """
    绘制 t-SNE 词向量图
    Args:
        embeddings: 词向量矩阵 (numpy array)，形状为 (num_words, embedding_dim)
        words: 词列表，对应 embeddings 的顺序
        save_dir: 保存图像的目录
        filename: 保存的文件名
    """
    print(f"Running t-SNE on {len(words)} words...")
    
    # 降维
    try:
        perp = min(30, len(words) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        print(f"t-SNE failed: {e}")
        return

    # 绘图
    plt.figure(figsize=(16, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='steelblue', alpha=0.6, s=40)

    # 标注文字
    texts_to_plot = []
    for i, word in enumerate(words):
        t = plt.text(
            embeddings_2d[i, 0], 
            embeddings_2d[i, 1], 
            word, 
            fontsize=9, 
            color='darkslategray'
        )
        texts_to_plot.append(t)

    # 自动避让
    if HAS_ADJUST_TEXT:
        print("Adjusting text positions...")
        adjust_text(texts_to_plot, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    plt.title(f"Word Embeddings t-SNE (Top {len(words)} words)", fontsize=16)
    plt.grid(True, alpha=0.2)
    
    # 保存
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()