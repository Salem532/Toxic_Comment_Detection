import torch
import argparse
import os
import sys
from src.utils.model_utils import load_config, seed_everything
from src.models.lstm import LSTMClassifier
from src.data.tokenizer import TextTokenizer

def predict_cli():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="恶意评论检测 - 单句预测")
    parser.add_argument("--text", type=str, required=True, help="输入要检测的英文句子")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        return
    config = load_config(args.config)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")

    # 加载 Tokenizer (关键步骤)
    tokenizer_path = os.path.join(config['train']['model_save_dir'], "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        print(f"❌ 错误: 找不到 Tokenizer 文件: {tokenizer_path}")
        print("   请先运行 'python train.py' 来生成并保存 Tokenizer。")
        return
    
    try:
        tokenizer = TextTokenizer.load(tokenizer_path)
        print(f"✅ 成功加载 Tokenizer (Vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        return

    # 加载模型
    model_path = os.path.join(config['train']['model_save_dir'], "best_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型权重: {model_path}")
        return

    # 这里的参数必须和 config.yaml 中训练时的一致
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

    # 5. 处理输入并预测
    text = args.text
    input_ids = tokenizer.convert_tokens_to_ids(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device) # [1, max_len]

    print("\n" + "="*40)
    print(f"📝 输入文本: \"{text}\"")
    print("="*40)

    with torch.no_grad():
        # 输出是概率值 (0~1)
        probs = model(input_tensor).squeeze().cpu().numpy()

    # 6. 格式化输出结果
    labels = ["Toxic (恶意)", "Severe Toxic (严重恶意)", "Obscene (淫秽)", 
              "Threat (威胁)", "Insult (侮辱)", "Identity Hate (身份仇恨)"]
    
    print("📊 预测结果:")
    print("-" * 40)
    for label, prob in zip(labels, probs):
        # 简单的进度条可视化
        bar_len = 20
        filled_len = int(bar_len * prob)
        bar = '█' * filled_len + '░' * (bar_len - filled_len)
        
        # 概率 > 50% 标红 (如果终端支持)
        prob_percent = prob * 100
        status = "⚠️ 检出" if prob > 0.5 else "✅ 安全"
        
        print(f"{label:<25} | {bar} | {prob_percent:5.1f}% | {status}")
    print("-" * 40)

    # 综合判断
    if any(probs > 0.5):
        print("\n🚫 结论: 这条评论包含恶意内容！")
    else:
        print("\n✨ 结论: 这是一条友善的评论。")

if __name__ == "__main__":
    predict_cli()