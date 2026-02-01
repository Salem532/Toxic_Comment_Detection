import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import os
from src.data import TextTokenizer, ToxicDataset
from src.models.lstm import LSTMClassifier

class ToxicEvaluator:
    """恶意评论检测模型评估器：封装完整评估流程，可复用"""
    def __init__(self, config, device):
        self.config = config  
        self.device = device  
        self.label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]  

    def load_and_clean_test_data(self):
        """加载并清洗测试集数据"""
        print("-> 正在处理测试集数据...")
        test_path = self.config['data']['test_path']
        test_labels_path = self.config['data']['test_labels_path']
        
        # 检查文件是否存在
        if not os.path.exists(test_path) or not os.path.exists(test_labels_path):
            raise FileNotFoundError("错误：找不到 test.csv 或 test_labels.csv")

        # 加载并合并文本与标签（通过id对齐）
        df_test = pd.read_csv(test_path)
        df_labels = pd.read_csv(test_labels_path)
        test_data = pd.merge(df_test, df_labels, on='id')
        
        # 剔除标签为-1的无效数据（Kaggle不参与评分的样本）
        test_data = test_data[test_data['toxic'] != -1]
        print(f"-> 清洗完成。有效测试样本数: {len(test_data)} (剔除了标记为-1的数据)")

        # 提取文本和标签（填充空值防止报错）
        test_texts = test_data['comment_text'].fillna("").values
        test_labels = test_data[self.label_cols].values
        
        return test_data['id'].values, test_texts, test_labels

    def build_tokenizer(self):
        """构建Tokenizer（基于训练集，保证词ID与训练时一致）"""
        print("-> 正在利用训练集重建词表 (Tokenizer)...")
        train_path = self.config['data']['train_path']
        if not os.path.exists(train_path):
            raise FileNotFoundError("错误：找不到训练集，无法构建词表。")
        
        train_df = pd.read_csv(train_path)
        train_texts = train_df['comment_text'].fillna("").values
        
        # 初始化并构建词表
        tokenizer = TextTokenizer(
            max_len=self.config['data']['max_len'],
            min_freq=self.config['data']['min_freq']
        )
        tokenizer.build_vocab(train_texts)
        print(f"-> 词表构建完成，词汇量: {tokenizer.vocab_size}")
        return tokenizer

    def load_model(self, tokenizer):
        """加载训练好的模型"""
        print("-> 正在加载模型权重...")
        model_path = os.path.join(self.config['train']['model_save_dir'], "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError("错误：找不到模型文件 best_model.pth，请先运行 train.py")

        # 初始化模型（参数与训练时一致）
        model = LSTMClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=self.config['model']['embed_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            bidirectional=self.config['model']['bidirectional'],
            num_classes=self.config['model']['num_classes'],
            dropout=self.config['model']['dropout']
        )
        
        # 加载权重并移至设备
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # 切换到评估模式（禁用Dropout）
        return model

    def predict(self, model, test_loader):
        """模型推理（批量预测，避免内存溢出）"""
        print("-> 开始推理...")
        all_preds = []
        all_labels = []

        # 禁用梯度计算，加速推理
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                inputs = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 模型输出（logits，未经过sigmoid）
                outputs = model(inputs)  # Shape: [batch_size, 6]
                
                # 保存结果（移至CPU并转为numpy）
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # 拼接所有批次结果
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        return all_preds, all_labels

    def calculate_metrics(self, all_labels, all_preds):
        """计算评估指标（AUC、准确率、Macro F1）"""
        metrics = {}
        print("\n" + "="*30)
        print("       测试集最终评估报告       ")
        print("="*30)
        
        # 1. ROC-AUC Score（多标签任务核心指标）
        try:
            auc_score = roc_auc_score(all_labels, all_preds, average='macro')
            metrics['auc'] = auc_score
            print(f"🏆 ROC-AUC Score: {auc_score:.5f}")
        except ValueError:
            metrics['auc'] = None
            print("⚠️ 警告: 无法计算 ROC-AUC (可能是某个类别的标签全为0)")

        # 2. 全局准确率（硬分类：概率>0.5视为1）
        hard_preds = (all_preds > 0.5).astype(int)
        acc = accuracy_score(all_labels.flatten(), hard_preds.flatten())
        metrics['accuracy'] = acc
        print(f"📊 Global Accuracy: {acc:.5f}")

        # 3. Macro F1 Score（多标签任务核心指标，平衡正负样本）
        f1 = f1_score(all_labels.flatten(), hard_preds.flatten(), average='macro')
        metrics['macro_f1'] = f1
        print(f"📉 Macro F1 Score : {f1:.5f}")
        
        print("-" * 30)
        return metrics

    def save_predictions(self, ids, all_preds):
        """保存预测结果到CSV文件"""
        # 构建结果DataFrame
        submission_df = pd.DataFrame({'id': ids})
        for i, col in enumerate(self.label_cols):
            submission_df[col] = all_preds[:, i]
        
        # 保存路径（从config读取，确保目录存在）
        save_dir = os.path.dirname(self.config['train']['result_save_dir'])
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "test_predictions.csv")
        
        submission_df.to_csv(save_file, index=False)
        print(f"✅ 详细预测结果已保存至: {save_file}")
        return save_file

    def run(self):
        """执行完整评估流程（主入口）"""
        try:
            # 构建Tokenizer
            tokenizer = self.build_tokenizer()
            
            # 加载并清洗测试集数据
            test_ids, test_texts, test_labels = self.load_and_clean_test_data()
            
            # 构建测试集DataLoader
            test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['train']['batch_size'],
                shuffle=False,
                num_workers=0  
            )
            
            # 加载模型
            model = self.load_model(tokenizer)
            
            # 批量预测
            all_preds, all_labels = self.predict(model, test_loader)
            
            # 计算指标
            metrics = self.calculate_metrics(all_labels, all_preds)
            
            # 保存结果
            self.save_predictions(test_ids, all_preds)
            
            print("\n=== 评估完成 ===")
            return metrics
        
        except Exception as e:
            print(f"\n❌ 评估失败：{str(e)}")
            raise  # 抛出异常，方便调试