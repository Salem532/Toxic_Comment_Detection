import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    """
    模型训练器：封装训练与验证逻辑
    Args:
        model: 待训练的PyTorch模型
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备（CPU或GPU）
        save_dir: 模型保存目录
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        """
        执行单个训练周期
        Args:
            epoch: 当前训练周期数
        Returns:
            avg_loss: 平均训练损失
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in progress_bar:
            inputs = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze(1) 
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                inputs = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 预测 > 0.5 为 1
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Validation Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # 保存最佳模型
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            save_path = f"{self.save_dir}/best_model.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
        return avg_loss
    
    