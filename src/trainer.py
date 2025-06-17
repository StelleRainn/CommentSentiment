import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_scheduler, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from src.model import FocalLoss

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳权重"""
        self.best_weights = model.state_dict().copy()

class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int], 
                         labels: Optional[List[str]] = None) -> Dict[str, float]:
        """计算各种评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true: List[int], y_pred: List[int], 
                                  target_names: Optional[List[str]] = None):
        """打印分类报告"""
        print("\n=== 分类报告 ===")
        print(classification_report(y_true, y_pred, target_names=target_names))
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                            target_names: Optional[List[str]] = None, 
                            save_path: Optional[str] = None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names or range(len(cm)),
                   yticklabels=target_names or range(len(cm)))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class SentimentTrainer:
    """情感分析训练器"""
    
    def __init__(self, model: nn.Module, config: Config, use_focal_loss: bool = True):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # 将模型移到设备
        self.model.to(self.device)
        
        # 损失函数
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器（稍后在训练时初始化）
        self.scheduler = None
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
                loss, logits = self.model(input_ids, attention_mask, labels)
            else:
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader) -> Tuple[float, float, float, List[int], List[int]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                if hasattr(self.model, 'forward') and 'labels' in self.model.forward.__code__.co_varnames:
                    loss, logits = self.model(input_ids, attention_mask, labels)
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_metrics(all_labels, all_predictions)
        accuracy = metrics['accuracy']
        f1_score = metrics['f1']
        
        return avg_loss, accuracy, f1_score, all_labels, all_predictions
    
    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None) -> Dict:
        """完整训练流程"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        
        # 初始化学习率调度器
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"总训练步数: {total_steps}")
        
        best_f1 = 0
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc, val_f1, val_labels, val_predictions = self.validate_epoch(val_loader)
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['learning_rates'].append(current_lr)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 验证F1: {val_f1:.4f}")
            print(f"学习率: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.save_model(f"best_model_epoch_{epoch+1}_f1_{val_f1:.4f}.pth")
                print(f"保存最佳模型，F1: {val_f1:.4f}")
            
            # 早停检查
            if self.early_stopping(val_f1, self.model):
                print(f"早停触发，在epoch {epoch + 1}")
                break
        
        # 最终评估
        print("\n=== 最终评估 ===")
        final_val_loss, final_val_acc, final_val_f1, final_labels, final_predictions = self.validate_epoch(val_loader)
        
        # 打印详细报告
        target_names = [self.config.ID_TO_LABEL[i] for i in range(self.config.NUM_CLASSES)]
        self.metrics_calculator.print_classification_report(final_labels, final_predictions, target_names)
        
        # 绘制混淆矩阵
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        confusion_matrix_path = os.path.join(self.config.LOG_PATH, "confusion_matrix.png")
        self.metrics_calculator.plot_confusion_matrix(
            final_labels, final_predictions, target_names, confusion_matrix_path
        )
        
        # 保存训练历史
        self.save_training_history()
        
        return {
            'best_f1': best_f1,
            'final_f1': final_val_f1,
            'final_accuracy': final_val_acc,
            'train_history': self.train_history
        }
    
    def save_model(self, filename: str):
        """保存模型"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        filepath = os.path.join(self.config.MODEL_SAVE_PATH, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_history': self.train_history
        }, filepath)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint.get('train_history', {})
        
        print(f"模型已从 {filepath} 加载")
    
    def save_training_history(self):
        """保存训练历史"""
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        history_path = os.path.join(self.config.LOG_PATH, "training_history.json")
        
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.train_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        axes[1, 0].plot(self.train_history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_title('F1 Score Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(self.train_history['learning_rates'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图片
        curves_path = os.path.join(self.config.LOG_PATH, "training_curves.png")
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练曲线已保存到: {curves_path}")

if __name__ == "__main__":
    print("训练器模块已创建！")