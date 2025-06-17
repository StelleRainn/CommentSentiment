# -*- coding: utf-8 -*-
"""
改进的训练脚本 - 解决过拟合问题，支持整合数据集

改进点：
1. 更强的正则化
2. 学习率调度
3. 早停机制优化
4. 数据增强
5. 模型集成准备
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import TextPreprocessor, SentimentDataset
from model import create_model
from config.config import Config

class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # 早停相关
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
        print(f"使用设备: {self.device}")
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        print("初始化模型和分词器...")
        
        # 创建分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # 创建模型
        self.model = create_model(
            config=self.config,
            model_type='bert'
        )
        
        self.model.to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data_loaders(self, train_data_path, val_data_path=None, use_augmentation=True):
        """设置数据加载器"""
        print("加载和预处理数据...")
        
        # 读取训练数据
        if train_data_path.endswith('.csv'):
            train_df = pd.read_csv(train_data_path)
        else:
            train_df = pd.read_excel(train_data_path)
        
        print(f"训练数据大小: {len(train_df)}")
        
        # 如果没有提供验证数据，从训练数据中分割
        if val_data_path is None:
            train_df, val_df = train_test_split(
                train_df, test_size=0.2, random_state=42,
                stratify=train_df['Sentiment']
            )
            print(f"自动分割 - 训练集: {len(train_df)}, 验证集: {len(val_df)}")
        else:
            if val_data_path.endswith('.csv'):
                val_df = pd.read_csv(val_data_path)
            else:
                val_df = pd.read_excel(val_data_path)
            print(f"使用提供的验证集: {len(val_df)}")
        
        # 文本预处理
        preprocessor = TextPreprocessor()
        
        # 处理训练数据
        train_texts = []
        train_labels = []
        
        for _, row in train_df.iterrows():
            text = preprocessor.enhanced_clean_text(row['CommentText'])
            if len(text.strip()) > 0:
                train_texts.append(text)
                
                # 标签映射
                sentiment = str(row['Sentiment']).lower().strip()
                if sentiment == 'negative':
                    train_labels.append(0)
                elif sentiment == 'neutral':
                    train_labels.append(1)
                elif sentiment == 'positive':
                    train_labels.append(2)
                else:
                    continue
        
        # 数据增强（可选）
        if use_augmentation and len(train_texts) < 10000:
            print("应用数据增强...")
            augmented_texts, augmented_labels = self.apply_data_augmentation(
                train_texts, train_labels, augment_ratio=0.3
            )
            train_texts.extend(augmented_texts)
            train_labels.extend(augmented_labels)
            print(f"增强后训练数据大小: {len(train_texts)}")
        
        # 处理验证数据
        val_texts = []
        val_labels = []
        
        for _, row in val_df.iterrows():
            text = preprocessor.enhanced_clean_text(row['CommentText'])
            if len(text.strip()) > 0:
                val_texts.append(text)
                
                sentiment = str(row['Sentiment']).lower().strip()
                if sentiment == 'negative':
                    val_labels.append(0)
                elif sentiment == 'neutral':
                    val_labels.append(1)
                elif sentiment == 'positive':
                    val_labels.append(2)
                else:
                    continue
        
        print(f"最终数据大小 - 训练: {len(train_texts)}, 验证: {len(val_texts)}")
        
        # 创建数据集
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True  # 避免最后一个batch大小不一致
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # 打印类别分布
        train_label_counts = np.bincount(train_labels)
        val_label_counts = np.bincount(val_labels)
        
        print("\n训练集类别分布:")
        for i, count in enumerate(train_label_counts):
            print(f"  {['Negative', 'Neutral', 'Positive'][i]}: {count} ({count/len(train_labels)*100:.1f}%)")
        
        print("\n验证集类别分布:")
        for i, count in enumerate(val_label_counts):
            print(f"  {['Negative', 'Neutral', 'Positive'][i]}: {count} ({count/len(val_labels)*100:.1f}%)")
    
    def apply_data_augmentation(self, texts, labels, augment_ratio=0.3):
        """应用数据增强"""
        augmented_texts = []
        augmented_labels = []
        
        num_to_augment = int(len(texts) * augment_ratio)
        indices = np.random.choice(len(texts), num_to_augment, replace=True)
        
        for idx in indices:
            original_text = texts[idx]
            label = labels[idx]
            
            # 简单的数据增强：随机删除词汇
            words = original_text.split()
            if len(words) > 3:
                # 随机删除1-2个词
                num_to_remove = min(2, len(words) // 4)
                indices_to_remove = np.random.choice(
                    len(words), num_to_remove, replace=False
                )
                augmented_words = [w for i, w in enumerate(words) if i not in indices_to_remove]
                augmented_text = ' '.join(augmented_words)
                
                if len(augmented_text.strip()) > 0:
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def setup_training_components(self):
        """设置训练组件"""
        print("设置优化器和学习率调度器...")
        
        # 使用权重衰减减少过拟合
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,  # L2正则化
            eps=1e-8
        )
        
        # 计算总训练步数
        total_steps = len(self.train_loader) * self.config.EPOCHS
        warmup_steps = int(0.1 * total_steps)  # 10%的步数用于warmup
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 使用标签平滑的交叉熵损失
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            
            if isinstance(outputs, tuple):
                loss = outputs[0]
                logits = outputs[1]
            else:
                logits = outputs
                loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # 收集预测结果
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 打印进度
            if batch_idx % 50 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{self.config.EPOCHS}, "
                      f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / len(self.train_loader)
        train_f1 = f1_score(all_labels, all_predictions, average='weighted')
        current_lr = self.scheduler.get_last_lr()[0]
        
        return avg_loss, train_f1, current_lr
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return avg_loss, val_f1, all_predictions, all_labels
    
    def save_model(self, epoch, val_f1, model_type='checkpoint'):
        """保存模型"""
        os.makedirs('models', exist_ok=True)
        
        if model_type == 'best':
            filename = f"models/best_model_epoch_{epoch+1}_f1_{val_f1:.4f}.pth"
        else:
            filename = f"models/final_model_bert_f1_{val_f1:.4f}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_f1': val_f1,
            'config': self.config.__dict__
        }, filename)
        
        print(f"模型已保存: {filename}")
        return filename
    
    def train(self, train_data_path, val_data_path=None):
        """完整训练流程"""
        print("开始训练...")
        
        # 设置模型和数据
        self.setup_model_and_tokenizer()
        self.setup_data_loaders(train_data_path, val_data_path)
        self.setup_training_components()
        
        # 训练循环
        for epoch in range(self.config.EPOCHS):
            print(f"\n=== Epoch {epoch+1}/{self.config.EPOCHS} ===")
            
            # 训练
            train_loss, train_f1, current_lr = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_f1, val_predictions, val_labels = self.validate()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_f1'].append(train_f1)
            self.train_history['val_f1'].append(val_f1)
            self.train_history['learning_rates'].append(current_lr)
            
            print(f"训练损失: {train_loss:.4f}, 训练F1: {train_f1:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证F1: {val_f1:.4f}")
            print(f"学习率: {current_lr:.2e}")
            
            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                best_model_path = self.save_model(epoch, val_f1, 'best')
                print(f"新的最佳模型! F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"验证F1未提升，耐心计数: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
            
            # 早停检查
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n早停触发! 最佳验证F1: {self.best_val_f1:.4f}")
                break
        
        # 保存最终模型
        final_model_path = self.save_model(epoch, val_f1, 'final')
        
        # 保存训练历史
        self.save_training_history()
        
        # 生成最终报告
        self.generate_final_report(val_predictions, val_labels)
        
        print(f"\n训练完成!")
        print(f"最佳验证F1: {self.best_val_f1:.4f}")
        print(f"最终验证F1: {val_f1:.4f}")
        
        return best_model_path, final_model_path
    
    def save_training_history(self):
        """保存训练历史"""
        os.makedirs('logs', exist_ok=True)
        
        with open('logs/training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # F1分数曲线
        ax2.plot(epochs, self.train_history['train_f1'], 'b-', label='训练F1')
        ax2.plot(epochs, self.train_history['val_f1'], 'r-', label='验证F1')
        ax2.set_title('训练和验证F1分数')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1分数')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.train_history['learning_rates'], 'g-')
        ax3.set_title('学习率变化')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('学习率')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 过拟合检测
        overfitting = np.array(self.train_history['train_f1']) - np.array(self.train_history['val_f1'])
        ax4.plot(epochs, overfitting, 'purple', label='过拟合程度')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('过拟合检测 (训练F1 - 验证F1)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1差值')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("训练曲线已保存到 logs/training_curves.png")
    
    def generate_final_report(self, predictions, labels):
        """生成最终报告"""
        # 分类报告
        target_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(labels, predictions, target_names=target_names)
        
        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"results/training_results_{timestamp}.txt"
        
        os.makedirs('results', exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"训练完成时间: {datetime.now()}\n")
            f.write(f"最佳验证F1: {self.best_val_f1:.4f}\n")
            f.write(f"最终验证F1: {f1_score(labels, predictions, average='weighted'):.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
            f.write("\n\n混淆矩阵:\n")
            f.write(str(cm))
        
        print(f"训练报告已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='改进的BERT情感分析训练')
    parser.add_argument('--train_data', type=str, default='data/integrated_train_s.csv', 
                       help='训练数据路径')
    parser.add_argument('--val_data', type=str, default='data/integrated_val_s.csv', 
                       help='验证数据路径')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # 创建训练器
    trainer = ImprovedTrainer(config)
    
    # 开始训练
    try:
        best_model, final_model = trainer.train(args.train_data, args.val_data)
        print(f"\n训练成功完成!")
        print(f"最佳模型: {best_model}")
        print(f"最终模型: {final_model}")
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()