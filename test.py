#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 使用训练好的模型预测数据集情感

功能：
1. 加载训练好的模型
2. 读取验证数据集
3. 进行情感预测
4. 输出结果到txt文件
"""

import os
import sys
import torch
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import TextPreprocessor, DataLoader as DataPreprocessor, SentimentDataset
from model import create_model
from config.config import Config

class SentimentTester:
    """情感分析测试器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径
            device: 设备类型
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        
        print(f"使用设备: {self.device}")
        
    def load_model(self, model_type='bert'):
        """
        加载训练好的模型
        
        Args:
            model_type: 模型类型 ('bert' 或 'bilstm')
        """
        print(f"正在加载模型: {self.model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建配置实例
        config_instance = Config()
        
        # 创建模型
        self.model = create_model(config_instance, model_type=model_type)
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化预处理器
        self.preprocessor = TextPreprocessor()
        
        print("模型加载完成！")
    
    def predict_dataset(self, data_path, output_path=None, batch_size=32):
        """
        预测数据集
        
        Args:
            data_path: 数据文件路径
            output_path: 输出文件路径
            batch_size: 批处理大小
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        print(f"正在读取数据: {data_path}")
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 读取数据
        try:
            df = pd.read_excel(data_path)
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            print("尝试读取CSV文件...")
            df = pd.read_csv(data_path)
        
        print(f"数据集大小: {len(df)}")
        
        # 检查必要的列
        if 'CommentText' not in df.columns:
            raise ValueError("数据文件必须包含'CommentText'列")
        
        # 检查是否有真实标签用于计算F1分数
        has_true_labels = 'Sentiment' in df.columns
        true_labels = None
        if has_true_labels:
            print("检测到真实标签，将计算F1分数")
            # 增强的标签预处理：统一标签格式
            def normalize_label(label):
                """标签标准化函数"""
                if pd.isna(label) or label == '':
                    return None
                
                label_str = str(label).strip().lower()
                
                # 处理各种可能的标签格式
                if label_str in ['negative', 'neg', '0', 'bad', 'poor']:
                    return 'negative'
                elif label_str in ['neutral', 'neu', '1', 'ok', 'average']:
                    return 'neutral'
                elif label_str in ['positive', 'pos', '2', 'good', 'great', 'excellent']:
                    return 'positive'
                else:
                    return None
            
            # 应用标签标准化
            normalized_labels = df['Sentiment'].apply(normalize_label)
            
            # 将标准化后的标签映射为数字
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            true_labels = normalized_labels.map(label_map).tolist()
            
            # 检查是否有无效标签
            valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
            if len(valid_indices) < len(true_labels):
                invalid_count = len(true_labels) - len(valid_indices)
                print(f"警告: 发现 {invalid_count} 个无效标签，将被排除在F1计算之外")
                print(f"有效标签数量: {len(valid_indices)}/{len(true_labels)}")
                has_true_labels = len(valid_indices) > 0
        
        # 增强的数据预处理
        print("正在进行数据清洗...")
        
        # 创建文本预处理器
        text_preprocessor = TextPreprocessor()
        
        # 应用增强的文本清洗
        raw_comments = df['CommentText'].fillna('').astype(str).tolist()
        comments = []
        
        for comment in raw_comments:
            # 应用增强的清洗功能
            cleaned_comment = text_preprocessor.enhanced_clean_text(comment)
            comments.append(cleaned_comment)
        
        print(f"数据清洗完成，处理了 {len(comments)} 条评论")
        
        # 创建数据加载器（用于预测，不需要标签）
        print("正在预处理数据...")
        
        # 创建临时的标签（预测时不使用）
        temp_labels = [0] * len(comments)
        
        # 创建配置实例
        config_instance = Config()
        
        # 创建tokenizer
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(config_instance.MODEL_NAME)
        
        # 创建测试数据集
        test_dataset = SentimentDataset(
            comments, temp_labels, tokenizer, config_instance.MAX_LENGTH
        )
        
        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 进行预测
        print("正在进行预测...")
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # 获取预测结果
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        # 映射预测结果到情感标签
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_labels = [sentiment_map[pred] for pred in predictions]
        
        # 生成输出文件路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"prediction_results_{timestamp}.txt"
        
        # 保存结果
        print(f"正在保存结果到: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sentiment in enumerate(sentiment_labels, 1):
                f.write(f"{i} {sentiment}\n")
        
        print(f"预测完成！结果已保存到: {output_path}")
        
        # 打印统计信息
        sentiment_counts = pd.Series(sentiment_labels).value_counts()
        print("\n预测结果统计:")
        for sentiment, count in sentiment_counts.items():
            percentage = count / len(sentiment_labels) * 100
            print(f"{sentiment}: {count} ({percentage:.1f}%)")
        
        # 如果有真实标签，计算F1分数
        if has_true_labels and true_labels is not None:
            from sklearn.metrics import f1_score, classification_report
            
            # 将预测标签映射为数字
            pred_labels_numeric = [{'negative': 0, 'neutral': 1, 'positive': 2}[label] for label in sentiment_labels]
            
            # 过滤出有效标签的数据
            valid_indices = [i for i, label in enumerate(true_labels) if label is not None]
            if valid_indices:
                valid_true_labels = [true_labels[i] for i in valid_indices]
                valid_pred_labels = [pred_labels_numeric[i] for i in valid_indices]
                
                # 计算F1分数
                f1_macro = f1_score(valid_true_labels, valid_pred_labels, average='macro')
                f1_weighted = f1_score(valid_true_labels, valid_pred_labels, average='weighted')
                
                print(f"\nF1分数评估 (基于 {len(valid_indices)} 个有效样本):")
                print(f"Macro F1: {f1_macro:.4f}")
                print(f"Weighted F1: {f1_weighted:.4f}")
                
                # 打印详细的分类报告
                print("\n详细分类报告:")
                target_names = ['negative', 'neutral', 'positive']
                print(classification_report(valid_true_labels, valid_pred_labels, target_names=target_names))
            else:
                print("\n警告: 没有有效的真实标签，无法计算F1分数")
        
        return output_path, sentiment_labels, true_labels

def find_best_model():
    """
    自动查找最佳模型文件
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return None
    
    # 查找所有模型文件
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            model_files.append(os.path.join(models_dir, file))
    
    if not model_files:
        return None
    
    # 优先选择best_model，然后是final_model
    best_models = [f for f in model_files if 'best_model' in f]
    if best_models:
        # 按F1分数排序，选择最高的
        best_models.sort(key=lambda x: float(x.split('_f1_')[1].split('.pth')[0]) if '_f1_' in x else 0, reverse=True)
        return best_models[0]
    
    final_models = [f for f in model_files if 'final_model' in f]
    if final_models:
        final_models.sort(key=lambda x: float(x.split('_f1_')[1].split('.pth')[0]) if '_f1_' in x else 0, reverse=True)
        return final_models[0]
    
    # 如果都没有，返回最新的模型文件
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files[0]

def main():
    parser = argparse.ArgumentParser(description='情感分析模型测试')
    parser.add_argument('--model_path', type=str, default='models/best_model_epoch_2_f1_0.7327.pth', help='模型文件路径')
    parser.add_argument('--data_path', type=str, default='data/integrated_val_s.csv', help='测试数据路径')
    parser.add_argument('--output_path', type=str, help='输出文件路径')
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'bilstm'], help='模型类型')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    
    args = parser.parse_args()
    
    try:
        # 自动查找模型文件（如果未指定）
        if args.model_path is None:
            print("未指定模型路径，正在自动查找最佳模型...")
            args.model_path = find_best_model()
            if args.model_path is None:
                print("错误：未找到任何模型文件！")
                print("请确保models目录下有训练好的模型文件。")
                return
            print(f"找到模型: {args.model_path}")
        
        # 创建测试器
        tester = SentimentTester(args.model_path)
        
        # 加载模型
        tester.load_model(args.model_type)
        
        # 进行预测
        output_path, predictions, true_labels = tester.predict_dataset(
            data_path=args.data_path,
            output_path=args.output_path,
            batch_size=args.batch_size
        )
        
        print(f"\n测试完成！")
        print(f"模型: {args.model_path}")
        print(f"数据: {args.data_path}")
        print(f"结果: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()