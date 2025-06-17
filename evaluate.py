#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube评论情感分析 - 模型评估和预测脚本

用于评估训练好的模型并进行单条预测
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from src.data_preprocessing import TextPreprocessor
from src.model import create_model
from transformers import BertTokenizer

class SentimentPredictor:
    """情感预测器"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # 加载模型
        self.model = create_model(config, 'bert')
        self.load_model(model_path)
        self.model.eval()
        
        # 加载tokenizer和预处理器
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.preprocessor = TextPreprocessor()
        
        print(f"模型已加载: {model_path}")
        print(f"设备: {self.device}")
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"模型权重已加载")
    
    def preprocess_text(self, comment_text: str, video_title: str = "") -> Dict[str, torch.Tensor]:
        """预处理单条文本"""
        # 合并评论和视频标题
        if video_title:
            combined_text = f"{comment_text} [SEP] {video_title}"
        else:
            combined_text = comment_text
        
        # 清洗文本
        cleaned_text = self.preprocessor.clean_text(combined_text)
        
        # 编码
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict_single(self, comment_text: str, video_title: str = "") -> Dict[str, float]:
        """预测单条评论的情感"""
        # 预处理
        inputs = self.preprocess_text(comment_text, video_title)
        
        # 预测
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # 转换为标签
        predicted_label = self.config.ID_TO_LABEL[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        # 所有类别的概率
        all_probs = {}
        for i, prob in enumerate(probabilities[0]):
            label = self.config.ID_TO_LABEL[i]
            all_probs[label] = prob.item()
        
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': all_probs,
            'predicted_class': predicted_class
        }
    
    def predict_batch(self, texts: List[str], video_titles: List[str] = None) -> List[Dict[str, float]]:
        """批量预测"""
        if video_titles is None:
            video_titles = [""] * len(texts)
        
        results = []
        for text, title in zip(texts, video_titles):
            result = self.predict_single(text, title)
            results.append(result)
        
        return results
    
    def evaluate_on_data(self, data_path: str) -> Dict[str, float]:
        """在数据集上评估模型"""
        # 加载数据
        data = pd.read_excel(data_path)
        
        # 预测
        predictions = []
        true_labels = []
        
        print(f"正在评估 {len(data)} 条数据...")
        
        for idx, row in data.iterrows():
            if idx % 100 == 0:
                print(f"进度: {idx}/{len(data)}")
            
            result = self.predict_single(row['CommentText'], row['VideoTitle'])
            predictions.append(result['predicted_label'])
            true_labels.append(row['Sentiment'])
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        print("\n=== 评估结果 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        print("\n=== 详细分类报告 ===")
        target_names = list(self.config.LABEL_MAP.keys())
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }



def batch_prediction_from_file(predictor: SentimentPredictor, input_file: str, output_file: str):
    """从文件批量预测"""
    print(f"\n从文件批量预测: {input_file}")
    
    # 读取数据
    if input_file.endswith('.xlsx'):
        data = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    else:
        raise ValueError("不支持的文件格式，请使用.xlsx或.csv")
    
    # 检查必要列
    if 'CommentText' not in data.columns:
        raise ValueError("输入文件必须包含'CommentText'列")
    
    # 预测
    results = []
    for idx, row in data.iterrows():
        if idx % 100 == 0:
            print(f"进度: {idx}/{len(data)}")
        
        comment = row['CommentText']
        video_title = row.get('VideoTitle', '')
        
        result = predictor.predict_single(comment, video_title)
        
        results.append({
            'CommentText': comment,
            'VideoTitle': video_title,
            'PredictedSentiment': result['predicted_label'],
            'Confidence': result['confidence'],
            'Positive_Prob': result['probabilities']['Positive'],
            'Negative_Prob': result['probabilities']['Negative'],
            'Neutral_Prob': result['probabilities']['Neutral']
        })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    if output_file.endswith('.xlsx'):
        results_df.to_excel(output_file, index=False)
    else:
        results_df.to_csv(output_file, index=False)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    # 统计结果
    sentiment_counts = results_df['PredictedSentiment'].value_counts()
    print(f"\n情感分布:")
    for sentiment, count in sentiment_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YouTube评论情感分析 - 模型评估和预测')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'batch'], 
                       default='evaluate', help='运行模式')
    parser.add_argument('--data_path', type=str, help='评估数据路径')
    parser.add_argument('--input_file', type=str, help='批量预测输入文件')
    parser.add_argument('--output_file', type=str, help='批量预测输出文件')
    parser.add_argument('--comment', type=str, help='单条评论预测')
    parser.add_argument('--video_title', type=str, default='', help='视频标题')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        sys.exit(1)
    
    # 加载配置
    config = Config()
    
    # 创建预测器
    try:
        predictor = SentimentPredictor(args.model_path, config)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    # 根据模式执行不同操作
    if args.mode == 'evaluate':
        if not args.data_path:
            print("评估模式需要指定 --data_path")
            sys.exit(1)
        
        if not os.path.exists(args.data_path):
            print(f"错误: 数据文件不存在 {args.data_path}")
            sys.exit(1)
        
        predictor.evaluate_on_data(args.data_path)
    
    elif args.mode == 'batch':
        if not args.input_file or not args.output_file:
            print("批量预测模式需要指定 --input_file 和 --output_file")
            sys.exit(1)
        
        if not os.path.exists(args.input_file):
            print(f"错误: 输入文件不存在 {args.input_file}")
            sys.exit(1)
        
        batch_prediction_from_file(predictor, args.input_file, args.output_file)
    
    # 单条预测
    if args.comment:
        result = predictor.predict_single(args.comment, args.video_title)
        print(f"\n评论: {args.comment}")
        if args.video_title:
            print(f"视频标题: {args.video_title}")
        print(f"预测情感: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"各类别概率: {result['probabilities']}")

if __name__ == "__main__":
    print("YouTube评论情感分析 - 模型评估和预测")
    print("-" * 50)
    main()