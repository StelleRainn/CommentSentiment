# -*- coding: utf-8 -*-
"""
数据集划分工具 

功能：
1. 自动检测数据文件格式（CSV/Excel）
2. 智能识别文本和标签列
3. 划分训练集和测试集
4. 保存为标准格式供训练使用
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

class DataSplitter:
    """数据集划分器"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.text_column_names = [
            'CommentText', 'comment_text', 'text', 'Text', 'review', 'Review',
            'content', 'Content', 'message', 'Message', 'comment', 'Comment'
        ]
        self.sentiment_column_names = [
            'Sentiment', 'sentiment', 'label', 'Label', 'emotion', 'Emotion',
            'class', 'Class', 'category', 'Category', 'target', 'Target'
        ]
    
    def load_data(self, file_path):
        """加载数据文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式: {self.supported_formats}")
        
        print(f"正在加载文件: {file_path}")
        
        try:
            if file_ext == '.csv':
                # 尝试不同的编码
                for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"成功使用 {encoding} 编码加载CSV文件")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("无法使用常见编码读取CSV文件")
            else:
                df = pd.read_excel(file_path)
                print("成功加载Excel文件")
            
            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            raise Exception(f"加载文件失败: {e}")
    
    def detect_columns(self, df):
        """智能检测文本列和标签列"""
        columns = df.columns.tolist()
        
        # 检测文本列
        text_col = None
        for col_name in self.text_column_names:
            if col_name in columns:
                text_col = col_name
                break
        
        # 如果没有找到标准名称，寻找包含文本数据的列
        if text_col is None:
            for col in columns:
                if df[col].dtype == 'object':  # 字符串类型
                    # 检查是否包含较长的文本
                    sample_texts = df[col].dropna().astype(str)
                    if len(sample_texts) > 0:
                        avg_length = sample_texts.str.len().mean()
                        if avg_length > 10:  # 平均长度大于10认为是文本列
                            text_col = col
                            break
        
        # 检测标签列
        sentiment_col = None
        for col_name in self.sentiment_column_names:
            if col_name in columns:
                sentiment_col = col_name
                break
        
        # 如果没有找到标准名称，寻找可能的标签列
        if sentiment_col is None:
            for col in columns:
                if col != text_col:  # 不是文本列
                    unique_values = df[col].dropna().unique()
                    # 检查是否是分类标签（唯一值较少）
                    if len(unique_values) <= 10:
                        # 检查是否包含情感词汇
                        sentiment_keywords = [
                            'positive', 'negative', 'neutral', 'pos', 'neg', 'neu',
                            '正面', '负面', '中性', '积极', '消极', '0', '1', '2'
                        ]
                        
                        values_str = ' '.join([str(v).lower() for v in unique_values])
                        if any(keyword in values_str for keyword in sentiment_keywords):
                            sentiment_col = col
                            break
        
        return text_col, sentiment_col
    
    def normalize_sentiment_labels(self, df, sentiment_col):
        """标准化情感标签"""
        print(f"\n原始标签分布:")
        print(df[sentiment_col].value_counts())
        
        # 创建标签映射
        def map_sentiment(label):
            if pd.isna(label):
                return None
            
            label_str = str(label).lower().strip()
            
            # 负面情感映射
            if label_str in ['negative', 'neg', '0', 'bad', 'poor', '负面', '消极', '差']:
                return 'negative'
            # 中性情感映射
            elif label_str in ['neutral', 'neu', '1', 'ok', 'average', '中性', '一般']:
                return 'neutral'
            # 正面情感映射
            elif label_str in ['positive', 'pos', '2', 'good', 'great', '正面', '积极', '好']:
                return 'positive'
            else:
                return None
        
        # 应用映射
        df[sentiment_col] = df[sentiment_col].apply(map_sentiment)
        
        # 移除无效标签
        valid_mask = df[sentiment_col].notna()
        df = df[valid_mask].copy()
        
        print(f"\n标准化后标签分布:")
        print(df[sentiment_col].value_counts())
        
        return df
    
    def clean_text_data(self, df, text_col):
        """清理文本数据"""
        print("\n清理文本数据...")
        
        # 移除空文本
        df = df[df[text_col].notna()].copy()
        df[text_col] = df[text_col].astype(str)
        
        # 移除过短的文本
        df = df[df[text_col].str.len() >= 3].copy()
        
        # 基础清理
        df[text_col] = df[text_col].str.strip()
        
        print(f"清理后数据量: {len(df)}")
        
        return df
    
    def split_data(self, df, text_col, sentiment_col, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        print(f"\n划分数据集 (测试集比例: {test_size})...")
        
        # 确保列名标准化
        df_clean = df[[text_col, sentiment_col]].copy()
        df_clean.columns = ['CommentText', 'Sentiment']
        
        # 分层划分
        train_df, test_df = train_test_split(
            df_clean,
            test_size=test_size,
            random_state=random_state,
            stratify=df_clean['Sentiment']
        )
        
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        
        print("\n训练集标签分布:")
        print(train_df['Sentiment'].value_counts())
        
        print("\n测试集标签分布:")
        print(test_df['Sentiment'].value_counts())
        
        return train_df, test_df
    
    def save_data(self, train_df, test_df, output_dir='data'):
        """保存划分后的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        train_path = output_dir / 'train_split.csv'
        test_path = output_dir / 'test_split.csv'
        
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        print(f"\n数据已保存:")
        print(f"训练集: {train_path}")
        print(f"测试集: {test_path}")
        
        return str(train_path), str(test_path)
    
    def process(self, input_file, test_size=0.2, output_dir='data', random_state=42):
        """完整的数据处理流程"""
        try:
            # 1. 加载数据
            df = self.load_data(input_file)
            
            # 2. 检测列
            text_col, sentiment_col = self.detect_columns(df)
            
            if text_col is None:
                raise ValueError("未找到文本列！请确保数据包含文本内容列。")
            
            if sentiment_col is None:
                raise ValueError("未找到标签列！请确保数据包含情感标签列。")
            
            print(f"\n检测到的列:")
            print(f"文本列: {text_col}")
            print(f"标签列: {sentiment_col}")
            
            # 3. 标准化标签
            df = self.normalize_sentiment_labels(df, sentiment_col)
            
            # 4. 清理文本
            df = self.clean_text_data(df, text_col)
            
            # 5. 检查数据质量
            if len(df) < 100:
                print("警告: 有效数据量较少，可能影响模型性能")
            
            # 6. 划分数据
            train_df, test_df = self.split_data(df, text_col, sentiment_col, test_size, random_state)
            
            # 7. 保存数据
            train_path, test_path = self.save_data(train_df, test_df, output_dir)
            
            return train_path, test_path
            
        except Exception as e:
            print(f"处理失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='数据集划分工具 - 现场考核专用')
    parser.add_argument('input_file', type=str, help='输入数据文件路径 (CSV或Excel)')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例 (默认: 0.2)')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录 (默认: data)')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--preview', action='store_true', help='预览数据而不保存')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据集划分工具 - 现场考核专用")
    print("=" * 60)
    
    splitter = DataSplitter()
    
    try:
        if args.preview:
            # 预览模式
            df = splitter.load_data(args.input_file)
            text_col, sentiment_col = splitter.detect_columns(df)
            
            print(f"\n检测结果:")
            print(f"文本列: {text_col}")
            print(f"标签列: {sentiment_col}")
            
            if text_col and sentiment_col:
                print(f"\n数据预览:")
                print(df[[text_col, sentiment_col]].head())
                
                print(f"\n标签分布:")
                print(df[sentiment_col].value_counts())
            
        else:
            # 正常处理模式
            train_path, test_path = splitter.process(
                args.input_file, 
                args.test_size, 
                args.output_dir, 
                args.random_state
            )
            
            print("\n" + "=" * 60)
            print("处理完成！")
            print("=" * 60)
            print(f"训练集: {train_path}")
            print(f"测试集: {test_path}")
            print("\n现在可以使用以下命令训练模型:")
            print(f"python train_improved.py --train_data {train_path} --val_data {test_path}")
            
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()