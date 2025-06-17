import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from typing import List, Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self):
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """清洗文本数据"""
        if pd.isna(text):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除用户提及和话题标签
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 保留字母、数字、空格和基本标点
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def enhanced_clean_text(self, text: str) -> str:
        """增强的文本清洗功能"""
        if pd.isna(text) or text == '':
            return ""
        
        # 转换为字符串并去除首尾空格
        text = str(text).strip()
        
        # 移除HTML标签和实体
        text = re.sub(r'<[^>]+>', '', text)  # HTML标签
        text = re.sub(r'&[a-zA-Z]+;', '', text)  # HTML实体如&amp; &lt; &gt;
        text = re.sub(r'&#\d+;', '', text)  # 数字HTML实体
        
        # 移除URL和链接
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱地址
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 移除社交媒体相关内容
        text = re.sub(r'@\w+', '', text)  # 用户提及
        text = re.sub(r'#\w+', '', text)  # 话题标签
        text = re.sub(r'RT\s+', '', text)  # 转发标记
        
        # 处理表情符号和特殊字符
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # 表情符号
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)  # 符号和象形文字
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)  # 交通和地图符号
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)  # 旗帜
        
        # 移除多余的标点符号
        text = re.sub(r'[!]{2,}', '!', text)  # 多个感叹号
        text = re.sub(r'[?]{2,}', '?', text)  # 多个问号
        text = re.sub(r'[.]{3,}', '...', text)  # 多个句号
        
        # 移除非英文字符（保留基本标点）
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"\-]', '', text)
        
        # 处理缩写
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 转换为小写
        text = text.lower()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 确保文本不为空
        if len(text.strip()) == 0:
            return "no content"
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用词"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def augment_text(self, text: str) -> str:
        """简单的数据增强：同义词替换"""
        # 这里实现简单的数据增强，可以后续扩展
        words = text.split()
        if len(words) > 3:
            # 随机删除一个词（简单增强）
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        return ' '.join(words)

class SentimentDataset(Dataset):
    """情感分析数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT tokenizer编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataLoader:
    """数据加载和预处理主类"""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = TextPreprocessor()
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.label_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        try:
            data = pd.read_excel(self.config.DATA_PATH)
            print(f"成功加载数据，共 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        print("开始数据预处理...")
        
        # 复制数据避免修改原始数据
        processed_data = data.copy()
        
        # 合并评论文本和视频标题
        processed_data['combined_text'] = (
            processed_data['CommentText'].astype(str) + ' [SEP] ' + 
            processed_data['VideoTitle'].astype(str)
        )
        
        # 清洗文本
        processed_data['cleaned_text'] = processed_data['combined_text'].apply(
            self.preprocessor.clean_text
        )
        
        # 移除空文本
        processed_data = processed_data[processed_data['cleaned_text'].str.len() > 0]
        
        # 编码标签
        processed_data['encoded_labels'] = self.label_encoder.fit_transform(
            processed_data['Sentiment']
        )
        
        print(f"预处理完成，剩余 {len(processed_data)} 条有效记录")
        print(f"标签分布: {processed_data['Sentiment'].value_counts()}")
        
        return processed_data
    
    def create_datasets(self, data: pd.DataFrame, augment: bool = True) -> Tuple[SentimentDataset, SentimentDataset]:
        """创建训练和验证数据集"""
        texts = data['cleaned_text'].tolist()
        labels = data['encoded_labels'].tolist()
        
        # 数据增强
        if augment and self.config.AUGMENT_DATA:
            print("执行数据增强...")
            augmented_texts = []
            augmented_labels = []
            
            for text, label in zip(texts, labels):
                augmented_texts.append(text)
                augmented_labels.append(label)
                
                # 随机增强部分数据
                if random.random() < self.config.AUGMENT_RATIO:
                    aug_text = self.preprocessor.augment_text(text)
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
            
            texts = augmented_texts
            labels = augmented_labels
            print(f"增强后数据量: {len(texts)}")
        
        # 分割训练和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=self.config.VALIDATION_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=labels
        )
        
        # 创建数据集
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer, self.config.MAX_LENGTH
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def get_data_loaders(self, train_dataset: SentimentDataset, val_dataset: SentimentDataset) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """创建数据加载器"""
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Windows下设置为0避免多进程问题
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def load_custom_data(self, train_path: str, val_path: str) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """加载自定义的训练和验证数据文件"""
        print(f"加载自定义数据文件...")
        print(f"训练数据: {train_path}")
        print(f"验证数据: {val_path}")
        
        # 加载训练数据
        try:
            if train_path.endswith('.xlsx') or train_path.endswith('.xls'):
                train_data = pd.read_excel(train_path)
            else:
                train_data = pd.read_csv(train_path, encoding='utf-8')
            print(f"训练数据加载成功，共 {len(train_data)} 条记录")
        except Exception as e:
            print(f"训练数据加载失败: {e}")
            raise
        
        # 加载验证数据
        try:
            if val_path.endswith('.xlsx') or val_path.endswith('.xls'):
                val_data = pd.read_excel(val_path)
            else:
                val_data = pd.read_csv(val_path, encoding='utf-8')
            print(f"验证数据加载成功，共 {len(val_data)} 条记录")
        except Exception as e:
            print(f"验证数据加载失败: {e}")
            raise
        
        # 检查必要的列是否存在
        required_columns = ['CommentText', 'Sentiment']
        for df, name in [(train_data, '训练数据'), (val_data, '验证数据')]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{name}缺少必要的列: {missing_cols}")
        
        # 预处理数据
        def preprocess_custom_data(data: pd.DataFrame) -> pd.DataFrame:
            """预处理自定义数据"""
            processed_data = data.copy()
            
            # 如果有VideoTitle列，合并文本；否则只使用CommentText
            if 'VideoTitle' in processed_data.columns:
                processed_data['combined_text'] = (
                    processed_data['CommentText'].astype(str) + ' [SEP] ' + 
                    processed_data['VideoTitle'].astype(str)
                )
            else:
                processed_data['combined_text'] = processed_data['CommentText'].astype(str)
            
            # 清洗文本
            processed_data['cleaned_text'] = processed_data['combined_text'].apply(
                self.preprocessor.clean_text
            )
            
            # 移除空文本
            processed_data = processed_data[processed_data['cleaned_text'].str.len() > 0]
            
            return processed_data
        
        # 预处理训练和验证数据
        train_processed = preprocess_custom_data(train_data)
        val_processed = preprocess_custom_data(val_data)
        
        # 合并数据以统一编码标签
        all_data = pd.concat([train_processed, val_processed], ignore_index=True)
        self.label_encoder.fit(all_data['Sentiment'])
        
        # 编码标签
        train_processed['encoded_labels'] = self.label_encoder.transform(train_processed['Sentiment'])
        val_processed['encoded_labels'] = self.label_encoder.transform(val_processed['Sentiment'])
        
        print(f"训练数据预处理完成，剩余 {len(train_processed)} 条有效记录")
        print(f"验证数据预处理完成，剩余 {len(val_processed)} 条有效记录")
        print(f"训练数据标签分布: {train_processed['Sentiment'].value_counts()}")
        print(f"验证数据标签分布: {val_processed['Sentiment'].value_counts()}")
        
        # 创建数据集
        train_dataset = SentimentDataset(
            train_processed['cleaned_text'].tolist(),
            train_processed['encoded_labels'].tolist(),
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        
        val_dataset = SentimentDataset(
            val_processed['cleaned_text'].tolist(),
            val_processed['encoded_labels'].tolist(),
            self.tokenizer,
            self.config.MAX_LENGTH
        )
        
        # 创建数据加载器
        train_loader, val_loader = self.get_data_loaders(train_dataset, val_dataset)
        
        return train_loader, val_loader
    
    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """完整的数据准备流程"""
        # 加载数据
        raw_data = self.load_data()
        
        # 预处理
        processed_data = self.preprocess_data(raw_data)
        
        # 创建数据集
        train_dataset, val_dataset = self.create_datasets(processed_data)
        
        # 创建数据加载器
        train_loader, val_loader = self.get_data_loaders(train_dataset, val_dataset)
        
        return train_loader, val_loader

if __name__ == "__main__":
    # 测试数据预处理
    config = Config()
    data_loader = DataLoader(config)
    train_loader, val_loader = data_loader.prepare_data()
    
    print("数据预处理测试完成！")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")