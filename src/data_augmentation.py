#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块

提供多种数据增强技术来提升模型性能
"""

import random
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextAugmenter:
    """文本数据增强器"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 情感词典
        self.positive_words = [
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'wonderful',
            'brilliant', 'outstanding', 'superb', 'magnificent', 'incredible',
            'marvelous', 'perfect', 'beautiful', 'lovely', 'good', 'nice',
            'cool', 'sweet', 'fun', 'enjoyable', 'pleasant', 'delightful'
        ]
        
        self.negative_words = [
            'terrible', 'awful', 'horrible', 'disgusting', 'bad', 'worst',
            'hate', 'stupid', 'boring', 'annoying', 'disappointing', 'useless',
            'pathetic', 'ridiculous', 'nonsense', 'garbage', 'trash', 'lame',
            'dumb', 'ugly', 'gross', 'sick', 'wrong', 'failed'
        ]
        
        self.neutral_words = [
            'okay', 'fine', 'normal', 'average', 'standard', 'typical',
            'regular', 'ordinary', 'common', 'usual', 'moderate', 'fair',
            'decent', 'acceptable', 'reasonable', 'adequate', 'sufficient'
        ]
        
        # 停用词（简化版）
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'through', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        ])
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """随机删除单词"""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """随机插入同义词"""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            # 随机选择一个非停用词
            candidates = [w for w in words if w.lower() not in self.stop_words]
            if not candidates:
                continue
            
            random_word = random.choice(candidates)
            synonyms = self.get_synonyms(random_word)
            
            if synonyms:
                synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(new_words))
                new_words.insert(random_idx, synonym)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换单词位置"""
        words = text.split()
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """同义词替换"""
        words = text.split()
        new_words = words.copy()
        
        # 找到可以替换的单词（非停用词）
        candidates = [(i, w) for i, w in enumerate(words) 
                     if w.lower() not in self.stop_words]
        
        if not candidates:
            return text
        
        # 随机选择n个单词进行替换
        random.shuffle(candidates)
        
        for i, word in candidates[:n]:
            synonyms = self.get_synonyms(word)
            if synonyms:
                new_words[i] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    def get_synonyms(self, word: str) -> List[str]:
        """获取同义词"""
        synonyms = set()
        
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
        except:
            pass
        
        return list(synonyms)
    
    def back_translation_simulation(self, text: str) -> str:
        """模拟回译（简化版）"""
        # 这里使用简单的同义词替换来模拟回译效果
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() < 0.3 and word.lower() not in self.stop_words:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(synonyms))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def add_noise(self, text: str, noise_level: float = 0.05) -> str:
        """添加噪声（字符级别）"""
        chars = list(text)
        n_noise = int(len(chars) * noise_level)
        
        for _ in range(n_noise):
            if len(chars) == 0:
                break
            
            noise_type = random.choice(['insert', 'delete', 'substitute'])
            pos = random.randint(0, len(chars) - 1)
            
            if noise_type == 'insert':
                chars.insert(pos, random.choice('abcdefghijklmnopqrstuvwxyz'))
            elif noise_type == 'delete' and len(chars) > 1:
                chars.pop(pos)
            elif noise_type == 'substitute':
                chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
        
        return ''.join(chars)
    
    def sentiment_guided_augmentation(self, text: str, sentiment: str) -> str:
        """基于情感的增强"""
        words = text.split()
        new_words = words.copy()
        
        # 根据情感添加相应的词汇
        if sentiment.lower() == 'positive':
            target_words = self.positive_words
        elif sentiment.lower() == 'negative':
            target_words = self.negative_words
        else:
            target_words = self.neutral_words
        
        # 随机添加1-2个情感词
        n_add = random.randint(1, 2)
        for _ in range(n_add):
            if random.random() < 0.3:  # 30%概率添加
                word_to_add = random.choice(target_words)
                pos = random.randint(0, len(new_words))
                new_words.insert(pos, word_to_add)
        
        return ' '.join(new_words)
    
    def augment_text(self, text: str, sentiment: str = None, 
                    methods: List[str] = None, intensity: float = 0.1) -> str:
        """综合文本增强"""
        if methods is None:
            methods = ['deletion', 'insertion', 'swap', 'synonym']
        
        augmented_text = text
        
        for method in methods:
            if random.random() < intensity:
                if method == 'deletion':
                    augmented_text = self.random_deletion(augmented_text, p=0.1)
                elif method == 'insertion':
                    augmented_text = self.random_insertion(augmented_text, n=1)
                elif method == 'swap':
                    augmented_text = self.random_swap(augmented_text, n=1)
                elif method == 'synonym':
                    augmented_text = self.synonym_replacement(augmented_text, n=1)
                elif method == 'back_translation':
                    augmented_text = self.back_translation_simulation(augmented_text)
                elif method == 'noise':
                    augmented_text = self.add_noise(augmented_text, noise_level=0.02)
                elif method == 'sentiment_guided' and sentiment:
                    augmented_text = self.sentiment_guided_augmentation(augmented_text, sentiment)
        
        return augmented_text

class DataAugmenter:
    """数据集增强器"""
    
    def __init__(self, random_seed: int = 42):
        self.text_augmenter = TextAugmenter(random_seed)
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def balance_dataset(self, df: pd.DataFrame, text_col: str = 'CommentText', 
                       label_col: str = 'Sentiment', target_size: int = None) -> pd.DataFrame:
        """平衡数据集"""
        # 统计各类别数量
        label_counts = df[label_col].value_counts()
        print(f"原始数据分布: {dict(label_counts)}")
        
        if target_size is None:
            target_size = label_counts.max()
        
        balanced_dfs = []
        
        for label in label_counts.index:
            label_df = df[df[label_col] == label].copy()
            current_size = len(label_df)
            
            if current_size >= target_size:
                # 如果当前类别数量已经足够，随机采样
                balanced_dfs.append(label_df.sample(n=target_size, random_state=self.random_seed))
            else:
                # 需要增强数据
                need_size = target_size - current_size
                augmented_data = []
                
                for _ in range(need_size):
                    # 随机选择一个样本进行增强
                    sample = label_df.sample(n=1, random_state=None).iloc[0]
                    
                    # 增强文本
                    original_text = sample[text_col]
                    augmented_text = self.text_augmenter.augment_text(
                        original_text, 
                        sentiment=label,
                        methods=['deletion', 'insertion', 'swap', 'synonym', 'sentiment_guided'],
                        intensity=0.3
                    )
                    
                    # 创建新样本
                    new_sample = sample.copy()
                    new_sample[text_col] = augmented_text
                    augmented_data.append(new_sample)
                
                # 合并原始数据和增强数据
                augmented_df = pd.DataFrame(augmented_data)
                combined_df = pd.concat([label_df, augmented_df], ignore_index=True)
                balanced_dfs.append(combined_df)
        
        # 合并所有类别的数据
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # 打乱数据
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # 统计平衡后的分布
        new_label_counts = balanced_df[label_col].value_counts()
        print(f"平衡后数据分布: {dict(new_label_counts)}")
        
        return balanced_df
    
    def augment_minority_classes(self, df: pd.DataFrame, text_col: str = 'CommentText',
                                label_col: str = 'Sentiment', augment_ratio: float = 0.5) -> pd.DataFrame:
        """增强少数类别"""
        label_counts = df[label_col].value_counts()
        max_count = label_counts.max()
        
        augmented_dfs = [df.copy()]
        
        for label in label_counts.index:
            current_count = label_counts[label]
            if current_count < max_count:
                # 计算需要增强的数量
                target_count = int(current_count * (1 + augment_ratio))
                need_count = target_count - current_count
                
                label_df = df[df[label_col] == label]
                augmented_data = []
                
                for _ in range(need_count):
                    sample = label_df.sample(n=1).iloc[0]
                    original_text = sample[text_col]
                    
                    # 使用不同的增强方法
                    method = random.choice([
                        ['deletion', 'synonym'],
                        ['insertion', 'swap'],
                        ['synonym', 'sentiment_guided'],
                        ['back_translation'],
                        ['noise', 'deletion']
                    ])
                    
                    augmented_text = self.text_augmenter.augment_text(
                        original_text,
                        sentiment=label,
                        methods=method,
                        intensity=0.4
                    )
                    
                    new_sample = sample.copy()
                    new_sample[text_col] = augmented_text
                    augmented_data.append(new_sample)
                
                if augmented_data:
                    augmented_df = pd.DataFrame(augmented_data)
                    augmented_dfs.append(augmented_df)
        
        # 合并所有数据
        final_df = pd.concat(augmented_dfs, ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return final_df
    
    def create_augmented_dataset(self, input_file: str, output_file: str,
                                augmentation_type: str = 'balance',
                                **kwargs) -> None:
        """创建增强数据集"""
        print(f"加载数据: {input_file}")
        df = pd.read_excel(input_file)
        
        print(f"原始数据集大小: {len(df)}")
        
        if augmentation_type == 'balance':
            augmented_df = self.balance_dataset(df, **kwargs)
        elif augmentation_type == 'minority':
            augmented_df = self.augment_minority_classes(df, **kwargs)
        else:
            raise ValueError(f"未知的增强类型: {augmentation_type}")
        
        print(f"增强后数据集大小: {len(augmented_df)}")
        
        # 保存增强后的数据
        augmented_df.to_excel(output_file, index=False)
        print(f"增强数据集已保存: {output_file}")
        
        return augmented_df

def main():
    """主函数 - 用于测试和演示"""
    # 创建数据增强器
    augmenter = DataAugmenter()
    
    # 测试文本增强
    test_texts = [
        "This video is amazing and I love it!",
        "This is terrible and boring content",
        "It's okay, nothing special about it"
    ]
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    
    print("=== 文本增强测试 ===")
    for text, sentiment in zip(test_texts, sentiments):
        print(f"\n原始文本 ({sentiment}): {text}")
        
        for i in range(3):
            augmented = augmenter.text_augmenter.augment_text(
                text, 
                sentiment=sentiment,
                methods=['deletion', 'insertion', 'synonym', 'sentiment_guided'],
                intensity=0.3
            )
            print(f"增强文本 {i+1}: {augmented}")
    
    # 如果存在训练数据，进行数据集增强
    input_file = "data/train_data.xlsx"
    if os.path.exists(input_file):
        print(f"\n=== 数据集增强测试 ===")
        
        # 创建平衡数据集
        balanced_file = "data/train_data_balanced.xlsx"
        augmenter.create_augmented_dataset(
            input_file=input_file,
            output_file=balanced_file,
            augmentation_type='balance'
        )
        
        # 创建少数类增强数据集
        minority_file = "data/train_data_minority_augmented.xlsx"
        augmenter.create_augmented_dataset(
            input_file=input_file,
            output_file=minority_file,
            augmentation_type='minority',
            augment_ratio=0.3
        )
    else:
        print(f"\n训练数据文件不存在: {input_file}")

if __name__ == "__main__":
    import os
    main()