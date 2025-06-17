#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class Config:
    """项目配置类"""
    
    # 数据路径
    DATA_PATH = "data/train_data.xlsx"
    MODEL_SAVE_PATH = "models/"
    LOG_PATH = "logs/"
    
    # 模型参数
    MODEL_NAME = "bert-base-uncased"  # 使用较小的BERT模型适配6GB显存
    MAX_LENGTH = 128  # 序列最大长度
    NUM_CLASSES = 3   # 三分类：Positive, Negative, Neutral
    
    # 训练参数
    BATCH_SIZE = 16   # 适配RTX 3060 6GB显存
    LEARNING_RATE = 2e-5
    EPOCHS = 5  # 修改为EPOCHS以保持一致性
    NUM_EPOCHS = 5  # 保留兼容性
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.3
    
    # 验证参数
    VALIDATION_SPLIT = 0.2
    K_FOLDS = 5
    RANDOM_SEED = 42
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 标签映射
    LABEL_MAP = {
        "Positive": 0,
        "Negative": 1,
        "Neutral": 2
    }
    
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
    
    # 数据增强参数
    AUGMENT_DATA = True
    AUGMENT_RATIO = 0.3
    
    # 早停参数
    EARLY_STOPPING_PATIENCE = 3
    
    # 日志参数
    USE_WANDB = False  # 可选择是否使用wandb
    PROJECT_NAME = "youtube_sentiment_analysis"