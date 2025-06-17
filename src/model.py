import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class BertSentimentClassifier(nn.Module):
    """基于BERT的情感分类模型"""
    
    def __init__(self, config: Config, pretrained_model_name: str = None):
        super(BertSentimentClassifier, self).__init__()
        
        self.config = config
        self.num_classes = config.NUM_CLASSES
        
        # 加载预训练BERT模型
        model_name = pretrained_model_name or config.MODEL_NAME
        self.bert = BertModel.from_pretrained(model_name)
        
        # 获取BERT隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
        
        # 初始化分类头权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类头权重"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            labels: 标签（可选，用于计算损失）
        
        Returns:
            如果提供labels，返回(loss, logits)
            否则返回logits
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 获取[CLS]标记的表示
        pooled_output = outputs.pooler_output
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            # 计算损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits
    
    def freeze_bert_layers(self, num_layers_to_freeze: int = 6):
        """冻结BERT的前几层"""
        # 冻结embedding层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结前num_layers_to_freeze个encoder层
        for i in range(num_layers_to_freeze):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"已冻结BERT前{num_layers_to_freeze}层")
    
    def unfreeze_all_layers(self):
        """解冻所有层"""
        for param in self.parameters():
            param.requires_grad = True
        print("已解冻所有层")

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BiLSTMSentimentClassifier(nn.Module):
    """基于BiLSTM的情感分类模型（备选方案）"""
    
    def __init__(self, config: Config, vocab_size: int, embedding_dim: int = 300):
        super(BiLSTMSentimentClassifier, self).__init__()
        
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = 256
        self.num_layers = 2
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            embedding_dim, 
            self.hidden_dim, 
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=config.DROPOUT_RATE if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(self.hidden_dim, config.NUM_CLASSES)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        """
        # 词嵌入
        embedded = self.embedding(input_ids)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 注意力机制
        if attention_mask is not None:
            # 创建key_padding_mask
            key_padding_mask = ~attention_mask.bool()
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=key_padding_mask
            )
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(attn_out.size()).float()
            sum_embeddings = torch.sum(attn_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(attn_out, dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits

def create_model(config: Config, model_type: str = 'bert') -> nn.Module:
    """创建模型的工厂函数"""
    if model_type.lower() == 'bert':
        model = BertSentimentClassifier(config)
        print(f"创建BERT模型，参数量: {sum(p.numel() for p in model.parameters()):,}")
    elif model_type.lower() == 'bilstm':
        # 这里需要词汇表大小，实际使用时需要从数据中获取
        vocab_size = 30000  # 示例值
        model = BiLSTMSentimentClassifier(config, vocab_size)
        print(f"创建BiLSTM模型，参数量: {sum(p.numel() for p in model.parameters()):,}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model

if __name__ == "__main__":
    # 测试模型创建
    config = Config()
    
    # 测试BERT模型
    bert_model = create_model(config, 'bert')
    print("BERT模型创建成功")
    
    # 测试输入
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 3, (batch_size,))
    
    # 前向传播测试
    with torch.no_grad():
        loss, logits = bert_model(input_ids, attention_mask, labels)
        print(f"输出形状: {logits.shape}")
        print(f"损失: {loss.item():.4f}")
    
    print("模型测试完成！")