# YouTube评论情感分析项目

基于深度学习的YouTube评论情感分析系统，支持三分类（Positive、Negative、Neutral）情感识别。

## 项目特点

- 🚀 **高性能模型**: 基于BERT预训练模型，针对RTX 3060 6GB显存优化
- 📊 **完整流程**: 从数据预处理到模型训练、评估的完整pipeline
- 🎯 **高F1值**: 目标F1分数0.85-0.90
- 🔧 **易于使用**: 提供命令行接口和交互式预测
- 📈 **可视化**: 训练曲线、混淆矩阵等可视化分析

## 项目结构

```
CommentSentiment/
├── config/
│   └── config.py              # 项目配置文件
├── src/
│   ├── data_preprocessing.py   # 数据预处理模块
│   ├── model.py               # 模型定义
│   └── trainer.py             # 训练器
├── data/
│   └── train_data.xlsx        # 训练数据
├── codes/
│   └── data_exploration.ipynb # 数据探索代码
├── models/                    # 保存的模型文件
├── logs/                      # 训练日志和可视化
├── results/                   # 训练结果
├── checkpoints/               # 模型检查点
├── train_improved.py          # 改进的训练脚本（主要训练脚本）
├── test.py                    # 模型测试脚本
├── evaluate.py                # 模型评估和批量预测脚本
├── requirements.txt           # 依赖包列表
├── 项目技术文档.md             # 详细技术文档
└── README.md                  # 项目说明文档
```

## 环境要求

### 硬件要求
- **推荐**: NVIDIA RTX 3060 6GB或更高
- **最低**: 8GB RAM，支持CUDA的GPU（可选）

### 软件要求
- Python 3.8+
- CUDA 11.0+ (如果使用GPU)

## 安装说明

### 1. 克隆项目
```bash
cd CommentSentiment
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载NLTK数据（首次运行时自动下载）
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 使用方法

### 1. 训练模型

#### 基础训练
```bash
python train_improved.py
```

#### 自定义参数训练
```bash
python train_improved.py --model_type bert --batch_size 16 --learning_rate 2e-5 --num_epochs 5 --use_focal_loss
```

#### 参数说明
- `--model_type`: 模型类型 (bert/bilstm)
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数
- `--max_length`: 最大序列长度
- `--use_focal_loss`: 使用Focal Loss处理类别不平衡
- `--freeze_layers`: 冻结BERT前N层
- `--seed`: 随机种子

### 2. 模型评估

#### 在测试集上评估
```bash
python evaluate.py --model_path models/best_model.pth --mode evaluate --data_path data/train_data.xlsx
```

#### 批量预测
```bash
python evaluate.py --model_path models/best_model.pth --mode batch --data_path data/test_data.xlsx --output_file results/predictions.xlsx
```

#### 单条评论预测
```bash
python evaluate.py --model_path models/best_model.pth --comment "This is a great video!"
```

### 3. 模型测试

#### 使用测试脚本
```bash
python test.py
```

#### 批量预测
```bash
python evaluate.py --model_path models/best_model.pth --mode batch --input_file input.xlsx --output_file predictions.xlsx
```

#### 单条预测
```bash
python evaluate.py --model_path models/best_model.pth --comment "This video is amazing!" --video_title "Great Tutorial"
```

## 配置说明

主要配置在 `config/config.py` 中：

```python
class Config:
    # 模型参数
    MODEL_NAME = "bert-base-uncased"  # BERT模型
    MAX_LENGTH = 128                  # 序列最大长度
    NUM_CLASSES = 3                   # 分类数量
    
    # 训练参数
    BATCH_SIZE = 16                   # 批次大小
    LEARNING_RATE = 2e-5              # 学习率
    NUM_EPOCHS = 5                    # 训练轮数
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 数据格式

训练数据应包含以下列：
- `CommentText`: 评论文本
- `VideoTitle`: 视频标题
- `Sentiment`: 情感标签 (Positive/Negative/Neutral)
- `Likes`: 点赞数（可选）
- `Replies`: 回复数（可选）
- `CountryCode`: 国家代码（可选）
- `CategoryID`: 类别ID（可选）

## 模型架构

### BERT模型（主推方案）
```
BERT-base-uncased
├── 12层Transformer编码器
├── 768维隐藏层
├── Dropout层 (0.3)
└── 分类头
    ├── Linear(768 → 384)
    ├── ReLU激活
    ├── Dropout(0.3)
    └── Linear(384 → 3)
```

### BiLSTM模型（备选方案）
```
词嵌入层 (300维)
├── BiLSTM (256×2层)
├── 多头注意力机制
├── 全局平均池化
└── 分类头
```

## 训练策略

1. **数据预处理**
   - 文本清洗（移除URL、特殊字符等）
   - 合并评论文本和视频标题
   - 数据增强（同义词替换、随机删除）

2. **模型优化**
   - AdamW优化器
   - 线性学习率预热
   - 梯度裁剪
   - 早停机制

3. **损失函数**
   - Focal Loss（处理类别不平衡）
   - 交叉熵损失（备选）

## 性能指标

预期性能（基于类似任务）：
- **准确率**: 87-92%
- **F1分数**: 85-90%
- **训练时间**: 约30-60分钟（RTX 3060）

## 可视化输出

训练完成后会生成：
- 训练曲线图 (`logs/training_curves.png`)
- 混淆矩阵 (`logs/confusion_matrix.png`)
- 训练历史 (`logs/training_history.json`)

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train.py --batch_size 8
   ```

2. **模型加载失败**
   - 检查模型文件路径
   - 确保模型文件完整

3. **数据加载错误**
   - 检查数据文件格式
   - 确保必要列存在

4. **依赖包问题**
   ```bash
   pip install --upgrade transformers torch
   ```

### 性能优化建议

1. **GPU优化**
   - 使用混合精度训练
   - 调整批次大小
   - 启用CUDA优化

2. **模型优化**
   - 冻结BERT前几层
   - 使用较小的BERT模型
   - 调整序列长度

## 扩展功能

### 1. 添加新的模型
在 `src/model.py` 中添加新的模型类：
```python
class CustomSentimentModel(nn.Module):
    # 自定义模型实现
    pass
```

### 2. 自定义数据增强
在 `src/data_preprocessing.py` 中扩展 `augment_text` 方法。

### 3. 集成学习
训练多个模型并进行投票集成。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 邮件联系

## 更新日志

### v1.0.0 (2024)
- 初始版本发布
- 支持BERT和BiLSTM模型
- 完整的训练和评估流程
- 可视化分析功能

---

**祝您使用愉快！如果这个项目对您有帮助，请给个⭐️**