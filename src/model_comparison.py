#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型比较和分析模块

提供多个模型的训练、比较和分析功能
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelComparator:
    """模型比较器"""
    
    def __init__(self, results_dir: str = "results/comparison"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.models_info = {}
        self.comparison_results = {}
        
    def add_model_result(self, model_name: str, model_path: str, 
                        metrics: Dict[str, float], training_time: float,
                        model_config: Dict[str, Any] = None):
        """添加模型结果"""
        self.models_info[model_name] = {
            'model_path': model_path,
            'metrics': metrics,
            'training_time': training_time,
            'config': model_config or {},
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"已添加模型: {model_name}")
        print(f"F1分数: {metrics.get('f1_score', 'N/A'):.4f}")
        print(f"准确率: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"训练时间: {training_time:.2f}秒")
    
    def load_model_results(self, results_file: str):
        """从文件加载模型结果"""
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                self.models_info = json.load(f)
            print(f"已加载 {len(self.models_info)} 个模型结果")
        else:
            print(f"结果文件不存在: {results_file}")
    
    def save_model_results(self, results_file: str):
        """保存模型结果到文件"""
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.models_info, f, indent=2, ensure_ascii=False)
        print(f"模型结果已保存: {results_file}")
    
    def compare_models(self) -> pd.DataFrame:
        """比较所有模型"""
        if not self.models_info:
            print("没有模型结果可比较")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, info in self.models_info.items():
            metrics = info['metrics']
            config = info['config']
            
            row = {
                '模型名称': model_name,
                'F1分数': metrics.get('f1_score', 0),
                '准确率': metrics.get('accuracy', 0),
                '精确率': metrics.get('precision', 0),
                '召回率': metrics.get('recall', 0),
                '训练时间(秒)': info['training_time'],
                '模型类型': config.get('model_type', 'Unknown'),
                '批次大小': config.get('batch_size', 'N/A'),
                '学习率': config.get('learning_rate', 'N/A'),
                '最大长度': config.get('max_length', 'N/A'),
                '使用Focal Loss': config.get('use_focal_loss', False),
                '时间戳': info['timestamp']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # 按F1分数排序
        df = df.sort_values('F1分数', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_model_comparison(self, save_path: str = None):
        """绘制模型比较图"""
        df = self.compare_models()
        
        if df.empty:
            print("没有数据可绘制")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能比较', fontsize=16, fontweight='bold')
        
        # 1. F1分数比较
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(df)), df['F1分数'], color='skyblue', alpha=0.7)
        ax1.set_title('F1分数比较')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('F1分数')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['模型名称'], rotation=45, ha='right')
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. 准确率比较
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(df)), df['准确率'], color='lightgreen', alpha=0.7)
        ax2.set_title('准确率比较')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('准确率')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['模型名称'], rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. 训练时间比较
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(df)), df['训练时间(秒)'], color='orange', alpha=0.7)
        ax3.set_title('训练时间比较')
        ax3.set_xlabel('模型')
        ax3.set_ylabel('训练时间(秒)')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['模型名称'], rotation=45, ha='right')
        
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{height:.0f}s', ha='center', va='bottom')
        
        # 4. 综合性能雷达图
        ax4 = axes[1, 1]
        
        # 选择前3个模型进行雷达图比较
        top_models = df.head(3)
        
        if len(top_models) > 0:
            metrics = ['F1分数', '准确率', '精确率', '召回率']
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            ax4 = plt.subplot(2, 2, 4, projection='polar')
            
            colors = ['red', 'blue', 'green']
            
            for i, (_, model) in enumerate(top_models.iterrows()):
                if i >= 3:  # 最多显示3个模型
                    break
                
                values = [model[metric] for metric in metrics]
                values += values[:1]  # 闭合图形
                
                ax4.plot(angles, values, 'o-', linewidth=2, 
                        label=model['模型名称'], color=colors[i])
                ax4.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_ylim(0, 1)
            ax4.set_title('Top 3 模型综合性能')
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较图已保存: {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, model_name: str, log_file: str = None):
        """绘制训练曲线"""
        if log_file and os.path.exists(log_file):
            # 从日志文件读取训练历史
            training_history = self.parse_training_log(log_file)
        else:
            print(f"日志文件不存在: {log_file}")
            return
        
        if not training_history:
            print("没有找到训练历史数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} 训练曲线', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='训练损失')
        if 'val_loss' in training_history:
            axes[0, 0].plot(epochs, training_history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        if 'train_acc' in training_history:
            axes[0, 1].plot(epochs, training_history['train_acc'], 'b-', label='训练准确率')
        if 'val_acc' in training_history:
            axes[0, 1].plot(epochs, training_history['val_acc'], 'r-', label='验证准确率')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        if 'train_f1' in training_history:
            axes[1, 0].plot(epochs, training_history['train_f1'], 'b-', label='训练F1')
        if 'val_f1' in training_history:
            axes[1, 0].plot(epochs, training_history['val_f1'], 'r-', label='验证F1')
        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        if 'learning_rate' in training_history:
            axes[1, 1].plot(epochs, training_history['learning_rate'], 'g-', label='学习率')
            axes[1, 1].set_title('学习率曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'{model_name}_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
        
        plt.show()
    
    def parse_training_log(self, log_file: str) -> Dict[str, List[float]]:
        """解析训练日志文件"""
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 这里需要根据实际的日志格式来解析
                    # 示例格式: "Epoch 1/5 - Train Loss: 0.5, Val Loss: 0.4, Train Acc: 0.8, Val Acc: 0.85"
                    if 'Epoch' in line and 'Train Loss' in line:
                        parts = line.strip().split(' - ')
                        if len(parts) > 1:
                            metrics_part = parts[1]
                            
                            # 解析各种指标
                            if 'Train Loss:' in metrics_part:
                                train_loss = float(metrics_part.split('Train Loss:')[1].split(',')[0].strip())
                                training_history['train_loss'].append(train_loss)
                            
                            if 'Val Loss:' in metrics_part:
                                val_loss = float(metrics_part.split('Val Loss:')[1].split(',')[0].strip())
                                training_history['val_loss'].append(val_loss)
                            
                            # 可以继续添加其他指标的解析
        except Exception as e:
            print(f"解析日志文件时出错: {e}")
        
        return training_history
    
    def generate_report(self, output_file: str = None):
        """生成详细的比较报告"""
        df = self.compare_models()
        
        if df.empty:
            print("没有模型数据可生成报告")
            return
        
        report = []
        report.append("# YouTube评论情感分析 - 模型比较报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n总共比较了 {len(df)} 个模型\n")
        
        # 最佳模型
        best_model = df.iloc[0]
        report.append("## 🏆 最佳模型")
        report.append(f"- **模型名称**: {best_model['模型名称']}")
        report.append(f"- **F1分数**: {best_model['F1分数']:.4f}")
        report.append(f"- **准确率**: {best_model['准确率']:.4f}")
        report.append(f"- **精确率**: {best_model['精确率']:.4f}")
        report.append(f"- **召回率**: {best_model['召回率']:.4f}")
        report.append(f"- **训练时间**: {best_model['训练时间(秒)']:.2f}秒")
        report.append(f"- **模型类型**: {best_model['模型类型']}")
        
        # 性能统计
        report.append("\n## 📊 性能统计")
        report.append(f"- **平均F1分数**: {df['F1分数'].mean():.4f} ± {df['F1分数'].std():.4f}")
        report.append(f"- **平均准确率**: {df['准确率'].mean():.4f} ± {df['准确率'].std():.4f}")
        report.append(f"- **平均训练时间**: {df['训练时间(秒)'].mean():.2f} ± {df['训练时间(秒)'].std():.2f}秒")
        
        # 详细比较表
        report.append("\n## 📋 详细比较")
        report.append("\n| 排名 | 模型名称 | F1分数 | 准确率 | 训练时间(秒) | 模型类型 |")
        report.append("|------|----------|--------|--------|--------------|----------|")
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            report.append(
                f"| {i} | {row['模型名称']} | {row['F1分数']:.4f} | "
                f"{row['准确率']:.4f} | {row['训练时间(秒)']:.2f} | {row['模型类型']} |"
            )
        
        # 建议
        report.append("\n## 💡 建议")
        
        if len(df) > 1:
            f1_diff = df.iloc[0]['F1分数'] - df.iloc[1]['F1分数']
            if f1_diff > 0.05:
                report.append(f"- 最佳模型 '{best_model['模型名称']}' 明显优于其他模型")
            else:
                report.append("- 多个模型性能相近，可以考虑集成学习")
        
        if best_model['训练时间(秒)'] > df['训练时间(秒)'].median() * 2:
            report.append("- 最佳模型训练时间较长，可以考虑模型压缩或优化")
        
        if best_model['F1分数'] < 0.85:
            report.append("- F1分数还有提升空间，建议尝试数据增强或超参数调优")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"比较报告已保存: {output_file}")
        
        print("\n" + report_text)
        
        return report_text
    
    def recommend_best_model(self) -> Dict[str, Any]:
        """推荐最佳模型"""
        df = self.compare_models()
        
        if df.empty:
            return {}
        
        # 综合评分：F1分数权重0.4，准确率权重0.3，训练时间权重0.3（越短越好）
        df['综合评分'] = (
            df['F1分数'] * 0.4 + 
            df['准确率'] * 0.3 + 
            (1 - df['训练时间(秒)'] / df['训练时间(秒)'].max()) * 0.3
        )
        
        best_model = df.loc[df['综合评分'].idxmax()]
        
        recommendation = {
            'model_name': best_model['模型名称'],
            'model_path': self.models_info[best_model['模型名称']]['model_path'],
            'f1_score': best_model['F1分数'],
            'accuracy': best_model['准确率'],
            'training_time': best_model['训练时间(秒)'],
            'composite_score': best_model['综合评分'],
            'config': self.models_info[best_model['模型名称']]['config']
        }
        
        return recommendation

def main():
    """主函数 - 演示模型比较功能"""
    comparator = ModelComparator()
    
    # 示例：添加一些模型结果
    comparator.add_model_result(
        model_name="BERT-base",
        model_path="models/bert_base_model.pth",
        metrics={
            'f1_score': 0.8756,
            'accuracy': 0.8923,
            'precision': 0.8834,
            'recall': 0.8679
        },
        training_time=1245.6,
        model_config={
            'model_type': 'bert',
            'batch_size': 16,
            'learning_rate': 2e-5,
            'max_length': 128,
            'use_focal_loss': True
        }
    )
    
    comparator.add_model_result(
        model_name="BiLSTM-Attention",
        model_path="models/bilstm_attention_model.pth",
        metrics={
            'f1_score': 0.8234,
            'accuracy': 0.8456,
            'precision': 0.8123,
            'recall': 0.8345
        },
        training_time=856.3,
        model_config={
            'model_type': 'bilstm',
            'batch_size': 32,
            'learning_rate': 1e-3,
            'max_length': 128,
            'use_focal_loss': False
        }
    )
    
    # 比较模型
    print("=== 模型比较 ===")
    comparison_df = comparator.compare_models()
    print(comparison_df)
    
    # 绘制比较图
    comparator.plot_model_comparison()
    
    # 生成报告
    report_file = os.path.join(comparator.results_dir, "model_comparison_report.md")
    comparator.generate_report(report_file)
    
    # 推荐最佳模型
    recommendation = comparator.recommend_best_model()
    print(f"\n=== 推荐模型 ===")
    print(f"推荐模型: {recommendation.get('model_name', 'N/A')}")
    print(f"综合评分: {recommendation.get('composite_score', 0):.4f}")

if __name__ == "__main__":
    main()