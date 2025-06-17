#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优模块

使用网格搜索、随机搜索和贝叶斯优化进行超参数调优
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from itertools import product
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from src.data_preprocessing import DataLoader
from src.model import ModelFactory
from src.trainer import SentimentTrainer

class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config: Config, results_dir: str = "results/tuning"):
        self.config = config
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.tuning_results = []
        self.best_params = None
        self.best_score = 0.0
        
    def define_search_space(self, model_type: str = 'bert') -> Dict[str, List]:
        """定义搜索空间"""
        if model_type == 'bert':
            search_space = {
                'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
                'batch_size': [8, 16, 32],
                'num_epochs': [3, 5, 8],
                'max_length': [64, 128, 256],
                'dropout_rate': [0.1, 0.2, 0.3],
                'weight_decay': [0.01, 0.1, 0.2],
                'warmup_steps': [100, 200, 500],
                'use_focal_loss': [True, False],
                'freeze_bert_layers': [0, 2, 4]
            }
        elif model_type == 'bilstm':
            search_space = {
                'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3],
                'batch_size': [16, 32, 64],
                'num_epochs': [10, 15, 20],
                'max_length': [64, 128, 256],
                'hidden_size': [128, 256, 512],
                'num_layers': [1, 2, 3],
                'dropout_rate': [0.2, 0.3, 0.5],
                'bidirectional': [True],
                'use_attention': [True, False],
                'use_focal_loss': [True, False]
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return search_space
    
    def grid_search(self, model_type: str = 'bert', max_trials: int = 50) -> Dict[str, Any]:
        """网格搜索"""
        print(f"开始网格搜索 - 模型类型: {model_type}")
        
        search_space = self.define_search_space(model_type)
        
        # 生成所有参数组合
        param_grid = list(ParameterGrid(search_space))
        
        # 如果组合太多，随机选择一部分
        if len(param_grid) > max_trials:
            param_grid = random.sample(param_grid, max_trials)
            print(f"参数组合过多，随机选择 {max_trials} 个组合进行搜索")
        
        print(f"总共需要测试 {len(param_grid)} 个参数组合")
        
        best_score = 0.0
        best_params = None
        
        for i, params in enumerate(param_grid, 1):
            print(f"\n=== 试验 {i}/{len(param_grid)} ===")
            print(f"参数: {params}")
            
            try:
                score = self._evaluate_params(params, model_type)
                
                # 记录结果
                result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat()
                }
                self.tuning_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎉 发现更好的参数! F1分数: {score:.4f}")
                
                print(f"当前F1分数: {score:.4f}, 最佳F1分数: {best_score:.4f}")
                
            except Exception as e:
                print(f"❌ 试验失败: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\n=== 网格搜索完成 ===")
        print(f"最佳F1分数: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def random_search(self, model_type: str = 'bert', n_trials: int = 30) -> Dict[str, Any]:
        """随机搜索"""
        print(f"开始随机搜索 - 模型类型: {model_type}, 试验次数: {n_trials}")
        
        search_space = self.define_search_space(model_type)
        
        best_score = 0.0
        best_params = None
        
        for i in range(1, n_trials + 1):
            print(f"\n=== 随机试验 {i}/{n_trials} ===")
            
            # 随机采样参数
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            
            print(f"参数: {params}")
            
            try:
                score = self._evaluate_params(params, model_type)
                
                # 记录结果
                result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'model_type': model_type,
                    'search_type': 'random',
                    'timestamp': datetime.now().isoformat()
                }
                self.tuning_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎉 发现更好的参数! F1分数: {score:.4f}")
                
                print(f"当前F1分数: {score:.4f}, 最佳F1分数: {best_score:.4f}")
                
            except Exception as e:
                print(f"❌ 试验失败: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\n=== 随机搜索完成 ===")
        print(f"最佳F1分数: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def bayesian_optimization(self, model_type: str = 'bert', n_trials: int = 20) -> Dict[str, Any]:
        """贝叶斯优化（简化版）"""
        print(f"开始贝叶斯优化 - 模型类型: {model_type}, 试验次数: {n_trials}")
        
        # 这里实现一个简化的贝叶斯优化
        # 实际项目中可以使用 optuna 或 hyperopt 库
        
        search_space = self.define_search_space(model_type)
        
        # 初始随机试验
        n_random = min(5, n_trials // 2)
        print(f"进行 {n_random} 次随机初始化试验")
        
        best_score = 0.0
        best_params = None
        
        # 随机初始化
        for i in range(1, n_random + 1):
            print(f"\n=== 初始化试验 {i}/{n_random} ===")
            
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            
            print(f"参数: {params}")
            
            try:
                score = self._evaluate_params(params, model_type)
                
                result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'model_type': model_type,
                    'search_type': 'bayesian_init',
                    'timestamp': datetime.now().isoformat()
                }
                self.tuning_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎉 发现更好的参数! F1分数: {score:.4f}")
                
            except Exception as e:
                print(f"❌ 试验失败: {e}")
                continue
        
        # 基于历史结果的智能搜索
        for i in range(n_random + 1, n_trials + 1):
            print(f"\n=== 贝叶斯试验 {i}/{n_trials} ===")
            
            # 基于历史最佳结果生成新参数
            params = self._generate_smart_params(search_space, best_params)
            print(f"参数: {params}")
            
            try:
                score = self._evaluate_params(params, model_type)
                
                result = {
                    'trial': i,
                    'params': params,
                    'score': score,
                    'model_type': model_type,
                    'search_type': 'bayesian',
                    'timestamp': datetime.now().isoformat()
                }
                self.tuning_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"🎉 发现更好的参数! F1分数: {score:.4f}")
                
            except Exception as e:
                print(f"❌ 试验失败: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\n=== 贝叶斯优化完成 ===")
        print(f"最佳F1分数: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return {'best_params': best_params, 'best_score': best_score}
    
    def _generate_smart_params(self, search_space: Dict[str, List], 
                              best_params: Dict[str, Any]) -> Dict[str, Any]:
        """基于最佳参数生成新的参数组合"""
        if best_params is None:
            # 如果没有最佳参数，随机生成
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            return params
        
        params = best_params.copy()
        
        # 随机扰动一些参数
        n_perturb = random.randint(1, min(3, len(search_space)))
        params_to_perturb = random.sample(list(search_space.keys()), n_perturb)
        
        for param_name in params_to_perturb:
            param_values = search_space[param_name]
            current_value = params[param_name]
            
            if isinstance(current_value, (int, float)):
                # 数值参数：在邻近值中选择
                try:
                    current_idx = param_values.index(current_value)
                    # 选择邻近的值
                    candidates = []
                    if current_idx > 0:
                        candidates.append(param_values[current_idx - 1])
                    if current_idx < len(param_values) - 1:
                        candidates.append(param_values[current_idx + 1])
                    
                    if candidates:
                        params[param_name] = random.choice(candidates)
                    else:
                        params[param_name] = random.choice(param_values)
                except ValueError:
                    params[param_name] = random.choice(param_values)
            else:
                # 其他类型参数：随机选择
                params[param_name] = random.choice(param_values)
        
        return params
    
    def _evaluate_params(self, params: Dict[str, Any], model_type: str) -> float:
        """评估参数组合"""
        # 更新配置
        config = self.config.copy()
        
        # 更新模型参数
        config.MODEL_TYPE = model_type
        for key, value in params.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
            elif key == 'learning_rate':
                config.LEARNING_RATE = value
            elif key == 'batch_size':
                config.BATCH_SIZE = value
            elif key == 'num_epochs':
                config.NUM_EPOCHS = value
            elif key == 'max_length':
                config.MAX_LENGTH = value
            elif key == 'dropout_rate':
                config.DROPOUT_RATE = value
            elif key == 'weight_decay':
                config.WEIGHT_DECAY = value
            elif key == 'warmup_steps':
                config.WARMUP_STEPS = value
            elif key == 'use_focal_loss':
                config.USE_FOCAL_LOSS = value
            elif key == 'freeze_bert_layers':
                config.FREEZE_BERT_LAYERS = value
            elif key == 'hidden_size':
                config.HIDDEN_SIZE = value
            elif key == 'num_layers':
                config.NUM_LAYERS = value
            elif key == 'bidirectional':
                config.BIDIRECTIONAL = value
            elif key == 'use_attention':
                config.USE_ATTENTION = value
        
        # 调整批次大小以适应GPU内存
        if not torch.cuda.is_available() and config.BATCH_SIZE > 16:
            config.BATCH_SIZE = 16
        
        try:
            # 准备数据
            data_loader = DataLoader(config)
            train_loader, val_loader, test_loader = data_loader.get_data_loaders()
            
            # 创建模型
            model = ModelFactory.create_model(config.MODEL_TYPE, config)
            
            # 训练模型
            trainer = SentimentTrainer(model, config)
            
            # 使用较少的epoch进行快速评估
            original_epochs = config.NUM_EPOCHS
            config.NUM_EPOCHS = min(3, config.NUM_EPOCHS)  # 最多3个epoch
            
            history = trainer.train(train_loader, val_loader)
            
            # 恢复原始epoch数
            config.NUM_EPOCHS = original_epochs
            
            # 返回最佳验证F1分数
            best_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
            
            # 清理GPU内存
            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return best_f1
            
        except Exception as e:
            print(f"评估参数时出错: {e}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0
    
    def save_results(self, filename: str = None):
        """保存调优结果"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tuning_results_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.tuning_results,
            'summary': {
                'total_trials': len(self.tuning_results),
                'best_trial': max(self.tuning_results, key=lambda x: x['score']) if self.tuning_results else None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"调优结果已保存: {filepath}")
    
    def load_results(self, filename: str):
        """加载调优结果"""
        filepath = os.path.join(self.results_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.best_params = results.get('best_params')
            self.best_score = results.get('best_score', 0.0)
            self.tuning_results = results.get('all_results', [])
            
            print(f"已加载调优结果: {filepath}")
            print(f"最佳F1分数: {self.best_score:.4f}")
        else:
            print(f"结果文件不存在: {filepath}")
    
    def plot_tuning_results(self, save_path: str = None):
        """绘制调优结果"""
        if not self.tuning_results:
            print("没有调优结果可绘制")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.tuning_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('超参数调优结果分析', fontsize=16, fontweight='bold')
        
        # 1. 分数随试验次数的变化
        axes[0, 0].plot(df['trial'], df['score'], 'b-o', alpha=0.7)
        axes[0, 0].axhline(y=self.best_score, color='r', linestyle='--', 
                          label=f'最佳分数: {self.best_score:.4f}')
        axes[0, 0].set_title('F1分数随试验次数变化')
        axes[0, 0].set_xlabel('试验次数')
        axes[0, 0].set_ylabel('F1分数')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 分数分布直方图
        axes[0, 1].hist(df['score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=self.best_score, color='r', linestyle='--', 
                          label=f'最佳分数: {self.best_score:.4f}')
        axes[0, 1].set_title('F1分数分布')
        axes[0, 1].set_xlabel('F1分数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # 3. 参数重要性分析（以学习率为例）
        if 'params' in df.columns:
            learning_rates = []
            scores = []
            
            for _, row in df.iterrows():
                params = row['params']
                if 'learning_rate' in params:
                    learning_rates.append(params['learning_rate'])
                    scores.append(row['score'])
            
            if learning_rates:
                axes[1, 0].scatter(learning_rates, scores, alpha=0.7)
                axes[1, 0].set_title('学习率 vs F1分数')
                axes[1, 0].set_xlabel('学习率')
                axes[1, 0].set_ylabel('F1分数')
                axes[1, 0].set_xscale('log')
                axes[1, 0].grid(True)
        
        # 4. Top 10 试验结果
        top_results = df.nlargest(min(10, len(df)), 'score')
        axes[1, 1].barh(range(len(top_results)), top_results['score'])
        axes[1, 1].set_title('Top 10 试验结果')
        axes[1, 1].set_xlabel('F1分数')
        axes[1, 1].set_ylabel('试验排名')
        axes[1, 1].set_yticks(range(len(top_results)))
        axes[1, 1].set_yticklabels([f'试验 {t}' for t in top_results['trial']])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"调优结果图已保存: {save_path}")
        
        plt.show()
    
    def generate_tuning_report(self, output_file: str = None) -> str:
        """生成调优报告"""
        if not self.tuning_results:
            return "没有调优结果可生成报告"
        
        df = pd.DataFrame(self.tuning_results)
        
        report = []
        report.append("# 超参数调优报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n总试验次数: {len(self.tuning_results)}")
        
        # 最佳结果
        report.append("\n## 🏆 最佳结果")
        report.append(f"- **最佳F1分数**: {self.best_score:.4f}")
        report.append(f"- **最佳参数**: {json.dumps(self.best_params, indent=2, ensure_ascii=False)}")
        
        # 统计信息
        report.append("\n## 📊 统计信息")
        report.append(f"- **平均F1分数**: {df['score'].mean():.4f}")
        report.append(f"- **F1分数标准差**: {df['score'].std():.4f}")
        report.append(f"- **最高F1分数**: {df['score'].max():.4f}")
        report.append(f"- **最低F1分数**: {df['score'].min():.4f}")
        
        # Top 5 结果
        top_5 = df.nlargest(5, 'score')
        report.append("\n## 🥇 Top 5 结果")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            report.append(f"\n### 第 {i} 名")
            report.append(f"- **F1分数**: {row['score']:.4f}")
            report.append(f"- **试验编号**: {row['trial']}")
            report.append(f"- **参数**: {json.dumps(row['params'], indent=2, ensure_ascii=False)}")
        
        report_text = "\n".join(report)
        
        if output_file:
            filepath = os.path.join(self.results_dir, output_file)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"调优报告已保存: {filepath}")
        
        return report_text

def main():
    """主函数 - 演示超参数调优"""
    # 加载配置
    config = Config()
    
    # 创建调优器
    tuner = HyperparameterTuner(config)
    
    print("=== 超参数调优演示 ===")
    print("1. 随机搜索 (快速)")
    print("2. 网格搜索 (全面)")
    print("3. 贝叶斯优化 (智能)")
    
    choice = input("请选择调优方法 (1-3): ").strip()
    model_type = input("请选择模型类型 (bert/bilstm): ").strip().lower()
    
    if model_type not in ['bert', 'bilstm']:
        model_type = 'bert'
    
    start_time = time.time()
    
    if choice == '1':
        result = tuner.random_search(model_type=model_type, n_trials=10)
    elif choice == '2':
        result = tuner.grid_search(model_type=model_type, max_trials=20)
    elif choice == '3':
        result = tuner.bayesian_optimization(model_type=model_type, n_trials=15)
    else:
        print("无效选择，使用随机搜索")
        result = tuner.random_search(model_type=model_type, n_trials=10)
    
    end_time = time.time()
    
    print(f"\n=== 调优完成 ===")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"最佳F1分数: {result['best_score']:.4f}")
    print(f"最佳参数: {result['best_params']}")
    
    # 保存结果
    tuner.save_results()
    
    # 绘制结果
    plot_path = os.path.join(tuner.results_dir, "tuning_analysis.png")
    tuner.plot_tuning_results(plot_path)
    
    # 生成报告
    report_path = "tuning_report.md"
    tuner.generate_tuning_report(report_path)

if __name__ == "__main__":
    import sys
    main()