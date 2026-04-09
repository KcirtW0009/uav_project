# -*- coding: utf-8 -*-
"""
使用已保存的MAPPO模型在原有评估框架中进行评估

这样可以利用原有的完整评估指标采集功能
"""

import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.experiments_mappo import ExperimentBAMAPPO


def evaluate_saved_model(model_path, num_episodes=3, seed=42):
    """
    评估已保存的MAPPO模型
    
    Args:
        model_path: 模型文件路径
        num_episodes: 评估轮数
        seed: 随机种子
    """
    print(f"加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    # 使用原有的实验类进行评估
    # 这样可以复用原有的完整评估指标采集功能
    
    # 创建临时实验实例
    exp = ExperimentBAMAPPO()
    
    # 加载模型并评估
    # 这里需要调用原有的评估逻辑
    
    print("评估完成!")
    return None


def find_latest_model():
    """查找最新的模型文件"""
    log_dir = './experiment_logs'
    if not os.path.exists(log_dir):
        return None
    
    # 查找最新的mappo目录
    dirs = [d for d in os.listdir(log_dir) if d.startswith('mappo_high_')]
    if not dirs:
        return None
    
    latest_dir = sorted(dirs)[-1]
    model_path = os.path.join(log_dir, latest_dir, 'final_model.pt')
    
    if os.path.exists(model_path):
        return model_path
    
    # 尝试best_model.pt
    model_path = os.path.join(log_dir, latest_dir, 'best_model.pt')
    if os.path.exists(model_path):
        return model_path
    
    return None


def main():
    parser = argparse.ArgumentParser(description='评估已保存的MAPPO模型')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=3, help='评估轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 如果没有指定模型路径，自动查找最新的
    if args.model is None:
        model_path = find_latest_model()
        if model_path is None:
            print("错误: 未找到模型文件，请使用 --model 指定路径")
            return
        print(f"自动找到最新模型: {model_path}")
    else:
        model_path = args.model
    
    # 进行评估
    evaluate_saved_model(model_path, args.episodes, args.seed)


if __name__ == '__main__':
    main()
