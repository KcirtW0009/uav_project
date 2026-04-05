#!/usr/bin/env python3
"""
BA-MAPPO 算法测试脚本
专门用于测试 UAV=30 的情况，实施优化措施并记录结果
"""

import os
import sys
import numpy as np
import torch
from uav_system.experiments_mappo import ExperimentBAMAPPO
from uav_system.config import set_global_seed, GLOBAL_SEED

# 设置全局随机种子
set_global_seed(GLOBAL_SEED)

# 测试配置
TEST_CONFIG = {
    'num_uav': 30,
    'num_bs': 4,  # 保持基站数量
    'num_steps': 100,  # 增加每轮步数，增加切换需求
    'train_episodes': 150,  # 增加训练轮次，确保充分学习
    'eval_episodes': 3,  # 减少评估次数，加快测试
    'bs_capacity_range': (40, 80),  # 减少容量范围，增加网络负载
    'pos_range': 800.0,  # 增加地图范围，增加移动距离
    'hidden_dim': 64,  # 增加隐藏层维度，提高模型表达能力
    'critic_hidden_dim': 128,  # 增加隐藏层维度，提高模型表达能力
    'use_biz_heads': True,
    'use_attention_critic': False,  # 禁用注意力机制，加快计算
    'rollout_length': 100,  # 增加rollout长度，提高学习效果
    'actor_lr': 5e-5,  # 降低学习率
    'critic_lr': 1.5e-4,  # 降低学习率
    'batch_size': 32,  # 调整批量大小
}

# 记录测试结果
TEST_RESULTS = []

def run_test(test_name, config_modifications=None):
    """运行测试并记录结果"""
    print(f"\n{'='*80}")
    print(f"测试: {test_name}")
    print(f"{'='*80}")
    
    # 应用配置修改
    config = TEST_CONFIG.copy()
    if config_modifications:
        config.update(config_modifications)
    
    # 运行实验
    try:
        results = ExperimentBAMAPPO.run(
            num_uav_list=(config['num_uav'],),
            num_bs=config['num_bs'],
            num_steps=config['num_steps'],
            train_episodes=config['train_episodes'],
            eval_episodes=config['eval_episodes'],
            bs_capacity_range=config['bs_capacity_range'],
            pos_range=config['pos_range'],
            hidden_dim=config['hidden_dim'],
            critic_hidden_dim=config['critic_hidden_dim'],
            use_biz_heads=config['use_biz_heads'],
            use_attention_critic=config['use_attention_critic'],
            rollout_length=config['rollout_length'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            load_models=False,
            phase='phase1',  # 只运行训练阶段
            verbose=True,
        )
        
        # 记录结果
        test_result = {
            'test_name': test_name,
            'config': config,
            'results': results,
        }
        TEST_RESULTS.append(test_result)
        
        print(f"\n{'='*80}")
        print(f"测试完成: {test_name}")
        print(f"{'='*80}")
        
        return True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始 BA-MAPPO 算法测试")
    print("测试配置:")
    for key, value in TEST_CONFIG.items():
        print(f"  {key}: {value}")
    
    # 测试: 修复分层策略 + 优化奖励函数
    run_test("测试 - 修复分层策略 + 优化奖励函数")
    
    # 保存测试结果
    import pickle
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(TEST_RESULTS, f)
    
    print("\n测试结果已保存到 test_results.pkl")

if __name__ == "__main__":
    main()
