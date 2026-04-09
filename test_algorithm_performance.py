# -*- coding: utf-8 -*-
"""
算法性能测试脚本

测试MAPPO、增强算法和传统算法的性能排序，确保MAPPO > 增强 > 传统

Author: Performance Tester
Date: 2026-04-08
"""

import os
import sys
import numpy as np
from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm

def test_algorithm_performance():
    """测试算法性能排序"""
    print("\n" + "="*80)
    print("算法性能测试")
    print("="*80)
    
    # 设置种子
    set_global_seed(GLOBAL_SEED)
    
    # 测试配置 - 85%负载率
    test_configs = [
        {'name': '小规模', 'num_uav': 128, 'num_bs': 3, 'num_steps': 50},  # 128/3=85%
        {'name': '标准规模', 'num_uav': 200, 'num_bs': 4, 'num_steps': 50},  # 200/4=85%
        {'name': '大规模', 'num_uav': 280, 'num_bs': 5, 'num_steps': 50},  # 280/5=85%
    ]
    
    for config in test_configs:
        print(f"\n>>> 测试 {config['name']} (UAV={config['num_uav']}, BS={config['num_bs']}) <<<")
        print("-"*60)
        
        # 创建环境
        env = QMixHandoverEnv(
            num_bs=config['num_bs'],
            num_uav=config['num_uav'],
            max_steps=config['num_steps'],
            seed=GLOBAL_SEED,
            bs_capacity_range=(500, 1000),
            pos_range=1000,
        )
        
        # 1. 测试传统算法
        print("  测试传统算法...")
        env.reset()
        traditional_algo = IntegratedHandoverAlgorithm(env)
        for step in range(config['num_steps']):
            traditional_algo.run_step()
            env.advance_env_only()
        traditional_sat = env.get_average_satisfaction()
        print(f"  传统算法满意度: {traditional_sat:.4f}")
        
        # 2. 测试增强算法
        print("  测试增强算法...")
        env.reset()
        enhanced_algo = EnhancedHandoverAlgorithm(env)
        for step in range(config['num_steps']):
            enhanced_algo.run_step()
            env.advance_env_only()
        enhanced_sat = env.get_average_satisfaction()
        print(f"  增强算法满意度: {enhanced_sat:.4f}")
        
        # 3. 测试MAPPO算法（使用预训练模型）
        print("  测试MAPPO算法...")
        # 这里我们创建一个简单的MAPPO代理进行测试
        # 注意：实际使用时应该加载训练好的模型
        agent = MAPPOAgent(
            num_agents=config['num_uav'],
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=64,
            critic_hidden_dim=128,
            use_biz_heads=True,
            use_attention_critic=True,
        )
        
        # 简单测试MAPPO（使用随机策略）
        env.reset()
        total_reward = 0
        for step in range(config['num_steps']):
            # 获取观测和业务类型
            obs_dict = {i: env.get_obs(i) for i in range(config['num_uav'])}
            biz_types = {i: env.uavs[i].business_type.value for i in range(config['num_uav'])}
            state = env.get_global_state()
            
            # 选择动作
            actions, _, _, _, _ = agent.select_actions(obs_dict, state, biz_types, training=False)
            
            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            total_reward += team_reward
        mappo_sat = env.get_average_satisfaction()
        print(f"  MAPPO算法满意度: {mappo_sat:.4f}")
        
        # 检查性能排序
        print("\n  性能排序:")
        algorithms = [
            ('传统算法', traditional_sat),
            ('增强算法', enhanced_sat),
            ('MAPPO算法', mappo_sat),
        ]
        
        # 按满意度排序
        sorted_algorithms = sorted(algorithms, key=lambda x: x[1], reverse=True)
        
        for i, (name, sat) in enumerate(sorted_algorithms, 1):
            print(f"    {i}. {name}: {sat:.4f}")
        
        # 验证排序是否正确
        expected_order = ['MAPPO算法', '增强算法', '传统算法']
        actual_order = [name for name, _ in sorted_algorithms]
        
        if actual_order == expected_order:
            print("  ✅ 性能排序正确: MAPPO > 增强 > 传统")
        else:
            print("  ❌ 性能排序不正确，需要进一步优化")
        
        # 验证增强算法是否超过传统算法
        if enhanced_sat > traditional_sat:
            print("  ✅ 增强算法超过传统算法")
        else:
            print("  ❌ 增强算法未超过传统算法，需要优化")
        
        # 验证MAPPO是否超过增强算法
        if mappo_sat > enhanced_sat:
            print("  ✅ MAPPO算法超过增强算法")
        else:
            print("  ❌ MAPPO算法未超过增强算法，需要优化")
    
    print("\n" + "="*80)
    print("算法性能测试完成")
    print("="*80)


def main():
    """主函数"""
    test_algorithm_performance()


if __name__ == "__main__":
    main()
