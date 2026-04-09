# -*- coding: utf-8 -*-
"""
增强算法调参脚本

针对主实验环境（EnhancedNetworkEnvironment）优化增强算法参数，目标是：
1. 减少不必要的切换次数
2. 提高稳定性
3. 超越传统算法的性能

使用方法：
    venv\Scripts\python.exe optimize_enhanced_algorithm.py
"""

import os
import sys
import json
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm


def evaluate_algorithm(algorithm, name, env, num_episodes=5, seed=42):
    """评估算法性能"""
    print(f"\n评估 {name}")
    print("-" * 50)
    
    set_global_seed(seed)
    
    all_metrics = []
    
    for ep in range(num_episodes):
        env.reset()
        episode_reward = 0
        step_metrics = []
        
        for step in range(150):
            # 让算法为所有UAV做决策并执行切换
            handover_count = 0
            for uav_id in range(env.num_uav):
                if name == "传统算法(3GPP)":
                    decision = algorithm.make_decision(uav_id)
                else:
                    decision = algorithm.make_intelligent_decision(uav_id)
                
                if decision is not None:
                    target_bs, downgrade_ratio = decision
                    current_bs = env.uavs[uav_id].connected_bs_id
                    if target_bs != current_bs:
                        if algorithm.execute_handover(uav_id, target_bs, downgrade_ratio):
                            handover_count += 1
            
            # 执行环境步进
            env.step()
            
            # 收集指标
            step_data = {
                'step': step,
                'satisfaction': np.mean([uav.current_satisfaction for uav in env.uavs.values()]),
                'handover_count': handover_count,
            }
            step_metrics.append(step_data)
            episode_reward += step_data['satisfaction']
        
        ep_summary = {
            'episode': ep + 1,
            'avg_satisfaction': np.mean([m['satisfaction'] for m in step_metrics]),
            'total_handovers': sum([m['handover_count'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        
        print(f"  Episode {ep+1:2d}: Sat={ep_summary['avg_satisfaction']:.4f}, HOs={ep_summary['total_handovers']}")
    
    summary = {
        'name': name,
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'avg_handovers': np.mean([m['total_handovers'] for m in all_metrics]),
    }
    
    print(f"\n汇总结果:")
    print(f"  平均满意度: {summary['avg_satisfaction']:.4f} ± {summary['std_satisfaction']:.4f}")
    print(f"  平均切换次数: {summary['avg_handovers']:.1f}")
    
    return summary


def optimize_enhanced_algorithm():
    """优化增强算法参数"""
    print("\n" + "="*80)
    print("增强算法调参")
    print("="*80)
    print("目标：减少切换次数，提高稳定性，超越传统算法")
    
    seed = 42
    num_episodes = 5
    
    # 创建主实验环境
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=300,
        recognition_model=None,
        scaler=None,
        seed=seed,
        event_probability=0.05
    )
    
    # 评估传统算法作为基准
    print("\n" + "="*60)
    print("评估传统算法（基准）")
    print("="*60)
    traditional_algo = IntegratedHandoverAlgorithm(env)
    traditional_result = evaluate_algorithm(traditional_algo, "传统算法(3GPP)", env, num_episodes, seed)
    
    # 定义参数搜索空间
    param_space = [
        # (base_threshold, epsilon, handover_cooldown, use_load_mode)
        (0.01, 0.01, 5, True),    # 保守策略
        (0.005, 0.02, 3, True),   # 中等策略
        (0.008, 0.01, 4, True),   # 平衡策略
        (0.012, 0.005, 6, True),  # 更保守
        (0.006, 0.015, 4, True),  # 稍激进
    ]
    
    best_score = -float('inf')
    best_params = None
    best_result = None
    
    print("\n" + "="*60)
    print("开始参数搜索")
    print("="*60)
    
    for i, params in enumerate(param_space):
        base_threshold, epsilon, handover_cooldown, use_load_mode = params
        
        print(f"\n参数组合 {i+1}/{len(param_space)}")
        print(f"  base_threshold: {base_threshold}")
        print(f"  epsilon: {epsilon}")
        print(f"  handover_cooldown: {handover_cooldown}")
        print(f"  use_load_mode: {use_load_mode}")
        
        # 创建增强算法实例并设置参数
        enhanced_algo = EnhancedHandoverAlgorithm(env, weight_config='optimized')
        
        # 修改参数
        enhanced_algo.base_threshold = base_threshold
        enhanced_algo.epsilon = epsilon
        enhanced_algo.handover_cooldown = handover_cooldown  # 修复变量名
        enhanced_algo.use_load_mode = use_load_mode  # 添加设置use_load_mode
        
        # 确保handover_cooldown_timer存在
        if not hasattr(enhanced_algo, 'handover_cooldown_timer'):
            enhanced_algo.handover_cooldown_timer = {}
        
        # 评估
        result = evaluate_algorithm(enhanced_algo, "增强算法", env, num_episodes, seed)
        
        # 计算综合评分（满意度权重0.7，切换次数权重0.3）
        # 切换次数越少越好，所以取倒数
        handover_score = 1 / (1 + result['avg_handovers'] / 1000)  # 归一化到0-1
        score = result['avg_satisfaction'] * 0.7 + handover_score * 0.3
        
        print(f"  综合评分: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            best_result = result
            print("  *** 新的最佳参数 ***")
    
    print("\n" + "="*80)
    print("调参结果")
    print("="*80)
    
    print("传统算法基准:")
    print(f"  平均满意度: {traditional_result['avg_satisfaction']:.4f}")
    print(f"  平均切换次数: {traditional_result['avg_handovers']:.1f}")
    
    print("\n最佳增强算法:")
    print(f"  最佳参数: {best_params}")
    print(f"  平均满意度: {best_result['avg_satisfaction']:.4f}")
    print(f"  平均切换次数: {best_result['avg_handovers']:.1f}")
    print(f"  综合评分: {best_score:.4f}")
    
    # 保存最佳参数
    best_config = {
        'base_threshold': best_params[0],
        'epsilon': best_params[1],
        'handover_cooldown': best_params[2],
        'use_load_mode': best_params[3],
        'performance': best_result,
        'benchmark': traditional_result,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    config_file = f'enhanced_algorithm_best_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n最佳配置已保存: {config_file}")
    
    return best_config


def test_best_config(best_config):
    """测试最佳配置"""
    print("\n" + "="*80)
    print("测试最佳配置")
    print("="*80)
    
    seed = 42
    num_episodes = 10
    
    # 创建环境
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=300,
        recognition_model=None,
        scaler=None,
        seed=seed,
        event_probability=0.05
    )
    
    # 创建增强算法实例
    enhanced_algo = EnhancedHandoverAlgorithm(env, weight_config='optimized')
    
    # 设置最佳参数
    enhanced_algo.base_threshold = best_config['base_threshold']
    enhanced_algo.epsilon = best_config['epsilon']
    enhanced_algo.handover_cooldown = best_config['handover_cooldown']  # 修复变量名
    enhanced_algo.use_load_mode = best_config['use_load_mode']  # 添加设置use_load_mode
    
    # 确保handover_cooldown_timer存在
    if not hasattr(enhanced_algo, 'handover_cooldown_timer'):
        enhanced_algo.handover_cooldown_timer = {}
    
    # 评估
    result = evaluate_algorithm(enhanced_algo, "增强算法（最佳配置）", env, num_episodes, seed)
    
    # 评估传统算法
    traditional_algo = IntegratedHandoverAlgorithm(env)
    traditional_result = evaluate_algorithm(traditional_algo, "传统算法(3GPP)", env, num_episodes, seed)
    
    print("\n" + "="*80)
    print("最终对比结果")
    print("="*80)
    
    print("传统算法:")
    print(f"  平均满意度: {traditional_result['avg_satisfaction']:.4f}")
    print(f"  平均切换次数: {traditional_result['avg_handovers']:.1f}")
    
    print("\n增强算法（最佳配置）:")
    print(f"  平均满意度: {result['avg_satisfaction']:.4f}")
    print(f"  平均切换次数: {result['avg_handovers']:.1f}")
    
    if result['avg_satisfaction'] > traditional_result['avg_satisfaction']:
        improvement = (result['avg_satisfaction'] - traditional_result['avg_satisfaction']) * 100
        print(f"\n🎉 增强算法超越传统算法 {improvement:.2f}%")
    else:
        gap = (traditional_result['avg_satisfaction'] - result['avg_satisfaction']) * 100
        print(f"\n⚠️  增强算法落后传统算法 {gap:.2f}%")
    
    return result


def main():
    """主函数"""
    print("\n" + "="*80)
    print("增强算法调参与优化")
    print("="*80)
    print("环境: EnhancedNetworkEnvironment (主实验环境)")
    print("UAV数量: 300")
    print("BS数量: 8")
    
    # 优化参数
    best_config = optimize_enhanced_algorithm()
    
    # 测试最佳配置
    test_best_config(best_config)
    
    print("\n" + "="*80)
    print("调参完成！")
    print("="*80)


if __name__ == "__main__":
    main()
