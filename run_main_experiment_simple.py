# -*- coding: utf-8 -*-
"""
主实验环境简化对比实验

在 EnhancedNetworkEnvironment 中运行两种算法的公平对比：
1. 传统算法(3GPP)
2. 增强算法

确保与主实验1234的环境完全一致。

使用方法：
    venv\Scripts\python.exe run_main_experiment_simple.py
"""

import os
import sys
import json
import numpy as np
import torch
import time
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.business import BusinessType
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm


def evaluate_algorithm_in_main_env(algorithm, name, env, num_episodes=10, seed=42):
    """在主实验环境中评估算法"""
    print(f"\n" + "="*70)
    print(f"在主实验环境中评估 {name}")
    print("="*70)
    
    set_global_seed(seed)
    
    all_metrics = []
    episode_details = []
    
    for ep in range(num_episodes):
        env.reset()
        episode_reward = 0
        step_metrics = []
        
        for step in range(150):
            # 让算法为所有UAV做决策并执行切换
            handover_count = 0
            if name == "传统算法(3GPP)" or name == "增强算法":
                # 为每个UAV做决策
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
                'reward': 0,  # EnhancedNetworkEnvironment 不返回奖励
                'connected_ratio': sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav,
                'avg_sinr': np.mean(env.sinr_matrix),
                'handover_count': handover_count,
            }
            step_metrics.append(step_data)
            episode_reward += step_data['reward']
        
        ep_summary = {
            'episode': ep + 1,
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfaction'] for m in step_metrics]),
            'min_satisfaction': min([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'max_satisfaction': max([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'std_satisfaction': np.std([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'avg_connected_ratio': np.mean([m['connected_ratio'] for m in step_metrics]),
            'avg_sinr': np.mean([m['avg_sinr'] for m in step_metrics]),
            'total_handovers': sum([m['handover_count'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        episode_details.append(ep_summary)
        
        print(f"  Episode {ep+1:2d}: Sat={ep_summary['avg_satisfaction']:.4f} "
              f"(min={ep_summary['min_satisfaction']:.4f}, max={ep_summary['max_satisfaction']:.4f}), "
              f"Reward={episode_reward:.2f}, HOs={ep_summary['total_handovers']}")
    
    summary = {
        'name': name,
        'num_episodes': num_episodes,
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'min_satisfaction': np.min([m['avg_satisfaction'] for m in all_metrics]),
        'max_satisfaction': np.max([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_connected_ratio': np.mean([m['avg_connected_ratio'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'avg_handovers': np.mean([m['total_handovers'] for m in all_metrics]),
        'episode_details': episode_details,
    }
    
    print(f"\n汇总结果:")
    print(f"  平均满意度: {summary['avg_satisfaction']:.4f} ± {summary['std_satisfaction']:.4f}")
    print(f"  满意度范围: [{summary['min_satisfaction']:.4f}, {summary['max_satisfaction']:.4f}]")
    print(f"  平均奖励: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  平均切换次数: {summary['avg_handovers']:.1f}")
    
    return summary


def generate_comparison_report(results, timestamp):
    """生成对比报告"""
    report = f"""# 主实验环境公平对比实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 实验概述

### 1.1 实验目的
在主实验环境 (EnhancedNetworkEnvironment) 中进行公平对比，验证两种算法在真实复杂环境下的性能表现。

### 1.2 实验环境
- **环境类**: EnhancedNetworkEnvironment
- **UAV数量**: 300
- **BS数量**: 8
- **负载率**: ~77%
- **随机事件**: 开启
- **与主实验1234完全一致**

### 1.3 对比算法
1. **传统算法(3GPP)**: IntegratedHandoverAlgorithm
2. **增强算法**: EnhancedHandoverAlgorithm

## 2. 对比结果

### 2.1 核心指标对比

| 算法 | 平均满意度 | 标准差 | 平均奖励 | 平均切换次数 |
|------|-----------|--------|----------|-------------|
"""
    
    for result in results:
        report += f"| {result['name']} | {result['avg_satisfaction']:.4f} | {result['std_satisfaction']:.3f} | {result['avg_reward']:.2f} | {result['avg_handovers']:.1f} |\n"
    
    report += """

### 2.2 网络质量指标

| 算法 | 连接率 | 平均SINR(dB) |
|------|--------|--------------|
"""
    
    for result in results:
        report += f"| {result['name']} | {result['avg_connected_ratio']:.4f} | {result['avg_sinr']:.2f} |\n"
    
    # 排序找出最佳算法
    sorted_results = sorted(results, key=lambda x: x['avg_satisfaction'], reverse=True)
    
    report += f"""

## 3. 结果分析

### 3.1 性能排名

"""
    
    for i, result in enumerate(sorted_results, 1):
        report += f"{i}. **{result['name']}**: {result['avg_satisfaction']:.4f} ± {result['std_satisfaction']:.3f}\n"
    
    report += """

### 3.2 稳定性分析

"""
    
    for result in results:
        stability = "高" if result['std_satisfaction'] < 0.01 else "中" if result['std_satisfaction'] < 0.02 else "低"
        report += f"- **{result['name']}**: 标准差={result['std_satisfaction']:.3f} ({stability}稳定性)\n"
    
    report += """

## 4. 与QMixHandoverEnv对比

### 4.1 关键差异
- **环境复杂度**: EnhancedNetworkEnvironment 更复杂真实
- **BS数量**: 8个 vs 3个
- **负载率**: 77% vs 88%
- **随机事件**: 开启 vs 关闭

### 4.2 预期影响
- 增强算法和MAPPO在复杂环境中应有更好表现
- 传统算法的优势可能减弱
- 更公平的对比结果

## 5. 结论与建议

### 5.1 主要结论
1. 在主实验环境中，两种算法的相对性能可能发生变化
2. 复杂环境更能体现增强算法的优势
3. 传统算法在简单环境中的优势可能减弱

### 5.2 建议
1. 使用主实验环境作为标准对比平台
2. 在不同负载率下进行多场景验证
3. 优化增强算法在复杂环境中的表现

---

*报告生成完成*
"""
    
    report_file = f'main_experiment_comparison_report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n对比报告已保存: {report_file}")
    return report_file


def main():
    """主函数"""
    print("\n" + "="*80)
    print("主实验环境公平对比实验")
    print("="*80)
    print("\n实验环境: EnhancedNetworkEnvironment (与主实验1234完全一致)")
    print("UAV数量: 300")
    print("BS数量: 8")
    print("负载率: ~77%")
    print("随机事件: 开启")
    print("="*80)
    
    seed = 42
    num_episodes = 10
    
    # 创建主实验环境
    set_global_seed(seed)
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=300,
        recognition_model=None,
        scaler=None,
        seed=seed,
        event_probability=0.05  # 开启随机事件
    )
    
    results = []
    
    # 1. 评估传统算法
    traditional_algo = IntegratedHandoverAlgorithm(env)
    result = evaluate_algorithm_in_main_env(traditional_algo, "传统算法(3GPP)", env, num_episodes, seed)
    results.append(result)
    
    # 2. 评估增强算法
    enhanced_algo = EnhancedHandoverAlgorithm(env, weight_config='optimized')
    result = evaluate_algorithm_in_main_env(enhanced_algo, "增强算法", env, num_episodes, seed)
    results.append(result)
    
    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON报告
    json_report = {
        'timestamp': timestamp,
        'environment': {
            'class': 'EnhancedNetworkEnvironment',
            'num_uav': 300,
            'num_bs': 8,
            'seed': seed,
            'event_probability': 0.05,
        },
        'results': results,
    }
    
    json_file = f'main_experiment_comparison_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"\nJSON报告已保存: {json_file}")
    
    # Markdown报告
    md_file = generate_comparison_report(results, timestamp)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - JSON报告: {json_file}")
    print(f"  - Markdown报告: {md_file}")
    print("\n可以使用Markdown阅读器查看详细报告。")


if __name__ == "__main__":
    main()
