# -*- coding: utf-8 -*-
"""
多负载率实验结果分析
"""

import json
import os
import numpy as np

scenarios = ['low', 'medium', 'high']
results = {}

for scenario in scenarios:
    # 查找对应的日志文件
    log_dir = 'experiment_logs'
    dirs = [d for d in os.listdir(log_dir) if d.startswith(f'mappo_{scenario}_')]
    if not dirs:
        continue
    
    latest_dir = sorted(dirs)[-1]
    dir_path = os.path.join(log_dir, latest_dir)
    
    # 尝试找到任何json文件
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    if not json_files:
        continue
    
    json_file = os.path.join(dir_path, json_files[0])
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    history = data.get('history', {})
    
    # 计算关键指标
    rewards = history.get('reward', [])
    sats = history.get('satisfaction', [])
    
    if rewards and sats:
        results[scenario] = {
            'final_reward': rewards[-1] if rewards else 0,
            'avg_reward': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
            'final_sat': sats[-1] if sats else 0,
            'avg_sat': np.mean(sats[-20:]) if len(sats) >= 20 else np.mean(sats),
            'max_sat': max(sats) if sats else 0,
            'reward_variance': np.var(rewards[-20:]) if len(rewards) >= 20 else np.var(rewards),
        }

# 打印结果
print('='*70)
print('多负载率实验结果分析')
print('='*70)

scenario_names = {
    'low': '低负载 (32 UAVs, ~30%)',
    'medium': '中负载 (64 UAVs, ~60%)',
    'high': '高负载 (128 UAVs, ~88%)'
}

for scenario in scenarios:
    if scenario in results:
        r = results[scenario]
        print(f'\n{scenario_names.get(scenario, scenario)}:')
        print(f'  最终奖励: {r["final_reward"]:.2f}')
        print(f'  平均奖励(后20轮): {r["avg_reward"]:.2f}')
        print(f'  最终满意度: {r["final_sat"]:.4f}')
        print(f'  平均满意度(后20轮): {r["avg_sat"]:.4f}')
        print(f'  最高满意度: {r["max_sat"]:.4f}')
        print(f'  奖励方差: {r["reward_variance"]:.4f}')

print('\n' + '='*70)
print('对比分析')
print('='*70)

if len(results) >= 2:
    print('\n满意度对比:')
    for scenario in scenarios:
        if scenario in results:
            print(f'  {scenario_names.get(scenario, scenario)}: {results[scenario]["avg_sat"]:.4f}')
    
    print('\n奖励对比:')
    for scenario in scenarios:
        if scenario in results:
            print(f'  {scenario_names.get(scenario, scenario)}: {results[scenario]["avg_reward"]:.2f}')
    
    # 计算负载率影响
    if 'low' in results and 'high' in results:
        sat_drop = results['low']['avg_sat'] - results['high']['avg_sat']
        reward_drop = results['low']['avg_reward'] - results['high']['avg_reward']
        print(f'\n负载率影响 (低负载 → 高负载):')
        print(f'  满意度下降: {sat_drop:.4f} ({sat_drop/results["low"]["avg_sat"]*100:.1f}%)')
        print(f'  奖励下降: {reward_drop:.2f} ({reward_drop/results["low"]["avg_reward"]*100:.1f}%)')

print('\n' + '='*70)
print('结论')
print('='*70)

if len(results) == 3:
    print('\n✓ 所有三个负载率场景实验完成')
    
    # 找出最佳场景
    best_scenario = max(results.keys(), key=lambda x: results[x]['avg_sat'])
    print(f'✓ 最佳性能场景: {scenario_names.get(best_scenario, best_scenario)}')
    print(f'  满意度: {results[best_scenario]["avg_sat"]:.4f}')
    
    # 稳定性分析
    most_stable = min(results.keys(), key=lambda x: results[x]['reward_variance'])
    print(f'✓ 最稳定场景: {scenario_names.get(most_stable, most_stable)}')
    print(f'  奖励方差: {results[most_stable]["reward_variance"]:.4f}')

print('\n')
