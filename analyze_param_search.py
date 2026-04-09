# -*- coding: utf-8 -*-
"""分析参数搜索结果"""

import json
import os

# 读取结果文件
result_file = 'experiment_results/param_search/param_search_20260409_034053.json'

if not os.path.exists(result_file):
    print(f"Result file not found: {result_file}")
    exit(1)

with open(result_file, 'r') as f:
    results = json.load(f)

print(f'Total configurations completed: {len(results)}')
print()

# 按满意度排序
sorted_results = sorted(results, key=lambda x: x.get('final_sat', 0), reverse=True)

print('Top 5 configurations by satisfaction:')
print('=' * 80)
for i, r in enumerate(sorted_results[:5]):
    config_id = r['config_id']
    print(f'\nRank {i+1} (Config {config_id}):')
    print(f'  Final Satisfaction: {r["final_sat"]:.4f}')
    print(f'  Final Reward: {r["final_reward"]:.2f}')
    print(f'  Reward Variance: {r["reward_variance"]:.4f}')
    print(f'  Convergence Speed: {r["convergence_speed"]} episodes')
    print(f'  Key Params:')
    cfg = r['config']
    print(f'    actor_lr: {cfg.get("actor_lr")}')
    print(f'    critic_lr: {cfg.get("critic_lr")}')
    print(f'    clip_epsilon: {cfg.get("clip_epsilon")}')
    print(f'    entropy_coef: {cfg.get("entropy_coef")}')
    print(f'    gae_lambda: {cfg.get("gae_lambda")}')

# 参数敏感性分析
print('\n\nParameter Sensitivity Analysis:')
print('=' * 80)

param_names = ['actor_lr', 'critic_lr', 'clip_epsilon', 'entropy_coef', 'gae_lambda']

for param in param_names:
    param_values = {}
    for r in results:
        val = r['config'].get(param)
        if val not in param_values:
            param_values[val] = []
        param_values[val].append(r['final_sat'])
    
    print(f'\n{param}:')
    for val, sats in sorted(param_values.items()):
        import numpy as np
        mean_sat = sum(sats) / len(sats)
        print(f'  {val}: avg_sat={mean_sat:.4f} (n={len(sats)})')

print('\n\nAnalysis complete!')
