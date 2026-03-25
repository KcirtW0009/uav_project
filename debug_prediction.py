#!/usr/bin/env python
"""调试预测和实际执行的一致性"""

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm
import numpy as np

set_global_seed(GLOBAL_SEED)
recognition_model, scaler = train_or_load_recognition_model(force_retrain=False)

env = EnhancedNetworkEnvironment(
    num_bs=8, num_uav=180,
    recognition_model=recognition_model, scaler=scaler,
    seed=GLOBAL_SEED, event_probability=0.05
)
algo = EnhancedHandoverAlgorithm(env)

# 记录预测和实际的差异
prediction_vs_actual = []

print('=== 运行10步并记录预测vs实际 ===')
for step in range(10):
    env.step()
    handover_count, _ = algo.run_step(enable_load_balancing=True)
    if handover_count > 0:
        print(f"  步骤 {step}: 执行了 {handover_count} 次切换")

print(f'\n总切换尝试: {algo.handover_attempts}')
print(f'总切换成功: {algo.handover_successes}')
print(f'切换成功率: {algo.handover_successes / max(algo.handover_attempts, 1) * 100:.2f}%')

print(f'\n失败原因统计: {algo.failure_reasons}')

# 检查一个典型基站的负载情况
print(f'\n=== 基站负载情况 ===')
for bs_id, bs in env.base_stations.items():
    print(f"  基站 {bs_id}: 总容量={bs.total_capacity:.2f}, 已用={bs.used_capacity:.2f}, 可用={bs.available_capacity:.2f}, 负载率={bs.load_ratio:.2%}")
