"""
快速分析：切换成功率低的原因
检查决策阶段的过滤统计
"""

import numpy as np
from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm

# 加载模型
print("加载识别模型...")
recognition_model, scaler = train_or_load_recognition_model(force_retrain=False, compare_models=False, verbose=False)

# 快速测试
print("\n快速测试（正常场景，100步）...")
set_global_seed(GLOBAL_SEED)

env = EnhancedNetworkEnvironment(
    num_bs=8, num_uav=180,
    recognition_model=recognition_model, scaler=scaler,
    seed=GLOBAL_SEED, scenario='default', event_probability=0.05
)

algo = EnhancedHandoverAlgorithm(env)

# 运行100步
for step in range(100):
    env.step()
    algo.run_step(enable_load_balancing=True)

# 统计决策日志
print(f"\n决策阶段统计：")
print(f"  总决策次数: {len(algo.decision_log)}")

if algo.decision_log:
    filtered_low_load = sum(1 for log in algo.decision_log if log.get('filter_reason') == 'low_load')
    filtered_low_prob = sum(1 for log in algo.decision_log if log.get('filter_reason') == 'low_success_prob')
    filtered_low_gain = sum(1 for log in algo.decision_log if log.get('filter_reason') == 'low_gain')
    executed = sum(1 for log in algo.decision_log if 'filter_reason' not in log or log.get('filter_reason') is None)

    print(f"  过滤-低负载: {filtered_low_load} ({filtered_low_load/len(algo.decision_log)*100:.1f}%)")
    print(f"  过滤-低成功率: {filtered_low_prob} ({filtered_low_prob/len(algo.decision_log)*100:.1f}%)")
    print(f"  过滤-低增益: {filtered_low_gain} ({filtered_low_gain/len(algo.decision_log)*100:.1f}%)")
    print(f"  执行切换: {executed} ({executed/len(algo.decision_log)*100:.1f}%)")

    # 过滤率
    total_filtered = filtered_low_load + filtered_low_prob + filtered_low_gain
    print(f"\n总过滤率: {total_filtered/len(algo.decision_log)*100:.1f}%")
    print(f"执行率: {executed/len(algo.decision_log)*100:.1f}%")

# 切换统计
print(f"\n执行阶段统计：")
print(f"  切换尝试次数: {algo.handover_attempts}")
print(f"  切换成功次数: {algo.handover_successes}")
print(f"  切换成功率: {algo.handover_successes/max(algo.handover_attempts,1)*100:.1f}%")
print(f"  错失机会次数: {algo.missed_opportunity}")

print(f"\n失败原因分布:")
for reason, count in algo.failure_reasons.items():
    print(f"  {reason}: {count}")

# 对比传统算法
print(f"\n{'='*60}")
print(f"传统算法对比（相同环境）...")
set_global_seed(GLOBAL_SEED)

env_trad = EnhancedNetworkEnvironment(
    num_bs=8, num_uav=180,
    recognition_model=recognition_model, scaler=scaler,
    seed=GLOBAL_SEED, scenario='default', event_probability=0.05
)

algo_trad = IntegratedHandoverAlgorithm(env_trad)

for step in range(100):
    env_trad.step()
    algo_trad.run_step()

print(f"\n传统算法统计：")
print(f"  切换尝试次数: {algo_trad.handover_attempts}")
print(f"  切换成功次数: {algo_trad.handover_successes}")
print(f"  切换成功率: {algo_trad.handover_successes/max(algo_trad.handover_attempts,1)*100:.1f}%")
print(f"  错失机会次数: {algo_trad.missed_opportunity}")

print(f"\n失败原因分布:")
for reason, count in algo_trad.failure_reasons.items():
    print(f"  {reason}: {count}")
