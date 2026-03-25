"""
测试优化后的效果 - 短版本
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("优化效果测试（150步）")
print("=" * 80)

# 创建环境
env = EnhancedNetworkEnvironment(
    num_bs=10,
    num_uav=220,
    recognition_model=None,
    scaler=None,
    seed=GLOBAL_SEED,
    event_probability=0.0
)

# 创建增强算法
enh_algo = EnhancedHandoverAlgorithm(env)

# 运行增强算法
print("\n运行增强算法（150步）...")
for step in range(150):
    env.step()
    enh_algo.run_step(enable_load_balancing=True)

# 收集结果
enh_stats = enh_algo.get_detailed_stats()
env_stats = env.get_state_statistics()

print("\n" + "=" * 80)
print("增强算法结果")
print("=" * 80)

print(f"\n【切换成功率】")
print(f"决策调用次数: {enh_algo.decision_calls}")
print(f"切换尝试次数: {enh_algo.handover_attempts}")
print(f"切换成功次数: {enh_algo.handover_successes}")
print(f"切换成功率: {enh_stats['handover_success_rate']*100:.2f}%")

print(f"\n【错失机会】")
print(f"错失机会率: {enh_stats['missed_opportunity_rate']*100:.2f}%")

print(f"\n【失败原因】")
for reason, count in enh_stats['failure_reasons'].items():
    print(f"  {reason}: {count}")

print(f"\n【满足率】")
print(f"整体满足率: {env_stats['avg_satisfaction']:.3f}")
print(f"关键业务满足率: {env_stats['critical_satisfaction']:.3f}")

# 创建传统算法对比
env_trad = EnhancedNetworkEnvironment(
    num_bs=10,
    num_uav=220,
    recognition_model=None,
    scaler=None,
    seed=GLOBAL_SEED,
    event_probability=0.0
)

trad_algo = IntegratedHandoverAlgorithm(env_trad)

print("\n" + "=" * 80)
print("运行传统算法（150步）...")
for step in range(150):
    env_trad.step()
    trad_algo.run_step()

trad_stats = trad_algo.get_detailed_stats()
env_trad_stats = env_trad.get_state_statistics()

print("\n" + "=" * 80)
print("传统算法结果")
print("=" * 80)

print(f"\n【切换成功率】")
print(f"切换成功率: {trad_stats['handover_success_rate']*100:.2f}%")

print(f"\n【满足率】")
print(f"整体满足率: {env_trad_stats['avg_satisfaction']:.3f}")
print(f"关键业务满足率: {env_trad_stats['critical_satisfaction']:.3f}")

print("\n" + "=" * 80)
print("对比总结")
print("=" * 80)

print(f"\n切换成功率:")
print(f"  增强算法: {enh_stats['handover_success_rate']*100:.2f}%")
print(f"  传统算法: {trad_stats['handover_success_rate']*100:.2f}%")
print(f"  差异: {(enh_stats['handover_success_rate'] - trad_stats['handover_success_rate'])*100:+.2f}%")

print(f"\n整体满足率:")
print(f"  增强算法: {env_stats['avg_satisfaction']:.3f}")
print(f"  传统算法: {env_trad_stats['avg_satisfaction']:.3f}")
print(f"  提升: {(env_stats['avg_satisfaction'] - env_trad_stats['avg_satisfaction'])*100:+.2f}%")

print("\n" + "=" * 80)
