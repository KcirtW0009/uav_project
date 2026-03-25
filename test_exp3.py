#!/usr/bin/env python
"""快速测试实验3的切换成功率"""

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)
recognition_model, scaler = train_or_load_recognition_model(force_retrain=False)

print('=== 快速测试：实验3场景（220 UAV, 350步）===')
env = EnhancedNetworkEnvironment(
    num_bs=8, num_uav=220,
    recognition_model=recognition_model, scaler=scaler,
    seed=GLOBAL_SEED, event_probability=0.05
)
algo = EnhancedHandoverAlgorithm(env)

for step in range(350):
    env.step()
    algo.run_step(enable_load_balancing=True)

stats = env.get_state_statistics()
stats.update(algo.get_detailed_stats())
print(f"切换成功率: {stats['handover_success_rate']*100:.2f}%")
print(f"切换尝试: {stats.get('handover_attempts', 'N/A')}, 切换成功: {stats.get('handover_successes', 'N/A')}")
print(f"决策调用: {stats.get('decision_calls', 'N/A')}, 过滤次数: {stats.get('missed_opportunity', 'N/A')}")
print(f"决策过滤原因: {stats.get('decision_filters', {})}")
