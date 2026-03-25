#!/usr/bin/env python
"""快速测试切换成功率"""

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model
from uav_system.experiments_exp5 import Experiment5

set_global_seed(GLOBAL_SEED)
recognition_model, scaler = train_or_load_recognition_model(force_retrain=False)

print('=== 快速测试：实验5 baseline场景 ===')
env, stats = Experiment5._run_single_scenario('baseline', 0, recognition_model, scaler, 350, 'enhanced')
print(f"切换成功率: {stats['handover_success_rate']*100:.2f}%")
print(f"切换尝试: {stats.get('handover_attempts', 'N/A')}, 切换成功: {stats.get('handover_successes', 'N/A')}")
print(f"决策调用: {stats.get('decision_calls', 'N/A')}, 过滤次数: {stats.get('missed_opportunity', 'N/A')}")
print(f"决策过滤原因: {stats.get('decision_filters', {})}")
