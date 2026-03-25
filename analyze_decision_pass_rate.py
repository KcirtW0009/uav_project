"""
分析决策通过率为什么这么低
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("决策通过率分析")
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
algo = EnhancedHandoverAlgorithm(env)

# 统计
decision_return_none = 0  # 返回None的次数
decision_with_candidate = 0  # 有候选但被过滤的次数
decision_pass = 0  # 通过的次数
no_candidates_count = 0  # 无候选的次数
below_threshold_count = 0  # 候选低于阈值的次数

# 保存原始方法
original_make_decision = algo.make_intelligent_decision

def patched_make_decision(uav_id):
    global decision_return_none, decision_with_candidate, decision_pass
    global no_candidates_count, below_threshold_count

    from time import time
    t_start = time()
    algo.decision_calls += 1
    uav = env.uavs[uav_id]
    current_bs_id = uav.connected_bs_id

    if current_bs_id is None:
        result = original_make_decision(uav_id)
        if result is None:
            decision_return_none += 1
        else:
            decision_pass += 1
        return result

    emergency = False
    if current_bs_id is not None:
        current_sinr = env.sinr_matrix[uav_id, current_bs_id]
        if current_sinr < algo.emergency_sinr_threshold or uav.current_satisfaction < algo.emergency_satisfaction_threshold:
            emergency = True

    if emergency:
        result = original_make_decision(uav_id)
        if result is None:
            decision_return_none += 1
        else:
            decision_pass += 1
        return result

    feasible_ratios = uav.qos_profile.get_feasible_downgrade_ratios()
    candidates = []
    for bs_id in env.base_stations.keys():
        if bs_id == current_bs_id:
            continue
        for ratio in feasible_ratios:
            utility, is_feasible = algo.calculate_utility_with_downgrade(uav, bs_id, ratio)
            if is_feasible and ratio >= 0.6:
                success_prob = algo.predict_handover_success(uav, bs_id, ratio)
                candidates.append((bs_id, ratio, utility, success_prob))

    if not candidates:
        no_candidates_count += 1
        t_end = time()
        algo.decision_time_history.append((t_end - t_start) * 1000)
        return None

    # 统计候选
    UNIFIED_THRESHOLD = 0.52
    high_success_candidates = [c for c in candidates if c[3] >= UNIFIED_THRESHOLD]

    if not high_success_candidates:
        below_threshold_count += 1

    # 调用原始方法
    result = original_make_decision(uav_id)

    if result is None:
        decision_return_none += 1
        if candidates and not high_success_candidates:
            decision_with_candidate += 1
    else:
        decision_pass += 1

    return result

# 应用补丁
algo.make_intelligent_decision = patched_make_decision

# 运行仿真
print("\n运行仿真（20步）...")
for step in range(20):
    env.step()
    algo.run_step(enable_load_balancing=True)

# 恢复原始方法
algo.make_intelligent_decision = original_make_decision

# 输出统计
total_decisions = algo.decision_calls

print(f"\n【决策通过率分析】")
print(f"总决策次数: {total_decisions}")
print(f"返回None（未切换）: {decision_return_none} ({decision_return_none/total_decisions*100:.2f}%)")
print(f"  - 无候选: {no_candidates_count} ({no_candidates_count/total_decisions*100:.2f}%)")
print(f"  - 有候选但被过滤: {decision_with_candidate} ({decision_with_candidate/total_decisions*100:.2f}%)")
print(f"  - 其他（emergency等）: {decision_return_none - no_candidates_count - decision_with_candidate}")
print(f"返回切换决策（通过）: {decision_pass} ({decision_pass/total_decisions*100:.2f}%)")
print(f"候选<0.52阈值: {below_threshold_count} ({below_threshold_count/total_decisions*100:.2f}%)")

print(f"\n【关键指标】")
print(f"决策通过率: {decision_pass/total_decisions*100:.2f}%")
print(f"切换尝试次数: {algo.handover_attempts}")
print(f"切换成功次数: {algo.handover_successes}")
print(f"切换成功率: {algo.handover_successes/algo.handover_attempts*100:.2f}%")

print("\n" + "=" * 80)
