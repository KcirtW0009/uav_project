"""
分析决策阶段的兜底机制
查看为什么会有预测<0.55的候选被选中
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("决策阶段兜底机制分析")
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
fallback_count = 0  # 使用兜底的次数
tier1_count = 0      # 第一层（≥0.55）
tier2_count = 0      # 第二层（≥0.4）
tier3_count = 0      # 第三层（全部）
total_with_candidates = 0  # 有候选的决策次数

# 保存原始方法
original_make_decision = algo.make_intelligent_decision

def patched_make_decision(uav_id):
    global fallback_count, tier1_count, tier2_count, tier3_count, total_with_candidates

    from time import time
    t_start = time()
    algo.decision_calls += 1
    uav = env.uavs[uav_id]
    current_bs_id = uav.connected_bs_id

    # 复制决策逻辑（用于统计）
    if current_bs_id is None:
        return original_make_decision(uav_id)

    emergency = False
    if current_bs_id is not None:
        current_sinr = env.sinr_matrix[uav_id, current_bs_id]
        if current_sinr < algo.emergency_sinr_threshold or uav.current_satisfaction < algo.emergency_satisfaction_threshold:
            emergency = True

    if emergency:
        return original_make_decision(uav_id)

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
        t_end = time()
        algo.decision_time_history.append((t_end - t_start) * 1000)
        return None

    total_with_candidates += 1

    # 兜底机制统计
    high_success_candidates = [c for c in candidates if c[3] >= 0.55]
    if high_success_candidates:
        tier1_count += 1
    else:
        tier2_candidates = [c for c in candidates if c[3] >= 0.4]
        if tier2_candidates:
            tier2_count += 1
        else:
            tier3_count += 1
            fallback_count += 1

    # 调用原始方法
    result = original_make_decision(uav_id)

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
print(f"\n【决策阶段兜底机制统计】")
print(f"有候选的决策次数: {total_with_candidates}")
print(f"第一层（≥0.55）: {tier1_count} ({tier1_count/total_with_candidates*100:.1f}%)")
print(f"第二层（≥0.4但<0.55）: {tier2_count} ({tier2_count/total_with_candidates*100:.1f}%)")
print(f"第三层（全部兜底）: {tier3_count} ({tier3_count/total_with_candidates*100:.1f}%)")

# 总使用兜底的次数
total_fallback = tier2_count + tier3_count
print(f"\n使用兜底的总次数: {total_fallback} ({total_fallback/total_with_candidates*100:.1f}%)")

# 预测值分布
print(f"\n【执行阶段过滤统计】")
print(f"进入execute的切换: {algo.handover_attempts + algo.execution_filter_stats.get('below_threshold', 0)}")
print(f"预测≥0.55（计入）: {algo.handover_attempts}")
print(f"预测<0.55（被过滤）: {algo.execution_filter_stats.get('below_threshold', 0)}")

# 计算真实成功率
if algo.execution_filter_stats.get('below_threshold', 0) > 0:
    real_rate = algo.handover_successes / (algo.handover_attempts + algo.execution_filter_stats.get('below_threshold', 0))
    print(f"\n【真实切换成功率】")
    print(f"修正后成功率: {real_rate*100:.2f}%")
    print(f"当前显示成功率: {algo.handover_successes/algo.handover_attempts*100:.2f}%")
    print(f"统计偏差: {(algo.handover_successes/algo.handover_attempts - real_rate)*100:.2f}%")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("决策阶段的兜底机制（line 328-330）导致：")
print("1. 当没有≥0.55的候选时，会选择<0.55的候选")
print("2. 这些候选传给执行阶段，因预测<0.55被过滤")
print("3. 导致统计偏差：低预测切换不计入分母")
print("\n建议:")
print("修改 execute_handover 的统计逻辑：")
print("  - 所有进入execute的切换都计入 handover_attempts")
print("  - 预测<0.55的切换可以不执行，但应记录为失败")
print("=" * 80)
