"""
分析资源压力，理解为什么兜底机制被频繁使用
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm
from uav_system.business import QOS_PROFILES

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("资源压力分析：为什么兜底机制频繁使用？")
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

# 计算容量利用率
print(f"\n【系统容量分析】")
print(f"基站数量: {len(env.base_stations)}")
print(f"UAV数量: {len(env.uavs)}")

# 计算总需求
total_demand = 0
demand_by_business = {}
for uav in env.uavs.values():
    qos = QOS_PROFILES[uav.business_type]
    demand = qos.ideal_rate
    total_demand += demand
    business_type = uav.business_type.name
    demand_by_business[business_type] = demand_by_business.get(business_type, 0) + demand

# 计算总容量
total_capacity = sum(bs.capacity for bs in env.base_stations.values())

print(f"\n按业务类型的需求:")
for business, demand in demand_by_business.items():
    print(f"  {business}: {demand:.1f} Mbps")

print(f"\n总需求: {total_demand:.1f} Mbps")
print(f"总容量: {total_capacity:.1f} Mbps")
print(f"容量利用率: {total_demand/total_capacity*100:.1f}%")

# 创建增强算法并运行
algo = EnhancedHandoverAlgorithm(env)

# 监控资源状态
load_history = []
candidate_success_probs = []
candidates_per_decision = []

# 保存原始方法
original_make_decision = algo.make_intelligent_decision

def patched_make_decision(uav_id):
    from time import time
    t_start = time()
    algo.decision_calls += 1
    uav = env.uavs[uav_id]
    current_bs_id = uav.connected_bs_id

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

    # 记录候选信息
    candidates_per_decision.append(len(candidates))
    all_probs = [c[3] for c in candidates]
    candidate_success_probs.extend(all_probs)

    # 调用原始方法
    result = original_make_decision(uav_id)

    return result

# 应用补丁
algo.make_intelligent_decision = patched_make_decision

# 运行仿真
print(f"\n运行仿真（50步）...")
for step in range(50):
    env.step()

    # 记录负载
    load_ratios = [bs.load_ratio for bs in env.base_stations.values()]
    load_history.append(load_ratios)

    algo.run_step(enable_load_balancing=True)

# 恢复原始方法
algo.make_intelligent_decision = original_make_decision

# 分析负载
print(f"\n【基站负载分析】")
load_arr = np.array(load_history)
avg_load = np.mean(load_arr, axis=0)
std_load = np.std(load_arr, axis=0)
max_load = np.max(load_arr, axis=0)
min_load = np.min(load_arr, axis=0)

print(f"基站\t平均负载\t标准差\t最大负载\t最小负载")
for i, (avg, std, mx, mn) in enumerate(zip(avg_load, std_load, max_load, min_load)):
    print(f"  BS{i}\t{avg:.3f}\t\t{std:.3f}\t{mx:.3f}\t{mn:.3f}")

print(f"\n整体平均负载: {np.mean(avg_load):.3f}")
print(f"负载标准差: {np.mean(std_load):.3f}")

# 分析候选预测成功率
if candidate_success_probs:
    print(f"\n【候选预测成功率分布】")
    print(f"总候选数: {len(candidate_success_probs)}")
    print(f"每次决策平均候选数: {np.mean(candidates_per_decision):.1f}")
    print(f"最高预测: {max(candidate_success_probs):.3f}")
    print(f"平均预测: {np.mean(candidate_success_probs):.3f}")
    print(f"最低预测: {min(candidate_success_probs):.3f}")

    # 统计分布
    bins = [0, 0.3, 0.4, 0.55, 0.7, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for p in candidate_success_probs if bins[i] <= p < bins[i+1])
        percentage = count / len(candidate_success_probs) * 100
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} ({percentage:.1f}%)")

print(f"\n【关键指标】")
print(f"决策调用次数: {algo.decision_calls}")
print(f"切换尝试次数: {algo.handover_attempts}")
print(f"执行过滤次数: {algo.execution_filter_stats.get('below_threshold', 0)}")

print(f"\n【结论】")
print(f"1. 容量利用率: {total_demand/total_capacity*100:.1f}%")
print(f"   如果>100%，说明资源确实紧张")
print(f"\n2. 基站负载分布:")
print(f"   平均: {np.mean(avg_load):.3f}")
print(f"   标准差: {np.mean(std_load):.3f}")
print(f"   标准差大说明负载不均衡")
print(f"\n3. 候选预测成功率:")
if candidate_success_probs:
    print(f"   平均: {np.mean(candidate_success_probs):.3f}")
    print(f"   如果平均<0.55，说明大多数基站负载高")

print("\n" + "=" * 80)
