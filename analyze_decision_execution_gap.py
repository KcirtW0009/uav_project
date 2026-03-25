"""
分析决策阶段和执行阶段预测成功率的不一致
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("决策阶段 vs 执行阶段 预测不一致分析")
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

# 临时修改：记录决策阶段的预测
decision_success_probs = []
execution_success_probs = []

# 保存原始的execute_handover方法
original_execute_handover = algo.execute_handover

def patched_execute_handover(uav_id, target_bs_id, downgrade_ratio):
    # 记录执行阶段的预测
    exec_prob = algo.predict_handover_success(env.uavs[uav_id], target_bs_id, downgrade_ratio)
    execution_success_probs.append(exec_prob)
    # 调用原始方法
    return original_execute_handover(uav_id, target_bs_id, downgrade_ratio)

# 保存原始的make_intelligent_decision方法
original_make_decision = algo.make_intelligent_decision

def patched_make_decision(uav_id):
    # 调用原始方法
    result = original_make_decision(uav_id)

    # 如果有决策结果，记录预测
    if result is not None:
        target_bs_id, ratio = result
        uav = env.uavs[uav_id]
        dec_prob = algo.predict_handover_success(uav, target_bs_id, ratio)
        decision_success_probs.append(dec_prob)

    return result

# 应用补丁
algo.execute_handover = patched_execute_handover
algo.make_intelligent_decision = patched_make_decision

# 运行仿真
print("\n运行仿真（20步）...")
for step in range(20):
    env.step()
    algo.run_step(enable_load_balancing=True)

# 恢复原始方法
algo.execute_handover = original_execute_handover
algo.make_intelligent_decision = original_make_decision

# 分析数据
print(f"\n【统计】")
print(f"决策阶段记录的预测数量: {len(decision_success_probs)}")
print(f"执行阶段记录的预测数量: {len(execution_success_probs)}")

# 找出对应的决策和执行预测
min_len = min(len(decision_success_probs), len(execution_success_probs))
decision_probs_matched = decision_success_probs[:min_len]
execution_probs_matched = execution_success_probs[:min_len]

# 转换为数组
decision_probs_arr = np.array(decision_probs_matched)
execution_probs_arr = np.array(execution_probs_matched)

# 计算差异
differences = execution_probs_arr - decision_probs_arr

print(f"\n【预测差异分析】")
print(f"决策阶段平均预测成功率: {np.mean(decision_probs_arr):.4f}")
print(f"执行阶段平均预测成功率: {np.mean(execution_probs_arr):.4f}")
print(f"平均差异: {np.mean(differences):.4f}")
print(f"差异标准差: {np.std(differences):.4f}")

# 统计超过阈值的情况
print(f"\n【阈值0.55过滤分析】")
decision_above_055 = np.sum(decision_probs_arr >= 0.55)
execution_above_055 = np.sum(execution_probs_arr >= 0.55)

print(f"决策阶段预测≥0.55: {decision_above_055}/{min_len} ({decision_above_055/min_len*100:.1f}%)")
print(f"执行阶段预测≥0.55: {execution_above_055}/{min_len} ({execution_above_055/min_len*100:.1f}%)")

# 分析决策≥0.55但执行<0.55的情况
decision_ge_exec_lt = np.sum((decision_probs_arr >= 0.55) & (execution_probs_arr < 0.55))
print(f"\n决策≥0.55 但 执行<0.55 的数量: {decision_ge_exec_lt}")
print(f"占比: {decision_ge_exec_lt/min_len*100:.1f}%")

# 分析决策<0.55但执行≥0.55的情况
decision_lt_exec_ge = np.sum((decision_probs_arr < 0.55) & (execution_probs_arr >= 0.55))
print(f"决策<0.55 但 执行≥0.55 的数量: {decision_lt_exec_ge}")
print(f"占比: {decision_lt_exec_ge/min_len*100:.1f}%")

# 统计差异分布
print(f"\n【差异分布】")
print(f"差异 > 0.1 (执行高于决策): {np.sum(differences > 0.1)}")
print(f"差异 < -0.1 (执行低于决策): {np.sum(differences < -0.1)}")
print(f"差异 > 0.3 (执行显著高于决策): {np.sum(differences > 0.3)}")
print(f"差异 < -0.3 (执行显著低于决策): {np.sum(differences < -0.3)}")

# 显示一些具体例子
print(f"\n【具体例子（前10个）】")
for i in range(min(10, min_len)):
    print(f"#{i+1}: 决策={decision_probs_arr[i]:.3f}, 执行={execution_probs_arr[i]:.3f}, 差异={differences[i]:+.3f}")

# 统计原始计数器
print(f"\n【原始计数器】")
print(f"决策调用次数: {algo.decision_calls}")
print(f"切换尝试次数: {algo.handover_attempts}")
print(f"切换成功次数: {algo.handover_successes}")
print(f"执行阶段过滤次数: {algo.execution_filter_stats.get('below_threshold', 0)}")

# 验证
total_execute_calls = algo.handover_attempts + algo.execution_filter_stats.get('below_threshold', 0)
print(f"\n【验证】")
print(f"决策返回结果并进入execute的次数（估算）: {total_execute_calls}")
print(f"决策调用次数 × 决策通过率: {algo.decision_calls} × {total_execute_calls/algo.decision_calls*100:.1f}%")

print("\n" + "=" * 80)
