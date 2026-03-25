"""
测试脚本：查看执行阶段过滤统计数据

运行一个简短的实验，输出详细的切换成功率统计信息
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

set_global_seed(GLOBAL_SEED)

print("=" * 80)
print("执行阶段过滤统计测试")
print("=" * 80)

# 创建环境（使用实验3的配置）
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

# 运行仿真
print("\n运行仿真（50步）...")
for step in range(50):
    env.step()
    algo.run_step(enable_load_balancing=True)

# 获取详细统计
stats = algo.get_detailed_stats()

print("\n" + "=" * 80)
print("切换成功率统计信息")
print("=" * 80)

# 核心计数器
print(f"\n【核心计数器】")
print(f"决策调用次数 (decision_calls): {algo.decision_calls}")
print(f"切换尝试次数 (handover_attempts, 分母): {algo.handover_attempts}")
print(f"切换成功次数 (handover_successes, 分子): {algo.handover_successes}")
print(f"切换成功率: {stats['handover_success_rate']*100:.2f}%")

# 执行阶段过滤统计
print(f"\n【执行阶段过滤统计】")
execution_filter = stats.get('execution_filter_stats', {})
below_threshold = execution_filter.get('below_threshold', 0)
if below_threshold == 0:
    # 如果stats中没有，直接访问algo的属性
    below_threshold = algo.execution_filter_stats.get('below_threshold', 0)
print(f"低于阈值的切换 (below_threshold): {below_threshold}")

# 错失机会统计
print(f"\n【决策阶段过滤统计】")
print(f"错失机会率 (missed_opportunity_rate): {stats['missed_opportunity_rate']*100:.2f}%")

# 失败原因
print(f"\n【失败原因统计】")
failure_reasons = stats.get('failure_reasons', {})
if failure_reasons:
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count}")
else:
    print("  无失败记录")

# 计算真实切换成功率
attempts = algo.handover_attempts
successes = algo.handover_successes
decision_calls = algo.decision_calls

print("\n" + "=" * 80)
print("切换成功率分析")
print("=" * 80)

if below_threshold > 0:
    total_requests = attempts + below_threshold
    real_success_rate = successes / total_requests

    print(f"\n当前统计方式:")
    print(f"  切换成功率 = handover_successes / handover_attempts")
    print(f"             = {successes} / {attempts}")
    print(f"             = {stats['handover_success_rate']*100:.2f}%")

    print(f"\n如果将所有进入execute阶段的切换都计入分母:")
    print(f"  切换成功率 = handover_successes / (handover_attempts + below_threshold)")
    print(f"             = {successes} / {attempts + below_threshold}")
    print(f"             = {real_success_rate*100:.2f}%")

    print(f"\n统计偏差:")
    print(f"  偏差 = {stats['handover_success_rate']*100:.2f}% - {real_success_rate*100:.2f}%")
    print(f"       = {(stats['handover_success_rate'] - real_success_rate)*100:.2f}%")

    print(f"\n过滤比例:")
    print(f"  低于阈值的切换占比: {below_threshold/total_requests*100:.2f}%")

    if below_threshold / total_requests > 0.1:
        print(f"\n警告: 过滤比例超过10%，统计偏差明显！")
else:
    print(f"\n未检测到执行阶段的过滤")
    print(f"当前切换成功率 {stats['handover_success_rate']*100:.2f}% 是真实的")

# 分析决策阶段
if decision_calls > 0:
    decision_pass_rate = (attempts + below_threshold) / decision_calls
    print(f"\n决策阶段分析:")
    print(f"  决策通过率: {decision_pass_rate*100:.2f}%")
    print(f"  说明: 每{decision_calls:.0f}次决策，有{(attempts + below_threshold):.0f}次进入execute阶段")

print("\n" + "=" * 80)
