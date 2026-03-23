"""
快速测试修复效果 - 只运行1次重复,较少步数
"""

from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.business import BusinessType, QOS_PROFILES
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

def test_single_condition(target_accuracy, condition_seed):
    """测试单个准确率条件"""
    set_global_seed(condition_seed)
    env = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=50,
        recognition_model=None, scaler=None,
        seed=condition_seed,
        event_probability=0.0
    )

    # 确定性设置识别准确率
    import numpy as np
    rng = np.random.RandomState(condition_seed)
    correct_count = 0
    for uav in env.uavs.values():
        true_type = uav.true_business_type
        if rng.random() < target_accuracy:
            recognized_type = true_type
            correct_count += 1
        else:
            other_types = [t for t in BusinessType if t != true_type]
            error_index = (uav.uav_id + int(target_accuracy * 1000)) % len(other_types)
            recognized_type = other_types[error_index]
        uav.business_type = recognized_type
        uav.qos_profile = QOS_PROFILES[recognized_type]
        uav.recognition_confidence = 0.825

    actual_accuracy = correct_count / len(env.uavs)

    env.recognition_updater = None
    algo = EnhancedHandoverAlgorithm(env)
    algo.epsilon = 0.0

    # 运行50步快速测试
    for step in range(50):
        env.step()
        algo.run_step(enable_load_balancing=True)

    stats = env.get_state_statistics()
    algo_stats = algo.get_detailed_stats()
    stats.update(algo_stats)
    stats['actual_recognition_accuracy'] = actual_accuracy

    return stats

print("="*80)
print("快速测试修复效果")
print("="*80)

accuracies = {
    'perfect': 1.00,
    'high': 0.85,
    'medium': 0.70,
    'random': 0.33,
}

results = {}
for condition_name, target_accuracy in accuracies.items():
    seed = GLOBAL_SEED + int(target_accuracy * 1000)
    stats = test_single_condition(target_accuracy, seed)
    results[condition_name] = stats
    print(f"\n{condition_name:8s} (目标{target_accuracy*100:3.0f}%, 实际{stats['actual_recognition_accuracy']*100:5.1f}%)")
    print(f"  真实满足率: {stats['avg_true_satisfaction']:.3f}")
    print(f"  资源匹配度: {stats['resource_match_ratio']:.3f}")
    print(f"  关键业务满足率: {stats['critical_satisfaction']:.3f}")
    print(f"  切换成功率: {stats['handover_success_rate']*100:.1f}%")
    print(f"  系统吞吐量: {stats['total_load']:.1f} Mbps")

print("\n" + "="*80)
print("验证单调性:")
print("="*80)
true_sat_values = [results[c]['avg_true_satisfaction'] for c in ['perfect', 'high', 'medium', 'random']]
print(f"真实满足率序列: {' → '.join([f'{v:.3f}' for v in true_sat_values])}")

if true_sat_values == sorted(true_sat_values, reverse=True):
    print("OK 真实满足率随准确率降低而单调下降 (修复成功!)")
else:
    print("[!] 真实满足率仍未单调下降 (需要进一步分析)")

# 计算性能损失
perfect_sat = results['perfect']['avg_true_satisfaction']
print(f"\n相对于100%准确率的性能损失:")
for condition_name in ['high', 'medium', 'random']:
    loss = (perfect_sat - results[condition_name]['avg_true_satisfaction']) * 100
    print(f"  {condition_name}: {loss:+.2f}%")

print("="*80)
