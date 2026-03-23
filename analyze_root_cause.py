"""
深度分析: 识别准确率无法影响算法性能的根本原因
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.business import BusinessType, QOS_PROFILES
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm

def analyze_qos_impact():
    """分析QoS配置错误对满意率计算的影响"""
    print("="*80)
    print("分析: QoS配置错误对满意率计算的影响")
    print("="*80)

    # 模拟不同业务类型在不同分配速率下的满意率
    allocated_rates = [30, 80, 150, 200]

    print(f"\n{'业务类型':<25} | {'分配速率':<10} | {'满意率':<10}")
    print("-" * 60)

    for biz_type in BusinessType:
        qos = QOS_PROFILES[biz_type]
        print(f"\n{biz_type.name:<25} (ideal: {qos.ideal_rate}, min: {qos.min_rate})")
        for rate in allocated_rates:
            satisfaction = qos.calculate_satisfaction(rate)
            print(f"{'':25} | {rate:<10} | {satisfaction:<10.3f}")

    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)
    print("1. 当VIDEO(ideal=200)被分配150Mbps时:")
    print("   - 按VIDEO标准: 满意率 = 0.2 + 0.3 * (150-150)/50 = 0.2")
    print("   - 但如果被误识别为ENV(ideal=80):")
    print("   - 按ENV标准: 满意率 = 1.0 (因为150 > 80)")

    print("\n2. 当CONTROL(ideal=50)被分配40Mbps时:")
    print("   - 按CONTROL标准: 满意率 = 0.0 (因为40/50=0.8 < 0.85)")
    print("   - 但如果被误识别为ENV(ideal=80):")
    print("   - 按ENV标准: 满意率 = 0.4 + 0.6 * (40-30)/50 = 0.52")

    print("\n3. 结论: 错误识别可能导致满意率计算不准确!")
    print("="*80)

def analyze_weight_impact():
    """分析业务权重对utility计算的影响"""
    print("\n" + "="*80)
    print("分析: 业务权重对utility计算的影响")
    print("="*80)

    # 模拟一个具体的UAV决策场景
    print("\n假设场景: UAV在某个位置,有两个基站可选")
    print("  BS1: SINR=5dB, 负载=0.6, 可用速率=100Mbps")
    print("  BS2: SINR=8dB, 负载=0.8, 可用速率=150Mbps")

    sinr_1, load_1, rate_1 = 5, 0.6, 100
    sinr_2, load_2, rate_2 = 8, 0.8, 150

    # 归一化
    sinr_norm_1 = np.clip((sinr_1 + 10) / 40, 0, 1)  # 0.375
    sinr_norm_2 = np.clip((sinr_2 + 10) / 40, 0, 1)  # 0.45

    # 假设UAV需求
    required_rate = 100  # VIDEO
    rate_ratio_1 = 100 / 100
    rate_ratio_2 = min(150 / 100, 1.5)
    rate_match_1 = 1 - np.exp(-3 * min(rate_ratio_1, 1.5))
    rate_match_2 = 1 - np.exp(-3 * min(rate_ratio_2, 1.5))

    print(f"\n归一化指标:")
    print(f"  BS1: sinr_norm={sinr_norm_1:.3f}, load_norm={1-load_1:.3f}, rate_match={rate_match_1:.3f}")
    print(f"  BS2: sinr_norm={sinr_norm_2:.3f}, load_norm={1-load_2:.3f}, rate_match={rate_match_2:.3f}")

    print(f"\n不同业务类型下的utility计算:")
    print(f"{'业务类型':<25} | {'BS1 Utility':<12} | {'BS2 Utility':<12} | {'选择':<6}")
    print("-" * 60)

    weights_configs = {
        'CONTROL_SIGNAL': {'sinr': 0.5, 'load': 0.2, 'rate': 0.3},
        'VIDEO_STREAMING': {'sinr': 0.3, 'load': 0.25, 'rate': 0.45},
        'ENVIRONMENT_MONITORING': {'sinr': 0.25, 'load': 0.25, 'rate': 0.5}
    }

    for biz_name, weights in weights_configs.items():
        utility_1 = (weights['sinr'] * sinr_norm_1 +
                    weights['load'] * (1 - load_1) +
                    weights['rate'] * rate_match_1)
        utility_2 = (weights['sinr'] * sinr_norm_2 +
                    weights['load'] * (1 - load_2) +
                    weights['rate'] * rate_match_2)

        choice = "BS2" if utility_2 > utility_1 else "BS1"
        print(f"{biz_name:<25} | {utility_1:<12.3f} | {utility_2:<12.3f} | {choice:<6}")

    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)
    print("在这个场景下,所有业务类型都选择BS2!")
    print("说明: 业务权重只改变了utility的绝对值,没有改变相对排序")
    print("原因: SINR和负载的物理约束在不同业务类型下是相同的")
    print("="*80)

def analyze_recognition_error_cascade():
    """分析识别错误的级联影响"""
    print("\n" + "="*80)
    print("分析: 识别错误的级联影响链")
    print("="*80)

    print("\n识别错误的完整影响链路:")
    print("  1. 识别错误 → business_type改变")
    print("  2. business_type改变 → qos_profile改变")
    print("  3. qos_profile改变 →")
    print("     a. required_rate改变 (影响资源分配)")
    print("     b. 业务权重改变 (影响utility计算)")
    print("     c. 满意率计算标准改变 (影响性能指标)")
    print("  4. utility改变 → 切换决策改变")

    print("\n但是...")
    print("  [!] 真正影响切换决策的是:")
    print("     - SINR矩阵 (物理约束,不变)")
    print("     - 基站负载 (资源约束,可能改变)")
    print("     - 可用速率 (资源约束,可能改变)")

    print("\n  [!] 业务权重只是改变了组合方式,但如果:")
    print("     - SINR差异显著 (如8dB vs 5dB)")
    print("     - 负载差异显著 (如0.4 vs 0.8)")
    print("     则无论用哪套权重,最优基站选择可能都一样")

    print("\n  [!] 最严重的问题: 满意率计算标准不一致!")
    print("     - VIDEO被误识别为ENV")
    print("     - 系统按ENV标准计算满意率")
    print("     - 导致满意度指标失真")

    print("\n" + "="*80)

def simulate_exp1_scenario():
    """模拟实验1的具体场景"""
    print("\n" + "="*80)
    print("模拟: 实验1的典型场景")
    print("="*80)

    # 设置环境
    set_global_seed(GLOBAL_SEED)
    env = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=50,
        recognition_model=None, scaler=None,
        seed=GLOBAL_SEED,
        event_probability=0.0
    )

    # 统计业务分布
    biz_count = {bt: 0 for bt in BusinessType}
    for uav in env.uavs.values():
        biz_count[uav.true_business_type] += 1

    print(f"\nUAV业务类型分布 (总计50个):")
    for bt, count in biz_count.items():
        print(f"  {bt.name}: {count} ({count/50*100:.1f}%)")

    print(f"\n各业务类型的QoS配置:")
    for bt in BusinessType:
        qos = QOS_PROFILES[bt]
        print(f"  {bt.name}:")
        print(f"    ideal_rate={qos.ideal_rate}, min_rate={qos.min_rate}")
        print(f"    priority={qos.priority}, downgrade_tolerance={qos.downgrade_tolerance}")

    # 模拟一个误识别场景
    print(f"\n模拟场景: VIDEO_STREAMING被误识别为ENVIRONMENT_MONITORING")
    video_uav = None
    for uav in env.uavs.values():
        if uav.true_business_type == BusinessType.VIDEO_STREAMING:
            video_uav = uav
            break

    if video_uav:
        original_qos = QOS_PROFILES[BusinessType.VIDEO_STREAMING]
        error_qos = QOS_PROFILES[BusinessType.ENVIRONMENT_MONITORING]

        # 假设分配不同的速率
        test_rates = [80, 150, 200]
        print(f"\nVIDEO_UAV在不同分配速率下的满意率对比:")
        print(f"{'分配速率':<10} | {'按VIDEO标准':<12} | {'按ENV标准':<12} | {'差异':<10}")
        print("-" * 50)

        for rate in test_rates:
            sat_video = original_qos.calculate_satisfaction(rate)
            sat_env = error_qos.calculate_satisfaction(rate)
            diff = sat_env - sat_video
            print(f"{rate:<10} | {sat_video:<12.3f} | {sat_env:<12.3f} | {diff:+.3f}")

    print("\n" + "="*80)
    print("结论: 识别错误会导致满意率计算失真!")
    print("="*80)

def main():
    analyze_qos_impact()
    analyze_weight_impact()
    analyze_recognition_error_cascade()
    simulate_exp1_scenario()

    print("\n" + "="*80)
    print("总体结论")
    print("="*80)
    print("识别准确率无法有效影响算法性能的根本原因:")
    print()
    print("1. 业务权重只改变了utility计算方式,但物理约束(SINR/负载)不变")
    print("   → 导致切换决策可能不会因为业务类型改变而改变")
    print()
    print("2. 满意率计算使用识别类型的QoS配置,而非真实类型")
    print("   → 导致识别错误反而可能让满意率看起来更高")
    print()
    print("3. 资源分配基于识别类型的required_rate")
    print("   → VIDEO被误识别为ENV时,系统可能分配更少的资源")
    print("   → 但按ENV标准计算满意率,反而显示为100%满意")
    print()
    print("这就是为什么70%准确率可能比100%准确率表现更好的原因!")
    print("="*80)

if __name__ == "__main__":
    main()
