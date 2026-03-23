"""
深入诊断: 为什么85%准确率可能比100%准确率性能更好?
"""

from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.business import BusinessType, QOS_PROFILES
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm
import numpy as np

def analyze_recognition_distribution(target_accuracy, seed):
    """分析识别错误的分布情况"""
    set_global_seed(seed)
    env = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=50,
        recognition_model=None, scaler=None,
        seed=seed,
        event_probability=0.0
    )

    rng = np.random.RandomState(seed)
    correct_count = 0
    error_matrix = {bt: {bt2: 0 for bt2 in BusinessType} for bt in BusinessType}

    for uav in env.uavs.values():
        true_type = uav.true_business_type
        if rng.random() < target_accuracy:
            recognized_type = true_type
            correct_count += 1
        else:
            other_types = [t for t in BusinessType if t != true_type]
            error_index = (uav.uav_id + int(target_accuracy * 1000)) % len(other_types)
            recognized_type = other_types[error_index]

        error_matrix[true_type][recognized_type] += 1
        uav.business_type = recognized_type
        uav.qos_profile = QOS_PROFILES[recognized_type]
        uav.recognition_confidence = 0.825

    actual_accuracy = correct_count / len(env.uavs)

    print(f"\n目标准确率: {target_accuracy*100:.0f}%, 实际: {actual_accuracy*100:.1f}%")
    print("\n识别错误分布矩阵 (行=真实类型, 列=识别类型):")
    print(f"{'':<25}", end="")
    for bt in BusinessType:
        print(f"{bt.name[:15]:>15}", end="")
    print()

    for true_bt in BusinessType:
        print(f"{true_bt.name:<25}", end="")
        for recog_bt in BusinessType:
            count = error_matrix[true_bt][recog_bt]
            if true_bt == recog_bt:
                print(f"{count:>15}", end="")
            else:
                print(f"{count:>15}!", end="")
        print()

    return env, actual_accuracy, error_matrix

def analyze_qos_mismatch():
    """分析QoS不匹配的影响"""
    print("\n" + "="*80)
    print("QoS配置分析: 识别错误的影响")
    print("="*80)

    print(f"\n各业务类型的理想速率:")
    for bt in BusinessType:
        qos = QOS_PROFILES[bt]
        print(f"  {bt.name:<25}: ideal={qos.ideal_rate}, min={qos.min_rate}, "
              f"priority={qos.priority}")

    print(f"\n误识别场景分析:")
    scenarios = [
        (BusinessType.VIDEO_STREAMING, BusinessType.ENVIRONMENT_MONITORING),
        (BusinessType.CONTROL_SIGNAL, BusinessType.ENVIRONMENT_MONITORING),
        (BusinessType.ENVIRONMENT_MONITORING, BusinessType.VIDEO_STREAMING),
    ]

    for true_bt, recog_bt in scenarios:
        true_qos = QOS_PROFILES[true_bt]
        recog_qos = QOS_PROFILES[recog_bt]

        print(f"\n  {true_bt.name} -> {recog_bt.name}:")
        print(f"    真实需求: ideal={true_qos.ideal_rate}, min={true_qos.min_rate}")
        print(f"    识别为:   ideal={recog_qos.ideal_rate}, min={recog_qos.min_rate}")
        print(f"    优先级差异: {true_qos.priority} -> {recog_qos.priority}")

        # 模拟分配情况
        if recog_qos.ideal_rate < true_qos.ideal_rate:
            print(f"    [!] 识别后系统可能分配较少资源 (按{recog_qos.ideal_rate}而非{true_qos.ideal_rate})")
            print(f"    [!] 但按真实标准计算满意率会下降")
        elif recog_qos.ideal_rate > true_qos.ideal_rate:
            print(f"    [OK] 识别后系统可能分配更多资源 (按{recog_qos.ideal_rate}而非{true_qos.ideal_rate})")
            print(f"    [OK] 按真实标准计算满意率会上升")

def main():
    print("="*80)
    print("深入诊断: 为什么85%准确率可能比100%准确率性能更好?")
    print("="*80)

    analyze_qos_mismatch()

    # 分析100%准确率的错误分布
    print("\n" + "="*80)
    print("100%准确率情况")
    print("="*80)
    env_100, acc_100, matrix_100 = analyze_recognition_distribution(1.0, GLOBAL_SEED)

    # 分析85%准确率的错误分布
    print("\n" + "="*80)
    print("85%准确率情况")
    print("="*80)
    env_85, acc_85, matrix_85 = analyze_recognition_distribution(0.85, GLOBAL_SEED + int(0.85*1000))

    # 对比分析
    print("\n" + "="*80)
    print("关键洞察")
    print("="*80)

    print("\n1. 识别错误的不均衡分布:")
    for true_bt in BusinessType:
        correct_100 = matrix_100[true_bt][true_bt]
        correct_85 = matrix_85[true_bt][true_bt]
        print(f"   {true_bt.name}:")
        print(f"     100%准确率: 全部正确 ({correct_100}个)")
        print(f"     85%准确率: {correct_85}个正确, "
              f"{sum(matrix_85[true_bt][bt2] for bt2 in BusinessType if bt2 != true_bt)}个错误")

    print("\n2. 资源分配的影响:")
    print("   - 当VIDEO(ideal=200)被误识别为ENV(ideal=80)时:")
    print("     * 系统按ENV需求分配资源")
    print("     * 可能分配80Mbps而非200Mbps")
    print("     * 按VIDEO真实标准计算满意率会很低")
    print()
    print("   - 当ENV(ideal=80)被误识别为VIDEO(ideal=200)时:")
    print("     * 系统按VIDEO需求分配资源")
    print("     * 可能分配200Mbps而非80Mbps")
    print("     * 按ENV真实标准计算满意率会很高 (超额分配!)")

    print("\n3. 关键发现:")
    print("   [OK] 如果85%准确率时,高需求业务(VIDEO)被误识别为低需求(ENV)")
    print("     -> 系统分配不足资源")
    print("     -> 按真实标准满意率下降")
    print()
    print("   [OK] 但如果低需求业务(ENV)被误识别为高需求(VIDEO)")
    print("     -> 系统分配过多资源")
    print("     -> 按真实标准满意率上升 (因为超额分配)")
    print("     -> 这些'意外红利'可能抵消了识别错误的影响")

    print("\n4. 业务权重的影响:")
    print("   - 识别错误后,业务权重改变")
    print("   - VIDEO权重(rate=0.45) vs ENV权重(rate=0.5)")
    print("   - 如果ENV被误识别为VIDEO,系统会更注重rate匹配")
    print("   - 可能导致更好的资源分配决策")

    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("85%准确率可能比100%准确率表现更好的原因:")
    print()
    print("1. 识别错误的随机性可能导致部分UAV被超额分配资源")
    print("2. 低需求业务被误识别为高需求时,系统会分配更多资源")
    print("3. 这些'超额分配'的UAV满意率很高(按真实标准)")
    print("4. 可能抵消或超过高需求业务被误识别带来的性能损失")
    print()
    print("这反映了:")
    print("- 识别准确率的降低不一定是纯负面的")
    print("- 误识别可能带来意外的资源分配优势")
    print("- 真实满意率指标已经能正确反映用户体验")
    print("="*80)

if __name__ == "__main__":
    main()
