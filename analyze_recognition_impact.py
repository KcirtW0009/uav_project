"""
深入分析识别准确率对系统的影响
"""

import numpy as np
from collections import defaultdict
from uav_system.business import BusinessType
from uav_system.config import set_global_seed, GLOBAL_SEED


def analyze_recognition_error_impact():
    """分析识别错误的具体影响"""
    print("\n" + "="*80)
    print("识别错误对系统性能的影响分析")
    print("="*80)

    # 模拟识别错误的场景
    set_global_seed(GLOBAL_SEED)

    # 假设有50个UAV，不同准确率下的错误分布
    num_uav = 50

    # 业务类型分布 (根据环境初始化)
    ratios = [0.4, 0.3, 0.3]
    business_types = [BusinessType.CONTROL_SIGNAL,
                    BusinessType.VIDEO_STREAMING,
                    BusinessType.ENVIRONMENT_MONITORING]

    # 生成真实的业务类型分布
    uav_true_types = []
    for i in range(num_uav):
        rand = np.random.rand()
        if rand < ratios[0]:
            biz_type = BusinessType.CONTROL_SIGNAL
        elif rand < ratios[0] + ratios[1]:
            biz_type = BusinessType.VIDEO_STREAMING
        else:
            biz_type = BusinessType.ENVIRONMENT_MONITORING
        uav_true_types.append(biz_type)

    # 统计真实分布
    true_distribution = defaultdict(int)
    for bt in uav_true_types:
        true_distribution[bt] += 1

    print(f"\n【真实业务类型分布】")
    print(f"  控制信令: {true_distribution[BusinessType.CONTROL_SIGNAL]} ({true_distribution[BusinessType.CONTROL_SIGNAL]/num_uav*100:.1f}%)")
    print(f"  视频流: {true_distribution[BusinessType.VIDEO_STREAMING]} ({true_distribution[BusinessType.VIDEO_STREAMING]/num_uav*100:.1f}%)")
    print(f"  环境监测: {true_distribution[BusinessType.ENVIRONMENT_MONITORING]} ({true_distribution[BusinessType.ENVIRONMENT_MONITORING]/num_uav*100:.1f}%)")

    # 模拟不同准确率下的识别结果
    accuracy_levels = {
        'perfect': 1.00,
        'high': 0.85,
        'medium': 0.70,
        'random': 0.33
    }

    print("\n【识别错误模式分析】")

    for acc_name, accuracy in accuracy_levels.items():
        print(f"\n--- {acc_name} (准确率: {accuracy*100:.0f}%) ---")

        uav_recognized_types = []
        error_matrix = defaultdict(lambda: defaultdict(int))

        for true_type in uav_true_types:
            if np.random.rand() < accuracy:
                # 正确识别
                recognized_type = true_type
            else:
                # 错误识别：随机选择其他类型
                other_types = [t for t in BusinessType if t != true_type]
                recognized_type = np.random.choice(other_types)
                error_matrix[true_type][recognized_type] += 1

            uav_recognized_types.append(recognized_type)

        # 统计错误矩阵
        print(f"\n  识别错误矩阵 (行=真实, 列=识别为):")
        print(f"                控制信令   视频流   环境监测")

        for true_type in business_types:
            row = [true_type.name]
            for rec_type in business_types:
                if true_type == rec_type:
                    count = sum(1 for t, r in zip(uav_true_types, uav_recognized_types)
                               if t == true_type and r == rec_type)
                else:
                    count = error_matrix[true_type][rec_type]

                if true_type == rec_type:
                    # 正确识别数量
                    percentage = count / sum(1 for t in uav_true_types if t == true_type) * 100
                    row.append(f"{count:3d} ({percentage:5.1f}%)")
                else:
                    # 错误识别数量
                    if count > 0:
                        row.append(f"✗{count:2d}")
                    else:
                        row.append(f"  - ")
            print(f"  {row[0]:12s}: {' | '.join(row[1:])}")

        # 分析潜在影响
        print(f"\n  潜在影响分析:")

        # 1. 控制信令被误识别
        control_misidentified = error_matrix[BusinessType.CONTROL_SIGNAL]
        total_control_errors = sum(control_misidentified.values())
        if total_control_errors > 0:
            print(f"    ⚠ 控制信令误识别次数: {total_control_errors}")
            for rec_type, count in control_misidentified.items():
                if count > 0:
                    print(f"      - 误识别为 {rec_type.name}: {count}次")
                    if rec_type == BusinessType.ENVIRONMENT_MONITORING:
                        print(f"        → 可能导致过度降级 (环境监测接受更多降级)")
                    elif rec_type == BusinessType.VIDEO_STREAMING:
                        print(f"        → 可能导致切换过于保守 (视频流需要高SINR)")

        # 2. 环境监测被误识别为控制信令
        env_to_control = error_matrix[BusinessType.ENVIRONMENT_MONITORING][BusinessType.CONTROL_SIGNAL]
        if env_to_control > 0:
            print(f"    ⚠ 环境监测误识别为控制信令: {env_to_control}次")
            print(f"      → 可能导致资源过度分配 (控制信令有最高优先级)")

        # 3. 视频流误识别的影响
        video_misidentified = error_matrix[BusinessType.VIDEO_STREAMING]
        total_video_errors = sum(video_misidentified.values())
        if total_video_errors > 0:
            print(f"    ⚠ 视频流误识别次数: {total_video_errors}")
            for rec_type, count in video_misidentified.items():
                if count > 0:
                    print(f"      - 误识别为 {rec_type.name}: {count}次")

        # 4. 估计满足率影响
        print(f"\n  满足率影响估计:")
        base_satisfaction = 0.85  # 假设基准
        error_penalty = (1 - accuracy) * 0.3  # 错误率影响系数

        # 考虑到某些错误可能"意外"改善性能的权重
        # 例如：环境监测误识别为控制信令，获得更多资源
        accidental_benefit_ratio = 0.0

        # 环境监测->控制信令可能获益
        env_to_control = error_matrix[BusinessType.ENVIRONMENT_MONITORING][BusinessType.CONTROL_SIGNAL]
        if env_to_control > 0:
            accidental_benefit_ratio += env_to_control / num_uav * 0.15

        # 控制信令->环境监测可能受损
        control_to_env = error_matrix[BusinessType.CONTROL_SIGNAL][BusinessType.ENVIRONMENT_MONITORING]
        if control_to_env > 0:
            accidental_benefit_ratio -= control_to_env / num_uav * 0.10

        estimated_satisfaction = base_satisfaction - error_penalty + accidental_benefit_ratio

        print(f"    基准满足率: {base_satisfaction:.3f}")
        print(f"    错误惩罚: -{error_penalty:.3f}")
        print(f"    意外增益: {accidental_benefit_ratio:+.3f}")
        print(f"    估计满足率: {estimated_satisfaction:.3f}")

    print("\n【总结】")
    print("  识别错误的影响是复杂的：")
    print("  1. 正常情况下，错误识别会导致性能下降")
    print("  2. 但某些错误组合可能'意外'改善某些指标：")
    print("     - 低优先级业务误识别为高优先级 → 获得更多资源")
    print("     - 高优先级业务误识别为低优先级 → 性能下降")
    print("  3. 这可能导致实验结果不符合单调递减的预期")
    print("\n【建议】")
    print("  1. 在实验1中降低epsilon (已实施，从0.05降到0.01)")
    print("  2. 增加重复次数以减小随机误差")
    print("  3. 考虑分析识别错误的具体影响模式")
    print("  4. 可以添加惩罚机制，让错误识别更明显地反映在性能上")


if __name__ == "__main__":
    analyze_recognition_error_impact()
