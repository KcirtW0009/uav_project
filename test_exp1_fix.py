"""
测试实验1的修复效果
验证识别准确率对性能的影响是否符合预期
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.experiments import Experiment1
from uav_system.recognition import train_or_load_recognition_model


def test_exp1_fix():
    """测试修复后的实验1"""
    print("\n" + "="*80)
    print("测试修复后的实验1")
    print("="*80)

    print("\n【修复内容】")
    print("  1. 关闭随机事件 (event_probability=0.0)")
    print("  2. 使用确定性识别设置")
    print("  3. 完全禁用epsilon-greedy探索 (epsilon=0.0)")
    print("  4. 为每个准确率条件使用独立seed")

    # 准备或加载识别模型
    print("\n准备识别模型...")
    recognition_model, scaler = train_or_load_recognition_model()

    # 运行实验
    print("\n运行实验1 (修复版本)...")
    summary = Experiment1.run(recognition_model, scaler, num_steps=150, repeats=5)

    # 验证结果
    print("\n" + "="*80)
    print("结果验证")
    print("="*80)

    conditions = ['perfect', 'high', 'medium', 'random']
    condition_labels = ['100%', '85%', '70%', '33%']

    # 1. 检查满足率单调性
    print("\n【1. 满足率单调性检查】")
    satisfactions = [summary[c]['satisfaction'][0] for c in conditions]
    sat_errors = [summary[c]['satisfaction'][1] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  满足率: " + " → ".join([f"{s:.3f}±{e:.3f}" for s, e in zip(satisfactions, sat_errors)]))

    expected_monotonic = sorted(satisfactions, reverse=True)
    if satisfactions == expected_monotonic:
        print("  ✓✓✓ 满足率随准确率降低单调递减 (修复成功！)")
    else:
        print("  ✗✗✗ 满足率未随准确率单调递减 (仍有问题)")
        for i in range(len(satisfactions)-1):
            if satisfactions[i] < satisfactions[i+1]:
                print(f"    - {condition_labels[i]} ({satisfactions[i]:.3f}) < {condition_labels[i+1]} ({satisfactions[i+1]:.3f})")

    # 2. 检查切换成功率
    print("\n【2. 切换成功率检查】")
    handover_successes = [summary[c]['handover_success'][0] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  切换成功率: " + " → ".join([f"{h*100:.1f}%" for h in handover_successes]))

    expected_handover = sorted(handover_successes, reverse=True)
    if handover_successes == expected_handover:
        print("  ✓ 切换成功率随准确率降低单调递减")
    else:
        print("  ⚠ 切换成功率未完全单调递减（可能受其他因素影响）")

    # 3. 检查关键业务满足率
    print("\n【3. 关键业务满足率检查】")
    critical_sats = [summary[c]['critical_sat'][0] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  关键业务满足率: " + " → ".join([f"{c:.3f}" for c in critical_sats]))

    expected_critical = sorted(critical_sats, reverse=True)
    if critical_sats == expected_critical:
        print("  ✓ 关键业务满足率随准确率降低单调递减")
    else:
        print("  ⚠ 关键业务满足率未完全单调递减")

    # 4. 计算性能损失并验证单调性
    print("\n【4. 性能损失单调性检查】")
    perfect_sat = summary['perfect']['satisfaction'][0]
    performance_losses = []

    for i, c in enumerate(conditions[1:], 1):
        sat = summary[c]['satisfaction'][0]
        loss = (perfect_sat - sat) * 100
        performance_losses.append(loss)
        print(f"  {condition_labels[i]}: {perfect_sat:.3f} → {sat:.3f} = {loss:+.2f}%")

    # 性能损失应该随准确率降低而递增（负损失变成正损失）
    # 即: 85% < 70% < 33%
    if all(performance_losses[i] <= performance_losses[i+1] for i in range(len(performance_losses)-1)):
        print("  ✓ 性能损失随准确率降低而递增 (符合预期)")
    else:
        print("  ⚠ 性能损失未完全递增")

    # 5. 分析吞吐量
    print("\n【5. 吞吐量检查】")
    throughputs = [summary[c]['throughput'][0] for c in conditions]
    throughput_errors = [summary[c]['throughput'][1] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  吞吐量: " + " → ".join([f"{t:.1f}±{e:.1f} Mbps"
                                        for t, e in zip(throughputs, throughput_errors)]))

    # 吞吐量应该随准确率降低而降低（因为识别错误导致资源分配不当）
    expected_throughput = sorted(throughputs, reverse=True)
    if throughputs == expected_throughput:
        print("  ✓ 吞吐量随准确率降低单调递减")
    else:
        print("  ⚠ 吞吐量未完全单调递减")

    # 总结
    print("\n" + "="*80)
    print("修复效果总结")
    print("="*80)

    all_monotonic = (
        satisfactions == expected_monotonic and
        handover_successes == expected_handover and
        critical_sats == expected_critical and
        throughputs == expected_throughput
    )

    if all_monotonic:
        print("✓✓✓ 所有指标均呈现预期的单调趋势")
        print("✓✓✓ 实验1修复成功！")
        print("\n修复措施：")
        print("  1. 关闭随机事件，消除外部干扰")
        print("  2. 使用确定性识别，确保错误模式一致")
        print("  3. 完全禁用探索，消除随机决策")
        print("  4. 独立seed，保证条件间独立性")
    else:
        print("⚠ 部分指标仍未呈现预期趋势")
        print("\n可能原因：")
        print("  1. 识别错误的影响确实存在非线性效应")
        print("  2. 需要更多重复次数以减小方差")
        print("  3. 算法对业务类型的敏感度可能不足")

    print("="*80)

    return summary


if __name__ == "__main__":
    results = test_exp1_fix()
