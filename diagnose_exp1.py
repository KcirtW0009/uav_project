"""
实验1结果诊断脚本
用于分析识别准确率实验中的异常结果
"""

import numpy as np
import matplotlib.pyplot as plt
from uav_system.experiments import Experiment1
from uav_system.recognition import train_or_load_recognition_model


def diagnose_experiment1():
    """诊断实验1可能的问题"""
    print("\n" + "="*80)
    print("实验1结果诊断")
    print("="*80)

    # 准备或加载识别模型
    print("\n加载识别模型...")
    recognition_model, scaler = train_or_load_recognition_model()

    # 运行实验
    print("\n运行实验1 (详细诊断模式)...")
    summary = Experiment1.run(recognition_model, scaler, num_steps=150, repeats=5)

    # 分析结果
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)

    # 1. 检查满足率单调性
    print("\n【1. 满足率单调性检查】")
    conditions = ['perfect', 'high', 'medium', 'random']
    condition_labels = ['100%', '85%', '70%', '33%']

    satisfactions = [summary[c]['satisfaction'][0] for c in conditions]
    sat_errors = [summary[c]['satisfaction'][1] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  满足率: " + " → ".join([f"{s:.3f}±{e:.3f}" for s, e in zip(satisfactions, sat_errors)]))

    # 检查是否单调递减
    expected_monotonic = sorted(satisfactions, reverse=True)
    if satisfactions == expected_monotonic:
        print("  ✓ 满足率随准确率降低单调递减 (符合预期)")
    else:
        print("  ✗ 满足率未随准确率单调递减 (异常!)")
        for i in range(len(satisfactions)-1):
            if satisfactions[i] < satisfactions[i+1]:
                print(f"    - {condition_labels[i]} ({satisfactions[i]:.3f}) < {condition_labels[i+1]} ({satisfactions[i+1]:.3f})")

    # 2. 检查切换成功率
    print("\n【2. 切换成功率检查】")
    handover_successes = [summary[c]['handover_success'][0] for c in conditions]
    handover_errors = [summary[c]['handover_success'][1] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  切换成功率: " + " → ".join([f"{h*100:.1f}%±{e*100:.1f}%" for h, e in zip(handover_successes, handover_errors)]))

    # 切换成功率应该随准确率降低而降低（因为错误识别导致错误的QoS配置）
    expected_handover = sorted(handover_successes, reverse=True)
    if handover_successes == expected_handover:
        print("  ✓ 切换成功率随准确率降低单调递减")
    else:
        print("  ✗ 切换成功率异常")

    # 3. 检查关键业务满足率
    print("\n【3. 关键业务满足率检查】")
    critical_sats = [summary[c]['critical_sat'][0] for c in conditions]
    critical_errors = [summary[c]['critical_sat'][1] for c in conditions]

    print("  条件: " + " → ".join(condition_labels))
    print(f"  关键业务满足率: " + " → ".join([f"{c:.3f}±{e:.3f}" for c, e in zip(critical_sats, critical_errors)]))

    expected_critical = sorted(critical_sats, reverse=True)
    if critical_sats == expected_critical:
        print("  ✓ 关键业务满足率随准确率降低单调递减")
    else:
        print("  ✗ 关键业务满足率异常")

    # 4. 分析性能损失计算
    print("\n【4. 性能损失计算验证】")
    perfect_sat = summary['perfect']['satisfaction'][0]
    for i, c in enumerate(conditions[1:], 1):
        sat = summary[c]['satisfaction'][0]
        loss = (perfect_sat - sat) * 100
        print(f"  {condition_labels[i]}: {perfect_sat:.3f} → {sat:.3f} = {loss:+.2f}%")

    # 5. 识别可能的原因
    print("\n【5. 可能的问题分析】")
    issues = []

    if satisfactions != expected_monotonic:
        issues.append("满足率单调性异常：可能的原因包括：")
        issues.append("  - 实验设计问题：不同准确率条件使用了相同的seed，可能导致随机数状态污染")
        issues.append("  - 识别错误的随机性：某些错误的识别组合可能意外提升性能")
        issues.append("  - 算法敏感度低：切换算法可能对业务类型不敏感")

    if handover_successes != expected_handover:
        issues.append("切换成功率异常：可能的原因包括：")
        issues.append("  - 切换算法实现问题：ε-greedy探索导致性能波动")
        issues.append("  - QoS配置问题：错误识别后的QoS降级策略可能不合理")

    for issue in issues:
        print(f"  {issue}")

    # 6. 可视化结果
    print("\n【6. 生成可视化图表】")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('实验1结果诊断', fontsize=14, fontweight='bold')

    # 图1: 满足率
    ax = axes[0, 0]
    x = np.arange(len(conditions))
    bars = ax.bar(x, satisfactions, yerr=sat_errors, capsize=5,
                  color=['#4ade80', '#60a5fa', '#fbbf24', '#f87171'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('满足率')
    ax.set_title('整体满足率 (应单调递减)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # 标注异常
    for i, (s, e) in enumerate(zip(satisfactions, satisfactions[1:])):
        if s < e:
            ax.annotate(f'异常!', xy=(i+1, e), xytext=(i+1, e+0.05),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, ha='center', color='red')

    # 图2: 切换成功率
    ax = axes[0, 1]
    bars = ax.bar(x, [h*100 for h in handover_successes], yerr=[he*100 for he in handover_errors],
                  capsize=5, color=['#4ade80', '#60a5fa', '#fbbf24', '#f87171'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('切换成功率 (%)')
    ax.set_title('切换成功率 (应单调递减)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 图3: 关键业务满足率
    ax = axes[1, 0]
    bars = ax.bar(x, critical_sats, yerr=critical_errors, capsize=5,
                  color=['#4ade80', '#60a5fa', '#fbbf24', '#f87171'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('关键业务满足率')
    ax.set_title('关键业务满足率 (应单调递减)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 图4: 综合对比
    ax = axes[1, 1]
    x = np.arange(len(conditions))
    width = 0.25
    ax.bar(x - width, satisfactions, width, label='满足率', color='#60a5fa', alpha=0.8)
    ax.bar(x, critical_sats, width, label='关键业务满足率', color='#f87171', alpha=0.8)
    ax.bar(x + width, [h for h in handover_successes], width, label='切换成功率', color='#4ade80', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel('数值')
    ax.set_title('综合指标对比', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('experiment_results/exp1_diagnosis.png', dpi=200, bbox_inches='tight')
    print("  ✓ 图表已保存到 experiment_results/exp1_diagnosis.png")

    return summary


if __name__ == "__main__":
    results = diagnose_experiment1()

    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    print("\n建议：")
    print("1. 如果满足率单调性异常，检查实验设计的seed设置")
    print("2. 如果切换成功率异常，检查算法实现中的随机因素")
    print("3. 增加重复次数以减小随机误差")
    print("4. 检查QoS配置是否合理")
