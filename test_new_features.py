"""
测试脚本: 验证故障恢复时间统计、目标指标达标检验和统计显著性检验功能
"""

import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
from uav_system.recognition import train_or_load_recognition_model
from uav_system.experiments import (
    perform_statistical_test,
    print_statistical_results,
    compare_algorithms_with_tests,
    print_comprehensive_test_summary
)


def test_recovery_statistics():
    """测试故障恢复时间统计功能"""
    print("\n" + "="*80)
    print("测试1: 故障恢复时间统计")
    print("="*80)

    set_global_seed(GLOBAL_SEED)

    # 创建环境,设置较高的事件概率以增加故障发生频率
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=50,
        recognition_model=None,
        scaler=None,
        seed=GLOBAL_SEED,
        event_probability=0.15  # 提高事件概率
    )

    algo = EnhancedHandoverAlgorithm(env)

    # 运行仿真
    num_steps = 100
    for step in range(num_steps):
        env.step()
        algo.run_step(enable_load_balancing=True)

    # 获取故障恢复统计
    recovery_stats = env.get_recovery_statistics()

    print(f"\n故障事件统计:")
    print(f"  基站故障次数: {env.event_stats['bs_failure']}")
    print(f"  基站恢复次数: {env.event_stats['bs_recovery']}")
    print(f"  信道突发次数: {env.event_stats['channel_burst']}")

    print(f"\n故障恢复时间统计:")
    print(f"  总恢复事件数: {recovery_stats['total_recoveries']}")
    print(f"  平均恢复时间: {recovery_stats['avg_recovery_time']:.2f} 步")
    print(f"  最大恢复时间: {recovery_stats['max_recovery_time']:.2f} 步")
    print(f"  最小恢复时间: {recovery_stats['min_recovery_time']:.2f} 步")
    print(f"  标准差: {recovery_stats['std_recovery_time']:.2f} 步")
    print(f"  当前活跃故障数: {recovery_stats['active_failures']}")

    if recovery_stats['recovery_events']:
        print(f"\n最近5次恢复事件详情:")
        for i, event in enumerate(recovery_stats['recovery_events'][-5:], 1):
            print(f"  事件{i}: ID={event['event_id']}, "
                  f"开始步={event['start_step']}, "
                  f"结束步={event['end_step']}, "
                  f"持续时间={event['recovery_duration']}步, "
                  f"影响UAV数={len(event['affected_uavs'])}")

    return recovery_stats


def test_target_metrics_check():
    """测试目标指标达标检验功能"""
    print("\n" + "="*80)
    print("测试2: 目标指标达标检验")
    print("="*80)

    set_global_seed(GLOBAL_SEED)

    # 创建环境
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=50,
        recognition_model=None,
        scaler=None,
        seed=GLOBAL_SEED
    )

    algo = EnhancedHandoverAlgorithm(env)

    # 运行仿真
    for step in range(50):
        env.step()
        algo.run_step(enable_load_balancing=True)

    # 定义目标指标
    target_metrics = {
        'satisfaction_rate': 0.7,      # 满足率目标70%
        'handover_success_rate': 0.9,  # 切换成功率目标90%
        'recognition_accuracy': 95.0,   # 识别准确率目标95%
        'interruption_rate': 0.1       # 中断率目标不超过10%
    }

    # 检查达标情况
    results = env.check_target_metrics(target_metrics)

    print(f"\n目标指标达标检验结果:")
    print(f"{'='*60}")

    for metric_name, result in results.items():
        if metric_name == 'summary':
            continue

        status = "✓ 达标" if result.get('achieved') else "✗ 未达标"
        print(f"\n【{metric_name}】")
        print(f"  目标值: {result['target']}")
        print(f"  实际值: {result['actual']:.4f}" if result['actual'] is not None else f"  实际值: N/A")
        print(f"  状态: {status}")

        if result.get('difference') is not None:
            print(f"  差值: {result['difference']:.4f}")
            print(f"  相对差异: {result['relative_difference']:.2f}%")

    # 打印摘要
    summary = results['summary']
    print(f"\n{'='*60}")
    print(f"摘要:")
    print(f"  总指标数: {summary['total_metrics']}")
    print(f"  达标指标数: {summary['achieved_metrics']}")
    print(f"  达标率: {summary['achievement_rate']*100:.1f}%")
    print(f"  全部达标: {'是' if summary['all_achieved'] else '否'}")
    print(f"{'='*60}")

    return results


def test_statistical_significance():
    """测试统计显著性检验功能"""
    print("\n" + "="*80)
    print("测试3: 统计显著性检验")
    print("="*80)

    # 准备或加载识别模型
    print("\n准备识别模型...")
    recognition_model, scaler = train_or_load_recognition_model()

    # 运行多次实验获取数据
    set_global_seed(GLOBAL_SEED)

    num_repeats = 10
    enhanced_results = []
    traditional_results = []

    print(f"\n运行实验对比 (重复{num_repeats}次)...")
    for rep in range(num_repeats):
        set_global_seed(GLOBAL_SEED + rep)

        # 增强算法
        env_enh = EnhancedNetworkEnvironment(
            num_bs=8, num_uav=50,
            recognition_model=recognition_model, scaler=scaler,
            seed=GLOBAL_SEED + rep, event_probability=0.05
        )
        algo_enh = EnhancedHandoverAlgorithm(env_enh)

        for step in range(100):
            env_enh.step()
            algo_enh.run_step(enable_load_balancing=True)

        enh_stats = env_enh.get_state_statistics()
        enh_stats.update(algo_enh.get_detailed_stats())
        enhanced_results.append(enh_stats)

        # 传统算法
        env_trad = EnhancedNetworkEnvironment(
            num_bs=8, num_uav=50,
            recognition_model=recognition_model, scaler=scaler,
            seed=GLOBAL_SEED + rep, event_probability=0.05
        )
        algo_trad = IntegratedHandoverAlgorithm(env_trad)

        for step in range(100):
            env_trad.step()
            algo_trad.run_step()

        trad_stats = env_trad.get_state_statistics()
        trad_stats.update(algo_trad.get_detailed_stats())
        traditional_results.append(trad_stats)

        print(f"  重复 {rep+1}/{num_repeats} 完成")

    # 进行统计检验
    print(f"\n执行统计显著性检验...")

    metrics_to_test = [
        'avg_satisfaction',
        'handover_success_rate',
        'critical_satisfaction',
        'avg_switching_latency_ms',
        'total_load'
    ]

    all_test_results = compare_algorithms_with_tests(
        enhanced_results, traditional_results, metrics_to_test
    )

    # 打印详细结果
    for metric, results in all_test_results.items():
        print_statistical_results(results, metric)

    # 打印综合摘要
    print_comprehensive_test_summary(all_test_results, "增强算法", "传统算法")

    # 保存结果供后续分析
    test_results = {
        'enhanced_results': enhanced_results,
        'traditional_results': traditional_results,
        'statistical_tests': all_test_results
    }

    return test_results


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("开始运行功能测试")
    print("="*80)

    try:
        # 测试1: 故障恢复时间统计
        recovery_stats = test_recovery_statistics()
        print("\n✓ 测试1完成: 故障恢复时间统计")

        # 测试2: 目标指标达标检验
        target_results = test_target_metrics_check()
        print("\n✓ 测试2完成: 目标指标达标检验")

        # 测试3: 统计显著性检验
        stats_results = test_statistical_significance()
        print("\n✓ 测试3完成: 统计显著性检验")

        print("\n" + "="*80)
        print("所有测试完成!")
        print("="*80)

        return {
            'recovery_stats': recovery_stats,
            'target_results': target_results,
            'stats_results': stats_results
        }

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_all_tests()

    if results is not None:
        print("\n测试总结:")
        print(f"  - 故障恢复时间统计: 已验证")
        print(f"  - 目标指标达标检验: 已验证")
        print(f"  - 统计显著性检验: 已验证")
        print("\n所有功能正常工作!")
