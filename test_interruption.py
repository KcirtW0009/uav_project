"""
测试中断率统计功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm
from uav_system.recognition import train_or_load_recognition_model

def test_interruption_stats():
    print("="*80)
    print("测试中断率统计功能")
    print("="*80)

    # 初始化随机种子
    set_global_seed(GLOBAL_SEED)

    # 创建环境
    print("\n创建仿真环境...")
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=50,
        recognition_model=None,  # 不使用识别模型以简化测试
        scaler=None,
        seed=GLOBAL_SEED,
        event_probability=0.05
    )

    # 创建算法
    algo = EnhancedHandoverAlgorithm(env)

    # 运行仿真
    print("运行仿真 (50步)...")
    for step in range(50):
        env.step()
        algo.run_step(enable_load_balancing=True)
        if (step + 1) % 10 == 0:
            print(f"  已完成 {step + 1} 步")

    # 获取统计信息
    print("\n" + "="*80)
    print("中断率统计结果")
    print("="*80)

    stats = env.get_state_statistics()
    print(f"\n基本统计:")
    print(f"  总中断次数: {stats['total_interruptions']}")
    print(f"  当前活跃中断数: {stats['active_interruptions_count']}")
    print(f"  当前中断率: {stats['interruption_rate']*100:.2f}%")
    print(f"  平均中断持续时间: {stats['avg_interruption_duration']:.1f} 步")

    print(f"\n系统状态:")
    print(f"  整体满足率: {stats['avg_satisfaction']:.3f}")
    print(f"  连接UAV数: {stats['connected_count']}/{env.num_uav}")
    print(f"  系统吞吐量: {stats['total_load']:.1f} Mbps")

    # 获取详细中断统计
    print("\n" + "="*80)
    print("详细中断分析")
    print("="*80)

    detailed_stats = env.get_interruption_statistics()

    if detailed_stats['total_interruptions'] > 0:
        print(f"\n按业务类型统计:")
        for bt_name, bt_stats in detailed_stats['by_business_type'].items():
            print(f"  {bt_name}:")
            print(f"    中断次数: {bt_stats['count']}")
            print(f"    平均持续时间: {bt_stats['avg_duration']:.1f} 步")

        print(f"\n中断事件详情 (前5个):")
        for i, event in enumerate(env.interruption_events[:5], 1):
            print(f"  事件 {i}:")
            print(f"    UAV ID: {event['uav_id']}")
            print(f"    业务类型: {event['business_type']}")
            print(f"    开始步: {event['start_step']}, 结束步: {event['end_step']}")
            print(f"    持续时间: {event['duration']} 步")
            print(f"    满足率: {event['satisfaction']:.3f}")
            print(f"    中断阈值: {event['threshold']:.2f}")
    else:
        print("\n在本次仿真中未检测到中断事件。")

    print("\n" + "="*80)
    print("中断率定义说明")
    print("="*80)
    print(f"\n控制信令业务:")
    print(f"  中断阈值: 满足率 < {env.control_signal_threshold}")
    print(f"  持续步数: ≥ {env.control_signal_duration} 步")
    print(f"\n其他业务 (视频/环境监测):")
    print(f"  中断阈值: 满足率 < {env.interruption_threshold}")
    print(f"  持续步数: ≥ {env.interruption_duration} 步")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)

if __name__ == "__main__":
    test_interruption_stats()
