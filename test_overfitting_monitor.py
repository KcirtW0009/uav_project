#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OverfittingMonitor 测试脚本
验证过拟合监控器的输出格式和功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# 导入监控器
from uav_system.experiments_mappo import OverfittingMonitor


def generate_mock_episode_data(ep_num, base_sat=0.97, noise_level=0.002):
    """
    生成模拟的episode数据
    
    Args:
        ep_num: episode编号
        base_sat: 基准satisfaction
        noise_level: 噪声水平
    
    Returns:
        dict: 模拟的episode数据
    """
    # 添加一些真实的波动模式
    if ep_num < 10:
        sat = base_sat + np.random.normal(0, noise_level)
    elif ep_num < 30:
        sat = base_sat + 0.001 * (ep_num - 10) / 20 + np.random.normal(0, noise_level)
    else:
        sat = base_sat + 0.001 + np.random.normal(0, noise_level * 1.2)
    
    reward = 280 + np.random.normal(0, 12)
    stay_pct = 43.0 + np.random.normal(0, 1.5)
    
    return {
        'satisfaction': max(0.95, min(0.99, sat)),
        'reward': reward,
        'reward_std': abs(np.random.normal(11, 2)),
        'connected_ratio': 1.0,
        'switch_success_rate': 0.995 + np.random.uniform(-0.01, 0.005),
        'load_variance': 0.1114,
        'composite_score': sat * 0.997,
        'stay_percentage': max(38, min(48, stay_pct)),
        'switch_attempts': int(120000 + np.random.normal(0, 2000)),
        'switch_success': int(118000 + np.random.normal(0, 1500)),
        'switch_rollback': int(500 + np.random.normal(0, 200)),
        'switch_disconnect': 0,
        'biz_statistics': {
            bt: {
                'avg_satisfaction': 0.94 + bt * 0.03,
                'stay_count': int(50000 * (0.9 - bt * 0.3)),
                'switch_count': int(70000 * (0.1 + bt * 0.3)),
            } for bt in range(3)
        },
        'delta_sum': 0.001,
        'value_reward_sum': 0,
        'biz_reward_sum': 0,
        'action_reward_sum': 0,
        'connect_reward_sum': 0,
        'load_adaptive_sum': 0,
        'sample_count': 100,
        'actor_loss': abs(np.random.normal(0.147, 0.003)),
        'critic_loss': 36.5 + np.random.normal(0, 0.5),
        'entropy': 0.87 + np.random.normal(0, 0.02),
        'grad_norm': 1.0 + np.random.normal(0, 0.05),
        'value_mse': 73.0 + np.random.normal(0, 1.5),
    }


def test_overfitting_monitor():
    """测试过拟合监控器"""
    print("="*80)
    print("[TEST] OverfittingMonitor Function Test")
    print("="*80)
    
    # 初始化监控器
    monitor = OverfittingMonitor(window_size=20, alert_threshold=0.65)
    
    print("\n[OK] Monitor initialized")
    print(f"   Window size: {monitor.window_size}")
    print(f"   Alert threshold: {monitor.alert_threshold}")
    
    # 模拟30个episode的数据
    print(f"\n[DATA] Simulating {30} episodes...")
    print("-"*80)
    
    for ep in range(1, 31):
        episode_data = generate_mock_episode_data(ep)
        
        # 每5个episode执行一次完整检查（与实际训练一致）
        if ep % 5 == 0:
            risk_report = monitor.check(episode_data)
            
            print(f"\n{'='*80}")
            print(f"[CHECK] Episode {ep}/30 - Overfitting Check")
            print(f"{'='*80}")
            
            # 输出格式化报告
            print(monitor.format_report(risk_report))
            
            # 简要摘要
            total_risk = risk_report['total_risk']
            alert_status = "[!! ALERT !!]" if risk_report['is_alert'] else "[OK]"
            print(f"   [SUMMARY] Risk={total_risk:.3f} | Status={alert_status}")
    
    # 最终统计
    print("\n" + "="*80)
    print("[STATS] Test Complete - Summary")
    print("="*80)
    summary = monitor.check(generate_mock_episode_data(31))['summary']
    print(f"   Total checks: {summary['checks_performed']}")
    print(f"   Alerts triggered: {summary['total_alerts']}")
    print(f"   Alert rate: {summary['alert_rate']*100:.1f}%")
    print(f"   Data points analyzed: {summary['data_points']}")
    
    # 验证各维度风险评分
    latest_report = monitor.check(generate_mock_episode_data(32))
    print(f"\n   Latest dimension risk scores:")
    for dim_name, dim_data in latest_report['dimensions'].items():
        risk = dim_data['risk']
        status = dim_data.get('status', 'N/A')
        bar_len = int(risk * 20)
        bar = '#' * bar_len + '-' * (20 - bar_len)
        print(f"     * {dim_name:<20}: {risk:>5.3f} [{bar}] {status}")
    
    print(f"\n   Overall risk score: {latest_report['total_risk']:.3f}")
    print(f"   Risk level: {latest_report['alert_level']}")
    print(f"   Recommendation: {latest_report['recommendation']}")
    
    print("\n[SUCCESS] Test passed! OverfittingMonitor working correctly")
    print("="*80)


if __name__ == '__main__':
    test_overfitting_monitor()
