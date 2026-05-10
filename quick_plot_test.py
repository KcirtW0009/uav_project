#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试: 验证实验3的绘图功能是否正常

只运行1轮MAPPO评估(约28分钟)，然后立即绘图
如果绘图出错，可以快速发现并修复，不用等10轮!

用法:
    python quick_plot_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.experiments import Experiment3, evaluate_mappo_in_experiment, set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model
import numpy as np


def main():
    print("\n" + "="*80)
    print("  [QUICK TEST] 实验3绘图功能验证")
    print("  只运行1轮MAPPO评估 + 绘图 (约28分钟)")
    print("="*80)

    # 加载识别模型
    print("\n[Step1] 加载业务识别模型...")
    recognition_model, scaler = train_or_load_recognition_model(force_compare=False)
    if recognition_model is None:
        print("[ERROR] 无法加载识别模型!")
        return False

    mappo_model_path = r"experiment_results/mappo_models/mappo_8bs_300uav_best.pt"
    num_steps = 350

    print(f"\n[Step2] 运行1轮MAPPO评估 (种子=30047)...")
    set_global_seed(30047)

    mappo_stats = evaluate_mappo_in_experiment(
        num_bs=8,
        num_uav=300,
        num_steps=num_steps,
        recognition_model=recognition_model,
        scaler=scaler,
        seed=30047,
        model_path=mappo_model_path,
    )

    if mappo_stats is None:
        print("[ERROR] MAPPO评估失败!")
        return False

    print(f"\n[OK] MAPPO评估完成!")
    print(f"  满意度: {mappo_stats['avg_satisfaction']:.4f}")
    print(f"  成功率: {mappo_stats['handover_success_rate']*100:.2f}%")

    # 构造假的传统/增强算法数据（用于绘图）
    print("\n[Step3] 测试绘图功能...")

    enhanced_results = [{
        'avg_satisfaction': 0.9252,
        'handover_success_rate': 0.9384,
        'critical_satisfaction': 0.9990,
        'connected_ratio': 0.9805,
        'load_variance': 0.0020,
        'total_throughput': 4200.0,
        'avg_decision_time_ms': 0.0517,
        'avg_switching_latency_ms': 0.0165,
        'max_switching_latency_ms': 1.7806,
        'avg_sinr_db': 20.9485,
        'recognition_accuracy': 1.0,
        'migration_success_rate': 0.4131,
        'missed_opportunity_rate': 0.0,
        'rate_satisfaction': 0.9299,
        'latency_satisfaction': 1.0,
        'weighted_satisfaction': 0.6702,
    }]

    traditional_results = [{
        'avg_satisfaction': 0.8216,
        'handover_success_rate': 0.6337,
        'critical_satisfaction': 0.9366,
        'connected_ratio': 0.7755,
        'load_variance': 0.0369,
        'total_throughput': 3800.0,
        'avg_decision_time_ms': 0.0052,
        'avg_switching_latency_ms': 0.0018,
        'max_switching_latency_ms': 1.5823,
        'avg_sinr_db': 22.2701,
        'recognition_accuracy': 1.0,
        'migration_success_rate': 0.0,
        'missed_opportunity_rate': 0.0000,
        'rate_satisfaction': 0.7755,
        'latency_satisfaction': 1.0,
        'weighted_satisfaction': 0.5549,
    }]

    mappo_results = [mappo_stats]

    try:
        summary = Experiment3._summarize(enhanced_results, traditional_results, mappo_results)
        print("\n[OK] Summary计算完成!")

        print("\n--- 测试统计表格打印 ---")
        Experiment3._print_results_table(summary)

        print("\n--- 测试绘图功能 (安全模式: 不保存数据) ---")
        # [FIX] 备份原始数据，防止_plot()覆盖
        import shutil
        exp3_json_path = os.path.join('experiment_results', 'exp3_data.json')
        backup_path = exp3_json_path + '.before_test_backup'
        if os.path.exists(exp3_json_path):
            shutil.copy2(exp3_json_path, backup_path)
            print(f"  [BACKUP] 已备份原始数据 -> {backup_path}")

        try:
            Experiment3._plot(summary)
            print("  [OK] 绘图成功!")
        finally:
            # 恢复原始数据（无论绘图是否成功）
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, exp3_json_path)
                os.remove(backup_path)
                print("  [RESTORE] 已恢复原始数据 (防止被测试数据覆盖)")

        print("\n" + "="*80)
        print("  [SUCCESS] 绘图功能正常! 可以安全运行完整实验")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("  [FAILED] 绘图出错! 需要先修复bug再运行完整实验")
        print("="*80)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
