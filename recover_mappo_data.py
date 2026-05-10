#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MAPPO数据恢复工具

用法:
  1. 检查是否有已保存的数据:
     python recover_mappo_data.py --check

  2. 从已有数据生成报告 (不运行评估):
     python recover_mappo_data.py --report

  3. 快速重新运行 (只保存数据，跳过绘图):
     python recover_mappo_data.py --rerun
"""

import os
import sys
import json
import argparse
from datetime import datetime

RESULT_DIR = r"experiment_results"

def check_saved_data():
    """检查是否有已保存的MAPPO数据"""
    print("\n" + "="*80)
    print("检查已保存的MAPPO数据...")
    print("="*80)

    files_to_check = [
        ('exp3_mappo_raw_results.json', '实验3-原始结果(每轮自动保存)'),
        ('exp3_mappo_summary.json', '实验3-完整summary(最终保存)'),
        ('exp4_mappo_raw_results.json', '实验4-原始结果(每轮自动保存)'),
        ('exp4_mappo_summary.json', '实验4-完整summary(最终保存)'),
        ('exp3_data.json', '传统/增强算法缓存数据(实验3)'),
        ('exp4_data.json', '传统/增强算法缓存数据(实验4)'),
    ]

    found_any = False
    for filename, desc in files_to_check:
        filepath = os.path.join(RESULT_DIR, filename)
        if os.path.exists(filepath):
            found_any = True
            stat = os.stat(filepath)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            size_kb = stat.st_size / 1024

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"\n[OK] 找到: {filename}")
            print(f"   描述: {desc}")
            print(f"   大小: {size_kb:.1f} KB")
            print(f"   时间: {mtime}")

            if 'total_completed' in data:
                print(f"   包含: {data.get('total_completed', '?')} 轮MAPPO结果")
            if 'raw_results' in data:
                print(f"   原始数据: {len(data['raw_results'])} 轮")
            if 'summary' in data:
                print(f"   包含完整统计summary")
        else:
            print(f"\n[MISSING] 未找到: {filename}")

    if not found_any:
        print("\n[WARN] 未找到任何MAPPO数据文件")
        print("   需要重新运行实验3评估")

    return found_any


def generate_report_from_saved():
    """从保存的数据生成报告"""
    summary_path = os.path.join(RESULT_DIR, 'exp3_mappo_summary.json')
    raw_path = os.path.join(RESULT_DIR, 'exp3_mappo_raw_results.json')

    if not os.path.exists(summary_path) and not os.path.exists(raw_path):
        print("\n[ERROR] 没有找到可用的数据文件!")
        return False

    # 优先使用summary
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("\n" + "="*80)
        print("从保存的Summary生成报告...")
        print("="*80)
        print(f"时间戳: {data.get('timestamp', 'N/A')}")
        print(f"总轮数: {data.get('total_mappo_runs', 0)}")

        if 'summary' in data:
            summary = data['summary']
            print("\n--- MAPPO统计结果 ---")

            if 'mappo' in summary:
                print("\n指标                          |  均值      |  标准差")
                print("-"*60)
                for metric, (mean, std) in summary['mappo'].items():
                    if 'ratio' in metric or 'rate' in metric or 'accuracy' in metric:
                        print(f"{metric:30s}| {mean*100:8.2f}% | {std*100:8.4f}")
                    else:
                        print(f"{metric:30s}| {mean:10.4f} | {std:10.4f}")

        return True

    # 否则使用raw results
    elif os.path.exists(raw_path):
        with open(raw_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("\n" + "="*80)
        print("从原始结果计算统计...")
        print("="*80)
        print(f"时间戳: {data.get('timestamp', 'N/A')}")
        print(f"已完成轮数: {data.get('total_completed', 0)}")

        raw_results = data.get('results', [])
        if len(raw_results) > 0:
            import numpy as np

            print(f"\n共 {len(raw_results)} 轮结果")
            print("\n指标                          |  均值      |  标准差    |  最小值    |  最大值")
            print("-"*80)

            all_keys = set()
            for r in raw_results:
                all_keys.update(r.keys())

            for key in sorted(all_keys):
                values = [r[key] for r in raw_results if key in r and r[key] is not None]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)

                    if 'ratio' in key or 'rate' in key or 'accuracy' in key:
                        print(f"{key:30s}| {mean_val*100:8.2f}% | {std_val*100:8.4f}% | {min_val*100:8.2f}% | {max_val*100:8.2f}%")
                    else:
                        print(f"{key:30s}| {mean_val:10.4f} | {std_val:10.4f} | {min_val:10.4f} | {max_val:10.4f}")

        return True

    return False


def main():
    parser = argparse.ArgumentParser(description='MAPPO数据恢复工具')
    parser.add_argument('--check', action='store_true', help='检查是否有已保存的数据')
    parser.add_argument('--report', action='store_true', help='从已有数据生成报告')
    parser.add_argument('--rerun', action='store_true', help='提示如何重新运行')
    args = parser.parse_args()

    if args.check:
        check_saved_data()
    elif args.report:
        if not generate_report_from_saved():
            print("\n没有找到数据，请先运行 --check 确认状态")
            sys.exit(1)
    elif args.rerun:
        print("\n" + "="*80)
        print("重新运行MAPPO评估 (带自动保存)")
        print("="*80)
        print("\n运行命令:")
        print('  cd "f:\\桌面\\本科毕业论文\\结题\\uav_project"')
        print("  .\\venv\\Scripts\\python.exe main.py \\")
        print("    --exp 3 \\")
        print("    --include-mappo \\")
        print("    --use-cache \\")
        print('    --mappo-model "experiment_results/mappo_models/mappo_8bs_300uav_best.pt"')
        print("\n新功能:")
        print("  ✅ 每轮完成后自动保存到 exp3_mappo_raw_results.json")
        print("  ✅ 最终汇总保存到 exp3_mappo_summary.json")
        print("  ✅ 绘图出错不会丢失数据")
        print("  ✅ 支持Ctrl+C中断后从断点恢复")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
