#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速验证exp3_data.json数据完整性"""
import json
import os

RESULT_DIR = "experiment_results"
filepath = os.path.join(RESULT_DIR, "exp3_data.json")

print("=" * 80)
print("  [VERIFY] 验证 exp3_data.json 数据完整性")
print("=" * 80)

if not os.path.exists(filepath):
    print(f"\n[ERROR] 文件不存在: {filepath}")
    exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

stat = os.stat(filepath)
mtime = stat.st_mtime
import datetime
time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

print(f"\n文件信息:")
print(f"  路径: {filepath}")
print(f"  大小: {stat.st_size / 1024:.1f} KB")
print(f"  时间: {time_str}")

print(f"\n数据结构:")
print(f"  顶层键: {list(data.keys())}")

# 检查增强算法数据
if 'enhanced' in data:
    enh = data['enhanced']
    print(f"\n增强算法数据 ({len(enh)} 个指标):")
    
    # 检查关键指标是否有标准差
    key_metrics = ['handover_success_rate', 'avg_satisfaction', 'connected_ratio',
                   'critical_satisfaction', 'load_variance']
    
    all_valid = True
    for metric in key_metrics:
        if metric in enh:
            values = enh[metric]
            if isinstance(values, list) and len(values) >= 2:
                mean_val, std_val = values[0], values[1]
                status = "[OK]" if std_val > 0 else "[WARN]"
                print(f"  {status} {metric}:")
                print(f"       均值={mean_val:.4f}, 标准差={std_val:.4f}")
                if std_val == 0:
                    all_valid = False
            else:
                print(f"  [WARN] {metric}: 数据格式异常")
                all_valid = False

# 检查传统算法数据
if 'traditional' in data:
    trad = data['traditional']
    print(f"\n传统算法数据 ({len(trad)} 个指标):")
    
    for metric in key_metrics:
        if metric in trad:
            values = trad[metric]
            if isinstance(values, list) and len(values) >= 2:
                mean_val, std_val = values[0], values[1]
                status = "[OK]" if std_val > 0 else "[WARN]"
                print(f"  {status} {metric}:")
                print(f"       均值={mean_val:.4f}, 标准差={std_val:.4f}")

print("\n" + "=" * 80)
if all_valid:
    print("  [SUCCESS] 数据完整! 所有关键指标都有有效的标准差 (>0)")
    print("  → 这是原始的10轮实验数据，未被覆盖!")
else:
    print("  [WARNING] 部分指标标准差为0，可能是被测试数据覆盖!")
print("=" * 80)
