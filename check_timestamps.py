import os
from datetime import datetime
import json

print("=" * 60)
print("检查数据文件时间戳和内容")
print("=" * 60)

# 检查exp4_data.json
path1 = 'experiment_results/exp4_data.json'
if os.path.exists(path1):
    ts1 = os.path.getmtime(path1)
    print(f"\n[1] exp4_data.json")
    print(f"    最后修改: {datetime.fromtimestamp(ts1)}")
    with open(path1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    print(f"    场景数量: {len(data1) - 1}")  # 减去_meta
    for scenario in ['smart_city', 'agriculture']:
        if scenario in data1:
            if 'enhanced' in data1[scenario]:
                avg_sat = data1[scenario]['enhanced'].get('avg_satisfaction', [None, None])
                conn = data1[scenario]['enhanced'].get('connected_ratio', None)
                print(f"    {scenario}-enhanced: satisfaction={avg_sat[0]:.4f}, connected_ratio={conn}")
else:
    print(f"\n[1] exp4_data.json 不存在!")

# 检查exp4_mappo_summary.json
path2 = 'experiment_results/exp4_mappo_summary.json'
if os.path.exists(path2):
    ts2 = os.path.getmtime(path2)
    print(f"\n[2] exp4_mappo_summary.json")
    print(f"    最后修改: {datetime.fromtimestamp(ts2)}")
    with open(path2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    print(f"    MAPPO运行次数: {data2.get('total_mappo_runs', 0)}")
    raw = data2.get('raw_results_by_scenario', {})
    for scenario in ['smart_city', 'agriculture']:
        if scenario in raw and len(raw[scenario]) > 0:
            first_result = raw[scenario][0]
            avg_sat = first_result.get('avg_satisfaction', None)
            conn = first_result.get('connected_ratio', None)
            print(f"    {scenario}-mappo(第1轮): satisfaction={avg_sat:.4f}, connected_ratio={conn}")
else:
    print(f"\n[2] exp4_mappo_summary.json 不存在!")

# 检查latest_figures目录中的图片
path3 = 'experiment_results/latest_figures'
if os.path.exists(path3):
    print(f"\n[3] latest_figures 目录中的图片:")
    files = os.listdir(path3)
    png_files = [f for f in files if f.endswith('.png')]
    for f in sorted(png_files)[-5:]:  # 显示最新的5个图片
        full_path = os.path.join(path3, f)
        ts = os.path.getmtime(full_path)
        size = os.path.getsize(full_path) / 1024  # KB
        print(f"    {f}: {datetime.fromtimestamp(ts)}, {size:.1f}KB")
else:
    print(f"\n[3] latest_figures 目录不存在!")

print("\n" + "=" * 60)
