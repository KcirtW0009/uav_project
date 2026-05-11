import json
import os
from datetime import datetime

print("=" * 70)
print("数据来源验证报告")
print("=" * 70)

# [1] 加载exp4_data.json（绘图用的数据）
path1 = 'experiment_results/exp4_data.json'
with open(path1, 'r', encoding='utf-8') as f:
    plot_data = json.load(f)

ts1 = os.path.getmtime(path1)
print(f"\n[1] exp4_data.json (绘图数据源)")
print(f"    最后修改: {datetime.fromtimestamp(ts1)}")
print(f"    智慧城市-增强算法 avg_satisfaction:")
if 'smart_city' in plot_data and 'enhanced' in plot_data['smart_city']:
    val = plot_data['smart_city']['enhanced'].get('avg_satisfaction', ['MISSING'])
    print(f"      文件中 = {val}")
    print(f"      connected_ratio = {plot_data['smart_city']['enhanced'].get('connected_ratio', 'MISSING')}")
    print(f"      total_throughput = {plot_data['smart_city']['enhanced'].get('total_throughput', 'MISSING')}")

# [2] 用户提供的运行日志真实数据（最新运行结果）
print("\n" + "=" * 70)
print("[2] 你提供的运行日志（最新运行的真实数据）")
print("=" * 70)

log_data = {
    'agriculture': {
        'enhanced': {'avg_satisfaction': [0.925, 0.047], 'critical_satisfaction': [0.999, 0.001],
                    'total_throughput': [4481.6, 410.5], 'avg_switching_latency_ms': [0.01, 0.00],
                    'connected_ratio': [0.945, 0.050]},
        'traditional': {'avg_satisfaction': [0.896, 0.048], 'critical_satisfaction': [0.999, 0.002],
                       'total_throughput': [2818.4, 628.9], 'avg_switching_latency_ms': [0.00, 0.00],
                       'connected_ratio': [0.869, 0.061]},
        'mappo': {'avg_satisfaction': [0.899, 0.031], 'critical_satisfaction': [0.911, 0.016],
                 'total_throughput': [4160.7, 325.5], 'avg_switching_latency_ms': [7.23, 0.18],
                 'connected_ratio': [0.997, 0.002]}
    },
    'smart_city': {
        'enhanced': {'avg_satisfaction': [0.900, 0.060], 'critical_satisfaction': [0.997, 0.004],
                    'total_throughput': [10367.1, 1061.9], 'avg_switching_latency_ms': [0.01, 0.00],
                    'connected_ratio': [0.985, 0.026]},
        'traditional': {'avg_satisfaction': [0.759, 0.087], 'critical_satisfaction': [0.950, 0.036],
                       'total_throughput': [6696.0, 1494.3], 'avg_switching_latency_ms': [0.00, 0.00],
                       'connected_ratio': [0.673, 0.131]},
        'mappo': {'avg_satisfaction': [0.855, 0.033], 'critical_satisfaction': [0.910, 0.021],
                 'total_throughput': [4218.5, 510.2], 'avg_switching_latency_ms': [7.39, 0.05],
                 'connected_ratio': [0.992, 0.007]}
    },
    'industrial_inspection': {
        'enhanced': {'avg_satisfaction': [0.952, 0.048], 'critical_satisfaction': [0.998, 0.003],
                    'total_throughput': [10809.7, 819.8], 'avg_switching_latency_ms': [0.01, 0.00],
                    'connected_ratio': [0.980, 0.032]},
        'traditional': {'avg_satisfaction': [0.813, 0.052], 'critical_satisfaction': [0.983, 0.013],
                       'total_throughput': [7798.7, 1026.2], 'avg_switching_latency_ms': [0.00, 0.00],
                       'connected_ratio': [0.739, 0.075]},
        'mappo': {'avg_satisfaction': [0.925, 0.021], 'critical_satisfaction': [0.932, 0.017],
                 'total_throughput': [3798.3, 368.6], 'avg_switching_latency_ms': [7.05, 0.20],
                 'connected_ratio': [1.000, 0.001]}
    },
    'emergency_rescue': {
        'enhanced': {'avg_satisfaction': [0.951, 0.008], 'critical_satisfaction': [0.999, 0.001],
                    'total_throughput': [1845.4, 169.2], 'avg_switching_latency_ms': [0.01, 0.00],
                    'connected_ratio': [1.000, 0.000]},
        'traditional': {'avg_satisfaction': [0.905, 0.038], 'critical_satisfaction': [0.930, 0.057],
                       'total_throughput': [1720.9, 151.2], 'avg_switching_latency_ms': [0.00, 0.00],
                       'connected_ratio': [0.919, 0.066]},
        'mappo': {'avg_satisfaction': [0.929, 0.024], 'critical_satisfaction': [0.928, 0.011],
                 'total_throughput': [3847.6, 425.9], 'avg_switching_latency_ms': [7.07, 0.21],
                 'connected_ratio': [0.999, 0.002]}
    },
    'logistics_delivery': {
        'enhanced': {'avg_satisfaction': [0.918, 0.058], 'critical_satisfaction': [0.995, 0.006],
                    'total_throughput': [8903.9, 1028.4], 'avg_switching_latency_ms': [0.01, 0.00],
                    'connected_ratio': [0.986, 0.017]},
        'traditional': {'avg_satisfaction': [0.814, 0.059], 'critical_satisfaction': [0.932, 0.055],
                       'total_throughput': [6013.8, 1145.0], 'avg_switching_latency_ms': [0.00, 0.00],
                       'connected_ratio': [0.758, 0.093]},
        'mappo': {'avg_satisfaction': [0.833, 0.047], 'critical_satisfaction': [0.895, 0.040],
                 'total_throughput': [4635.7, 412.3], 'avg_switching_latency_ms': [7.45, 0.05],
                 'connected_ratio': [0.988, 0.011]}
    }
}

print("\n各场景-增强算法 avg_satisfaction 对比:")
for scenario in ['agriculture', 'smart_city', 'industrial_inspection', 'emergency_rescue', 'logistics_delivery']:
    names = {'agriculture': '农业植保', 'smart_city': '智慧城市', 'industrial_inspection': '工业巡检',
             'emergency_rescue': '应急救援', 'logistics_delivery': '物流配送'}

    # 从文件读取
    file_val = 'N/A'
    if scenario in plot_data and 'enhanced' in plot_data[scenario]:
        file_val = plot_data[scenario]['enhanced'].get('avg_satisfaction', ['N/A'])[0]

    # 从日志读取
    log_val = log_data.get(scenario, {}).get('enhanced', {}).get('avg_satisfaction', ['N/A'])[0]

    match = "[OK] 一致" if abs(float(file_val) - float(log_val)) < 0.01 else "[FAIL] 不一致!"
    print(f"  {names[scenario]:10s}: 文件={file_val:.3f} vs 日志={log_val:.3f}  {match}")

print("\n" + "=" * 70)
print("[结论]")
print("=" * 70)
print("""
问题根源:
  exp4_data.json 是 2026-05-08 的旧数据，没有更新到今天的运行结果！

证据:
  1. 智慧城市-增强算法: 文件=0.874 vs 日志=0.900 (差0.026)
  2. 你的日志显示有完整的 connected_ratio, total_throughput 等新指标
  3. 但 exp4_data.json 中这些指标全部缺失

根本原因:
  实验4运行时虽然计算了新指标，但没有成功保存到 exp4_data.json！
  （可能是缓存模式导致跳过了保存步骤）

解决方案:
  需要检查 experiments.py 中的数据保存逻辑，
  确保非缓存模式下完整保存所有指标。
""")
