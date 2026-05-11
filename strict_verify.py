import json
import numpy as np

print("=" * 70)
print("严格验证：运行日志 vs 实际保存文件 vs 绘图数据")
print("=" * 70)

# [1] 加载用户的运行日志数据（你提供的）
log_data = {
    'agriculture': {
        'mappo': {'avg_satisfaction': [0.899, 0.031], 'critical_satisfaction': [0.911, 0.016],
                 'connected_ratio': [0.997, 0.002], 'total_throughput': [4160.7, 325.5]},
    },
    'smart_city': {
        'mappo': {'avg_satisfaction': [0.855, 0.033], 'critical_satisfaction': [0.910, 0.021],
                 'connected_ratio': [0.992, 0.007], 'total_throughput': [4218.5, 510.2]},
    },
    'emergency_rescue': {
        'mappo': {'avg_satisfaction': [0.929, 0.024], 'critical_satisfaction': [0.928, 0.011],
                 'connected_ratio': [0.999, 0.002], 'total_throughput': [3847.6, 425.9]},
    },
    'logistics_delivery': {
        'mappo': {'avg_satisfaction': [0.833, 0.047], 'critical_satisfaction': [0.895, 0.040],
                 'connected_ratio': [0.988, 0.011], 'total_throughput': [4635.7, 412.3]},
    }
}

# [2] 加载exp4_mappo_summary.json（实际保存的MAPPO数据）
with open('experiment_results/exp4_mappo_summary.json', 'r', encoding='utf-8') as f:
    mappo_summary = json.load(f)

print("\n[1] MAPPO数据对比: 运行日志 vs exp4_mappo_summary.json")
print("-" * 80)

raw_results = mappo_summary.get('raw_results_by_scenario', {})

for scenario in ['agriculture', 'smart_city', 'emergency_rescue', 'logistics_delivery']:
    names = {'agriculture': '农业植保', 'smart_city': '智慧城市',
             'emergency_rescue': '应急救援', 'logistics_delivery': '物流配送'}

    # 从日志获取
    log_sat = log_data.get(scenario, {}).get('mappo', {}).get('avg_satisfaction', ['N/A'])[0]

    # 从文件获取
    if scenario in raw_results and len(raw_results[scenario]) > 0:
        results = raw_results[scenario]
        file_sats = [r.get('avg_satisfaction') for r in results if r.get('avg_satisfaction') is not None]
        if file_sats:
            file_sat_mean = np.mean(file_sats)
            file_sat_std = np.std(file_sats)
            diff = abs(float(log_sat) - float(file_sat_mean))
            status = "[OK]" if diff < 0.01 else f"[FAIL] 差={diff:.3f}"
            print(f"  {names[scenario]:10s}: 日志={log_sat:.3f} vs 文件={file_sat_mean:.3f}+/-{file_sat_std:.3f} {status}")
        else:
            print(f"  {names[scenario]:10s}: 日志={log_sat:.3f} vs 文件=[无satisfaction数据]")
    else:
        print(f"  {names[scenario]:10s}: 日志={log_sat:.3f} vs 文件=[场景不存在]")

# [3] 检查绘图脚本如何加载MAPPO数据
print("\n" + "=" * 70)
print("[2] 检查绘图脚本的load_exp4_data()逻辑")
print("=" * 70)

with open('experiment_results/exp4_data.json', 'r', encoding='utf-8') as f:
    plot_data_source = json.load(f)

print("\n绘图脚本读取的 exp4_data.json 中的MAPPO数据:")
for scenario in ['agriculture', 'smart_city', 'emergency_rescue', 'logistics_delivery']:
    names = {'agriculture': '农业植保', 'smart_city': '智慧城市',
             'emergency_rescue': '应急救援', 'logistics_delivery': '物流配送'}

    if scenario in plot_data_source and 'mappo' in plot_data_source[scenario]:
        mappo_in_file = plot_data_source[scenario]['mappo']
        sat_in_file = mappo_in_file.get('avg_satisfaction', ['MISSING'])
        print(f"  {names[scenario]:10s}: satisfaction={sat_in_file}")
    else:
        print(f"  {names[scenario]:10s}: [无MAPPO数据]")

print("\n" + "=" * 70)
print("[结论]")
print("=" * 70)
print("""
如果 [1] 中日志和文件不一致：
  → 说明 exp4_mappo_summary.json 保存的不是你看到的那次运行的数据

如果 [2] 中绘图数据和 [1] 的文件数据不一致：
  → 说明绘图脚本有额外的数据处理逻辑，导致数据被篡改

无论哪种情况，都需要深入检查数据流的每个环节。
""")
