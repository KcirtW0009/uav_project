import json

print("=" * 70)
print("检查 exp4_mappo_raw_results.json 的内容")
print("=" * 70)

with open('experiment_results/exp4_mappo_raw_results.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"\n总共有 {len(raw_data)} 条记录")

# 按场景分组
from collections import defaultdict
by_scenario = defaultdict(list)

for record in raw_data:
    scenario = record.get('scenario', 'unknown')
    by_scenario[scenario].append(record)

print("\n各场景的记录数:")
for scenario, records in by_scenario.items():
    print(f"  {scenario}: {len(records)} 轮")

# 显示第一条记录的所有字段
if len(raw_data) > 0:
    print("\n[示例] 第一条记录的所有字段:")
    first = raw_data[0]
    for key, value in first.items():
        if key != 'scenario':
            print(f"  {key}: {value}")

print("\n[重点] 检查是否有 connected_ratio:")
for scenario in ['agriculture', 'smart_city', 'emergency_rescue', 'logistics_delivery']:
    if scenario in by_scenario and len(by_scenario[scenario]) > 0:
        sample = by_scenario[scenario][0]
        has_connected = 'connected_ratio' in sample
        has_throughput = 'total_throughput' in sample
        avg_sat = sample.get('avg_satisfaction', 'N/A')
        print(f"  {scenario:20s}: satisfaction={avg_sat}, connected_ratio={'YES' if has_connected else 'MISSING'}, total_throughput={'YES' if has_throughput else 'MISSING'}")
