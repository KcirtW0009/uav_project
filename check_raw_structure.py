import json

print("=" * 70)
print("检查 exp4_mappo_raw_results.json 的实际结构")
print("=" * 70)

with open('experiment_results/exp4_mappo_raw_results.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"\n数据类型: {type(raw_data)}")

if isinstance(raw_data, dict):
    print(f"顶层键: {list(raw_data.keys())[:10]}")
    for key in list(raw_data.keys())[:3]:
        val = raw_data[key]
        print(f"\n  键 '{key}': 类型={type(val)}, 值预览={str(val)[:200]}")

elif isinstance(raw_data, list):
    print(f"列表长度: {len(raw_data)}")
    if len(raw_data) > 0:
        print(f"\n第一个元素: {raw_data[0]}")
        if isinstance(raw_data[0], dict):
            print(f"字段: {list(raw_data[0].keys())}")
