"""
查看切换成功率计数机制的统计数据

分析增强算法的切换成功率是否因为执行阶段的统计过滤而虚高
"""

import pickle
import numpy as np

# 先查看数据结构
print("=" * 80)
print("查看数据结构")
print("=" * 80)

with open('all_model_results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"\nresults的类型: {type(results)}")

if isinstance(results, dict):
    print(f"\nkeys: {list(results.keys())}")
    for key, value in results.items():
        print(f"\n{key}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, dict):
            print(f"  子keys: {list(value.keys())}")
elif isinstance(results, list):
    print(f"\n列表长度: {len(results)}")
    if len(results) > 0:
        print(f"第一个元素: {type(results[0])}")
        if isinstance(results[0], dict):
            print(f"  keys: {list(results[0].keys())}")
elif isinstance(results, np.ndarray):
    print(f"\n数组形状: {results.shape}")
    print(f"数组类型: {results.dtype}")
else:
    print(f"\n无法识别的类型")
