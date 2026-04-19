#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
业务识别模型重新比较脚本
重新训练并比较所有分类器（决策树、SVM、MLP、随机森林、GBDT），输出详细对比结果。

使用方法：
    .\venv\Scripts\python.exe rerun_model_comparison.py

注意：
    - 会强制重新训练所有模型，忽略已保存的模型文件
    - 会生成新的 all_model_results.pkl 和 business_recognition_model.pkl
    - 多目标优化权重：准确性40% + 稳定性30% + 实时性30%
"""

import os
import sys
import pickle

# 添加当前目录到路径，确保能导入 uav_system 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.recognition import train_or_load_recognition_model, RECOGNITION_SEED

def main():
    """主函数：强制重新训练并比较所有模型"""
    print("=" * 80)
    print("业务识别模型重新比较")
    print("多目标优化：准确性40% + 稳定性30% + 实时性30%")
    print("=" * 80)

    # 设置业务识别模块专用随机种子以确保结果可复现
    set_global_seed(RECOGNITION_SEED)
    
    # 强制重新训练并比较所有模型
    # force_retrain=True: 忽略已保存的模型，重新训练
    # compare_models=True: 比较所有模型并选择最优
    # verbose=True: 输出详细对比信息
    model, all_results = train_or_load_recognition_model(
        force_retrain=True,
        compare_models=True,
        verbose=True
    )
    
    # 打印最佳模型信息
    print("\n" + "=" * 80)
    print("最佳模型信息：")
    model.print_model_info()
    
    # 保存结果到文件
    with open("all_model_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n模型比较结果已保存至：{os.path.abspath('all_model_results.pkl')}")
    
    # 显示所有模型的排名
    if all_results:
        print("\n" + "=" * 80)
        print("所有模型综合得分排名：")
        sorted_results = sorted(all_results, key=lambda x: x['combined_score'], reverse=True)
        for i, r in enumerate(sorted_results, 1):
            marker = " *" if r['type'] == model.model_type else ""
            print(f"  {i}. {r['type']}: 综合得分={r['combined_score']:.4f}, "
                  f"F1={r['f1']:.3f}, 延迟={r['inference_latency_ms']:.3f}ms{marker}")
    
    print("\n" + "=" * 80)
    print("模型重新比较完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()