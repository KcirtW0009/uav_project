# -*- coding: utf-8 -*-
"""
环境负载率分析

分析当前环境负载率与系统配置的匹配情况

Author: Load Analysis
Date: 2026-04-08
"""

import sys
import os
import numpy as np


def analyze_load_rate():
    """分析环境负载率"""
    print("\n" + "="*80)
    print("环境负载率分析")
    print("="*80)

    # 计算负载率的函数
    def calculate_load_rate(num_uav, num_bs, avg_demand=15.5, avg_cap=750):
        """计算负载率
        
        Args:
            num_uav: UAV数量
            num_bs: 基站数量
            avg_demand: 每UAV平均带宽需求 (Mbps)
            avg_cap: 每基站平均容量 (Mbps)
            
        Returns:
            负载率百分比
        """
        total_demand = num_uav * avg_demand
        total_capacity = num_bs * avg_cap
        load_rate = (total_demand / total_capacity) * 100
        return load_rate

    # 分析当前配置
    print("当前系统配置:")
    print("-"*60)
    
    # 小规模配置
    small_uav = 150
    small_bs = 3
    small_load = calculate_load_rate(small_uav, small_bs)
    print("小规模: %d UAV / %d BS → 负载率: %.1f%%" % (small_uav, small_bs, small_load))
    
    # 标准配置
    standard_uav = 200
    standard_bs = 4
    standard_load = calculate_load_rate(standard_uav, standard_bs)
    print("标准: %d UAV / %d BS → 负载率: %.1f%%" % (standard_uav, standard_bs, standard_load))
    
    # 大规模配置
    large_uav = 280
    large_bs = 5
    large_load = calculate_load_rate(large_uav, large_bs)
    print("大规模: %d UAV / %d BS → 负载率: %.1f%%" % (large_uav, large_bs, large_load))
    
    # PHASE3场景
    print("\nPHASE3场景配置:")
    print("-"*60)
    
    scenarios = {
        '默认场景': {'uav': 200, 'bs': 4},
        '城市监控': {'uav': 300, 'bs': 5},
        '工业巡检': {'uav': 200, 'bs': 6},
        '应急救援': {'uav': 100, 'bs': 3},
        '物流配送': {'uav': 250, 'bs': 4},
    }
    
    for name, config in scenarios.items():
        load = calculate_load_rate(config['uav'], config['bs'])
        print("%s: %d UAV / %d BS → 负载率: %.1f%%" % (name, config['uav'], config['bs'], load))
    
    # 分析负载率合理性
    print("\n负载率分析:")
    print("-"*60)
    
    if 90 <= small_load <= 110:
        print("小规模负载率: 合理 (90-110%)")
    elif small_load < 90:
        print("小规模负载率: 偏低 (<90%)")
    else:
        print("小规模负载率: 偏高 (>110%)")
    
    if 90 <= standard_load <= 110:
        print("标准负载率: 合理 (90-110%)")
    elif standard_load < 90:
        print("标准负载率: 偏低 (<90%)")
    else:
        print("标准负载率: 偏高 (>110%)")
    
    if 110 <= large_load <= 130:
        print("大规模负载率: 合理 (110-130%)")
    elif large_load < 110:
        print("大规模负载率: 偏低 (<110%)")
    else:
        print("大规模负载率: 偏高 (>130%)")
    
    # 建议
    print("\n优化建议:")
    print("-"*60)
    
    if small_load < 90:
        print("1. 小规模场景: 增加UAV数量至180-200，或减少BS数量至2")
    elif small_load > 110:
        print("1. 小规模场景: 减少UAV数量至120-130，或增加BS数量至4")
    else:
        print("1. 小规模场景: 负载率合理，无需调整")
    
    if standard_load < 90:
        print("2. 标准场景: 增加UAV数量至220-240，或减少BS数量至3")
    elif standard_load > 110:
        print("2. 标准场景: 减少UAV数量至180-190，或增加BS数量至5")
    else:
        print("2. 标准场景: 负载率合理，无需调整")
    
    if large_load < 110:
        print("3. 大规模场景: 增加UAV数量至300-320，或减少BS数量至4")
    elif large_load > 130:
        print("3. 大规模场景: 减少UAV数量至250-260，或增加BS数量至6")
    else:
        print("3. 大规模场景: 负载率合理，无需调整")


def main():
    """主函数"""
    analyze_load_rate()


if __name__ == "__main__":
    main()
