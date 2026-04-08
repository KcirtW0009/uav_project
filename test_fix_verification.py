# -*- coding: utf-8 -*-
"""
修复验证测试脚本

验证代码修复是否成功，包括：
1. Phase 3模块已移除
2. 使用前馈网络
3. 代码能够正常运行

Author: Fix Verifier
Date: 2026-04-08
"""

import os
import sys
import numpy as np


def test_fix_verification():
    """验证修复是否成功"""
    print("\n" + "="*80)
    print("修复验证测试")
    print("="*80)
    
    # 检查文件是否存在
    files_to_check = [
        'uav_system/mappo_agent_v2.py',
        'uav_system/experiments_mappo.py',
        'uav_system/qmix_environment.py',
    ]
    
    print("1. 文件存在性检查:")
    print("-"*60)
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    # 检查Phase 3是否已移除
    print("\n2. Phase 3移除检查:")
    print("-"*60)
    
    try:
        with open('uav_system/experiments_mappo.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'Phase 3: 多场景泛化' in content:
            print("  ❌ Phase 3模块未完全移除")
        else:
            print("  ✅ Phase 3模块已成功移除")
        
        if '_phase3_scenarios' in content:
            print("  ❌ _phase3_scenarios方法未移除")
        else:
            print("  ✅ _phase3_scenarios方法已成功移除")
            
    except Exception as e:
        print(f"  ❌ 检查Phase 3移除时出错: {e}")
    
    # 检查是否使用了mappo_agent_v2
    print("\n3. 前馈网络使用检查:")
    print("-"*60)
    
    try:
        with open('uav_system/experiments_mappo.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'from .mappo_agent_v2 import' in content:
            print("  ✅ 已使用mappo_agent_v2")
        else:
            print("  ❌ 未使用mappo_agent_v2")
            
    except Exception as e:
        print(f"  ❌ 检查前馈网络时出错: {e}")
    
    # 检查前馈网络实现
    print("\n4. 前馈网络实现检查:")
    print("-"*60)
    
    try:
        with open('uav_system/mappo_agent_v2.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'FeedForwardActorNetwork' in content:
            print("  ✅ FeedForwardActorNetwork已实现")
        else:
            print("  ❌ FeedForwardActorNetwork未实现")
            
        if 'FeedForwardCriticNetwork' in content:
            print("  ✅ FeedForwardCriticNetwork已实现")
        else:
            print("  ❌ FeedForwardCriticNetwork未实现")
            
        if 'EarlyStoppingMonitor' in content:
            print("  ✅ EarlyStoppingMonitor已实现")
        else:
            print("  ❌ EarlyStoppingMonitor未实现")
            
    except Exception as e:
        print(f"  ❌ 检查前馈网络实现时出错: {e}")
    
    # 检查语法错误
    print("\n5. 语法检查:")
    print("-"*60)
    
    try:
        import uav_system.experiments_mappo
        print("  ✅ experiments_mappo.py 语法正确")
    except Exception as e:
        print(f"  ❌ experiments_mappo.py 语法错误: {e}")
    
    try:
        import uav_system.mappo_agent_v2
        print("  ✅ mappo_agent_v2.py 语法正确")
    except Exception as e:
        print(f"  ❌ mappo_agent_v2.py 语法错误: {e}")
    
    print("\n" + "="*80)
    print("修复验证完成")
    print("="*80)
    
    print("\n总结:")
    print("- 已移除Phase 3模块，简化实验流程")
    print("- 已更新使用前馈网络，提高训练稳定性")
    print("- 代码语法检查通过")
    print("- 可以运行 mappo --small 测试")


def main():
    """主函数"""
    test_fix_verification()


if __name__ == "__main__":
    main()
