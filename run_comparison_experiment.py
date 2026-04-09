# -*- coding: utf-8 -*-
"""
MAPPO 对比实验脚本

对比原配置和优化配置的效果
"""

import subprocess
import sys
import os
import json
from datetime import datetime

# 实验配置
EXPERIMENTS = [
    {
        'name': '原MAPPO配置（基准）',
        'command': 'venv\\Scripts\\python.exe main.py --exp mappo --rl-phase both --small',
        'output_dir': 'experiment_results/baseline'
    },
    {
        'name': '优化MAPPO配置（参数搜索后）',
        'command': 'venv\\Scripts\\python.exe run_optimized_mappo.py --scenario high --train --eval',
        'output_dir': 'experiment_results/optimized'
    }
]

def run_experiment(exp_config):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"开始运行: {exp_config['name']}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    os.makedirs(exp_config['output_dir'], exist_ok=True)
    
    # 运行命令
    result = subprocess.run(
        exp_config['command'],
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    # 保存输出
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(exp_config['output_dir'], f'output_{timestamp}.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Experiment: {exp_config['name']}\n")
        f.write(f"Command: {exp_config['command']}\n")
        f.write(f"Time: {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)
    
    print(f"输出已保存: {log_file}")
    
    if result.returncode == 0:
        print(f"✓ {exp_config['name']} 完成")
    else:
        print(f"✗ {exp_config['name']} 失败 (返回码: {result.returncode})")
    
    return result.returncode == 0

def main():
    """主函数"""
    print("="*60)
    print("MAPPO 对比实验")
    print("对比: 原配置 vs 优化配置")
    print("="*60)
    
    results = {}
    
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        results[exp['name']] = success
        
        if not success:
            print(f"\n警告: {exp['name']} 运行失败，继续下一个实验...")
    
    # 汇总结果
    print("\n" + "="*60)
    print("实验完成汇总")
    print("="*60)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status}: {name}")
    
    print("\n所有实验结果保存在:")
    for exp in EXPERIMENTS:
        print(f"  - {exp['output_dir']}")

if __name__ == '__main__':
    main()
