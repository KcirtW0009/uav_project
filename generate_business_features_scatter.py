#!/usr/bin/env python3
"""
生成三类典型UAV业务的QoS特征散点图 (图2-4)
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入项目模块
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.business import BusinessType, BUSINESS_FEATURE_PARAMS
from uav_system.config import GLOBAL_SEED

def generate_business_samples(num_samples=500, seed=GLOBAL_SEED):
    """
    生成三类业务的样本数据
    """
    np.random.seed(seed)
    samples = []
    labels = []
    colors = []
    business_names = []
    
    # 业务类型与颜色映射
    type_color_map = {
        BusinessType.CONTROL_SIGNAL: ('控制信令', '#FF6B6B'),
        BusinessType.VIDEO_STREAMING: ('视频回传', '#4ECDC4'),
        BusinessType.ENVIRONMENT_MONITORING: ('环境监测', '#95E1D3')
    }
    
    for bt, (name, color) in type_color_map.items():
        params = BUSINESS_FEATURE_PARAMS[bt]
        for _ in range(num_samples):
            # 生成时延 (ms)
            delay = np.clip(np.random.normal(params['delay'][0], params['delay'][1]), 0, 1000)
            # 生成带宽 (Mbps)
            bandwidth = np.clip(np.random.normal(params['bandwidth'][0], params['bandwidth'][1]), 0, 100)
            # 生成丢包率
            loss_rate = np.random.beta(params['loss_beta'][0], params['loss_beta'][1]) * params['loss_scale']
            # 生成抖动 (ms)
            jitter = np.clip(np.random.normal(params['jitter'][0], params['jitter'][1]), 0, 100)
            
            samples.append([delay, bandwidth, loss_rate, jitter])
            labels.append(bt.value)
            colors.append(color)
            business_names.append(name)
    
    return np.array(samples), np.array(labels), np.array(colors), np.array(business_names)

def plot_business_features_scatter(output_path='figures/business_features_scatter.png'):
    """
    绘制三类业务的QoS特征散点图
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成样本数据
    X, y, colors, names = generate_business_samples(num_samples=300, seed=GLOBAL_SEED)
    
    # 提取特征
    delays = X[:, 0]      # 时延 (ms)
    bandwidths = X[:, 1]  # 带宽 (Mbps)
    loss_rates = X[:, 2]  # 丢包率
    jitters = X[:, 3]     # 抖动 (ms)
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('三类典型UAV业务的QoS特征散点图', fontsize=16, fontweight='bold')
    
    # 子图1: 时延 vs 带宽
    ax1 = axes[0, 0]
    unique_names = ['控制信令', '视频回传', '环境监测']
    unique_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax1.scatter(delays[mask], bandwidths[mask], 
                   c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    
    ax1.set_xlabel('时延 (ms)', fontsize=12)
    ax1.set_ylabel('带宽 (Mbps)', fontsize=12)
    ax1.set_title('(a) 时延-带宽分布', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 子图2: 时延 vs 丢包率
    ax2 = axes[0, 1]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax2.scatter(delays[mask], loss_rates[mask], 
                   c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    
    ax2.set_xlabel('时延 (ms)', fontsize=12)
    ax2.set_ylabel('丢包率', fontsize=12)
    ax2.set_title('(b) 时延-丢包率分布', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 丢包率范围广，使用对数坐标
    ax2.legend(loc='upper right', fontsize=10)
    
    # 子图3: 带宽 vs 丢包率
    ax3 = axes[1, 0]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax3.scatter(bandwidths[mask], loss_rates[mask], 
                   c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    
    ax3.set_xlabel('带宽 (Mbps)', fontsize=12)
    ax3.set_ylabel('丢包率', fontsize=12)
    ax3.set_title('(c) 带宽-丢包率分布', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.legend(loc='upper right', fontsize=10)
    
    # 子图4: 抖动 vs 时延
    ax4 = axes[1, 1]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax4.scatter(jitters[mask], delays[mask], 
                   c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    
    ax4.set_xlabel('抖动 (ms)', fontsize=12)
    ax4.set_ylabel('时延 (ms)', fontsize=12)
    ax4.set_title('(d) 抖动-时延分布', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"QoS特征散点图已保存到: {output_path}")
    print(f"样本总数: {len(X)}")
    print(f"业务分布: 控制信令={np.sum(names=='控制信令')}, "
          f"视频回传={np.sum(names=='视频回传')}, "
          f"环境监测={np.sum(names=='环境监测')}")

if __name__ == '__main__':
    plot_business_features_scatter()