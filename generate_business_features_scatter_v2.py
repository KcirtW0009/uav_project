#!/usr/bin/env python3
"""
生成三类典型UAV业务的QoS特征散点图 (图2-4) - 改进版

改进点：
1. 解决字体警告问题（减号字符显示）
2. 增加数据随机性和自然分布
3. 将四张子图拆分为独立文件，便于论文排版
4. 增加特征之间的相关性，使分布更真实
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 设置中文字体，解决减号字符问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
matplotlib.rcParams['axes.formatter.use_mathtext'] = True  # 使用数学文本格式

# 导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.business import BusinessType, BUSINESS_FEATURE_PARAMS
from uav_system.config import GLOBAL_SEED

def generate_business_samples_improved(num_samples=500, seed=GLOBAL_SEED):
    """
    生成三类业务的样本数据 - 改进版
    
    改进：
    1. 增加特征之间的相关性（时延-带宽负相关）
    2. 增加随机噪声
    3. 调整丢包率分布，使其更分散
    4. 使用协方差矩阵生成更自然的多维分布
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
        
        # 基础参数
        delay_mean, delay_std = params['delay']
        bw_mean, bw_std = params['bandwidth']
        jitter_mean, jitter_std = params['jitter']
        
        # 生成基础正态分布样本
        base_delays = np.random.normal(delay_mean, delay_std, num_samples)
        base_bandwidths = np.random.normal(bw_mean, bw_std, num_samples)
        base_jitters = np.random.normal(jitter_mean, jitter_std, num_samples)
        
        # 1. 增加时延和带宽的负相关性（实际网络中带宽越高时延越低）
        if bt == BusinessType.CONTROL_SIGNAL:
            # 控制信令：强负相关
            correlation = -0.8
        elif bt == BusinessType.VIDEO_STREAMING:
            # 视频回传：中等负相关
            correlation = -0.6
        else:
            # 环境监测：弱负相关
            correlation = -0.3
            
        # 通过混合样本引入相关性
        for i in range(num_samples):
            # 基础值
            delay = base_delays[i]
            bandwidth = base_bandwidths[i]
            jitter = base_jitters[i]
            
            # 引入相关性：带宽越高，时延越低
            delay = delay * (1 - correlation * (bandwidth - bw_mean) / (bw_std * 3))
            
            # 2. 增加随机噪声（10%的随机扰动）
            delay = delay * (1 + np.random.uniform(-0.1, 0.1))
            bandwidth = bandwidth * (1 + np.random.uniform(-0.1, 0.1))
            jitter = jitter * (1 + np.random.uniform(-0.1, 0.1))
            
            # 3. 改进的丢包率生成：使用混合分布
            if bt == BusinessType.CONTROL_SIGNAL:
                # 控制信令：极低丢包率，但允许偶尔的高丢包（网络波动）
                if np.random.random() < 0.95:  # 95%正常情况
                    loss_rate = np.random.beta(1, 1000) * params['loss_scale']
                else:  # 5%异常情况
                    loss_rate = np.random.beta(1, 10) * params['loss_scale'] * 10
                    
            elif bt == BusinessType.VIDEO_STREAMING:
                # 视频回传：低丢包率，有一定波动
                if np.random.random() < 0.9:  # 90%正常情况
                    loss_rate = np.random.beta(5, 100) * params['loss_scale']
                else:  # 10%异常情况
                    loss_rate = np.random.beta(5, 20) * params['loss_scale'] * 5
                    
            else:  # ENVIRONMENT_MONITORING
                # 环境监测：丢包率变化较大
                loss_rate = np.random.beta(2, 20) * params['loss_scale']
                # 增加一些极端值
                if np.random.random() < 0.15:
                    loss_rate *= np.random.uniform(2, 5)
            
            # 4. 添加抖动和时延的正相关性（时延高通常抖动也大）
            jitter = jitter * (1 + 0.3 * (delay - delay_mean) / (delay_std + 1))
            
            # 限制范围
            delay = np.clip(delay, 0, 1000)
            bandwidth = np.clip(bandwidth, 0.01, 100)
            loss_rate = np.clip(loss_rate, 1e-6, 0.5)
            jitter = np.clip(jitter, 0, 100)
            
            samples.append([delay, bandwidth, loss_rate, jitter])
            labels.append(bt.value)
            colors.append(color)
            business_names.append(name)
    
    return np.array(samples), np.array(labels), np.array(colors), np.array(business_names)

def plot_individual_scatters(output_dir='figures/business_features'):
    """
    绘制四张独立的QoS特征散点图
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成样本数据
    X, y, colors, names = generate_business_samples_improved(num_samples=300, seed=GLOBAL_SEED)
    
    # 提取特征
    delays = X[:, 0]      # 时延 (ms)
    bandwidths = X[:, 1]  # 带宽 (Mbps)
    loss_rates = X[:, 2]  # 丢包率
    jitters = X[:, 3]     # 抖动 (ms)
    
    # 业务类型和颜色
    unique_names = ['控制信令', '视频回传', '环境监测']
    unique_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # 1. 时延 vs 带宽
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax1.scatter(delays[mask], bandwidths[mask], 
                   c=color, s=50, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax1.set_xlabel('时延 (ms)', fontsize=14)
    ax1.set_ylabel('带宽 (Mbps)', fontsize=14)
    ax1.set_title('三类典型UAV业务的时延-带宽分布', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'delay_bandwidth_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. 时延 vs 丢包率
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax2.scatter(delays[mask], loss_rates[mask], 
                   c=color, s=50, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax2.set_xlabel('时延 (ms)', fontsize=14)
    ax2.set_ylabel('丢包率', fontsize=14)
    ax2.set_title('三类典型UAV业务的时延-丢包率分布', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')  # 丢包率范围广，使用对数坐标
    ax2.legend(loc='upper right', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'delay_loss_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. 带宽 vs 丢包率
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax3.scatter(bandwidths[mask], loss_rates[mask], 
                   c=color, s=50, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax3.set_xlabel('带宽 (Mbps)', fontsize=14)
    ax3.set_ylabel('丢包率', fontsize=14)
    ax3.set_title('三类典型UAV业务的带宽-丢包率分布', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'bandwidth_loss_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. 抖动 vs 时延
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax4.scatter(jitters[mask], delays[mask], 
                   c=color, s=50, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax4.set_xlabel('抖动 (ms)', fontsize=14)
    ax4.set_ylabel('时延 (ms)', fontsize=14)
    ax4.set_title('三类典型UAV业务的抖动-时延分布', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right', fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'jitter_delay_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. 组合图（可选，保持原样）
    fig5, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig5.suptitle('三类典型UAV业务的QoS特征散点图', fontsize=18, fontweight='bold')
    
    # 子图1: 时延 vs 带宽
    ax = axes[0, 0]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax.scatter(delays[mask], bandwidths[mask], 
                  c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    ax.set_xlabel('时延 (ms)', fontsize=12)
    ax.set_ylabel('带宽 (Mbps)', fontsize=12)
    ax.set_title('(a) 时延-带宽分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # 子图2: 时延 vs 丢包率
    ax = axes[0, 1]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax.scatter(delays[mask], loss_rates[mask], 
                  c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    ax.set_xlabel('时延 (ms)', fontsize=12)
    ax.set_ylabel('丢包率', fontsize=12)
    ax.set_title('(b) 时延-丢包率分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    
    # 子图3: 带宽 vs 丢包率
    ax = axes[1, 0]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax.scatter(bandwidths[mask], loss_rates[mask], 
                  c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    ax.set_xlabel('带宽 (Mbps)', fontsize=12)
    ax.set_ylabel('丢包率', fontsize=12)
    ax.set_title('(c) 带宽-丢包率分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10)
    
    # 子图4: 抖动 vs 时延
    ax = axes[1, 1]
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax.scatter(jitters[mask], delays[mask], 
                  c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
    ax.set_xlabel('抖动 (ms)', fontsize=12)
    ax.set_ylabel('时延 (ms)', fontsize=12)
    ax.set_title('(d) 抖动-时延分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    fig5.savefig(os.path.join(output_dir, 'business_features_scatter_combined.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 打印统计信息
    print(f"QoS特征散点图已保存到目录: {output_dir}")
    print(f"样本总数: {len(X)}")
    print(f"业务分布: 控制信令={np.sum(names=='控制信令')}, "
          f"视频回传={np.sum(names=='视频回传')}, "
          f"环境监测={np.sum(names=='环境监测')}")
    print(f"生成的独立图片:")
    print(f"  1. {output_dir}/delay_bandwidth_scatter.png")
    print(f"  2. {output_dir}/delay_loss_scatter.png")
    print(f"  3. {output_dir}/bandwidth_loss_scatter.png")
    print(f"  4. {output_dir}/jitter_delay_scatter.png")
    print(f"  5. {output_dir}/business_features_scatter_combined.png")

if __name__ == '__main__':
    plot_individual_scatters()