#!/usr/bin/env python3
"""
生成三类典型UAV业务的QoS特征散点图 (图2-4) - 最终版

解决所有问题：
1. 字体警告问题：正确配置matplotlib，避免减号字符警告
2. 分布太规律：调整参数，增加随机性和相关性
3. 图片合并问题：生成独立图片，便于论文排版
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings

# 抑制字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# 强制设置matplotlib配置，解决字体问题
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,  # 使用ASCII减号，避免U+2212警告
    'text.usetex': False,  # 禁用LaTeX，避免字体问题
    'axes.formatter.use_mathtext': False,  # 禁用数学文本
    'axes.formatter.useoffset': False,
    'axes.formatter.limits': [-5, 5],
})

# 导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.business import BusinessType, BUSINESS_FEATURE_PARAMS
from uav_system.config import GLOBAL_SEED

def generate_business_samples_final(num_samples=500, seed=GLOBAL_SEED):
    """
    生成三类业务的样本数据 - 最终版
    
    改进策略：
    1. 扩大特征范围，增加多样性
    2. 引入特征之间的合理相关性
    3. 使用混合分布，避免过于规律的分布
    4. 增加异常值，模拟真实网络波动
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
    
    # 调整后的参数（扩大范围，增加多样性）
    enhanced_params = {
        BusinessType.CONTROL_SIGNAL: {
            'delay': (10, 5),           # 扩大标准差：10±5ms
            'bandwidth': (0.5, 0.2),    # 扩大标准差：0.5±0.2Mbps
            'loss_scale': 0.0001,       # 提高一个数量级
            'jitter': (1, 0.8)          # 扩大抖动范围
        },
        BusinessType.VIDEO_STREAMING: {
            'delay': (15, 8),           # 15±8ms
            'bandwidth': (50, 25),      # 50±25Mbps（更广范围）
            'loss_scale': 0.005,        # 提高丢包率
            'jitter': (3, 2)            # 3±2ms
        },
        BusinessType.ENVIRONMENT_MONITORING: {
            'delay': (500, 300),        # 500±300ms
            'bandwidth': (1, 0.5),      # 1±0.5Mbps
            'loss_scale': 0.1,          # 显著提高丢包率
            'jitter': (50, 30)          # 50±30ms
        }
    }
    
    for bt, (name, color) in type_color_map.items():
        base_params = BUSINESS_FEATURE_PARAMS[bt]
        enh_params = enhanced_params[bt]
        
        for i in range(num_samples):
            # 1. 时延生成：使用混合分布（80%正常，20%异常）
            if np.random.random() < 0.8:
                delay = np.random.normal(enh_params['delay'][0], enh_params['delay'][1])
            else:
                # 异常值：更大的时延波动
                delay = np.random.normal(enh_params['delay'][0], enh_params['delay'][1] * 2)
            
            # 2. 带宽生成：与时延负相关
            bandwidth = np.random.normal(enh_params['bandwidth'][0], enh_params['bandwidth'][1])
            # 引入负相关性：带宽越高，时延越低（相关系数-0.6）
            delay = delay * (1 - 0.6 * (bandwidth - enh_params['bandwidth'][0]) / 
                            (enh_params['bandwidth'][1] * 3 + 1))
            
            # 3. 丢包率生成：使用更复杂的分布
            # 基础Beta分布参数
            alpha, beta = base_params['loss_beta']
            # 根据时延和带宽调整丢包率：时延高或带宽低时丢包率更高
            delay_factor = max(0, (delay - enh_params['delay'][0]) / (enh_params['delay'][1] * 2))
            bw_factor = max(0, (enh_params['bandwidth'][0] - bandwidth) / (enh_params['bandwidth'][1] * 2))
            
            # 调整Beta参数，模拟网络条件变化
            adjusted_beta = beta * (1 + delay_factor + bw_factor)
            loss_rate = np.random.beta(alpha, adjusted_beta) * enh_params['loss_scale']
            
            # 10%的概率出现高丢包率事件
            if np.random.random() < 0.1:
                loss_rate *= np.random.uniform(5, 20)
            
            # 4. 抖动生成：与时延正相关
            jitter = np.random.normal(enh_params['jitter'][0], enh_params['jitter'][1])
            # 抖动与时延正相关：时延越高，抖动越大
            jitter = jitter * (1 + 0.4 * (delay - enh_params['delay'][0]) / 
                              (enh_params['delay'][1] * 2 + 1))
            
            # 5. 添加随机噪声（5-15%的随机扰动）
            noise_level = np.random.uniform(0.05, 0.15)
            delay *= (1 + np.random.uniform(-noise_level, noise_level))
            bandwidth *= (1 + np.random.uniform(-noise_level, noise_level))
            jitter *= (1 + np.random.uniform(-noise_level, noise_level))
            
            # 限制范围
            delay = np.clip(delay, 0.1, 2000)
            bandwidth = np.clip(bandwidth, 0.01, 200)
            loss_rate = np.clip(loss_rate, 1e-6, 0.5)
            jitter = np.clip(jitter, 0.1, 200)
            
            samples.append([delay, bandwidth, loss_rate, jitter])
            labels.append(bt.value)
            colors.append(color)
            business_names.append(name)
    
    return np.array(samples), np.array(labels), np.array(colors), np.array(business_names)

def plot_final_scatters(output_dir='figures/business_features_final'):
    """
    绘制最终版的QoS特征散点图
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成样本数据
    X, y, colors, names = generate_business_samples_final(num_samples=400, seed=GLOBAL_SEED)
    
    # 提取特征
    delays = X[:, 0]      # 时延 (ms)
    bandwidths = X[:, 1]  # 带宽 (Mbps)
    loss_rates = X[:, 2]  # 丢包率
    jitters = X[:, 3]     # 抖动 (ms)
    
    # 业务类型和颜色
    unique_names = ['控制信令', '视频回传', '环境监测']
    unique_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    # 打印统计信息
    print("=" * 60)
    print("三类典型UAV业务的QoS特征统计")
    print("=" * 60)
    for name in unique_names:
        mask = names == name
        count = np.sum(mask)
        if count > 0:
            print(f"\n{name} (样本数: {count}):")
            print(f"  时延: {delays[mask].mean():.1f} ± {delays[mask].std():.1f} ms")
            print(f"  带宽: {bandwidths[mask].mean():.1f} ± {bandwidths[mask].std():.1f} Mbps")
            print(f"  丢包率: {loss_rates[mask].mean():.6f} ± {loss_rates[mask].std():.6f}")
            print(f"  抖动: {jitters[mask].mean():.1f} ± {jitters[mask].std():.1f} ms")
    
    print("\n" + "=" * 60)
    print("开始生成散点图...")
    
    # 1. 时延 vs 带宽
    print("生成: 时延-带宽分布图")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax1.scatter(delays[mask], bandwidths[mask], 
                   c=color, s=60, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax1.set_xlabel('时延 (ms)', fontsize=14)
    ax1.set_ylabel('带宽 (Mbps)', fontsize=14)
    ax1.set_title('三类典型UAV业务的时延-带宽分布', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置合理的坐标轴范围
    ax1.set_xlim([0, max(100, delays.max() * 1.1)])
    ax1.set_ylim([0, max(10, bandwidths.max() * 1.1)])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'delay_bandwidth_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. 时延 vs 丢包率
    print("生成: 时延-丢包率分布图")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax2.scatter(delays[mask], loss_rates[mask], 
                   c=color, s=60, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax2.set_xlabel('时延 (ms)', fontsize=14)
    ax2.set_ylabel('丢包率', fontsize=14)
    ax2.set_title('三类典型UAV业务的时延-丢包率分布', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')  # 对数坐标
    ax2.legend(loc='upper right', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置对数坐标的合理范围
    ax2.set_ylim([max(1e-6, loss_rates.min() * 0.5), min(0.5, loss_rates.max() * 2)])
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'delay_loss_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. 带宽 vs 丢包率
    print("生成: 带宽-丢包率分布图")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax3.scatter(bandwidths[mask], loss_rates[mask], 
                   c=color, s=60, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax3.set_xlabel('带宽 (Mbps)', fontsize=14)
    ax3.set_ylabel('丢包率', fontsize=14)
    ax3.set_title('三类典型UAV业务的带宽-丢包率分布', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    ax3.set_ylim([max(1e-6, loss_rates.min() * 0.5), min(0.5, loss_rates.max() * 2)])
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'bandwidth_loss_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. 抖动 vs 时延
    print("生成: 抖动-时延分布图")
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    for name, color in zip(unique_names, unique_colors):
        mask = names == name
        ax4.scatter(jitters[mask], delays[mask], 
                   c=color, s=60, alpha=0.7, edgecolors='white', linewidth=0.8, label=name)
    
    ax4.set_xlabel('抖动 (ms)', fontsize=14)
    ax4.set_ylabel('时延 (ms)', fontsize=14)
    ax4.set_title('三类典型UAV业务的抖动-时延分布', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right', fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'jitter_delay_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. 组合图（可选）
    print("生成: 组合散点图")
    fig5, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig5.suptitle('三类典型UAV业务的QoS特征散点图', fontsize=18, fontweight='bold')
    
    # 子图配置
    subplot_configs = [
        (axes[0, 0], delays, bandwidths, '时延 (ms)', '带宽 (Mbps)', '(a) 时延-带宽分布'),
        (axes[0, 1], delays, loss_rates, '时延 (ms)', '丢包率', '(b) 时延-丢包率分布', 'log'),
        (axes[1, 0], bandwidths, loss_rates, '带宽 (Mbps)', '丢包率', '(c) 带宽-丢包率分布', 'log'),
        (axes[1, 1], jitters, delays, '抖动 (ms)', '时延 (ms)', '(d) 抖动-时延分布')
    ]
    
    for idx, config in enumerate(subplot_configs):
        ax, x_data, y_data, xlabel, ylabel, title = config[:6]
        scale = config[6] if len(config) > 6 else None
        
        for name, color in zip(unique_names, unique_colors):
            mask = names == name
            ax.scatter(x_data[mask], y_data[mask], 
                      c=color, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, label=name)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if scale == 'log':
            ax.set_yscale('log')
        
        if idx == 0:  # 只在第一个子图显示图例
            ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig5.savefig(os.path.join(output_dir, 'business_features_scatter_combined.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. 特征分布直方图（额外）
    print("生成: 特征分布直方图")
    fig6, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig6.suptitle('三类典型UAV业务的QoS特征分布直方图', fontsize=16, fontweight='bold')
    
    features = [delays, bandwidths, loss_rates, jitters]
    feature_names = ['时延 (ms)', '带宽 (Mbps)', '丢包率', '抖动 (ms)']
    feature_titles = ['时延分布', '带宽分布', '丢包率分布', '抖动分布']
    
    for idx, (ax, feat, name, title) in enumerate(zip(axes.flat, features, feature_names, feature_titles)):
        for biz_name, color in zip(unique_names, unique_colors):
            mask = names == biz_name
            if idx == 2:  # 丢包率使用对数坐标
                hist_range = (max(1e-6, feat.min()), min(0.5, feat.max()))
                bins = np.logspace(np.log10(hist_range[0]), np.log10(hist_range[1]), 30)
                ax.hist(feat[mask], bins=bins, alpha=0.6, color=color, label=biz_name, edgecolor='black')
                ax.set_xscale('log')
            else:
                ax.hist(feat[mask], bins=30, alpha=0.6, color=color, label=biz_name, edgecolor='black')
        
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig6.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print("\n" + "=" * 60)
    print("所有图片生成完成！")
    print(f"输出目录: {output_dir}")
    print("\n生成的图片:")
    print(f"  1. {output_dir}/delay_bandwidth_scatter.png")
    print(f"  2. {output_dir}/delay_loss_scatter.png")
    print(f"  3. {output_dir}/bandwidth_loss_scatter.png")
    print(f"  4. {output_dir}/jitter_delay_scatter.png")
    print(f"  5. {output_dir}/business_features_scatter_combined.png")
    print(f"  6. {output_dir}/feature_distributions.png")
    print("\n提示: 前四张独立图片适合论文排版，组合图用于整体展示。")
    print("=" * 60)

if __name__ == '__main__':
    # 检查字体配置
    print("检查字体配置...")
    print(f"axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
    print(f"font.sans-serif: {plt.rcParams['font.sans-serif'][:2]}")
    
    # 生成图片
    plot_final_scatters()