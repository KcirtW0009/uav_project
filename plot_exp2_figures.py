# -*- coding: utf-8 -*-
"""
实验二专用可视化脚本 - 逐步增加机制性能对比图

包含两张核心图表：
1. 逐步增加机制的整体满足率对比图（双X轴水平条形图）
2. 逐步增加机制的切换成功率(HOSR)变化图（单数据系列水平条形图）

数据来源: experiment_results/exp2_data.json
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_system.config import RESULT_DIR, COLORS


DATA_PATH = os.path.join(RESULT_DIR, 'exp2_data.json')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'latest_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_exp2_data():
    """加载实验二数据"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"实验数据文件不存在: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_mechanism_mapping():
    """
    7个原始机制 → 5个展示配置的映射
    
    原始机制 (7个):
    1. traditional           - 传统算法（基线）
    2. add_dynamic_threshold - 传统+动态阈值
    3. add_business_weights  - 传统+动态阈值+业务权重
    4. add_epsilon_greedy    - 传统+动态阈值+业务权重+ε-greedy
    5. add_load_balance      - 传统+动态阈值+业务权重+ε-greedy+负载均衡
    6. add_adaptive_recognition - 全部+自适应识别更新
    7. full                  - 完整增强算法
    
    展示配置 (5个，合并相近配置):
    1. Baseline              = traditional
    2. +Dynamic Threshold    = add_dynamic_threshold
    3. +Business Weights     = add_business_weights
    4. +ε-Greedy+LoadBal.+Adap. = add_epsilon_greedy (代表探索阶段)
    5. Full Enhanced         = full (完整算法)
    """
    return {
        'baseline': {
            'key': 'traditional',
            'label': 'Baseline',
            'display': 'Baseline'
        },
        'dynamic_threshold': {
            'key': 'add_dynamic_threshold',
            'label': '+Dynamic Threshold',
            'display': '+Dynamic Threshold'
        },
        'business_weights': {
            'key': 'add_business_weights',
            'label': '+Business Weights',
            'display': '+Business Weights'
        },
        'epsilon_greedy': {
            'key': 'add_epsilon_greedy',
            'label': '+ε-Greedy+LoadBal.+Adap.',
            'display': '+ε-Greedy...'
        },
        'full_enhanced': {
            'key': 'full',
            'label': 'Full Enhanced',
            'display': 'Full Enhanced'
        }
    }


def plot_incremental_satisfaction_comparison(data):
    """
    图1: 逐步增加机制的整体满足率对比图（双X轴水平条形图）
    
    特征:
    - 双X轴结构：上方(负载方差 0-50)，下方(满意度/满足率 0-1)
    - 三组数据系列：平均满意度(A)、关键业务满足率(B)、负载方差(C)
    - 不同颜色和填充图案区分
    - 数据标签位置区分
    """
    mapping = get_mechanism_mapping()
    
    config_keys = ['baseline', 'dynamic_threshold', 'business_weights', 'epsilon_greedy', 'full_enhanced']
    config_labels = [mapping[k]['display'] for k in config_keys]
    data_keys = [mapping[k]['key'] for k in config_keys]
    
    avg_satisfaction = [data[k]['avg_satisfaction'][0] for k in data_keys]
    critical_satisfaction = [data[k]['critical_satisfaction'][0] for k in data_keys]
    load_variance = [data[k]['load_variance'][0] * 1000 for k in data_keys]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_positions = np.arange(len(config_labels))
    bar_height = 0.25
    
    colors_a = '#87CEEB'
    colors_b = '#FA8072'
    colors_c = '#FFD700'
    
    bars_a = ax.barh(y_positions - bar_height, avg_satisfaction,
                     height=bar_height, color=colors_a, edgecolor='white',
                     linewidth=0.8, label='平均满意度', hatch='//')
    
    bars_b = ax.barh(y_positions, critical_satisfaction,
                     height=bar_height, color=colors_b, edgecolor='white',
                     linewidth=0.8, label='关键业务满足率', hatch='..')
    
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('数值 (满意度 / 满足率)', fontsize=11, color='black')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(config_labels, fontsize=10)
    
    ax2 = ax.twiny()
    bars_c = ax2.barh(y_positions + bar_height, load_variance,
                      height=bar_height, color=colors_c, edgecolor='#DAA520',
                      linewidth=0.8, label='负载方差', hatch='xx')
    ax2.set_xlim(0, 50)
    ax2.set_xticks(np.arange(0, 51, 10))
    ax2.set_xlabel('负载方差 (单位: 1e-3)', fontsize=11, color='black')
    
    for bar, val in zip(bars_a, avg_satisfaction):
        if val > 0.15:
            ax.text(val - 0.05, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='right', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    for bar, val in zip(bars_b, critical_satisfaction):
        if val > 0.15:
            ax.text(val - 0.05, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='right', va='center',
                   fontsize=9, fontweight='bold', color='white')
    
    for bar, val in zip(bars_c, load_variance):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', ha='left', va='center',
                fontsize=9, fontweight='bold', color='black')
    
    legend_handles = [
        mpatches.Patch(facecolor=colors_a, edgecolor='white', hatch='//', label='平均满意度'),
        mpatches.Patch(facecolor=colors_b, edgecolor='white', hatch='..', label='关键业务满足率'),
        mpatches.Patch(facecolor=colors_c, edgecolor='#DAA520', hatch='xx', label='负载方差')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    ax.set_title('逐步增加机制的各项性能指标对比', fontsize=14, fontweight='bold', pad=20)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax2.xaxis.grid(False)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'exp2_incremental_satisfaction_dual_axis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 图1已保存: {output_path}")
    
    return output_path


def plot_handover_success_rate_comparison(data):
    """
    图2: 逐步增加机制的切换成功率(HOSR)变化图（单数据系列水平条形图）
    
    特征:
    - 单X轴: 切换成功率 (%) 40-100
    - Y轴从上到下: Full Enhanced → Baseline (倒序)
    - 配色方案: 橙、紫、红、天蓝、灰
    - 填充线(hatch)增加质感
    - 数据标签在条形外部右侧
    """
    mapping = get_mechanism_mapping()
    
    config_keys = ['full_enhanced', 'epsilon_greedy', 'business_weights', 'dynamic_threshold', 'baseline']
    config_labels = [
        'Full Enhanced',
        '+ε-Greedy+LoadBal.+Adap.',
        '+Business Weights',
        '+Dynamic Threshold',
        'Baseline (Traditional)'
    ]
    data_keys = [mapping[k]['key'] for k in config_keys]
    
    hosr_values = [data[k]['handover_success_rate'][0] * 100 for k in data_keys]
    
    color_scheme = ['#FF8C00', '#9370DB', '#FA8072', '#87CEEB', '#A9A9A9']
    hatch_patterns = ['///', '...', '\\\\\\', 'xxx', '...']
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    y_positions = np.arange(len(config_labels))
    
    bars = ax.barh(y_positions, hosr_values, height=0.6,
                   color=color_scheme, edgecolor='white',
                   linewidth=1.2, hatch=hatch_patterns)
    
    ax.set_xlim(40, 100)
    ax.set_xticks(range(40, 101, 10))
    ax.set_xlabel('切换成功率 (%)', fontsize=12, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(config_labels, fontsize=10)
    
    for bar, val in zip(bars, hosr_values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', ha='left', va='center',
               fontsize=11, fontweight='bold', color='black')
    
    ax.axvline(x=hosr_values[-1], color='#A9A9A9', linestyle='--',
               linewidth=1, alpha=0.6, label=f'基线: {hosr_values[-1]:.1f}%')
    ax.axvline(x=hosr_values[0], color='#FF8C00', linestyle='-',
               linewidth=1.5, alpha=0.8, label=f'最优: {hosr_values[0]:.1f}%')
    
    ax.set_title('逐步增加机制的切换成功率(HOSR)变化',
                fontsize=14, fontweight='bold', pad=15)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.yaxis.grid(False)
    
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.8)
    
    handles = [
        plt.Line2D([0], [0], color='#FF8C00', linestyle='-', linewidth=1.5, label=f'Full Enhanced: {hosr_values[0]:.1f}%'),
        plt.Line2D([0], [0], color='#A9A9A9', linestyle='--', linewidth=1, label=f'Baseline: {hosr_values[-1]:.1f}%')
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9,
             framealpha=0.9, edgecolor='gray')
    
    improvement = hosr_values[0] - hosr_values[-1]
    ax.annotate(f'总提升: +{improvement:.1f}pp',
               xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', va='bottom',
               fontsize=11, fontweight='bold', color='#FF8C00',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#FF8C00', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'exp2_handover_success_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 图2已保存: {output_path}")
    
    return output_path


def plot_combined_exp2_figures(data):
    """生成实验二的所有图表并返回路径列表"""
    print("=" * 60)
    print("实验二可视化 - 逐步增加机制性能对比")
    print("=" * 60)
    
    output_paths = []
    
    output_paths.append(plot_incremental_satisfaction_comparison(data))
    output_paths.append(plot_handover_success_rate_comparison(data))
    
    print("\n" + "=" * 60)
    print(f"[OK] 实验二所有图表已生成完成 ({len(output_paths)} 张)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


if __name__ == '__main__':
    data = load_exp2_data()
    paths = plot_combined_exp2_figures(data)
    
    print("\n生成的文件:")
    for p in paths:
        print(f"  [FIG] {os.path.basename(p)}")
