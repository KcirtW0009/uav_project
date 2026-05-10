# -*- coding: utf-8 -*-
"""
实验四专用可视化脚本 - 各场景性能对比图

包含图表：
1. 各场景整体满足率对比图（分组柱状图）
   - X轴: 5个场景（智慧城市、工业巡检、农业植保、应急救援、物流配送）
   - Y轴: 平均满意度 (0.7-1.1)
   - 3个数据系列: 增强算法(天蓝色/斜线)、传统算法(灰色/实心)、MAPPO(橙黄色/网格)

2. 各场景平均SINR对比图（分组柱状图）
   - Y轴: 平均 SINR (dB) 10.0-30.0, 间隔 2.5
   
3. 各场景连接保持率对比图（分组柱状图）
   - Y轴: 连接保持率 (%) 0-100, 间隔 20

数据来源: experiment_results/exp4_data.json
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_system.config import RESULT_DIR


DATA_PATH = os.path.join(RESULT_DIR, 'exp4_data.json')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'latest_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


SCENARIOS = [
    {'key': 'smart_city', 'name': '智慧城市', 'num_uav': 400},
    {'key': 'industrial_inspection', 'name': '工业巡检', 'num_uav': 300},
    {'key': 'agriculture', 'name': '农业植保', 'num_uav': 350},
    {'key': 'emergency_rescue', 'name': '应急救援', 'num_uav': 300},
    {'key': 'logistics_delivery', 'name': '物流配送', 'num_uav': 500},
]

ALGORITHMS = ['enhanced', 'traditional', 'mappo']
ALGORITHM_LABELS = {
    'enhanced': '增强算法',
    'traditional': '传统算法',
    'mappo': 'MAPPO'
}

ALGORITHM_COLORS = {
    'enhanced': '#87CEEB',
    'traditional': '#A9A9A9',
    'mappo': '#FFD700'
}

ALGORITHM_HATCHES = {
    'enhanced': '/',
    'traditional': '',
    'mappo': 'x'
}


def load_exp4_data():
    """加载实验四数据"""
    if not os.path.exists(DATA_PATH):
        print(f"[WARN] 实验四数据文件不存在: {DATA_PATH}")
        print("       使用示例数据进行绘图预览...")
        return generate_sample_data()
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_sample_data():
    """生成示例数据用于预览（当真实数据不可用时）"""
    np.random.seed(42)
    sample = {}
    for scenario in SCENARIOS:
        key = scenario['key']
        sample[key] = {}
        for algo in ALGORITHMS:
            base_sat = np.random.uniform(0.82, 0.95)
            if algo == 'enhanced':
                base_sat += 0.02
            elif algo == 'mappo':
                base_sat += 0.03
            
            base_sinr = np.random.uniform(18, 28)
            if algo == 'mappo':
                base_sinr += 1.5
            elif algo == 'enhanced':
                base_sinr += 0.8
            
            base_conn = np.random.uniform(0.85, 0.97)
            if algo == 'mappo':
                base_conn = min(base_conn + 0.02, 0.99)
            
            sample[key][algo] = {
                'avg_satisfaction': [min(base_sat, 0.99), np.random.uniform(0.02, 0.06)],
                'handover_success_rate': [np.random.uniform(0.75, 0.92), np.random.uniform(0.05, 0.12)],
                'critical_satisfaction': [np.random.uniform(0.97, 1.0), np.random.uniform(0.005, 0.03)],
                'weighted_satisfaction': [base_sat * 0.75, np.random.uniform(0.03, 0.07)],
                'total_load': [np.random.uniform(8000, 12000), np.random.uniform(500, 1500)],
                'load_variance': [np.random.uniform(0.001, 0.015), np.random.uniform(0.0005, 0.008)],
                'avg_sinr': [base_sinr, np.random.uniform(0.8, 2.0)],
                'connected_ratio': [base_conn, np.random.uniform(0.03, 0.08)]
            }
    sample['_meta'] = {'sample': True, 'note': 'This is sample data for preview'}
    return sample


def _create_grouped_bar_chart(data, metric_key, title, ylabel, 
                               ylim, yticks, value_format='{:.3f}',
                               filename=None):
    """
    通用分组柱状图绘制函数
    
    Args:
        data: 实验四数据字典
        metric_key: 数据字段名 (如 'avg_satisfaction', 'avg_sinr', 'connected_ratio')
        title: 图表标题
        ylabel: Y轴标签
        ylim: Y轴范围 (min, max)
        yticks: Y轴刻度列表或元组 (start, stop, step)
        value_format: 数值标签格式字符串
        filename: 输出文件名（可选，默认根据metric_key生成）
    
    Returns:
        output_path: 生成的图片路径
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    
    x_positions = np.arange(len(SCENARIOS))
    bar_width = 0.25
    
    algorithm_data = {algo: [] for algo in ALGORITHMS}
    
    for scenario in SCENARIOS:
        key = scenario['key']
        if key in data:
            for algo in ALGORITHMS:
                if algo in data[key] and metric_key in data[key][algo]:
                    val = data[key][algo][metric_key][0]
                    
                    if metric_key == 'connected_ratio' and val <= 1.0:
                        val = val * 100
                    
                    algorithm_data[algo].append(val)
                else:
                    algorithm_data[algo].append(0)
        else:
            for algo in ALGORITHMS:
                algorithm_data[algo].append(0)
    
    bars_list = []
    for i, algo in enumerate(ALGORITHMS):
        offset = (i - 1) * bar_width
        bars = ax.bar(x_positions + offset,
                     algorithm_data[algo],
                     width=bar_width,
                     color=ALGORITHM_COLORS[algo],
                     edgecolor='white',
                     linewidth=1.2,
                     hatch=ALGORITHM_HATCHES[algo],
                     label=ALGORITHM_LABELS[algo],
                     zorder=3)
        bars_list.append(bars)
        
        y_range = ylim[1] - ylim[0]
        label_offset = y_range * 0.01
        
        for bar, val in zip(bars, algorithm_data[algo]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + label_offset,
                       value_format.format(val),
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       color='#333333')
    
    scenario_labels = [f"{s['name']}\n({s['num_uav']}架)" for s in SCENARIOS]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenario_labels, fontsize=10, ha='center')
    
    ax.set_ylim(ylim)
    if isinstance(yticks, tuple):
        ax.set_yticks(np.arange(yticks[0], yticks[1] + yticks[2]/2, yticks[2]))
    else:
        ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    legend_handles = [
        mpatches.Patch(facecolor=ALGORITHM_COLORS['enhanced'], edgecolor='white',
                      hatch=ALGORITHM_HATCHES['enhanced'], label=ALGORITHM_LABELS['enhanced']),
        mpatches.Patch(facecolor=ALGORITHM_COLORS['traditional'], edgecolor='white',
                      hatch=ALGORITHM_HATCHES['traditional'], label=ALGORITHM_LABELS['traditional']),
        mpatches.Patch(facecolor=ALGORITHM_COLORS['mappo'], edgecolor='#DAA520',
                      hatch=ALGORITHM_HATCHES['mappo'], label=ALGORITHM_LABELS['mappo'])
    ]
    ax.legend(handles=legend_handles, loc='upper right', ncol=3,
             fontsize=10, framealpha=0.9, edgecolor='gray')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
    ax.xaxis.grid(False)
    
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#CCCCCC')
        ax.spines[spine].set_linewidth(0.8)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
    
    is_sample = data.get('_meta', {}).get('sample', False)
    if is_sample:
        ax.text(0.98, 0.02, '[Preview] Sample Data',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=9, style='italic', color='gray',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor='orange', alpha=0.8))
    
    plt.tight_layout()
    
    if filename is None:
        safe_key = metric_key.replace('_', '_')
        filename = f'exp4_{safe_key}_comparison.png'
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")
    
    return output_path


def plot_scenario_satisfaction_comparison(data):
    """
    图1: 各场景整体满足率对比图（分组柱状图）
    
    特征:
    - 分组柱状图，X轴为场景名称（两行显示）
    - Y轴为平均满意度 (0.7-1.1)
    - 3个数据系列: 增强算法、传统算法、MAPPO
    """
    return _create_grouped_bar_chart(
        data=data,
        metric_key='avg_satisfaction',
        title='各场景整体满足率对比',
        ylabel='平均满意度',
        ylim=(0.70, 1.10),
        yticks=(0.7, 1.1, 0.1),
        value_format='{:.3f}',
        filename='exp4_scenario_satisfaction_comparison.png'
    )


def plot_scenario_sinr_comparison(data):
    """
    图2: 各场景平均SINR对比图（分组柱状图）
    
    特征:
    - 分组柱状图，X轴为场景名称（两行显示）
    - Y轴为平均 SINR (dB) 10.0-30.0, 刻度间隔 2.5
    - 3个数据系列: 增强算法、传统算法、MAPPO
    - 配色与图1一致
    """
    return _create_grouped_bar_chart(
        data=data,
        metric_key='avg_sinr',
        title='各场景平均SINR对比',
        ylabel='平均 SINR (dB)',
        ylim=(10.0, 30.0),
        yticks=(10.0, 30.0, 2.5),
        value_format='{:.1f}',
        filename='exp4_scenario_sinr_comparison.png'
    )


def plot_scenario_connected_ratio_comparison(data):
    """
    图3: 各场景连接保持率对比图（分组柱状图）
    
    特征:
    - 分组柱状图，X轴为场景名称（两行显示）
    - Y轴为连接保持率 (%) 0-100, 刻度间隔 20
    - 3个数据系列: 增强算法、传统算法、MAPPO
    - 配色与图1一致
    - 注意: connected_ratio 在数据中可能是0-1范围，需要转换为百分比
    """
    return _create_grouped_bar_chart(
        data=data,
        metric_key='connected_ratio',
        title='各场景连接保持率对比',
        ylabel='连接保持率 (%)',
        ylim=(0, 100),
        yticks=(0, 100, 20),
        value_format='{:.1f}%',
        filename='exp4_scenario_connected_ratio_comparison.png'
    )


def plot_combined_exp4_figures(data):
    """生成实验四的所有图表并返回路径列表"""
    print("=" * 60)
    print("Exp4 Visualization - Scenario Performance Comparison")
    print("=" * 60)
    
    is_sample = data.get('_meta', {}).get('sample', False)
    if is_sample:
        print("[INFO] Using sample data, re-run after exp4 completes")
    
    output_paths = []
    
    print("\n[1/3] Generating satisfaction comparison chart...")
    output_paths.append(plot_scenario_satisfaction_comparison(data))
    
    print("\n[2/3] Generating SINR comparison chart...")
    output_paths.append(plot_scenario_sinr_comparison(data))
    
    print("\n[3/3] Generating connected ratio comparison chart...")
    output_paths.append(plot_scenario_connected_ratio_comparison(data))
    
    print("\n" + "=" * 60)
    print(f"[OK] All Exp4 charts generated ({len(output_paths)} figures)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


if __name__ == '__main__':
    data = load_exp4_data()
    paths = plot_combined_exp4_figures(data)
    
    print("\nGenerated files:")
    for p in paths:
        print(f"  [FIG] {os.path.basename(p)}")
