# -*- coding: utf-8 -*-
"""
实验二/三/四图片重绘脚本 - 大字体版本

保持原图布局、数据与核心视觉元素不变，
将所有文本（坐标轴标签、图例、标题、注释）的字体大小显著放大。
输出目录: latest_figures_large_font/
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_system.config import RESULT_DIR

# ==================== 输出目录 ====================
OUTPUT_DIR = os.path.join(RESULT_DIR, 'latest_figures_large_font')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 全局字体配置 ====================
# 显著放大的字体设置
FONT_TITLE = 22          # 原始: 14-16
FONT_SUBTITLE = 20       # 原始: 14
FONT_AXIS_LABEL = 18     # 原始: 11-13
FONT_TICK = 15           # 原始: 9-11
FONT_LEGEND = 14         # 原始: 9-10
FONT_DATA_LABEL = 12     # 原始: 7.5-8.5 (数据标签)
FONT_ANNOTATION = 12     # 注释文字
FONT_FOOTNOTE = 10       # 脚注

# 全局 rcParams 设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 实验二：逐步增加机制性能对比
# ============================================================================

def load_exp2_data():
    data_path = os.path.join(RESULT_DIR, 'exp2_data.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"实验二数据文件不存在: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_mechanism_mapping():
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


def plot_exp2_incremental_satisfaction(data):
    """实验二 图1: 逐步增加机制的整体满足率对比 - 大字体版"""
    mapping = get_mechanism_mapping()
    
    config_keys = ['baseline', 'dynamic_threshold', 'business_weights', 'epsilon_greedy', 'full_enhanced']
    config_labels = [mapping[k]['display'] for k in config_keys]
    data_keys = [mapping[k]['key'] for k in config_keys]
    
    avg_satisfaction = [data[k]['avg_satisfaction'][0] for k in data_keys]
    critical_satisfaction = [data[k]['critical_satisfaction'][0] for k in data_keys]
    load_variance = [data[k]['load_variance'][0] * 1000 for k in data_keys]
    
    # 放大画布尺寸
    fig, ax = plt.subplots(figsize=(18, 12))
    
    y_positions = np.arange(len(config_labels))
    bar_height = 0.25
    
    colors_a = '#87CEEB'
    colors_b = '#FA8072'
    colors_c = '#FFD700'
    
    bars_a = ax.barh(y_positions - bar_height, avg_satisfaction,
                     height=bar_height, color=colors_a, edgecolor='white',
                     linewidth=1.0, label='平均满意度', hatch='//')
    
    bars_b = ax.barh(y_positions, critical_satisfaction,
                     height=bar_height, color=colors_b, edgecolor='white',
                     linewidth=1.0, label='关键业务满足率', hatch='..')
    
    ax.set_xlim(0, 1.22)
    ax.set_xticks(np.arange(0, 1.15, 0.1))
    ax.set_xlabel('数值 (满意度 / 满足率)', fontsize=FONT_AXIS_LABEL, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(config_labels, fontsize=FONT_TICK + 2)  # Y轴标签再大一点
    
    # 右侧X轴: 负载方差
    ax2 = ax.twiny()
    bars_c = ax2.barh(y_positions + bar_height, load_variance,
                      height=bar_height, color=colors_c, edgecolor='#DAA520',
                      linewidth=1.0, label='负载方差', hatch='xx')
    ax2.set_xlim(0, 58)
    ax2.set_xticks(np.arange(0, 60, 10))
    ax2.set_xlabel(r'负载方差 ($\times 10^{-3}$)', fontsize=FONT_AXIS_LABEL, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=FONT_TICK)
    
    # 数据标签 A: 平均满意度
    for bar, val in zip(bars_a, avg_satisfaction):
        y_center = bar.get_y() + bar.get_height() / 2
        if val >= 0.55:
            ax.text(val - 0.02, y_center, f'{val:.3f}',
                   ha='right', va='center', fontsize=FONT_DATA_LABEL,
                   fontweight='bold', color='#1a1a1a')
        else:
            ax.text(val + 0.02, y_center, f'{val:.3f}',
                   ha='left', va='center', fontsize=FONT_DATA_LABEL,
                   fontweight='bold', color='#1565C0')
    
    # 数据标签 B: 关键业务满足率
    for bar, val in zip(bars_b, critical_satisfaction):
        y_center = bar.get_y() + bar.get_height() / 2
        if val >= 0.55:
            ax.text(val - 0.02, y_center, f'{val:.3f}',
                   ha='right', va='center', fontsize=FONT_DATA_LABEL,
                   fontweight='bold', color='#1a1a1a')
        else:
            ax.text(val + 0.02, y_center, f'{val:.3f}',
                   ha='left', va='center', fontsize=FONT_DATA_LABEL,
                   fontweight='bold', color='#C62828')
    
    # 数据标签 C: 负载方差
    for bar, val in zip(bars_c, load_variance):
        y_center = bar.get_y() + bar.get_height() / 2
        ax2.text(val + 1.2, y_center, f'{val:.1f}',
                ha='left', va='center', fontsize=FONT_DATA_LABEL,
                fontweight='bold', color='#B8860B')
    
    # 图例
    legend_handles = [
        mpatches.Patch(facecolor=colors_a, edgecolor='white', hatch='//', label='平均满意度'),
        mpatches.Patch(facecolor=colors_b, edgecolor='white', hatch='..', label='关键业务满足率'),
        mpatches.Patch(facecolor=colors_c, edgecolor='#DAA520', hatch='xx', label='负载方差 (1e-3)')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=FONT_LEGEND + 2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    
    ax.set_title('逐步增加机制的各项性能指标对比', fontsize=FONT_TITLE, fontweight='bold', pad=25)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax2.xaxis.grid(False)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'exp2_incremental_satisfaction_dual_axis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 实验二图1已保存: {output_path}")


def plot_exp2_handover_success_rate(data):
    """实验二 图2: 逐步增加机制的切换成功率(HOSR)变化 - 大字体版"""
    mapping = get_mechanism_mapping()
    
    config_labels = [
        'Full Enhanced',
        '+ε-Greedy+LoadBal.+Adap.',
        '+Business Weights',
        '+Dynamic Threshold',
        'Baseline (Traditional)'
    ]
    data_keys = ['full', 'add_epsilon_greedy', 'add_business_weights', 
                 'add_dynamic_threshold', 'traditional']
    
    hosr_values = [data[k]['handover_success_rate'][0] * 100 for k in data_keys]
    
    color_scheme = ['#FF8C00', '#9370DB', '#FA8072', '#87CEEB', '#A9A9A9']
    hatch_patterns = ['///', '...', '\\\\\\', 'xxx', '...']
    
    # 放大画布
    fig, ax = plt.subplots(figsize=(16, 10))
    
    y_positions = np.arange(len(config_labels))
    
    bars = ax.barh(y_positions, hosr_values, height=0.6,
                   color=color_scheme, edgecolor='white',
                   linewidth=1.5, hatch=hatch_patterns)
    
    ax.set_xlim(40, 102)
    ax.set_xticks(range(40, 103, 10))
    ax.set_xlabel('切换成功率 (%)', fontsize=FONT_AXIS_LABEL, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(config_labels, fontsize=FONT_TICK + 2)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    
    for bar, val in zip(bars, hosr_values):
        ax.text(val + 1.2, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', ha='left', va='center',
               fontsize=FONT_DATA_LABEL + 2, fontweight='bold', color='black')
    
    ax.axvline(x=hosr_values[-1], color='#A9A9A9', linestyle='--',
               linewidth=1.5, alpha=0.6)
    ax.axvline(x=hosr_values[0], color='#FF8C00', linestyle='-',
               linewidth=2.0, alpha=0.8)
    
    ax.set_title('逐步增加机制的切换成功率 (HOSR) 变化',
                fontsize=FONT_TITLE, fontweight='bold', pad=20)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.yaxis.grid(False)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    handles = [
        plt.Line2D([0], [0], color='#FF8C00', linestyle='-', linewidth=2, 
                  label=f'Full Enhanced: {hosr_values[0]:.1f}%'),
        plt.Line2D([0], [0], color='#A9A9A9', linestyle='--', linewidth=1.5, 
                  label=f'Baseline: {hosr_values[-1]:.1f}%')
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=FONT_LEGEND,
             framealpha=0.9, edgecolor='gray')
    
    improvement = hosr_values[0] - hosr_values[-1]
    ax.annotate(f'总提升: +{improvement:.1f}pp',
               xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', va='bottom',
               fontsize=FONT_ANNOTATION + 2, fontweight='bold', color='#FF8C00',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='#FF8C00', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'exp2_handover_success_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 实验二图2已保存: {output_path}")


# ============================================================================
# 实验三：增强算法 vs 传统算法 vs MAPPO 多维度对比
# ============================================================================

def load_exp3_data():
    data_path = os.path.join(RESULT_DIR, 'exp3_data.json')
    if not os.path.exists(data_path):
        print(f"[WARN] Exp3 data not found: {data_path}, using sample data...")
        return generate_sample_exp3()
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


ALGORITHMS = ['enhanced', 'traditional', 'mappo']
ALGORITHM_LABELS = {'enhanced': 'Enhanced', 'traditional': 'Traditional', 'mappo': 'MAPPO'}
ALGORITHM_COLORS = {'enhanced': '#87CEEB', 'traditional': '#A9A9A9', 'mappo': '#FFD700'}
ALGORITHM_HATCHES = {'enhanced': '/', 'traditional': '', 'mappo': 'x'}

CATEGORY1_METRICS = [
    {'key': 'avg_satisfaction', 'name': 'Avg\nSatisfaction', 'unit': '',
     'is_percentage': False, 'is_reverse': False, 'normalization_range': (0.7, 1.0),
     'value_format': '{:.3f}'},
    {'key': 'critical_satisfaction', 'name': 'Critical Biz\nSatisfaction', 'unit': '',
     'is_percentage': False, 'is_reverse': False, 'normalization_range': (0.85, 1.0),
     'value_format': '{:.3f}'},
    {'key': 'weighted_satisfaction', 'name': 'Weighted\nSatisfaction', 'unit': '',
     'is_percentage': False, 'is_reverse': False, 'normalization_range': (0.5, 0.75),
     'value_format': '{:.3f}'},
    {'key': 'connected_ratio', 'name': 'Connected\nRatio (%)', 'unit': '%',
     'is_percentage': True, 'is_reverse': False, 'normalization_range': (60, 100),
     'value_format': '{:.1f}%'}
]

CATEGORY2_METRICS = [
    {'key': 'handover_success_rate', 'name': 'Handover\nSuccess Rate (%)', 'unit': '%',
     'is_percentage': True, 'is_reverse': False, 'normalization_range': (60, 95),
     'value_format': '{:.1f}%'},
    {'key': 'load_variance', 'name': 'Load Variance\n(x1e-3)', 'unit': 'e-3',
     'is_percentage': False, 'is_reverse': True, 'normalization_range': (0, 0.04),
     'value_format': '{:.4f}'},
    {'key': 'migration_success_rate', 'name': 'Migration\nSuccess Rate (%)', 'unit': '%',
     'is_percentage': True, 'is_reverse': False, 'normalization_range': (0, 100),
     'value_format': '{:.1f}%', 'enhanced_only': True}
]

CATEGORY3_METRICS = [
    {'key': 'avg_sinr', 'name': 'Avg SINR\n(dB)', 'unit': 'dB',
     'is_percentage': False, 'is_reverse': False, 'normalization_range': (15, 25),
     'value_format': '{:.1f}'},
    {'key': 'avg_switching_latency_ms', 'name': 'Switching\nLatency (ms)', 'unit': 'ms',
     'is_percentage': False, 'is_reverse': True, 'normalization_range': (0, 10),
     'value_format': '{:.2f}'},
    {'key': 'avg_decision_time_ms', 'name': 'Decision\nTime (ms)', 'unit': 'ms',
     'is_percentage': False, 'is_reverse': True, 'normalization_range': (0, 0.08),
     'value_format': '{:.4f}'}
]


def normalize_value(value, metric_config, all_values_for_metric=None):
    """归一化函数"""
    if all_values_for_metric is not None and len(all_values_for_metric) > 1:
        actual_min = min(all_values_for_metric)
        actual_max = max(all_values_for_metric)
        actual_range = actual_max - actual_min
        if actual_range < 1e-9:
            return 0.75
        cv = np.std(all_values_for_metric) / (np.mean(all_values_for_metric) + 1e-9)
        if cv < 0.1:
            baseline = 0.85
        elif cv < 0.3:
            baseline = 0.65
        else:
            baseline = 0.45
        display_range = 1.0 - baseline
        if metric_config['is_reverse']:
            norm_val = 1.0 - ((value - actual_min) / actual_range * display_range + baseline - 1.0)
        else:
            norm_val = (value - actual_min) / actual_range * display_range + baseline
        return np.clip(norm_val, 0.15, 1.08)
    else:
        min_val, max_val = metric_config['normalization_range']
        range_val = max_val - min_val
        if range_val == 0:
            return 0.5
        if metric_config['is_reverse']:
            norm_val = 1.0 - (value - min_val) / range_val
        else:
            norm_val = (value - min_val) / range_val
        return np.clip(norm_val, 0, 1.05)


def format_real_value(value, metric_config):
    formatted = metric_config['value_format'].format(value)
    if metric_config['is_percentage'] and not formatted.endswith('%'):
        formatted += '%'
    unit = metric_config.get('unit', '')
    if unit and unit != '%' and not formatted.endswith(unit):
        formatted = f"{formatted} {unit}"
    return formatted


def create_exp3_category_chart_large(data, category_metrics, title, filename):
    """实验三 分组柱状图 - 大字体版"""
    # 放大画布
    fig, ax = plt.subplots(figsize=(18, 10))
    
    num_metrics = len(category_metrics)
    x_positions = np.arange(num_metrics)
    bar_width = 0.26  # 稍微加宽
    
    algorithm_normalized_data = {algo: [] for algo in ALGORITHMS}
    algorithm_raw_data = {algo: [] for algo in ALGORITHMS}
    
    for metric in category_metrics:
        key = metric['key']
        is_enhanced_only = metric.get('enhanced_only', False)
        
        raw_values_for_metric = []
        for algo in ALGORITHMS:
            if algo in data and key in data[algo]:
                raw_value = data[algo][key][0]
                if metric['is_percentage'] and raw_value <= 1.0:
                    raw_value = raw_value * 100
                if is_enhanced_only and algo != 'enhanced':
                    raw_value = 0.0
                raw_values_for_metric.append(raw_value)
            else:
                raw_values_for_metric.append(0)
        
        for algo_idx, algo in enumerate(ALGORITHMS):
            raw_val = raw_values_for_metric[algo_idx]
            algorithm_raw_data[algo].append(raw_val)
            algorithm_normalized_data[algo].append(
                normalize_value(raw_val, metric, all_values_for_metric=raw_values_for_metric)
            )
    
    bars_list = []
    for i, algo in enumerate(ALGORITHMS):
        offset = (i - 1) * bar_width
        bars = ax.bar(x_positions + offset,
                     algorithm_normalized_data[algo],
                     width=bar_width,
                     color=ALGORITHM_COLORS[algo],
                     edgecolor='white',
                     linewidth=1.5,
                     hatch=ALGORITHM_HATCHES[algo],
                     label=ALGORITHM_LABELS[algo],
                     zorder=3)
        bars_list.append(bars)
        
        for bar_idx, (bar, raw_val, metric) in enumerate(zip(bars, algorithm_raw_data[algo], 
                                                               category_metrics)):
            is_enhanced_only = metric.get('enhanced_only', False)
            if raw_val > 0:
                label_text = format_real_value(raw_val, metric)
            elif is_enhanced_only and algo != 'enhanced':
                label_text = 'N/A*'
            else:
                continue
            
            bar_height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar_height + 0.025,
                   label_text,
                   ha='center', va='bottom',
                   fontsize=FONT_DATA_LABEL, fontweight='bold',
                   color='#333333')
    
    # 标注反向指标
    reverse_indices = [i for i, m in enumerate(category_metrics) if m['is_reverse']]
    for idx in reverse_indices:
        ax.text(idx, -0.08,
               '↓ lower is better',
               ha='center', va='top',
               fontsize=FONT_FOOTNOTE + 1, style='italic', color='#D32F2F',
               transform=ax.get_xaxis_transform())
    
    # 配置坐标轴 - 大字体
    metric_names = [m['name'] for m in category_metrics]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, fontsize=FONT_TICK + 2, ha='center')  # X轴标签加大
    
    ax.set_ylim(0.22, 1.15)
    ax.set_yticks(np.arange(0.2, 1.16, 0.2))
    ax.set_ylabel('Normalized Value', fontsize=FONT_AXIS_LABEL, fontweight='bold')
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=22)
    
    # 图例 - 加大
    legend_handles = [
        mpatches.Patch(facecolor=ALGORITHM_COLORS['enhanced'], edgecolor='white',
                      hatch=ALGORITHM_HATCHES['enhanced'], label=ALGORITHM_LABELS['enhanced']),
        mpatches.Patch(facecolor=ALGORITHM_COLORS['traditional'], edgecolor='white',
                      hatch=ALGORITHM_HATCHES['traditional'], label=ALGORITHM_LABELS['traditional']),
        mpatches.Patch(facecolor=ALGORITHM_COLORS['mappo'], edgecolor='#DAA520',
                      hatch=ALGORITHM_HATCHES['mappo'], label=ALGORITHM_LABELS['mappo'])
    ]
    ax.legend(handles=legend_handles, loc='upper right', ncol=3,
             fontsize=FONT_LEGEND + 1, framealpha=0.9, edgecolor='gray')
    
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
        ax.spines[spine].set_linewidth(1.2)
    
    has_enhanced_only = any(m.get('enhanced_only', False) for m in category_metrics)
    if has_enhanced_only:
        ax.text(0.98, -0.12,
               '*N/A: 该指标为增强算法独有（负载均衡迁移机制），传统/MAPPO无此功能',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=FONT_FOOTNOTE, style='italic', color='#666666',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7',
                        edgecolor='#FFB300', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] 已保存: {output_path}")
    return output_path


def plot_exp3_all_charts(data):
    """生成实验三的所有大字体图表"""
    print("\n" + "=" * 60)
    print("实验三可视化 - 大字体版")
    print("=" * 60)
    
    paths = []
    
    print("[1/3] Satisfaction-related Metrics Comparison...")
    paths.append(create_exp3_category_chart_large(
        data, CATEGORY1_METRICS,
        'Satisfaction-related Metrics Comparison',
        'exp3_satisfaction_metrics_comparison.png'))
    
    print("[2/3] Stability-related Metrics Comparison...")
    paths.append(create_exp3_category_chart_large(
        data, CATEGORY2_METRICS,
        'Stability-related Metrics Comparison',
        'exp3_stability_metrics_comparison.png'))
    
    print("[3/3] Performance Efficiency Metrics Comparison...")
    paths.append(create_exp3_category_chart_large(
        data, CATEGORY3_METRICS,
        'Performance Efficiency Metrics Comparison',
        'exp3_performance_metrics_comparison.png'))
    
    print(f"\n[OK] 实验三共生成 {len(paths)} 张大字体图表")
    print(f"    输出目录: {OUTPUT_DIR}\n")
    return paths


# ============================================================================
# 实验四：多场景性能对比
# ============================================================================

def load_exp4_data():
    data_path = os.path.join(RESULT_DIR, 'exp4_data.json')
    if not os.path.exists(data_path):
        print(f"[WARN] Exp4 data not found: {data_path}, using sample data...")
        return generate_sample_exp4()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


SCENARIOS = [
    {'key': 'smart_city', 'name': '智慧城市', 'num_uav': 400},
    {'key': 'industrial_inspection', 'name': '工业巡检', 'num_uav': 300},
    {'key': 'agriculture', 'name': '农业植保', 'num_uav': 350},
    {'key': 'emergency_rescue', 'name': '应急救援', 'num_uav': 300},
    {'key': 'logistics_delivery', 'name': '物流配送', 'num_uav': 500},
]

EXP4_ALGO_LABELS = {'enhanced': '增强算法', 'traditional': '传统算法', 'mappo': 'MAPPO'}
EXP4_ALGO_COLORS = {'enhanced': '#87CEEB', 'traditional': '#A9A9A9', 'mappo': '#FFD700'}
EXP4_ALGO_HATCHES = {'enhanced': '/', 'traditional': '', 'mappo': 'x'}


def create_exp4_grouped_bar_large(data, metric_key, title, ylabel,
                                   ylim, yticks, value_format='{:.3f}',
                                   filename=None, is_percentage=False,
                                   figsize=(18, 11)):
    """实验四 通用分组柱状图 - 大字体版"""
    fig, ax = plt.subplots(figsize=figsize)
    
    x_positions = np.arange(len(SCENARIOS))
    bar_width = 0.26
    
    algorithm_data = {algo: [] for algo in ALGORITHMS}
    
    for scenario in SCENARIOS:
        key = scenario['key']
        if key in data:
            for algo in ALGORITHMS:
                if algo in data[key] and metric_key in data[key][algo]:
                    val = data[key][algo][metric_key][0]
                    if is_percentage and val <= 1.0:
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
                     color=EXP4_ALGO_COLORS[algo],
                     edgecolor='white',
                     linewidth=1.5,
                     hatch=EXP4_ALGO_HATCHES[algo],
                     label=EXP4_ALGO_LABELS[algo],
                     zorder=3)
        bars_list.append(bars)
        
        y_range = ylim[1] - ylim[0]
        label_offset = y_range * 0.012
        
        for bar, val in zip(bars, algorithm_data[algo]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + label_offset,
                       value_format.format(val),
                       ha='center', va='bottom',
                       fontsize=FONT_DATA_LABEL, fontweight='bold',
                       color='#333333')
    
    scenario_labels = [f"{s['name']}\n({s['num_uav']}架)" for s in SCENARIOS]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenario_labels, fontsize=FONT_TICK + 2, ha='center')
    ax.set_ylim(ylim)
    
    if isinstance(yticks, tuple):
        ax.set_yticks(np.arange(yticks[0], yticks[1] + yticks[2]/2, yticks[2]))
    else:
        ax.set_yticks(yticks)
    
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL, fontweight='bold')
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold', pad=22)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    
    legend_handles = [
        Patch(facecolor=EXP4_ALGO_COLORS['enhanced'], edgecolor='white',
              hatch=EXP4_ALGO_HATCHES['enhanced'], label=EXP4_ALGO_LABELS['enhanced']),
        Patch(facecolor=EXP4_ALGO_COLORS['traditional'], edgecolor='white',
              hatch=EXP4_ALGO_HATCHES['traditional'], label=EXP4_ALGO_LABELS['traditional']),
        Patch(facecolor=EXP4_ALGO_COLORS['mappo'], edgecolor='#DAA520',
              hatch=EXP4_ALGO_HATCHES['mappo'], label=EXP4_ALGO_LABELS['mappo'])
    ]
    ax.legend(handles=legend_handles, loc='upper right', ncol=3,
             fontsize=FONT_LEGEND + 1, framealpha=0.9, edgecolor='gray')
    
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
        ax.spines[spine].set_linewidth(1.2)
    
    plt.tight_layout()
    
    if filename is None:
        filename = f'exp4_{metric_key}_comparison.png'
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      [OK] {os.path.basename(output_path)}")
    return output_path


def plot_exp4_all_charts(data):
    """生成实验四的所有大字体图表"""
    print("\n" + "=" * 70)
    print("实验四可视化 - 大字体版")
    print("=" * 70)
    
    paths = []
    
    print("\n[1/6] 各场景整体满足率对比...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='avg_satisfaction',
        title='各场景整体满足率对比', ylabel='平均满足度',
        ylim=(0.62, 1.06), yticks=(0.62, 1.06, 0.10),
        value_format='{:.3f}',
        filename='exp4_satisfaction_comparison.png'))
    
    print("[2/6] 各场景关键业务满足率对比...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='critical_satisfaction',
        title='各场景关键业务满足率对比', ylabel='关键业务满足率',
        ylim=(0.78, 1.02), yticks=(0.78, 1.02, 0.05),
        value_format='{:.3f}',
        filename='exp4_critical_satisfaction.png'))
    
    print("[3/6] 各场景连接保持率对比（系统稳定性）...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='connected_ratio',
        title='各场景连接保持率对比（系统稳定性）', ylabel='连接保持率 (%)',
        ylim=(58, 104), yticks=(58, 104, 10),
        value_format='{:.1f}%',
        filename='exp4_connected_ratio.png',
        is_percentage=True))
    
    print("[4/6] 各场景吞吐量对比...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='total_throughput',
        title='各场景吞吐量对比', ylabel='吞吐量 (Mbps)',
        ylim=(-200, 13500), yticks=(0, 13500, 2000),
        value_format='{:.1f}',
        filename='exp4_throughput_comparison.png',
        figsize=(19, 11)))  # 吞吐量图更大
    
    print("[5/6] 各场景平均SINR对比...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='avg_sinr',
        title='各场景平均SINR对比', ylabel='平均 SINR (dB)',
        ylim=(-1, 32), yticks=(0, 32, 5),
        value_format='{:.1f}',
        filename='exp4_sinr_comparison.png'))
    
    print("[6/6] 各场景负载方差对比（越小越均衡）...")
    paths.append(create_exp4_grouped_bar_large(
        data=data, metric_key='load_variance',
        title='各场景负载方差对比（越小越均衡）', ylabel='负载方差',
        ylim=(-0.005, 0.19), yticks=(0, 0.19, 0.03),
        value_format='{:.4f}',
        filename='exp4_load_variance.png'))
    
    print(f"\n[COMPLETE] 实验四共生成 {len(paths)} 张大字体图表")
    print(f"          输出目录: {OUTPUT_DIR}\n")
    return paths


# ==================== 示例数据生成 ====================

def generate_sample_exp3():
    np.random.seed(42)
    sample = {}
    for algo in ALGORITHMS:
        sample[algo] = {
            'avg_satisfaction': [np.random.uniform(0.80, 0.96), np.random.uniform(0.02, 0.06)],
            'critical_satisfaction': [np.random.uniform(0.92, 1.0), np.random.uniform(0.005, 0.03)],
            'weighted_satisfaction': [np.random.uniform(0.55, 0.70), np.random.uniform(0.03, 0.07)],
            'connected_ratio': [np.random.uniform(0.75, 0.99), np.random.uniform(0.02, 0.08)],
            'handover_success_rate': [np.random.uniform(0.72, 0.90), np.random.uniform(0.05, 0.15)],
            'load_variance': [np.random.uniform(0.001, 0.035), np.random.uniform(0.0005, 0.01)],
            'migration_success_rate': [np.random.uniform(0.30, 0.85), np.random.uniform(0.15, 0.35)],
            'avg_sinr': [np.random.uniform(17, 24), np.random.uniform(0.5, 2.0)],
            'avg_switching_latency_ms': [np.random.uniform(0.5, 8.0), np.random.uniform(0.1, 2.0)],
            'avg_decision_time_ms': [np.random.uniform(0.001, 0.07), np.random.uniform(0.0001, 0.015)],
        }
    sample['_meta'] = {'sample': True}
    return sample


def generate_sample_exp4():
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
                'critical_satisfaction': [np.random.uniform(0.95, 1.0), np.random.uniform(0.005, 0.03)],
                'connected_ratio': [base_conn, np.random.uniform(0.02, 0.05)],
                'total_throughput': [np.random.uniform(3500, 5500), np.random.uniform(300, 800)],
                'avg_sinr': [base_sinr, np.random.uniform(0.8, 2.0)],
                'avg_switching_latency_ms': [np.random.uniform(5, 15), np.random.uniform(1, 3)],
                'load_variance': [np.random.uniform(0.001, 0.01), np.random.uniform(0.0005, 0.003)],
                'handover_success_rate': [np.random.uniform(0.75, 0.95), np.random.uniform(0.05, 0.12)],
            }
    sample['_meta'] = {'sample': True}
    return sample


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("=" * 70)
    print("  实验 二 / 三 / 四 图片重绘 - 大字体版本")
    print("=" * 70)
    print(f"\n  字体设置:")
    print(f"    标题(Title):      {FONT_TITLE}pt")
    print(f"    坐标轴标签:       {FONT_AXIS_LABEL}pt")
    print(f"    刻度标签:         {FONT_TICK}pt")
    print(f"    图例:             {FONT_LEGEND}pt")
    print(f"    数据标签:         {FONT_DATA_LABEL}pt")
    print(f"\n  输出目录: {OUTPUT_DIR}\n")
    
    all_paths = []
    
    # ========== 实验二 ==========
    try:
        exp2_data = load_exp2_data()
        print("\n>>> 实验二：逐步增加机制性能对比 <<<")
        plot_exp2_incremental_satisfaction(exp2_data)
        plot_exp2_handover_success_rate(exp2_data)
        all_paths.extend([
            os.path.join(OUTPUT_DIR, 'exp2_incremental_satisfaction_dual_axis.png'),
            os.path.join(OUTPUT_DIR, 'exp2_handover_success_rate.png')
        ])
    except Exception as e:
        print(f"[ERROR] 实验二绘图失败: {e}")
    
    # ========== 实验三 ==========
    try:
        exp3_data = load_exp3_data()
        print(">>> 实验三：增强算法 vs 传统算法 vs MAPPO <<<")
        paths = plot_exp3_all_charts(exp3_data)
        all_paths.extend(paths)
    except Exception as e:
        print(f"[ERROR] 实验三绘图失败: {e}")
    
    # ========== 实验四 ==========
    try:
        exp4_data = load_exp4_data()
        print(">>> 实验四：多场景性能对比 <<<")
        paths = plot_exp4_all_charts(exp4_data)
        all_paths.extend(paths)
    except Exception as e:
        print(f"[ERROR] 实验四绘图失败: {e}")
    
    print("\n" + "=" * 70)
    print(f"  [完成] 共生成 {len(all_paths)} 张大字体图片")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)
    
    print("\nGenerated files:")
    for p in all_paths:
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"  [OK] {os.path.basename(p):50s} ({size_kb:.1f} KB)")
        else:
            print(f"  [FAIL] {os.path.basename(p):50s} (file not found)")
