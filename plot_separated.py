# -*- coding: utf-8 -*-
"""
分离绘图脚本：将实验2/3/4的每个指标单独绘制成独立图片

从 experiment_results/*.json 数据文件中读取实验结果，
为每个实验的每个指标生成单独的高清图片。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
COLORS = {
    'primary': '#2E86AB',      # 蓝色 - 增强算法
    'success': '#28A745',      # 绿色
    'warning': '#FF8C00',      # 橙色 - MAPPO
    'danger': '#DC3545',       # 红色
    'neutral': '#6C757D',      # 灰色 - 传统算法
}

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results', 'separated_figs')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(exp_name):
    """加载实验JSON数据"""
    path = os.path.join(RESULT_DIR, f'{exp_name}_data.json')
    if not os.path.exists(path):
        print(f"[警告] 文件不存在: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_fig(fig, filename, dpi=200):
    """保存图片到指定目录"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {filename}")


# ============================================================
# 实验2：机制有效性验证（逐步添加）
# ============================================================
def plot_exp2_separated(data):
    """
    实验2分离绘图
    
    配置顺序：traditional, add_dynamic_threshold, add_business_weights,
              add_epsilon_greedy, add_load_balance, add_adaptive_recognition, full
    """
    if data is None:
        return
    
    MECHANISM_NAMES = {
        'traditional': '传统算法',
        'add_dynamic_threshold': '+动态阈值',
        'add_business_weights': '+业务权重',
        'add_epsilon_greedy': '+ε-greedy',
        'add_load_balance': '+负载均衡',
        'add_adaptive_recognition': '+自适应识别',
        'full': '完整增强',
    }
    
    mechanism_order = list(MECHANISM_NAMES.keys())
    mechanisms = [m for m in mechanism_order if m in data]
    names = [MECHANISM_NAMES[m] for m in mechanisms]
    
    # 定义要绑制的指标
    metrics_config = [
        ('avg_satisfaction', '整体满足率', '整体满足率'),
        ('handover_success_rate', '切换成功率', '切换成功率'),
        ('critical_satisfaction', '关键业务满足率', '关键业务满足率'),
        ('weighted_satisfaction', '加权满足率', '加权满足率'),
        ('total_load', '系统总负载', '系统总负载'),
        ('load_variance', '负载方差', '负载方差'),
    ]
    
    print("\n" + "="*60)
    print("实验2：机制有效性验证 - 分离绘图")
    print("="*60)
    
    for key, ylabel, title in metrics_config:
        vals = [data[m][key][0] if key in data[m] else 0 for m in mechanisms]
        errs = [data[m][key][1] if key in data[m] else 0 for m in mechanisms]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 使用渐变色
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mechanisms)))
        
        bars = ax.barh(names, vals, xerr=errs, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=1.5, capsize=3)
        
        ax.set_xlabel(ylabel, fontsize=12)
        ax.set_title(f'实验2：{title}对比（逐步添加机制）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 在柱子末端标注数值
        for bar, val in zip(bars, vals):
            ax.text(val + (max(vals)*0.02), bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_fig(fig, f'exp2_{key}.png')


def plot_exp2_improvement_curve(data):
    """实验2：逐步提升曲线图"""
    if data is None:
        return
    
    MECHANISM_NAMES = {
        'traditional': '传统',
        'add_dynamic_threshold': '+动态阈值',
        'add_business_weights': '+业务权重',
        'add_epsilon_greedy': '+ε-greedy',
        'add_load_balance': '+负载均衡',
        'add_adaptive_recognition': '+自适应识别',
        'full': '完整增强',
    }
    
    mechanism_order = [k for k in MECHANISM_NAMES.keys() if k in data]
    short_names = [MECHANISM_NAMES[k] for k in mechanism_order]
    
    sats = [data[m]['avg_satisfaction'][0] for m in mechanism_order]
    stds = [data[m]['avg_satisfaction'][1] for m in mechanism_order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(mechanism_order))
    
    ax.plot(x, sats, 'o-', color=COLORS['primary'], linewidth=2.5, markersize=10)
    ax.fill_between(x, [s-std for s,std in zip(sats, stds)],
                   [s+std for s,std in zip(sats, stds)],
                   alpha=0.2, color=COLORS['primary'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylabel('整体满足率', fontsize=12)
    ax.set_title('实验2：逐步添加机制的性能提升曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注数值
    for i, (sat, std) in enumerate(zip(sats, stds)):
        ax.annotate(f'{sat:.4f}', xy=(i, sat), xytext=(i, sat + 0.015),
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    # 标注提升
    for i in range(1, len(sats)):
        improvement = sats[i] - sats[i-1]
        if abs(improvement) > 0.005:
            mid_x = (i-1 + i) / 2
            mid_y = (sats[i-1] + sats[i]) / 2
            ax.annotate(f'{improvement:+.4f}', xy=(mid_x, mid_y),
                       xytext=(mid_x, mid_y + improvement*3),
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', lw=0.8, color='red'))
    
    plt.tight_layout()
    save_fig(fig, 'exp2_improvement_curve.png')


def plot_exp2_contribution(data):
    """实验2：各机制的独立贡献"""
    if data is None:
        return
    
    contributions = []
    contrib_names_full = [
        ('traditional', 'add_dynamic_threshold', '动态阈值'),
        ('add_dynamic_threshold', 'add_business_weights', '业务权重'),
        ('add_business_weights', 'add_epsilon_greedy', 'ε-greedy'),
        ('add_epsilon_greedy', 'add_load_balance', '负载均衡'),
        ('add_load_balance', 'add_adaptive_recognition', '自适应识别'),
        ('add_adaptive_recognition', 'full', '完整验证'),
    ]
    
    for prev_key, curr_key, name in contrib_names_full:
        if prev_key in data and curr_key in data:
            prev_sat = data[prev_key]['avg_satisfaction'][0]
            curr_sat = data[curr_key]['avg_satisfaction'][0]
            improvement = curr_sat - prev_sat
            contributions.append((name, improvement))
    
    if not contributions:
        return
    
    names = [c[0] for c in contributions]
    values = [c[1] for c in contributions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in values]
    
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('满足率提升量', fontsize=12)
    ax.set_title('实验2：各机制的独立贡献分析', fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        ypos = val + 0.001 if val >= 0 else val - 0.001
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
               f'{val:+.4f}', ha='center',
               va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'exp2_mechanism_contribution.png')


# ============================================================
# 实验3：增强算法 vs 传统算法（含MAPPO三算法对比）
# ============================================================
def plot_exp3_separated(data):
    """
    实验3分离绘图
    
    包含三种算法：enhanced(增强), traditional(传统), mappo(MAPPO)
    """
    if data is None:
        return
    
    METRICS = {
        'handover_success_rate': ('切换成功率', True),  # 是否需要乘100显示百分比
        'avg_switching_latency_ms': ('平均切换时延(ms)', False),
        'max_switching_latency_ms': ('最大切换时延(ms)', False),
        'avg_decision_time_ms': ('平均决策时间(ms)', False),
        'missed_opportunity_rate': ('错失机会率', False),
        'avg_satisfaction': ('整体满足率', False),
        'critical_satisfaction': ('关键业务满足率', False),
        'weighted_satisfaction': ('加权满足率', False),
        'latency_satisfaction': ('时延满足率', False),
        'rate_satisfaction': ('速率满足率', False),
        'total_throughput': ('系统吞吐量(Mbps)', False),
        'load_variance': ('负载方差', False),
        'avg_sinr': ('平均SINR(dB)', False),
        'recognition_accuracy': ('识别准确率(%)', False),
        'migration_success_rate': ('迁移成功率', False),
        'connected_ratio': ('连接保持率', False),
    }
    
    has_mappo = 'mappo' in data and any(k in data.get('mappo', {}) for k in METRICS.keys())
    
    print("\n" + "="*60)
    print("实验3：三算法全面对比 - 分离绘图" + (" (含MAPPO)" if has_mappo else ""))
    print("="*60)
    
    for key, (ylabel, is_percent) in METRICS.items():
        if key not in data.get('enhanced', {}):
            continue
        
        enh_mean, enh_std = data['enhanced'][key]
        trad_mean, trad_std = data['traditional'][key]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        algo_names = ['增强算法', '传统算法']
        means = [enh_mean, trad_mean]
        stds = [enh_std, trad_std]
        bar_colors = [COLORS['primary'], COLORS['neutral']]
        
        if has_mappo and key in data.get('mappo', {}):
            map_mean, map_std = data['mappo'][key]
            algo_names.insert(1, 'MAPPO')
            means.insert(1, map_mean)
            stds.insert(1, map_std)
            bar_colors.insert(1, COLORS['warning'])
        
        x = np.arange(len(algo_names))
        width = 0.5
        
        bars = ax.bar(x, means, width, yerr=stds, color=bar_colors, alpha=0.85,
                     edgecolor='white', linewidth=1.5, capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'实验3：{ylabel}对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 标注数值
        for bar, mean, std in zip(bars, means, stds):
            display_val = mean * 100 if is_percent else mean
            unit = '%' if is_percent else ''
            ax.text(bar.get_x() + bar.get_width()/2, mean + max(stds)*0.1 + mean*0.01,
                   f'{display_val:.3f}{unit}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        save_fig(fig, f'exp3_{key}.png')


def plot_exp3_radar(data):
    """实验3雷达图"""
    if data is None:
        return
    
    has_mappo = 'mappo' in data and any(k in data.get('mappo', {}) for k in METRICS.keys()) if 'METRICS' in dir() else 'mappo' in data
    
    categories = ['切换成功率', '整体满足率', '关键业务满足率', '吞吐量', '连接保持率']
    metrics_map = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                   'total_throughput', 'connected_ratio']
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
    
    def get_val(algo, m):
        if algo in data and m in data[algo]:
            v = data[algo][m][0]
            if m == 'total_throughput':
                return min(v / 5000, 1.0)  # 归一化
            return v
        return 0
    
    enh_vals = [get_val('enhanced', m) for m in metrics_map]
    trad_vals = [get_val('traditional', m) for m in metrics_map]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    enh_vals_plot = enh_vals + [enh_vals[0]]
    trad_vals_plot = trad_vals + [trad_vals[0]]
    angles_plot = angles + [angles[0]]
    
    ax.plot(angles_plot, enh_vals_plot, 'o-', linewidth=2.5, label='增强算法', color=COLORS['primary'], markersize=8)
    ax.fill(angles_plot, enh_vals_plot, alpha=0.25, color=COLORS['primary'])
    ax.plot(angles_plot, trad_vals_plot, 'o-', linewidth=2.5, label='传统算法', color=COLORS['neutral'], markersize=8)
    ax.fill(angles_plot, trad_vals_plot, alpha=0.15, color=COLORS['neutral'])
    
    if has_mappo:
        mappo_vals = [get_val('mappo', m) for m in metrics_map]
        mappo_vals_plot = mappo_vals + [mappo_vals[0]]
        ax.plot(angles_plot, mappo_vals_plot, 'o-', linewidth=2.5, label='MAPPO', color=COLORS['warning'], markersize=8)
        ax.fill(angles_plot, mappo_vals_plot, alpha=0.15, color=COLORS['warning'])
    
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('实验3：综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=11)
    
    plt.tight_layout()
    save_fig(fig, 'exp3_radar.png')


def plot_exp3_improvement_bar(data):
    """实验3提升百分比横向柱状图"""
    if data is None or 'improvement' not in data:
        return
    
    improvements = [(k, v) for k, v in data['improvement'].items() if abs(v) > 0.1]
    improvements.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if len(improvements) > 12:
        improvements = improvements[:12]
    
    METRICS = {
        'handover_success_rate': '切换成功率',
        'avg_switching_latency_ms': '平均切换时延',
        'max_switching_latency_ms': '最大切换时延',
        'avg_decision_time_ms': '平均决策时间',
        'missed_opportunity_rate': '错失机会率',
        'avg_satisfaction': '整体满足率',
        'critical_satisfaction': '关键业务满足率',
        'weighted_satisfaction': '加权满足率',
        'latency_satisfaction': '时延满足率',
        'rate_satisfaction': '速率满足率',
        'total_throughput': '系统吞吐量',
        'load_variance': '负载方差',
        'avg_sinr': '平均SINR',
        'recognition_accuracy': '识别准确率',
        'migration_success_rate': '迁移成功率',
        'connected_ratio': '连接保持率',
    }
    
    names = [METRICS.get(k, k) for k, _ in improvements]
    values = [v for _, v in improvements]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('提升百分比 (%)', fontsize=12)
    ax.set_title('实验3：关键指标提升对比（增强 vs 传统）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        xpos = val + (max(abs(v) for v in values) * 0.03) if val > 0 else val - (max(abs(v) for v in values) * 0.03)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
               f'{val:+.1f}%', ha='left' if val > 0 else 'right', va='center',
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'exp3_improvement.png')


def plot_exp3_heatmap(data):
    """实验3热力图"""
    if data is None:
        return
    
    has_mappo = 'mappo' in data and any(k in data.get('mappo', {}) for k in ['avg_satisfaction'])
    
    metrics_subset = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                      'latency_satisfaction', 'rate_satisfaction', 'connected_ratio']
    metric_names = ['切换成功率', '整体满足率', '关键业务满足率', '时延满足率', '速率满足率', '连接保持率']
    
    rows_data = []
    row_labels = []
    
    rows_data.append([data['enhanced'].get(m, [0])[0] for m in metrics_subset])
    row_labels.append('增强算法')
    
    if has_mappo:
        rows_data.append([data.get('mappo', {}).get(m, [0])[0] for m in metrics_subset])
        row_labels.append('MAPPO')
    
    rows_data.append([data['traditional'].get(m, [0])[0] for m in metrics_subset])
    row_labels.append('传统算法')
    
    data_array = np.array(rows_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(metrics_subset)))
    ax.set_xticklabels(metric_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title('实验3：性能指标热力图', fontsize=14, fontweight='bold')
    
    for i in range(len(row_labels)):
        for j in range(len(metrics_subset)):
            val = data_array[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color='white' if val < 0.5 else 'black', fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig(fig, 'exp3_heatmap.png')


def plot_exp3_errorbar_comparison(data):
    """实验3误差线对比图"""
    if data is None:
        return
    
    has_mappo = 'mappo' in data and any(k in data.get('mappo', {}) for k in ['avg_satisfaction'])
    
    metrics = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction']
    metric_names = ['整体满足率', '切换成功率', '关键业务满足率']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(metrics))
    
    for i, (m, name) in enumerate(zip(metrics, metric_names)):
        if m not in data.get('enhanced', {}):
            continue
        
        enh_mean, enh_std = data['enhanced'][m]
        trad_mean, trad_std = data['traditional'][m]
        
        offset = 0.18 if has_mappo else 0.15
        
        ax.errorbar(i - offset, enh_mean, yerr=enh_std, fmt='o', color=COLORS['primary'],
                   markersize=12, capsize=6, linewidth=2, label='增强算法' if i == 0 else '')
        
        if has_mappo and m in data.get('mappo', {}):
            map_mean, map_std = data['mappo'][m]
            ax.errorbar(i, map_mean, yerr=map_std, fmt='^', color=COLORS['warning'],
                       markersize=12, capsize=6, linewidth=2, label='MAPPO' if i == 0 else '')
        
        ax.errorbar(i + offset, trad_mean, yerr=trad_std, fmt='s', color=COLORS['neutral'],
                   markersize=12, capsize=6, linewidth=2, label='传统算法' if i == 0 else '')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('实验3：关键指标分布对比（含误差棒）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, 'exp3_errorbar.png')


# ============================================================
# 实验4：多场景对比实验（含MAPPO三算法）
# ============================================================

SCENARIOS_CONFIG = {
    'smart_city': {'name': '智慧城市监控'},
    'industrial_inspection': {'name': '工业巡检'},
    'agriculture': {'name': '农业植保'},
    'emergency_rescue': {'name': '应急救援'},
    'logistics_delivery': {'name': '物流配送'},
}

def _get_val_4(summary, scenario, algo, key, fallback=0, scale=1):
    """安全获取实验4 summary中的值"""
    if scenario in summary and algo in summary[scenario] and key in summary[scenario][algo]:
        return summary[scenario][algo][key][0] * scale
    return fallback


def plot_exp4_metric_by_scenario(data, metric_key, metric_label, scale=1):
    """实验4：按场景绘制某指标的分组柱状图"""
    if data is None:
        return
    
    scenarios = list(SCENARIOS_CONFIG.keys())
    scenario_names = [SCENARIOS_CONFIG[s]['name'] for s in scenarios]
    
    # 检测是否有MAPPO
    has_mappo = any(
        'mappo' in data.get(s, {}) and data[s].get('mappo')
        for s in scenarios
    )
    
    x = np.arange(len(scenarios))
    
    if has_mappo:
        width = 0.25
        enh_vals = [_get_val_4(data, s, 'enhanced', metric_key, scale=scale) for s in scenarios]
        trad_vals = [_get_val_4(data, s, 'traditional', metric_key, scale=scale) for s in scenarios]
        map_vals = [_get_val_4(data, s, 'mappo', metric_key, scale=scale) for s in scenarios]
    else:
        width = 0.35
        enh_vals = [_get_val_4(data, s, 'enhanced', metric_key, scale=scale) for s in scenarios]
        trad_vals = [_get_val_4(data, s, 'traditional', metric_key, scale=scale) for s in scenarios]
        map_vals = None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    offset = width if has_mappo else width / 2
    
    bars_enh = ax.bar(x - offset, enh_vals, width, label='增强算法',
                      color=COLORS['primary'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars_trad = ax.bar(x + offset, trad_vals, width, label='传统算法',
                       color=COLORS['neutral'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    if has_mappo and map_vals:
        bars_map = ax.bar(x, map_vals, width, label='MAPPO',
                          color=COLORS['warning'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=11)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'实验4：各场景{metric_label}对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注数值
    for bar in bars_enh:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(max(enh_vals), max(trad_vals))*0.01,
               f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    suffix = '_percent' if scale == 100 else ''
    save_fig(fig, f'exp4_{metric_key}{suffix}.png')


def plot_exp4_all_metrics(data):
    """实验4全部分离图表"""
    if data is None:
        return
    
    has_mappo = any(
        'mappo' in data.get(s, {}) and data[s].get('mappo')
        for s in SCENARIOS_CONFIG.keys()
    )
    
    print("\n" + "="*60)
    print("实验4：多场景对比 - 分离绘图" + (" (含MAPPO)" if has_mappo else ""))
    print("="*60)
    
    # 各指标单独绘制
    metrics_to_plot = [
        ('avg_satisfaction', '整体满足率', 1),
        ('handover_success_rate', '切换成功率', 100),
        ('critical_satisfaction', '关键业务满足率', 1),
        ('weighted_satisfaction', '加权满足率', 1),
        ('total_load', '系统总负载', 1),
        ('load_variance', '负载方差', 1),
        ('avg_sinr', '平均SINR(dB)', 1),
    ]
    
    for key, label, scale in metrics_to_plot:
        plot_exp4_metric_by_scenario(data, key, label, scale)


def plot_exp4_improvement_by_scenario(data):
    """实验4：各场景提升百分比"""
    if data is None:
        return
    
    scenarios = list(SCENARIOS_CONFIG.keys())
    scenario_names = [SCENARIOS_CONFIG[s]['name'] for s in scenarios]
    
    improvements = []
    for s in scenarios:
        e = _get_val_4(data, s, 'enhanced', 'avg_satisfaction')
        t = _get_val_4(data, s, 'traditional', 'avg_satisfaction')
        improvements.append((e - t) / max(t, 0.001) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['success'] if i > 0 else COLORS['danger'] for i in improvements]
    
    bars = ax.bar(scenario_names, improvements, color=colors, alpha=0.85,
                 edgecolor='white', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('提升百分比 (%)', fontsize=12)
    ax.set_title('实验4：增强算法在各场景的满足率提升', fontsize=14, fontweight='bold')
    ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, val,
               f'{val:+.1f}%', ha='center',
               va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'exp4_improvement_by_scenario.png')


def plot_exp4_heatmap(data):
    """实验4热力图"""
    if data is None:
        return
    
    scenarios = list(SCENARIOS_CONFIG.keys())
    scenario_names = [SCENARIOS_CONFIG[s]['name'] for s in scenarios]
    
    has_mappo = any(
        'mappo' in data.get(s, {}) and data[s].get('mappo')
        for s in scenarios
    )
    
    heat_rows = [
        ('增强-满足率', lambda s: _get_val_4(data, s, 'enhanced', 'avg_satisfaction')),
        ('传统-满足率', lambda s: _get_val_4(data, s, 'traditional', 'avg_satisfaction')),
        ('增强-成功率', lambda s: _get_val_4(data, s, 'enhanced', 'handover_success_rate')),
        ('传统-成功率', lambda s: _get_val_4(data, s, 'traditional', 'handover_success_rate')),
    ]
    
    if has_mappo:
        heat_rows.extend([
            ('MAPPO-满足率', lambda s: _get_val_4(data, s, 'mappo', 'avg_satisfaction')),
            ('MAPPO-成功率', lambda s: _get_val_4(data, s, 'mappo', 'handover_success_rate')),
        ])
    
    data_array = np.array([[fn(s) for s in scenarios] for name, fn in heat_rows])
    row_labels = [name for name, _ in heat_rows]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title('实验4：场景适应性热力图' + (' (含MAPPO)' if has_mappo else ''), fontsize=14, fontweight='bold')
    
    for i in range(len(row_labels)):
        for j in range(len(scenarios)):
            val = data_array[i, j]
            text_str = f'{val*100:.0f}%' if '成功率' in row_labels[i] else f'{val:.2f}'
            ax.text(j, i, text_str, ha='center', va='center',
                   color='white' if val < 0.5 else 'black', fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_fig(fig, 'exp4_heatmap.png')


# ============================================================
# 主函数
# ============================================================
def main():
    print("="*70)
    print("  实验数据分离绑图工具")
    print("  从数据库加载实验2/3/4数据，为每个指标单独生成高清图片")
    print("="*70)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # ===== 实验2 =====
    exp2_data = load_json('exp2')
    if exp2_data:
        plot_exp2_separated(exp2_data)
        plot_exp2_improvement_curve(exp2_data)
        plot_exp2_contribution(exp2_data)
    
    # ===== 实验3 =====
    exp3_data = load_json('exp3')
    if exp3_data:
        plot_exp3_separated(exp3_data)
        plot_exp3_radar(exp3_data)
        plot_exp3_improvement_bar(exp3_data)
        plot_exp3_heatmap(exp3_data)
        plot_exp3_errorbar_comparison(exp3_data)
    
    # ===== 实验4 =====
    exp4_data = load_json('exp4')
    if exp4_data:
        plot_exp4_all_metrics(exp4_data)
        plot_exp4_improvement_by_scenario(exp4_data)
        plot_exp4_heatmap(exp4_data)
    
    print("\n" + "="*70)
    print("  全部绑图完成!")
    print(f"  图片保存位置: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
