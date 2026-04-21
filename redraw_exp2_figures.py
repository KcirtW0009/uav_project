# -*- coding: utf-8 -*-
"""
实验二消融实验图表重绘脚本 v2（基于exp2b最新数据）
=====================================================

生成两张高质量学术图片：
  图4-4: 分组柱状图展示各配置的多维指标（满意度 / 关键业务 / 负载方差）
  图4-5: 切换成功率独立成图，如实展示各配置的表现

配色方案与 redraw_exp1_figures.py 完全统一。
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
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# ============================================================
# 中文字体 + 全局样式
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

# ============================================================
# 统一配色方案
# ============================================================
COLORS = {
    'primary':    '#2E86AB',   # 蓝 - 平均满意度
    'success':    '#28A745',   # 绿 - 关键业务满足率
    'warning':    '#FF8C00',   # 橙 - MAPPO (备用)
    'danger':     '#DC3545',   # 红
    'neutral':    '#6C757D',   # 灰
    'purple':     '#9b59b6',   # 紫 - 切换成功率
}

METRIC_COLORS = {
    'satisfaction':    COLORS['primary'],   # 蓝 - 平均满意度
    'critical_sat':    COLORS['success'],   # 绿 - 关键业务满足率
    'load_variance':   COLORS['warning'],   # 橙 - 负载方差
    'handover_success': COLORS['purple'],   # 紫 - 切换成功率
}

# ============================================================
# 加载 exp2b 数据
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'experiment_results', 'exp2b_data.json'), 'r', encoding='utf-8') as f:
    raw = json.load(f)

# 消融配置顺序（按机制递进逻辑排列）
CONFIGS = [
    ('traditional',           '传统基线'),
    ('dyn_thresh',            '+动态阈值'),
    ('dyn_thresh_weights',    '+业务权重'),
    ('dyn_thresh_epsilon',    '+ε-greedy'),
    ('dyn_thresh_weights_epsilon_lb', '+负载均衡(完整)'),
]

# 额外显示几个关键配置
EXTRA_CONFIGS = [
    ('weights',              '仅权重'),
    ('weights_epsilon',      '权重+ε'),
    ('full',                 '全部(另一组合)'),
]

ALL_CONFIGS = CONFIGS + EXTRA_CONFIGS

def get_mean_std(config_name, metric):
    """获取某配置某指标的 [mean, std]"""
    if config_name in raw and metric in raw[config_name]:
        return raw[config_name][metric]
    return [0, 0]


OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename, dpi=300):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {filename}")


# ============================================================
# 图4-4: 多维指标分组柱状图
# 展示每个配置的平均满意度、关键业务满足率、负载方差(×100放大)
# ============================================================
def plot_figure_4():
    """
    分组柱状图：每个配置画3根柱子（满意度 / 关键业务 / 负载方差×100）
    
    核心设计思路：
    - 不再画单一递进曲线（因为数据不支持）
    - 用分组柱状图让读者一眼看出每个机制的主攻维度
    - 负载方差数值太小，需要放大100倍才能和其他指标同轴展示
    """
    fig, ax = plt.subplots(figsize=(13, 6.5))

    configs_short = ['传统基线', '+动态阈值', '+业务权重', '+ε-greedy', '+负载均衡\n(完整增强)']
    x = np.arange(len(configs_short))
    
    width = 0.24
    
    # ---- 数据提取 ----
    sat_means  = [get_mean_std(c, 'avg_satisfaction')[0]      for c, _ in CONFIGS]
    sat_stds   = [get_mean_std(c, 'avg_satisfaction')[1]       for c, _ in CONFIGS]
    crit_means = [get_mean_std(c, 'critical_satisfaction')[0]  for c, _ in CONFIGS]
    crit_stds  = [get_mean_std(c, 'critical_satisfaction')[1]   for c, _ in CONFIGS]
    # 负载方差放大100倍以便在同一Y轴上可视化
    lv_means   = [get_mean_std(c, 'load_variance')[0] * 100   for c, _ in CONFIGS]
    lv_stds    = [get_mean_std(c, 'load_variance')[1] * 100    for c, _ in CONFIGS]
    
    # ---- 绘制分组柱状图 ----
    bars1 = ax.bar(x - width, sat_means, width,
                   label='平均满意度', color=METRIC_COLORS['satisfaction'],
                   alpha=0.85, edgecolor='white', linewidth=1.0,
                   hatch='//')
    bars2 = ax.bar(x, crit_means, width,
                   label='关键业务满足率', color=METRIC_COLORS['critical_sat'],
                   alpha=0.85, edgecolor='white', linewidth=1.0)
    bars3 = ax.bar(x + width, lv_means, width,
                   label='负载方差 (x10^{-2})', color=METRIC_COLORS['load_variance'],
                   alpha=0.85, edgecolor='white', linewidth=1.0,
                   hatch='\\\\')
    
    # ---- 误差棒 ----
    ax.errorbar(x - width, sat_means, yerr=sat_stds, fmt='none',
                ecolor='#333', elinewidth=1.2, capsize=3, capthick=1.2)
    ax.errorbar(x, crit_means, yerr=crit_stds, fmt='none',
                ecolor='#333', elinewidth=1.2, capsize=3, capthick=1.2)
    ax.errorbar(x + width, lv_means, yerr=lv_stds, fmt='none',
                ecolor='#333', elinewidth=1.2, capsize=3, capthick=1.2)
    
    # ---- 数据标签（只标关键值）----
    for i, (s, c, l) in enumerate(zip(sat_means, crit_means, lv_means)):
        # 在满意度柱顶标值
        ax.annotate(f'{s:.3f}', xy=(x[i]-width, s),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=METRIC_COLORS['satisfaction'])
        # 关键业务：只有非1.0才标注
        if c < 0.999:
            ax.annotate(f'{c:.3f}', xy=(x[i], c),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=METRIC_COLORS['critical_sat'])
        else:
            ax.annotate('1.000*', xy=(x[i], c),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=METRIC_COLORS['critical_sat'])
        # 负载方差标值
        ax.annotate(f'{l:.2f}', xy=(x[i]+width, l),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=METRIC_COLORS['load_variance'])
    
    # ---- 坐标轴设置 ----
    ax.set_xlabel('消融配置（逐步添加机制）', fontweight='bold')
    ax.set_ylabel('指标值', fontweight='bold')
    ax.set_title('图4-4  各消融配置的多维性能指标对比', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_short, fontsize=9.5)
    ax.set_ylim(0, 1.15)
    
    # Y轴参考线：满意度=1.0, 关键业务=1.0
    ax.axhline(y=1.0, color='#bbb', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # 图例
    ax.legend(loc='upper left', framealpha=0.92, fontsize=10.5)
    ax.grid(True, axis='y', alpha=0.2)
    
    # 底部注释
    ax.text(0.5, -0.12, '* 关键业务满足率达到完美值(1.000)',
            transform=ax.transAxes, ha='center', fontsize=9,
            style='italic', color='#666')
    
    plt.tight_layout()
    save_fig(fig, 'fig4-4_ablation_multi_metric.png')


# ============================================================
# 图4-4b: 扩展版（包含所有16个配置）
# 可选：用于附录或详细讨论
# ============================================================
def plot_figure_4_extended():
    """包含所有配置的完整版本"""
    fig, ax = plt.subplots(figsize=(15, 6.5))
    
    all_keys = list(raw.keys())
    # 过滤掉_meta
    config_names = [k for k in all_keys if k != '_meta']
    
    # 简化标签（使用全局LABEL_MAP）
    labels = [LABEL_MAP.get(k, k[:8]) for k in config_names]
    x = np.arange(len(labels))
    width = 0.22
    
    sat_means = [get_mean_std(k, 'avg_satisfaction')[0] for k in config_names]
    crit_means = [get_mean_std(k, 'critical_satisfaction')[0] for k in config_names]
    lv_means = [get_mean_std(k, 'load_variance')[0] * 100 for k in config_names]
    
    bars1 = ax.bar(x - width, sat_means, width,
                   label='平均满意度', color=METRIC_COLORS['satisfaction'],
                   alpha=0.82, edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x, crit_means, width,
                   label='关键业务满足率', color=METRIC_COLORS['critical_sat'],
                   alpha=0.82, edgecolor='white', linewidth=0.8)
    bars3 = ax.bar(x + width, lv_means, width,
                   label='负载方差(x10^{-2})', color=METRIC_COLORS['load_variance'],
                   alpha=0.82, edgecolor='white', linewidth=0.8, hatch='\\\\')
    
    ax.set_xlabel('消融配置', fontweight='bold')
    ax.set_ylabel('指标值', fontweight='bold')
    ax.set_title('全部16种消融配置的多维指标对比（附录用）', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha='right')
    ax.set_ylim(0, 1.18)
    ax.axhline(y=1.0, color='#bbb', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(loc='upper right', framealpha=0.92, fontsize=10)
    ax.grid(True, axis='y', alpha=0.2)
    
    # 标注最优值
    best_sat_idx = np.argmax(sat_means)
    best_lv_idx = np.argmin(lv_means)
    ax.annotate('SAT最优', xy=(x[best_sat_idx]-width, sat_means[best_sat_idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=8, color=METRIC_COLORS['satisfaction'], fontweight='bold')
    ax.annotate('LV最优', xy=(x[best_lv_idx]+width, lv_means[best_lv_idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=8, color=METRIC_COLORS['load_variance'], fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, 'fig4-4b_ablation_full_configs.png')


# ============================================================
# 图4-5: 切换成功率独立成图
# 如实展示各配置的HOSR表现（含误差棒）
# ============================================================
# ============================================================
# 配置标签映射（全局共享）
# ============================================================
LABEL_MAP = {
    'traditional': '基线',
    'dyn_thresh': 'DT',
    'weights': 'W',
    'epsilon': 'E',
    'load_balance': 'LB',
    'dyn_thresh_weights': 'DT+W',
    'dyn_thresh_epsilon': 'DT+E',
    'weights_epsilon': 'W+E',
    'dyn_thresh_weights_epsilon_lb': 'DT+W+E+LB',
    'dyn_thresh_weights_lb': 'DT+W+LB',
    'dyn_thresh_epsilon_lb': 'DT+E+LB',
    'weights_epsilon_lb': 'W+E+LB',
    'full': 'Full',
}


def plot_figure_5():
    """
    切换成功率柱状图：
    - 如实展示每个配置的HOSR
    - 用不同颜色区分"主递进线"和"其他组合"
    - 突出显示最优值
    """
    fig, ax = plt.subplots(figsize=(12, 6.5))
    
    configs_short = ['传统基线', '+动态阈值', '+业务权重', '+ε-greedy', '+负载均衡\n(完整增强)']
    extra_labels = ['仅权重', '权重+ε', 'Full']
    all_labels = configs_short + extra_labels
    x = np.arange(len(all_labels))
    
    width = 0.65
    
    # 主配置（前5个）用深色，额外配置用浅色
    hosr_means = [get_mean_std(c, 'handover_success_rate')[0] * 100 
                  for c, _ in CONFIGS] + \
                 [get_mean_std(c, 'handover_success_rate')[0] * 100 
                  for c, _ in EXTRA_CONFIGS]
    hosr_stds  = [get_mean_std(c, 'handover_success_rate')[1] * 100 
                  for c, _ in CONFIGS] + \
                 [get_mean_std(c, 'handover_success_rate')[1] * 100 
                  for c, _ in EXTRA_CONFIGS]
    
    colors = ['#2E86AB']*5 + ['#95a5a6']*3
    
    bars = ax.bar(x, hosr_means, width,
                  color=colors, alpha=0.80, edgecolor='white', linewidth=1.2)
    
    # 误差棒
    ax.errorbar(x, hosr_means, yerr=hosr_stds, fmt='none',
                ecolor='#333', elinewidth=1.3, capsize=4, capthick=1.3)
    
    # ---- 数据标签 ----
    for i, (m, s) in enumerate(zip(hosr_means, hosr_stds)):
        # 柱顶标注均值±标准差
        label_text = f'{m:.1f}%'
        y_pos = m + s + 1.5
        
        # 特别标注最优值
        if i == 6:  # weights_epsilon 是最高的
            label_text += ' *'
            ax.annotate(label_text, xy=(x[i], m+s),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color='#DC3545')
            # 加箭头
            ax.annotate('', xy=(x[i], m), xytext=(x[i], m+s+7),
                       arrowprops=dict(arrowstyle='->', color='#DC3545', lw=1.5))
        else:
            ax.annotate(label_text, xy=(x[i], m+s),
                       xytext=(0, 6), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color='#34495e')
        
        # 在柱内部标注标准差（如果柱够高的话）
        if m > 20:
            ax.annotate(f'+-{s:.1f}', xy=(x[i], m/2),
                       ha='center', va='center', fontsize=7.5, color='white', alpha=0.9)
    
    # ---- 坐标轴设置 ----
    ax.set_xlabel('消融配置', fontweight='bold')
    ax.set_ylabel('切换成功率 (%)', fontweight='bold')
    ax.set_title('图4-5  各消融配置的切换成功率对比', fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylim(0, 105)
    
    # 参考线
    ax.axhline(y=80, color='#27ae60', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.text(len(all_labels)-0.5, 81, '80%', fontsize=8.5, color='#27ae60', alpha=0.7)
    
    ax.grid(True, axis='y', alpha=0.2)
    
    # 图例说明颜色含义
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.80, edgecolor='white', label='主递进配置'),
        Patch(facecolor='#95a5a6', alpha=0.80, edgecolor='white', label='其他组合配置'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.92, fontsize=10)
    
    # 底部注释
    ax.text(0.5, -0.10,
            '* weights_epsilon配置达到最高切换成功率(84.1%)。\n'
            '注：动态阈值单独使用时HOSR下降是因为它抑制了非必要切换（提升切换质量而非数量）。',
            transform=ax.transAxes, ha='center', fontsize=8.5,
            style='italic', color='#555', linespacing=1.4)
    
    plt.tight_layout()
    save_fig(fig, 'fig4-5_ablation_handover_success_rate.png')


# ============================================================
# 辅助图：热力图视图（可选，用于直观展示所有配置×所有指标）
# ============================================================
def plot_heatmap():
    """热力图：配置 × 指标，颜色深浅表示性能优劣"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics = [
        ('avg_satisfaction',         '平均满意度',       True),   # 越高越好
        ('handover_success_rate',    '切换成功率',       True),
        ('critical_satisfaction',    '关键业务满足率',    True),
        ('load_variance',            '负载方差(x10^{3})', False),  # 越低越好
    ]
    
    config_keys = [c for c, _ in CONFIGS + EXTRA_CONFIGS if c in raw]
    config_labels = [LABEL_MAP.get(k, k[:10]) for k in config_keys]
    
    # 构建矩阵
    matrix = []
    for metric_key, _, higher_better in metrics:
        row = []
        for ck in config_keys:
            mean_val = get_mean_std(ck, metric_key)[0]
            if metric_key == 'load_variance':
                mean_val *= 1000  # 放大
            row.append(mean_val)
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 对每行归一化到 [0, 1]，方便着色
    normalized = np.zeros_like(matrix)
    for i, (_, _, hb) in enumerate(metrics):
        col = matrix[i, :]
        if hb:
            normalized[i, :] = (col - col.min()) / max(col.max() - col.min(), 1e-10)
        else:
            # 越低越好，反转
            normalized[i, :] = 1 - (col - col.min()) / max(col.max() - col.min(), 1e-10)
    
    im = ax.imshow(normalized, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # 刻度
    ax.set_xticks(np.arange(len(config_labels)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(config_labels, fontsize=9.5, rotation=30, ha='right')
    ax.set_yticklabels([m[1] for m in metrics], fontsize=10)
    
    # 单元格文字（原始值）
    for i in range(len(metrics)):
        for j in range(len(config_labels)):
            val = matrix[i, j]
            if metrics[i][2]:  # 越高越好
                text_color = 'white' if normalized[i, j] < 0.4 else 'black'
            else:
                text_color = 'white' if normalized[i, j] < 0.4 else 'black'
            
            if abs(val) >= 1:
                fmt = '{:.2f}'
            elif abs(val) >= 0.01:
                fmt = '{:.3f}'
            else:
                fmt = '{:.4f}'
            
            text = fmt.format(val)
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=8.5, color=text_color, fontweight='bold')
    
    ax.set_title('消融配置热力图（颜色越绿表示越优）', fontweight='bold', pad=14)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('归一化性能 (绿=优, 红=劣)', fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, 'fig4-heatmap_ablation.png')


if __name__ == '__main__':
    print("=" * 60)
    print("  实验二 消融实验图表 v2 (基于exp2b数据)")
    print("=" * 60)
    
    print("\n[1/4] 图4-4: 多维指标分组柱状图...")
    plot_figure_4()
    
    print("\n[2/4] 图4-4b: 全部配置扩展版...")
    plot_figure_4_extended()
    
    print("\n[3/4] 图4-5: 切换成功率独立图...")
    plot_figure_5()
    
    print("\n[4/4] 热力图...")
    plot_heatmap()
    
    print("\n" + "=" * 60)
    print("  Done! => figures/")
    print("=" * 60)
