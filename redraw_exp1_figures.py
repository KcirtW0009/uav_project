# -*- coding: utf-8 -*-
"""
实验一图片重绘脚本 - 学术简洁版 v3
生成三张独立的高质量图片：
  图4-1: 识别准确率与平均满意度关系曲线（干净版）
  图4-2: 不同业务识别准确率的性能指标对比图（统一学术配色，双Y轴）
  图4-3: 不同准确率下的切换成功率对比图（折线图风格）

配色方案与 plot_separated.py 统一，去除所有结论性标注。
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
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

# ============================================================
# 配色方案 —— 与 plot_separated.py / 后续实验完全统一
# ============================================================
COLORS = {
    'primary':   '#2E86AB',   # 蓝色 - 增强算法 / 主色调
    'success':   '#28A745',   # 绿色
    'warning':   '#FF8C00',   # 橙色 - MAPPO
    'danger':    '#DC3545',   # 红色
    'neutral':   '#6C757D',   # 灰色 - 传统算法 / 辅助
}

# 实验1专用多指标配色（基于统一色板扩展）
METRIC_COLORS = {
    'satisfaction':    COLORS['primary'],     # 蓝 - 平均满意度
    'critical_sat':    COLORS['danger'],      # 红 - 关键业务满足率
    'handover_success':'#9b59b6',             # 紫 - 切换成功率
    'resource_match':  COLORS['warning'],     # 橙 - 资源匹配度(右Y轴)
}


# ============================================================
# 加载实验数据
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, 'experiment_results', 'exp1_data.json'), 'r', encoding='utf-8') as f:
    raw = json.load(f)

configs = ['perfect', 'high', 'medium', 'low', 'random']
labels = ['100%', '89.9%', '80.6%', '60%', '33.7%']
accuracy_values = [100, 89.9, 80.6, 60, 33.7]

data = {}
for cfg in configs:
    d = raw[cfg]
    data[cfg] = {
        'satisfaction': d['satisfaction'],
        'resource_match': d['resource_match'],
        'handover_success': d['handover_success'],
        'critical_sat': d['critical_sat'],
        'actual_accuracy': d['actual_accuracy'],
    }

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename, dpi=300):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {filename}")


def get_gradient_colors(n, start_color, end_color):
    import matplotlib.colors as mcolors
    rgb_s = mcolors.to_rgb(start_color)
    rgb_e = mcolors.to_rgb(end_color)
    return [tuple(rgb_s[j] + (i/max(n-1,1)) * (rgb_e[j]-rgb_s[j]) for j in range(3)) for i in range(n)]


# ============================================================
# 图4-1: 识别准确率与平均满意度关系曲线（干净版）
# ============================================================
def plot_figure_1():
    fig, ax = plt.subplots(figsize=(9, 6))

    means = [data[cfg]['satisfaction'][0] for cfg in configs]
    stds = [data[cfg]['satisfaction'][1] for cfg in configs]

    # 主曲线
    ax.plot(accuracy_values, means, marker='o', markersize=10,
            color=METRIC_COLORS['satisfaction'], linewidth=2.5,
            markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=METRIC_COLORS['satisfaction'])

    # 误差填充带
    ax.fill_between(accuracy_values,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color=METRIC_COLORS['satisfaction'], edgecolor='none',
                    label='均值 ± 标准差')

    # 数据标签
    for x, m in zip(accuracy_values, means):
        ax.annotate(f'{m:.3f}', xy=(x, m), xytext=(0, 12),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#34495e')

    ax.set_xlabel('业务识别准确率 (%)', fontweight='bold')
    ax.set_ylabel('平均满意度', fontweight='bold')
    ax.set_title('业务识别准确率与系统平均满意度的关系', fontweight='bold', pad=12)
    ax.set_xlim(22, 108)
    ax.set_ylim(0.52, 1.05)
    ax.set_xticks(accuracy_values)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True)

    save_fig(fig, 'fig4-1_recognition_satisfaction_curve.png')


# ============================================================
# 图4-2: 性能指标对比（双Y轴，统一学术配色）
# ============================================================
def plot_figure_2():
    fig, ax1 = plt.subplots(figsize=(10.5, 6.5))

    x = np.arange(len(configs))
    width = 0.20

    # ---- 左Y轴：三类指标 ----
    sat_means  = [data[cfg]['satisfaction'][0]      for cfg in configs]
    sat_stds   = [data[cfg]['satisfaction'][1]       for cfg in configs]
    crit_means = [data[cfg]['critical_sat'][0]       for cfg in configs]
    crit_stds  = [data[cfg]['critical_sat'][1]        for cfg in configs]
    hos_means  = [data[cfg]['handover_success'][0]   for cfg in configs]
    hos_stds   = [data[cfg]['handover_success'][1]    for cfg in configs]

    bars1 = ax1.bar(x - 1.5*width, sat_means, width,
                     label='平均满意度', color=METRIC_COLORS['satisfaction'],
                     alpha=0.82, edgecolor='white', linewidth=1.0,
                     hatch='//')
    bars2 = ax1.bar(x - 0.5*width, crit_means, width,
                     label='关键业务满足率', color=METRIC_COLORS['critical_sat'],
                     alpha=0.82, edgecolor='white', linewidth=1.0)
    bars3 = ax1.bar(x + 0.5*width, hos_means, width,
                     label='切换成功率', color=METRIC_COLORS['handover_success'],
                     alpha=0.82, edgecolor='white', linewidth=1.0,
                     hatch='\\\\')

    ax1.set_xlabel('业务识别准确率水平', fontweight='bold')
    ax1.set_ylabel('数值 (满意度 / 成功率)', fontweight='bold', color='#2c3e50')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax1.set_ylim(0, 1.20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10.5)

    # ---- 右Y轴：资源匹配度 ----
    ax2 = ax1.twinx()
    res_means = [data[cfg]['resource_match'][0] for cfg in configs]
    res_stds  = [data[cfg]['resource_match'][1] for cfg in configs]
    bars4 = ax2.bar(x + 1.5*width, res_means, width,
                     label='资源匹配度', color=METRIC_COLORS['resource_match'],
                     alpha=0.82, edgecolor='white', linewidth=1.0,
                     hatch='xx')
    ax2.set_ylabel('资源匹配度', fontweight='bold', color=METRIC_COLORS['resource_match'])
    ax2.tick_params(axis='y', labelcolor=METRIC_COLORS['resource_match'])
    ax2.set_ylim(0, 17)

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper center',
               framealpha=0.92, ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=10)

    ax1.set_title('不同业务识别准确率下各项性能指标对比',
                  fontweight='bold', pad=13)
    ax1.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    save_fig(fig, 'fig4-2_performance_comparison.png')


# ============================================================
# 图4-3: 切换成功率对比 —— 折线图风格（替代误差棒柱状图）
# ============================================================
def plot_figure_3():
    fig, ax = plt.subplots(figsize=(9, 6))

    means = [data[cfg]['handover_success'][0] * 100 for cfg in configs]
    stds  = [data[cfg]['handover_success'][1] * 100 for cfg in configs]

    # 折线 + 数据点（纯均值，不画标准差）
    ax.plot(accuracy_values, means, marker='o', markersize=10,
            color=METRIC_COLORS['handover_success'], linewidth=2.5,
            markerfacecolor='white', markeredgewidth=2.2,
            markeredgecolor=METRIC_COLORS['handover_success'])

    # 数据标签
    for x, m in zip(accuracy_values, means):
        ax.annotate(f'{m:.1f}%', xy=(x, m), xytext=(0, 12),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10.5, fontweight='bold', color='#34495e')

    ax.set_xlabel('业务识别准确率 (%)', fontweight='bold')
    ax.set_ylabel('切换成功率 (%)', fontweight='bold')
    ax.set_title('不同识别准确率下的切换成功率对比', fontweight='bold', pad=12)
    ax.set_xlim(22, 108)
    ax.set_ylim(45, 105)
    ax.set_xticks(accuracy_values)
    ax.grid(True)

    save_fig(fig, 'fig4-3_handover_success_rate.png')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  实验一美化学术图表 v3")
    print("=" * 55)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    print("\nDone! => figures/")
