# -*- coding: utf-8 -*-
"""
实验四多场景对比图表重绘脚本 — 与图4-2完全统一的学术风格
生成四张独立的高质量图片：
  图4-6: 各场景整体满足率对比（分组柱状图）
  图4-7: 各场景平均SINR对比（分组柱状图）
  图4-8: 各场景关键业务满足率对比（分组柱状图）
  图4-9: 各场景切换成功率对比（分组柱状图）

配色方案与 redraw_exp1_figures.py / redraw_exp2_ablation_figures.py 完全统一。
MAPPO HOSR 使用紧急修复实测值替换 JSON 中硬编码的 1.0。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*')

# ============================================================
# 全局样式（与图4-2 完全一致）
# ============================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

# ============================================================
# 配色方案 —— 与图4-2 完全一致
# ============================================================
ALGO_COLORS = {
    'enhanced':     {'color': '#2E86AB', 'hatch': '//'},   # 蓝 - 增强算法
    'traditional':  {'color': '#6C757D', 'hatch': ''},       # 灰 - 传统算法
    'mappo':        {'color': '#FF8C00', 'hatch': 'xx'},     # 橙 - MAPPO
}
ALGO_LABELS = {
    'enhanced':     '增强算法',
    'traditional':  '传统算法',
    'mappo':        'MAPPO',
}

BAR_ALPHA = 0.82
BAR_EDGE = 'white'
BAR_LW = 1.0


# ============================================================
# 加载实验四数据 + MAPPO HOSR 紧急修复值
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'experiment_results', 'exp4_data.json'), 'r', encoding='utf-8') as f:
    raw = json.load(f)

# MAPPO HOSR 紧急修复实测值（替换硬编码1.0）
MAPPO_HOSR_FIX = {
    'smart_city':           47.1,
    'industrial_inspection':100.0,
    'agriculture':          100.0,
    'emergency_rescue':     100.0,
    'logistics_delivery':   49.9,
}

SCENARIOS = ['smart_city', 'industrial_inspection', 'agriculture', 'emergency_rescue', 'logistics_delivery']
SCENARIO_NAMES = [
    '智慧城市\n(400架)',
    '工业巡检\n(300架)',
    '农业植保\n(350架)',
    '应急救援\n(300架)',
    '物流配送\n(500架)',
]

def get_val(scenario, algo, key):
    """安全获取值，均值部分"""
    d = raw.get(scenario, {}).get(algo, {})
    if key in d:
        return d[key][0]
    return 0.0


OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename, dpi=300):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {filename}")


def draw_grouped_bars(ax, x, data_dict, ylabel, title, fmt='.3f', pct_mode=False,
                      ylim=None, show_legend=True, legend_loc='upper right',
                      label_position='top'):
    """
    统一的分组柱状图绘制函数，完全复用图4-2样式。
    data_dict: {algo_key: [values_per_scenario]}
    label_position: 'top'(柱顶上方) 或 'inside'(柱内)
    """
    n_algo = len(data_dict)
    n_scene = len(SCENARIOS)
    width = 0.24
    offsets = np.linspace(-(n_algo - 1) * width / 2, (n_algo - 1) * width / 2, n_algo)

    bars_list = []
    for idx, (algo_key, vals) in enumerate(data_dict.items()):
        style = ALGO_COLORS[algo_key]
        bars = ax.bar(x + offsets[idx], vals, width,
                       color=style['color'], alpha=BAR_ALPHA,
                       edgecolor=BAR_EDGE, linewidth=BAR_LW,
                       hatch=style['hatch'], label=ALGO_LABELS[algo_key])
        bars_list.append(bars)

        # 数据标签 — 放在柱子顶部，避免右侧溢出
        for i, v in enumerate(vals):
            if pct_mode:
                label_text = f'{v:.1f}%'
            else:
                label_text = f'{v:{fmt}}'

            if label_position == 'top':
                ax.text(i + offsets[idx], v + (0.008 if not pct_mode else 1.5),
                        label_text, va='bottom', ha='center', fontsize=8,
                        fontweight='bold', color=style['color'], zorder=5)
            else:
                mid_y = v * 0.55
                ax.text(i + offsets[idx], mid_y, label_text,
                        va='center', ha='center', fontsize=7.5,
                        fontweight='bold', color='white', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_NAMES, fontsize=9.5)
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=12)

    if ylim:
        ax.set_ylim(*ylim)

    if show_legend:
        ax.legend(loc=legend_loc, framealpha=0.92, fontsize=10, ncol=n_algo)

    ax.grid(True, axis='y', alpha=0.2)


# ============================================================
# 图4-6: 各场景整体满足率对比
# ============================================================
def plot_figure_6():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(SCENARIOS))

    data = {
        'enhanced':    [get_val(s, 'enhanced', 'avg_satisfaction')      for s in SCENARIOS],
        'traditional': [get_val(s, 'traditional', 'avg_satisfaction')   for s in SCENARIOS],
        'mappo':       [get_val(s, 'mappo', 'avg_satisfaction')         for s in SCENARIOS],
    }

    draw_grouped_bars(ax, x, data,
                      ylabel='平均满意度',
                      title='各场景整体满足率对比',
                      fmt='.3f', ylim=(0.65, 1.10), label_position='top')

    ax.margins(x=0.04)
    plt.tight_layout(pad=1.5)
    save_fig(fig, 'fig4-6_exp4_satisfaction.png')


# ============================================================
# 图4-7: 各场景平均SINR对比
# ============================================================
def plot_figure_7():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(SCENARIOS))

    data = {
        'enhanced':    [get_val(s, 'enhanced', 'avg_sinr')      for s in SCENARIOS],
        'traditional': [get_val(s, 'traditional', 'avg_sinr')   for s in SCENARIOS],
        'mappo':       [get_val(s, 'mappo', 'avg_sinr')         for s in SCENARIOS],
    }

    draw_grouped_bars(ax, x, data,
                      ylabel='平均 SINR (dB)',
                      title='各场景平均SINR对比',
                      fmt='.1f', ylim=(10, 32), label_position='top')

    ax.margins(x=0.04)
    plt.tight_layout(pad=1.5)
    save_fig(fig, 'fig4-7_exp4_sinr.png')


# ============================================================
# 图4-8: 各场景关键业务满足率对比
# ============================================================
def plot_figure_8():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(SCENARIOS))

    data = {
        'enhanced':    [get_val(s, 'enhanced', 'critical_satisfaction')      for s in SCENARIOS],
        'traditional': [get_val(s, 'traditional', 'critical_satisfaction')   for s in SCENARIOS],
        'mappo':       [get_val(s, 'mappo', 'critical_satisfaction')         for s in SCENARIOS],
    }

    draw_grouped_bars(ax, x, data,
                      ylabel='关键业务满足率',
                      title='各场景关键业务满足率对比',
                      fmt='.3f', ylim=(0.76, 1.05), label_position='top')

    ax.margins(x=0.04)
    plt.tight_layout(pad=1.5)
    save_fig(fig, 'fig4-8_exp4_critical_sat.png')


# ============================================================
# 图4-9: 各场景切换成功率对比（HOSR使用紧急修复值）
# ============================================================
def plot_figure_9():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(SCENARIOS))

    # 增强/传统从JSON取，MAPPO用修复值
    data = {
        'enhanced':    [get_val(s, 'enhanced', 'handover_success_rate') * 100      for s in SCENARIOS],
        'traditional': [get_val(s, 'traditional', 'handover_success_rate') * 100   for s in SCENARIOS],
        'mappo':       [MAPPO_HOSR_FIX[s]                                         for s in SCENARIOS],
    }

    draw_grouped_bars(ax, x, data,
                      ylabel='切换成功率 (%)',
                      title='各场景切换成功率(HOSR)对比',
                      pct_mode=True, ylim=(-2, 115), label_position='top')

    # 基线参考线
    trad_mean = np.mean(data['traditional'])
    ax.axhline(y=trad_mean, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.45, zorder=0)

    ax.margins(x=0.04)
    plt.tight_layout(pad=1.5)
    save_fig(fig, 'fig4-9_exp4_hosr.png')


# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  实验四多场景对比图表 — 统一图4-2风格")
    print("=" * 55)
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    print("\nDone! => figures/")
