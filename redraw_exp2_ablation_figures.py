# -*- coding: utf-8 -*-
"""
实验二消融实验图表重绘脚本 v2 —— 与图4-2完全统一的学术风格
生成两张图：
  图4-4: 逐步增加机制的多指标满足度对比（横向分组柱状图 + 双X轴）
  图4-5: 逐步添加机制的切换成功率对比（横向柱状图）

配色/填充/线宽/透明度与 redraw_exp1_figures.py 的图4-2 完全统一。
布局：配置名称(Baseline/+DT等)在纵轴，指标数值在横轴。
无误差棒。负载方差用右X轴解决基线值过大的问题。
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
# 中文字体 + 全局样式（与实验一图4-2完全一致）
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
# 配色方案 —— 与图4-2 完全一致
# ============================================================
COLORS = {
    'primary':   '#2E86AB',   # 蓝 - 平均满意度
    'danger':    '#DC3545',   # 红 - 关键业务满足率
    'purple':    '#9b59b6',   # 紫 - 切换成功率
    'warning':   '#FF8C00',   # 橙 - 负载方差 / 资源匹配度
    'neutral':   '#6C757D',   # 灰色 - 基线
}

METRIC_STYLE = {
    'satisfaction':    {'color': COLORS['primary'], 'hatch': '//'},
    'critical_sat':    {'color': COLORS['danger'],  'hatch': ''},
    'handover_success':{'color': COLORS['purple'],  'hatch': '\\\\'},
    'load_variance':   {'color': COLORS['warning'], 'hatch': 'xx'},
}

BAR_ALPHA = 0.82
BAR_EDGE = 'white'
BAR_LW = 1.0


# ============================================================
# 加载数据
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'experiment_results', 'exp2_data.json'), 'r', encoding='utf-8') as f:
    raw = json.load(f)

config_keys = [
    'traditional',
    'add_dynamic_threshold',
    'add_business_weights',
    'add_epsilon_greedy',
    'full',
]
config_labels = [
    'Baseline\n(Traditional)',
    '+Dynamic\nThreshold',
    '+Business\nWeights',
    '+ε-Greedy+LoadBal.\n+Adap.',
    'Full Enhanced',
]

def extract(k):
    d = raw[k]
    return {
        'sat':     d['avg_satisfaction'][0],
        'crit':    d['critical_satisfaction'][0],
        'lv':      d['load_variance'][0] * 1000,
        'hosr':    d['handover_success_rate'][0] * 100,
    }

data_list = [extract(k) for k in config_keys]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(fig, filename, dpi=300):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {filename}")


# ============================================================
# 图4-4: 多指标对比（横向分组 + 右X轴放负载方差）
# ============================================================
def plot_figure_4():
    """
    横向分组柱状图，三组：
      - 左X轴：平均满意度（蓝 //）、关键业务满足率（红 实心），范围[0, 1.15]
      - 右X轴：负载方差（橙 xx），独立缩放范围[0, 50]，标签用纯文本无上标
    无误差棒。
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    y_idx = np.arange(len(config_labels))
    bar_h = 0.22
    offsets = np.array([-bar_h, 0, bar_h])

    sat_vals  = [d['sat']  for d in data_list]
    crit_vals = [d['crit'] for d in data_list]
    lv_vals   = [d['lv']   for d in data_list]

    s = METRIC_STYLE['satisfaction']
    c = METRIC_STYLE['critical_sat']
    l = METRIC_STYLE['load_variance']

    # ---- 左X轴：满意度 + 关键业务满足率 ----
    bars_sat = ax.barh(y_idx + offsets[0], sat_vals, bar_h,
                        color=s['color'], alpha=BAR_ALPHA,
                        edgecolor=BAR_EDGE, linewidth=BAR_LW,
                        hatch=s['hatch'], label='平均满意度')
    bars_crit = ax.barh(y_idx + offsets[1], crit_vals, bar_h,
                         color=c['color'], alpha=BAR_ALPHA,
                         edgecolor=BAR_EDGE, linewidth=BAR_LW,
                         hatch=c['hatch'], label='关键业务满足率')

    # ---- 右X轴：负载方差（独立缩放，纯文本标签） ----
    ax2 = ax.twiny()
    bars_lv = ax2.barh(y_idx + offsets[2], lv_vals, bar_h,
                        color=l['color'], alpha=BAR_ALPHA,
                        edgecolor=BAR_EDGE, linewidth=BAR_LW,
                        hatch=l['hatch'], label='负载方差')

    # 数据标签
    for i in range(len(config_labels)):
        ax.text(sat_vals[i] + 0.018, i - bar_h, f'{sat_vals[i]:.3f}',
                va='center', ha='left', fontsize=9, fontweight='bold', color=s['color'])
        ax.text(crit_vals[i] + 0.018, i, f'{crit_vals[i]:.3f}',
                va='center', ha='left', fontsize=9, fontweight='bold', color=c['color'])
        ax2.text(lv_vals[i] + 0.8, i + bar_h, f'{lv_vals[i]:.2f}',
                 va='center', ha='left', fontsize=9, fontweight='bold', color=l['color'])

    # ---- 坐标轴设置 ----
    ax.set_yticks(y_idx)
    ax.set_yticklabels(config_labels, fontsize=10.5)
    ax.set_xlabel('数值（满意度 / 满足率）', fontweight='bold', color='#2c3e50')
    ax.set_ylabel('消融配置', fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_title('逐步增加机制的各项性能指标对比', fontweight='bold', pad=12)

    # 右X轴 — 纯文本标签，不使用Unicode上标
    ax2.set_xlabel('负载方差 (单位: 1e-3)', fontweight='bold', color=l['color'])
    ax2.set_xlim(0, 50)
    ax2.tick_params(axis='x', labelcolor=l['color'])

    # 合并图例（左轴两项 + 右轴一项）
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower right',
              framealpha=0.92, fontsize=10.5, ncol=3)

    ax.grid(True, axis='x', alpha=0.2)

    # 基线区域淡灰底色
    ax.axhspan(-0.45, 0.45, facecolor='#f0f0f0', alpha=0.30, zorder=0)

    plt.tight_layout()
    save_fig(fig, 'fig4-4_ablation_multi_metric.png')


# ============================================================
# 图4-5: 切换成功率变化（横向柱状图，无误差棒）
# ============================================================
def plot_figure_5():
    hosr_vals = [d['hosr'] for d in data_list]

    fig, ax = plt.subplots(figsize=(10, 5.2))

    y_idx = np.arange(len(config_labels))
    bar_h = 0.50

    # 渐进式配色：灰(基线) → 蓝 → 红 → 紫 → 橙(Full)
    # 与图4-4的多指标色板统一，每步消融一个新颜色
    bar_colors = [
        COLORS['neutral'],           # Baseline: 灰
        METRIC_STYLE['satisfaction']['color'],   # +DT: 蓝
        METRIC_STYLE['critical_sat']['color'],    # +Weights: 红
        METRIC_STYLE['handover_success']['color'], # +Epsilon: 紫
        METRIC_STYLE['load_variance']['color'],    # Full: 橙
    ]

    bars = ax.barh(y_idx, hosr_vals, bar_h,
                   color=bar_colors, alpha=BAR_ALPHA,
                   edgecolor=BAR_EDGE, linewidth=BAR_LW)

    # 数据标签
    for i, v in enumerate(hosr_vals):
        ax.text(v + 1.2, i, f'{v:.1f}%', va='center', ha='left',
                fontsize=11, fontweight='bold', color='#2c3e50')

    # 基线参考虚线
    bl = hosr_vals[0]
    ax.axvline(x=bl, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.55, zorder=0)

    # ---- 坐标轴 ----
    ax.set_yticks(y_idx)
    ax.set_yticklabels(config_labels, fontsize=10.5)
    ax.set_xlabel('切换成功率 (%)', fontweight='bold')
    ax.set_ylabel('消融配置', fontweight='bold')
    ax.set_title('逐步增加机制的切换成功率(HOSR)变化', fontweight='bold', pad=12)
    ax.set_xlim(40, 102)
    ax.grid(True, axis='x', alpha=0.2)

    # 基线区域底色
    ax.axhspan(-0.42, 0.42, facecolor='#f0f0f0', alpha=0.30, zorder=0)

    plt.tight_layout()
    save_fig(fig, 'fig4-5_ablation_handover_success_rate.png')


# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  实验二消融实验图表 v2 — 统一图4-2风格")
    print("=" * 55)
    plot_figure_4()
    plot_figure_5()
    print("\nDone! => figures/")
