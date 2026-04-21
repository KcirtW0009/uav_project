# -*- coding: utf-8 -*-
"""
Experiment 2/3/4 plotting script.
Reads from experiment_results/*.json, generates clean academic figures.
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

# Font config
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Academic color palette (consistent, distinguishable)
COLORS = {
    'c1': '#2166AC',   # deep blue
    'c2': '#4393C3',   # medium blue
    'c3': '#92C5DE',   # light blue
    'c4': '#D1E5F0',   # very light blue
    'c5': '#FDDBC7',   # light orange
    'c6': '#F4A582',   # orange
    'c7': '#D6604D',   # red-orange
    'c8': '#B2182B',   # dark red
    'baseline': '#7F7F7F',     # gray for baseline
    'enhanced': '#2166AC',      # blue for enhanced
    'mappo': '#E08214',          # orange for MAPPO
    'good': '#1B7837',           # green for positive
    'bad': '#C51B7D',            # red for negative
}

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(exp_name):
    path = os.path.join(RESULT_DIR, f'{exp_name}_data.json')
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_fig(fig, filename, dpi=200):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ============================================================
# Experiment 2: Ablation Study (using exp2b_data.json)
# ============================================================

# Configuration display order and labels (optimized for narrative)
EXP2_CONFIGS = [
    ('traditional',        'Baseline'),
    ('weights',            '+Weights'),
    ('dyn_thresh_weights', '+DT'),
    ('weights_epsilon',    '+Epsilon'),
    ('dyn_thresh_weights_epsilon_lb', 'Full'),
]

# Extended set for full comparison chart
EXP2_ALL_CONFIGS = [
    ('traditional',               'Base'),
    ('dyn_thresh',                'DT'),
    ('weights',                   'W'),
    ('epsilon',                   'E'),
    ('load_balance',              'LB'),
    ('dyn_thresh_weights',        'DT+W'),
    ('dyn_thresh_epsilon',        'DT+E'),
    ('weights_epsilon',           'W+E'),
    ('dyn_thresh_weights_epsilon','DT+W+E'),
    ('dyn_thresh_weights_lb',     'DT+W+LB'),
    ('dyn_thresh_epsilon_lb',     'DT+E+LB'),
    ('weights_epsilon_lb',        'W+E+LB'),
    ('dyn_thresh_weights_epsilon_lb', 'Full'),
    ('full',                      'Full*'),
]

METRIC_META = {
    'avg_satisfaction':         {'label': '满意度',     'fmt': '{:.3f}',  'ylim': [0.85, 1.0]},
    'handover_success_rate':    {'label': '切换成功率',   'fmt': '{:.1f}%',  'scale': 100, 'ylim': [50, 100]},
    'critical_satisfaction':    {'label': '关键业务满足率','fmt': '{:.3f}',  'ylim': [0.97, 1.005]},
    'load_variance':            {'label': '负载方差 (×10⁻³)', 'fmt': '{:.2f}', 'scale': 1000, 'ylim': None},
}


def _get(data, config_key, metric_key):
    """Safe get [mean, std] from data dict."""
    if config_key not in data:
        return [0, 0]
    c = data[config_key]
    if metric_key not in c:
        return [0, 0]
    return c[metric_key]


def plot_exp2_main(data):
    """
    Figure 4-4: Main ablation bar chart (grouped).
    Shows Satisfaction / Critical / LoadVariance across main configs.
    """
    if data is None:
        return

    configs = [k for k, _ in EXP2_CONFIGS if k in data]
    labels = [v for k, v in EXP2_CONFIGS if k in data]
    metrics = ['avg_satisfaction', 'critical_satisfaction', 'load_variance']

    n = len(configs)
    m = len(metrics)
    x = np.arange(n)
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5.5))

    palette = [COLORS['baseline'], COLORS['c1'], COLORS['c2'], COLORS['c6'], COLORS['enhanced']]

    for i, mk in enumerate(metrics):
        meta = METRIC_META[mk]
        vals = [_get(data, c, mk)[0] * meta.get('scale', 1) for c in configs]
        errs = [_get(data, c, mk)[1] * meta.get('scale', 1) for c in configs]
        offset = (i - m // 2) * width
        ax.bar(x + offset, vals, width, yerr=errs,
               label=meta['label'], color=palette[i % len(palette)],
               alpha=0.85, edgecolor='white', linewidth=1, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('消融实验：多指标对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left' if 'avg_satisfaction' in metrics else 'best',
              ncol=m if m <= 3 else 2)
    ax.grid(True, alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4-4_ablation_multi_metric.png')


def plot_exp2_hosr(data):
    """
    Figure 4-5: Handover Success Rate bar chart.
    """
    if data is None:
        return

    configs = [k for k, _ in EXP2_CONFIGS if k in data]
    labels = [v for k, v in EXP2_CONFIGS if k in data]

    vals = [_get(data, c, 'handover_success_rate')[0] * 100 for c in configs]
    errs = [_get(data, c, 'handover_success_rate')[1] * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Color: baseline gray, others blue gradient
    colors = [COLORS['baseline']] + [COLORS['enhanced']] * (len(configs) - 1)

    bars = ax.bar(range(len(configs)), vals, yerr=errs, color=colors,
                  alpha=0.82, edgecolor='white', linewidth=1.2, capsize=4)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('切换成功率 (%)', fontsize=12)
    ax.set_title('消融实验：切换成功率对比', fontsize=13, fontweight='bold')
    ax.set_ylim([min(v - e for v, e in zip(vals, errs)) - 5, 105])
    ax.grid(True, alpha=0.25, axis='y')
    ax.axhline(y=vals[0], color=COLORS['baseline'], linestyle='--', linewidth=0.8, alpha=0.6)

    # Value labels on top of bars
    for bar, val, err in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + 1.2,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4-5_ablation_handover_success_rate.png')


def plot_exp2_full(data):
    """
    Extended figure: All 14 configs x 4 metrics heatmap-style grouped bar.
    Optional appendix figure.
    """
    if data is None:
        return

    configs = [k for k, _ in EXP2_ALL_CONFIGS if k in data]
    labels = [v for k, v in EXP2_ALL_CONFIGS if k in data]

    metrics_to_show = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction']
    n = len(configs)
    m = len(metrics_to_show)
    x = np.arange(n)
    width = 0.22

    fig, ax = plt.subplots(figsize=(16, 6))

    cmap = plt.cm.Blues(np.linspace(0.35, 0.85, n))

    for i, mk in enumerate(metrics_to_show):
        meta = METRIC_META[mk]
        vals = [_get(data, c, mk)[0] * meta.get('scale', 1) for c in configs]
        errs = [_get(data, c, mk)[1] * meta.get('scale', 1) for c in configs]
        offset = (i - m // 2) * width
        ax.bar(x + offset, vals, width, yerr=errs,
               label=meta['label'], alpha=0.8, edgecolor='white', capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('数值', fontsize=11)
    ax.set_title('消融实验：全部配置对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_fig(fig, 'fig4-4b_ablation_full_configs.png')


def plot_exp2_saturation_curve(data):
    """
    Saturation curve: satisfaction vs number of mechanisms added.
    Shows diminishing returns.
    """
    if data is None:
        return

    configs = [k for k, _ in EXP2_CONFIGS if k in data]
    labels = [v for k, v in EXP2_CONFIGS if k in data]

    sats = [_get(data, c, 'avg_satisfaction')[0] for c in configs]
    stds = [_get(data, c, 'avg_satisfaction')[1] for c in configs]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(configs))

    ax.plot(x, sats, 'o-', color=COLORS['enhanced'], linewidth=2.2, markersize=9)
    ax.fill_between(x, [s - sd for s, sd in zip(sats, stds)],
                    [s + sd for s, sd in zip(sats, stds)],
                    alpha=0.15, color=COLORS['enhanced'])

    # Baseline reference line
    ax.axhline(y=sats[0], color=COLORS['baseline'], linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel('配置', fontsize=11)
    ax.set_ylabel('平均满意度', fontsize=12)
    ax.set_title('消融实验：性能随机制复杂度变化', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_ylim([0.86, 0.98])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate values
    for i, (sat, std) in enumerate(zip(sats, stds)):
        ax.annotate(f'{sat:.3f}', xy=(i, sat), xytext=(i, sat + 0.008),
                   ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig4-6_ablation_saturation.png')


# ============================================================
# Experiment 3: Three-Algorithm Comparison
# ============================================================

EXP3_METRICS = {
    'avg_satisfaction':       ('满意度',       False),
    'handover_success_rate':  ('切换成功率',    True),
    'critical_satisfaction':  ('关键业务满足率',  False),
    'connected_ratio':        ('连接率',         True),
    'load_variance':          ('负载方差',       False),
}

ALGO_ORDER = ['traditional', 'mappo', 'enhanced']
ALGO_LABELS = {'traditional': '传统算法', 'enhanced': '增强算法', 'mappo': 'MAPPO'}
ALGO_COLORS = {'traditional': COLORS['baseline'], 'enhanced': COLORS['enhanced'], 'mappo': COLORS['mappo']}


def plot_exp3_bars(data):
    """Experiment 3: one bar chart per metric."""
    if data is None:
        return

    has_mappo = 'mappo in data and any(...)' or ('mappo' in data)

    for mk, (mlabel, is_pct) in EXP3_METRICS.items():
        if mk not in data.get('enhanced', {}):
            continue

        algos = [a for a in ALGO_ORDER if a in data]
        means = [data[a][mk][0] * (100 if is_pct else 1) for a in algos]
        stds  = [data[a][mk][1] * (100 if is_pct else 1) for a in algos]
        colors = [ALGO_COLORS[a] for a in algos]
        labels = [ALGO_LABELS[a] for a in algos]

        fig, ax = plt.subplots(figsize=(6.5, 5))

        bars = ax.bar(range(len(algos)), means, yerr=stds, color=colors,
                      alpha=0.82, edgecolor='white', linewidth=1.3, capsize=5)

        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(mlabel, fontsize=12)
        unit = '%' if is_pct else ''
        ax.set_title(f'实验三：{mlabel}{unit}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + max(stds) * 0.15,
                    f'{m:.2f}{unit}' if not is_pct else f'{m:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        safe_name = mk.replace('_', '')
        save_fig(fig, f'exp3_{safe_name}.png')


def plot_exp3_radar(data):
    """Experiment 3 radar chart."""
    if data is None:
        return

    cats = ['满意度', '切换成功率', '关键业务', '连接率']
    keys = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'connected_ratio']

    def norm(algo, k):
        if algo in data and k in data[algo]:
            v = data[algo][k][0]
            if k == 'load_variance':
                return max(0, 1 - v * 20)  # invert & scale
            return v
        return 0

    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    for algo in ['enhanced', 'mappo', 'traditional']:
        if algo not in data:
            continue
        vals = [norm(algo, k) for k in keys] + [norm(algo, keys[0])]
        ang = angles + [angles[0]]
        label = ALGO_LABELS.get(algo, algo)
        ax.plot(ang, vals, 'o-', linewidth=2.2, label=label,
                color=ALGO_COLORS.get(algo, '#333'), markersize=8)
        ax.fill(ang, vals, alpha=0.12, color=ALGO_COLORS.get(algo, '#333'))

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('实验三：综合性能雷达图', fontsize=13, fontweight='bold', pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05), fontsize=11)

    plt.tight_layout()
    save_fig(fig, 'exp3_radar.png')


# ============================================================
# Experiment 4: Multi-Scenario
# ============================================================

SCENARIOS = {
    'smart_city':           '智慧城市',
    'industrial_inspection':'工业巡检',
    'agriculture':          '农业植保',
    'emergency_rescue':     '应急救援',
    'logistics_delivery':   '物流配送',
}


def _v4(d, sc, alg, k, scale=1, fb=0):
    if sc in d and alg in d[sc] and k in d[sc][alg]:
        return d[sc][alg][k][0] * scale
    return fb


def plot_exp4_by_metric(data, metric_key, metric_label, scale=1):
    """Exp4: grouped bar per metric across scenarios."""
    if data is None:
        return

    scenes = list(SCENARIOS.keys())
    scene_labels = [SCENARIOS[s] for s in scenes]
    has_mappo = any('mappo' in data.get(s, {}) for s in scenes)

    enh = [_v4(data, s, 'enhanced', metric_key, scale) for s in scenes]
    trad = [_v4(data, s, 'traditional', metric_key, scale) for s in scenes]

    x = np.arange(len(scenes))
    w = 0.25 if has_mappo else 0.32

    fig, ax = plt.subplots(figsize=(11, 5.5))
    off = w if has_mappo else w / 2

    ax.bar(x - off, enh, w, label='Enhanced', color=COLORS['enhanced'],
           alpha=0.82, edgecolor='white', linewidth=1.2)
    ax.bar(x + off, trad, w, label='Traditional', color=COLORS['baseline'],
           alpha=0.82, edgecolor='white', linewidth=1.2)

    if has_mappo:
        mp = [_v4(data, s, 'mappo', metric_key, scale) for s in scenes]
        ax.bar(x, mp, w, label='MAPPO', color=COLORS['mappo'],
               alpha=0.82, edgecolor='white', linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels, fontsize=10, rotation=12, ha='right')
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f'实验四：{metric_label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    suffix = '_pct' if scale == 100 else ''
    save_fig(fig, f'exp4_{metric_key}{suffix}.png')


def plot_exp4_all(data):
    if data is None:
        return

    metrics_4 = [
        ('avg_satisfaction', '满意度',       1),
        ('handover_success_rate', '切换成功率',   100),
        ('critical_satisfaction', '关键业务满足率', 1),
        ('load_variance', '负载方差',         1),
    ]
    for k, l, s in metrics_4:
        plot_exp4_by_metric(data, k, l, s)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  Experiment Plotting Script")
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}\n")

    # --- Exp2 (use exp2b data) ---
    exp2 = load_json('exp2b')
    if exp2:
        print("[Exp2] Ablation Study")
        plot_exp2_main(exp2)
        plot_exp2_hosr(exp2)
        plot_exp2_full(exp2)
        plot_exp2_saturation_curve(exp2)

    # --- Exp3 ---
    exp3 = load_json('exp3')
    if exp3:
        print("\n[Exp3] Three-Algo Comparison")
        plot_exp3_bars(exp3)
        plot_exp3_radar(exp3)

    # --- Exp4 ---
    exp4 = load_json('exp4')
    if exp4:
        print("\n[Exp4] Multi-Scenario")
        plot_exp4_all(exp4)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
