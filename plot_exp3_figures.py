# -*- coding: utf-8 -*-
"""
Experiment 3 Visualization Script - Multi-dimensional Performance Comparison

Contains charts:
1. Satisfaction-related Metrics Comparison (Grouped Bar Chart)
   - X-axis: 4 metrics (avg_satisfaction, critical_satisfaction, weighted_satisfaction, connected_ratio)
   - Y-axis: Normalized (0-1) + data labels with real values
   - 3 algorithms: Enhanced(sky blue/), Traditional(gray), MAPPO(orange yellow/x)

2. Stability-related Metrics Comparison (Grouped Bar Chart)
   - X-axis: 3 metrics (handover_success_rate, load_variance, migration_success_rate)
   - Y-axis: Normalized (0-1) + data labels with real values
   - Special: Reverse indicators marked with ↓ annotation

3. Performance Efficiency Metrics Comparison (Grouped Bar Chart)
   - X-axis: 4 metrics (avg_sinr, avg_switching_latency_ms, avg_decision_time_ms, total_throughput)
   - Y-axis: Normalized (0-1) + data labels with real values and units
   - Special: Reverse indicators marked with ↓ annotation

Data source: experiment_results/exp3_data.json
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uav_system.config import RESULT_DIR


DATA_PATH = os.path.join(RESULT_DIR, 'exp3_data.json')
OUTPUT_DIR = os.path.join(RESULT_DIR, 'latest_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


ALGORITHMS = ['enhanced', 'traditional', 'mappo']
ALGORITHM_LABELS = {
    'enhanced': 'Enhanced',
    'traditional': 'Traditional',
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


# ==================== Metric Definitions ====================

CATEGORY1_METRICS = [
    {
        'key': 'avg_satisfaction',
        'name': 'Avg\nSatisfaction',
        'display_name': 'Avg Satisfaction',
        'unit': '',
        'is_percentage': False,
        'is_reverse': False,
        'normalization_range': (0.7, 1.0),
        'value_format': '{:.3f}'
    },
    {
        'key': 'critical_satisfaction',
        'name': 'Critical Biz\nSatisfaction',
        'display_name': 'Critical Biz Sat.',
        'unit': '',
        'is_percentage': False,
        'is_reverse': False,
        'normalization_range': (0.85, 1.0),
        'value_format': '{:.3f}'
    },
    {
        'key': 'weighted_satisfaction',
        'name': 'Weighted\nSatisfaction',
        'display_name': 'Weighted Sat.',
        'unit': '',
        'is_percentage': False,
        'is_reverse': False,
        'normalization_range': (0.5, 0.75),
        'value_format': '{:.3f}'
    },
    {
        'key': 'connected_ratio',
        'name': 'Connected\nRatio (%)',
        'display_name': 'Connected Ratio',
        'unit': '%',
        'is_percentage': True,
        'is_reverse': False,
        'normalization_range': (60, 100),
        'value_format': '{:.1f}%'
    }
]

CATEGORY2_METRICS = [
    {
        'key': 'handover_success_rate',
        'name': 'Handover\nSuccess Rate (%)',
        'display_name': 'HOSR',
        'unit': '%',
        'is_percentage': True,
        'is_reverse': False,
        'normalization_range': (60, 95),
        'value_format': '{:.1f}%'
    },
    {
        'key': 'load_variance',
        'name': 'Load Variance\n(x1e-3)',
        'display_name': 'Load Var.',
        'unit': 'e-3',
        'is_percentage': False,
        'is_reverse': True,
        'normalization_range': (0, 0.04),
        'value_format': '{:.4f}'
    },
    {
        'key': 'migration_success_rate',
        'name': 'Migration\nSuccess Rate (%)',
        'display_name': 'Migration SR',
        'unit': '%',
        'is_percentage': True,
        'is_reverse': False,
        'normalization_range': (0, 100),
        'value_format': '{:.1f}%',
        # 增强算法独有指标：传统/MAPPO无此机制，显示为N/A
        'enhanced_only': True
    }
]

CATEGORY3_METRICS = [
    {
        'key': 'avg_sinr',
        'name': 'Avg SINR\n(dB)',
        'display_name': 'Avg SINR',
        'unit': 'dB',
        'is_percentage': False,
        'is_reverse': False,
        'normalization_range': (15, 25),
        'value_format': '{:.1f}'
    },
    {
        'key': 'avg_switching_latency_ms',
        'name': 'Switching\nLatency (ms)',
        'display_name': 'Switch Lat.',
        'unit': 'ms',
        'is_percentage': False,
        'is_reverse': True,
        'normalization_range': (0, 10),
        'value_format': '{:.2f}'
    },
    {
        'key': 'avg_decision_time_ms',
        'name': 'Decision\nTime (ms)',
        'display_name': 'Decision Time',
        'unit': 'ms',
        'is_percentage': False,
        'is_reverse': True,
        'normalization_range': (0, 0.08),
        'value_format': '{:.4f}'
    },
    {
        'key': 'total_throughput',
        'name': 'Total\nThroughput (Mbps)',
        'display_name': 'Throughput',
        'unit': 'Mbps',
        'is_percentage': False,
        'is_reverse': False,
        'normalization_range': (0, 5000),
        'value_format': '{:.0f}'
    }
]


def load_exp3_data():
    """Load experiment 3 data"""
    if not os.path.exists(DATA_PATH):
        print(f"[WARN] Exp3 data not found: {DATA_PATH}")
        print("       Using sample data for preview...")
        return generate_sample_data()
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_sample_data():
    """Generate sample data for preview when real data unavailable"""
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
            'total_throughput': [np.random.uniform(3000, 5000), np.random.uniform(200, 600)]
        }
    
    sample['_meta'] = {'sample': True, 'note': 'Sample data for preview'}
    return sample


def normalize_value(value, metric_config, all_values_for_metric=None):
    """
    Normalize value to a reasonable display range
    
    Strategy (V2 - Adaptive):
    1. If all_values_for_metric provided: use min-max normalization within actual data range
       → Maps values to [baseline, 1.0] where baseline=0.5~0.7 depending on spread
    2. Otherwise: fall back to fixed range normalization
    
    For reverse indicators, the normalization is inverted.
    """
    if all_values_for_metric is not None and len(all_values_for_metric) > 1:
        # Adaptive normalization based on actual data spread
        actual_min = min(all_values_for_metric)
        actual_max = max(all_values_for_metric)
        actual_range = actual_max - actual_min
        
        if actual_range < 1e-9:
            return 0.75
        
        # Calculate baseline dynamically based on coefficient of variation
        cv = np.std(all_values_for_metric) / (np.mean(all_values_for_metric) + 1e-9)
        
        if cv < 0.1:
            # Very small variation → use tight range [0.85, 1.0] to show subtle differences
            baseline = 0.85
        elif cv < 0.3:
            # Moderate variation → use medium range [0.65, 1.0]
            baseline = 0.65
        else:
            # Large variation → use wider range [0.45, 1.0]
            baseline = 0.45
        
        display_range = 1.0 - baseline
        
        if metric_config['is_reverse']:
            norm_val = 1.0 - ((value - actual_min) / actual_range * display_range + baseline - 1.0)
        else:
            norm_val = (value - actual_min) / actual_range * display_range + baseline
        
        return np.clip(norm_val, 0.15, 1.08)
    
    else:
        # Fallback to fixed range normalization
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
    """Format the real value for display in data label"""
    formatted = metric_config['value_format'].format(value)
    
    if metric_config['is_percentage'] and not formatted.endswith('%'):
        formatted += '%'
    
    unit = metric_config.get('unit', '')
    if unit and unit != '%' and not formatted.endswith(unit):
        formatted = f"{formatted} {unit}"
    
    return formatted


def create_category_chart(data, category_metrics, title, filename):
    """
    Create a grouped bar chart for a specific category of metrics
    
    Args:
        data: Experiment 3 data dictionary
        category_metrics: List of metric configurations for this category
        title: Chart title
        filename: Output filename
    
    Returns:
        output_path: Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    num_metrics = len(category_metrics)
    x_positions = np.arange(num_metrics)
    bar_width = 0.25
    
    # Extract and normalize data for each algorithm
    algorithm_normalized_data = {algo: [] for algo in ALGORITHMS}
    algorithm_raw_data = {algo: [] for algo in ALGORITHMS}
    
    for metric in category_metrics:
        key = metric['key']
        is_enhanced_only = metric.get('enhanced_only', False)
        
        # First pass: collect all raw values for this metric across algorithms
        raw_values_for_metric = []
        for algo in ALGORITHMS:
            if algo in data and key in data[algo]:
                raw_value = data[algo][key][0]
                
                # Convert percentage if needed
                if metric['is_percentage'] and raw_value <= 1.0:
                    raw_value = raw_value * 100
                
                # Enhanced-only指标：非增强算法设为0（无此机制）
                if is_enhanced_only and algo != 'enhanced':
                    raw_value = 0.0
                
                raw_values_for_metric.append(raw_value)
            else:
                raw_values_for_metric.append(0)
        
        # Second pass: normalize using adaptive method with all values
        for algo_idx, algo in enumerate(ALGORITHMS):
            raw_val = raw_values_for_metric[algo_idx]
            
            algorithm_raw_data[algo].append(raw_val)
            algorithm_normalized_data[algo].append(
                normalize_value(raw_val, metric, all_values_for_metric=raw_values_for_metric)
            )
    
    # Draw bars for each algorithm
    bars_list = []
    for i, algo in enumerate(ALGORITHMS):
        offset = (i - 1) * bar_width
        bars = ax.bar(x_positions + offset,
                     algorithm_normalized_data[algo],
                     width=bar_width,
                     color=ALGORITHM_COLORS[algo],
                     edgecolor='white',
                     linewidth=1.2,
                     hatch=ALGORITHM_HATCHES[algo],
                     label=ALGORITHM_LABELS[algo],
                     zorder=3)
        bars_list.append(bars)
        
        # Add data labels with real values
        for bar_idx, (bar, raw_val, metric) in enumerate(zip(bars, 
                                                              algorithm_raw_data[algo], 
                                                              category_metrics)):
            is_enhanced_only = metric.get('enhanced_only', False)
            if raw_val > 0:
                label_text = format_real_value(raw_val, metric)
            elif is_enhanced_only and algo != 'enhanced':
                # Enhanced-only指标：非增强算法显示N/A
                label_text = 'N/A*'
            else:
                continue  # 跳过零值标签
            
            # Position label above bar
            bar_height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar_height + 0.02,
                   label_text,
                   ha='center', va='bottom',
                   fontsize=7.5, fontweight='bold',
                   color='#333333',
                   rotation=0)
    
    # Mark reverse indicators with special annotation
    reverse_indices = [i for i, m in enumerate(category_metrics) if m['is_reverse']]
    for idx in reverse_indices:
        metric = category_metrics[idx]
        x_pos = idx
        
        # Add ↓ symbol below the metric name to indicate "lower is better"
        ax.text(x_pos, -0.08,
               '↓ lower is better',
               ha='center', va='top',
               fontsize=7, style='italic', color='#D32F2F',
               transform=ax.get_xaxis_transform())
    
    # Configure axes
    metric_names = [m['name'] for m in category_metrics]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, fontsize=9, ha='center')
    
    ax.set_ylim(0.25, 1.12)
    ax.set_yticks(np.arange(0.2, 1.13, 0.2))
    ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=18)
    
    # Legend
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
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
    ax.xaxis.grid(False)
    
    # Background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('#CCCCCC')
        ax.spines[spine].set_linewidth(0.8)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
    
    # Sample mode indicator
    is_sample = data.get('_meta', {}).get('sample', False)
    if is_sample:
        ax.text(0.98, 0.02, '[Preview] Sample Data',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=9, style='italic', color='gray',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                        edgecolor='orange', alpha=0.8))
    
    # Enhanced-only指标脚注
    has_enhanced_only = any(m.get('enhanced_only', False) for m in category_metrics)
    if has_enhanced_only:
        ax.text(0.98, -0.12,
               '*N/A: 该指标为增强算法独有（负载均衡迁移机制），传统/MAPPO无此功能',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=7.5, style='italic', color='#666666',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                        edgecolor='#FFB300', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {output_path}")
    
    return output_path


def plot_category1_satisfaction(data):
    """Chart 1: Satisfaction-related Metrics Comparison"""
    return create_category_chart(
        data=data,
        category_metrics=CATEGORY1_METRICS,
        title='Satisfaction-related Metrics Comparison',
        filename='exp3_satisfaction_metrics_comparison.png'
    )


def plot_category2_stability(data):
    """Chart 2: Stability-related Metrics Comparison"""
    return create_category_chart(
        data=data,
        category_metrics=CATEGORY2_METRICS,
        title='Stability-related Metrics Comparison',
        filename='exp3_stability_metrics_comparison.png'
    )


def plot_category3_performance(data):
    """Chart 3: Performance Efficiency Metrics Comparison"""
    return create_category_chart(
        data=data,
        category_metrics=CATEGORY3_METRICS,
        title='Performance Efficiency Metrics Comparison',
        filename='exp3_performance_metrics_comparison.png'
    )


def plot_combined_exp3_figures(data):
    """Generate all experiment 3 charts and return path list"""
    print("=" * 60)
    print("Exp3 Visualization - Multi-dimensional Performance")
    print("=" * 60)
    
    is_sample = data.get('_meta', {}).get('sample', False)
    if is_sample:
        print("[INFO] Using sample data, re-run after exp3 completes")
    
    output_paths = []
    
    print("\n[1/3] Generating satisfaction metrics chart...")
    output_paths.append(plot_category1_satisfaction(data))
    
    print("\n[2/3] Generating stability metrics chart...")
    output_paths.append(plot_category2_stability(data))
    
    print("\n[3/3] Generating performance efficiency chart...")
    output_paths.append(plot_category3_performance(data))
    
    print("\n" + "=" * 60)
    print(f"[OK] All Exp3 charts generated ({len(output_paths)} figures)")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    
    return output_paths


if __name__ == '__main__':
    data = load_exp3_data()
    paths = plot_combined_exp3_figures(data)
    
    print("\nGenerated files:")
    for p in paths:
        print(f"  [FIG] {os.path.basename(p)}")
