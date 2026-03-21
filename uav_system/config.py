import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# 全局随机种子
GLOBAL_SEED = 42

def set_global_seed(seed=GLOBAL_SEED):
    """设置所有随机数生成器的种子（安全版本）"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass
    print(f"全局随机种子已设置为：{seed}")

# 结果保存目录
RESULT_DIR = "experiment_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 自定义颜色方案
COLORS = {
    'control': '#FF6B6B',
    'video': '#4ECDC4',
    'environment': '#95E1D3',
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'info': '#60a5fa',
    'neutral': '#9ca3af'
}

def create_gradient_cmap(color1, color2, name='custom'):
    """创建渐变色映射"""
    return LinearSegmentedColormap.from_list(name, [color1, color2])

CMAP_PRIMARY = create_gradient_cmap(COLORS['primary'], COLORS['secondary'], 'primary')
CMAP_SUCCESS = create_gradient_cmap('#86efac', COLORS['success'], 'success')
CMAP_WARNING = create_gradient_cmap('#fcd34d', COLORS['warning'], 'warning')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100