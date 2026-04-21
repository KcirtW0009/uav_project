#!/usr/bin/env python3
"""
生成无人机网络拓扑结构示意图
对应第二章图2-1

参数来源: uav_system/environment.py (3GPP标准)
- 宏基站(Macro): 高度25m, 覆盖半径500m
- 小基站(Small/微): 高度8m, 覆盖半径200m
- 默认场景小基站比例: 50%
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 来自 environment.py 的标准参数 ==========
MACRO_HEIGHT = 25.0      # 宏基站: 楼顶部署 (3GPP UMa h_BS=25m)
SMALL_HEIGHT = 8.0       # 小基站: 灯杆/墙面部署 (3GPP UMi h_BS≈10m, 取8m)
UAV_ALT_MIN = 80         # UAV最低飞行高度
UAV_ALT_MAX = 350        # UAV最高飞行高度
POS_RANGE = 1000         # 水平范围 m


def generate_topology(num_bs=8, num_uav=300, scenario='default', seed=42):
    """生成基站和UAV的位置，使用与environment.py一致的参数"""
    np.random.seed(seed)

    # 小基站比例（来自environment.py的small_cell_ratios）
    small_cell_ratios = {
        'smart_city': 0.70, 'industrial_inspection': 0.50,
        'agriculture': 0.30, 'emergency_rescue': 0.25,
        'logistics_delivery': 0.45, 'urban': 0.70,
        'default': 0.50, 'emergency': 0.40,
    }
    small_ratio = small_cell_ratios.get(scenario, 0.50)

    # ====== 生成基站位置 ======
    bs_positions = []
    bs_types = []
    for i in range(num_bs):
        if np.random.rand() < small_ratio:
            bs_type = 'small'
            bs_h = SMALL_HEIGHT + np.random.uniform(-2, 2)
        else:
            bs_type = 'macro'
            bs_h = MACRO_HEIGHT + np.random.uniform(-2, 2)
        x = np.random.uniform(0, POS_RANGE)
        y = np.random.uniform(0, POS_RANGE)
        bs_positions.append([x, y, bs_h])
        bs_types.append(bs_type)
    bs_positions = np.array(bs_positions)

    # ====== 生成UAV位置（飞行在空中，高于所有基站）=====
    uav_positions = np.zeros((num_uav, 3))
    uav_positions[:, 0] = np.random.uniform(0, POS_RANGE, num_uav)
    uav_positions[:, 1] = np.random.uniform(0, POS_RANGE, num_uav)
    uav_positions[:, 2] = np.random.uniform(UAV_ALT_MIN, UAV_ALT_MAX, num_uav)

    # ====== 为UAV分配业务类型 ======
    ratios_map = {
        'emergency': [0.3, 0.5, 0.2], 'agriculture': [0.15, 0.25, 0.60],
        'default': [0.38, 0.29, 0.33],   # 接近实际300架分布
        'smart_city': [0.30, 0.60, 0.10], 'industrial_inspection': [0.15, 0.75, 0.10],
        'emergency_rescue': [0.85, 0.10, 0.05], 'logistics_delivery': [0.50, 0.40, 0.10],
    }
    ratios = ratios_map.get(scenario, ratios_map['default'])
    business_types = []
    rand_vals = np.random.rand(num_uav)
    for r in rand_vals:
        if r < ratios[0]:
            business_types.append(0)
        elif r < ratios[0] + ratios[1]:
            business_types.append(1)
        else:
            business_types.append(2)

    return bs_positions, bs_types, uav_positions, business_types


def plot_topology(bs_positions, bs_types, uav_positions, business_types, output_path):
    """绘制3D网络拓扑图"""
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')

    # ====== 分离宏/微基站 ======
    macro_idx = [i for i, t in enumerate(bs_types) if t == 'macro']
    small_idx = [i for i, t in enumerate(bs_types) if t == 'small']

    # 宏基站：红色方块（大）
    if macro_idx:
        mp = bs_positions[macro_idx]
        ax.scatter(mp[:, 0], mp[:, 1], mp[:, 2], c='#E41A1C', marker='s', s=180,
                   label=f'宏基站 ({len(macro_idx)}个)', depthshade=False,
                   edgecolors='darkred', linewidths=1.5, alpha=0.9, zorder=10)

    # 小基站：蓝色三角
    if small_idx:
        sp = bs_positions[small_idx]
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c='#377EB8', marker='^', s=140,
                   label=f'微基站 ({len(small_idx)}个)', depthshade=False,
                   edgecolors='darkblue', linewidths=1.2, alpha=0.9, zorder=10)

    # ====== 分离三类UAV ======
    ctrl_idx = [i for i, b in enumerate(business_types) if b == 0]
    vid_idx  = [i for i, b in enumerate(business_types) if b == 1]
    mon_idx  = [i for i, b in enumerate(business_types) if b == 2]

    # 控制信令：绿色圆点
    if ctrl_idx:
        up = uav_positions[ctrl_idx]
        ax.scatter(up[:, 0], up[:, 1], up[:, 2], c='#4DAF4A', marker='o', s=35,
                   label=f'控制信令 ({len(ctrl_idx)}架)', depthshade=False, alpha=0.65)
    # 视频回传：橙色圆点
    if vid_idx:
        up = uav_positions[vid_idx]
        ax.scatter(up[:, 0], up[:, 1], up[:, 2], c='#FF7F00', marker='o', s=35,
                   label=f'视频回传 ({len(vid_idx)}架)', depthshade=False, alpha=0.65)
    # 环境监测：紫色圆点
    if mon_idx:
        up = uav_positions[mon_idx]
        ax.scatter(up[:, 0], up[:, 1], up[:, 2], c='#984EA3', marker='o', s=35,
                   label=f'环境监测 ({len(mon_idx)}架)', depthshade=False, alpha=0.65)

    # ====== 绘制地面参考平面（半透明）======
    xx, yy = np.meshgrid(np.linspace(0, POS_RANGE, 3), np.linspace(0, POS_RANGE, 3))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')

    # ====== 高度标注线（可选，标注两层高度的差异）======
    # 在图的边缘画两条虚线表示典型高度层
    mid_x = POS_RANGE * 0.08
    mid_y = POS_RANGE * 0.08
    # 宏基站高度线
    ax.plot([mid_x, mid_x], [mid_y, mid_y], [0, MACRO_HEIGHT], 'r--', linewidth=1.5, alpha=0.6)
    ax.text(mid_x + 20, mid_y, MACRO_HEIGHT + 5, f'{int(MACRO_HEIGHT)}m',
            fontsize=8, color='darkred', ha='left')
    # 小基站高度线
    ax.plot([mid_x + 60, mid_x + 60], [mid_y, mid_y], [0, SMALL_HEIGHT], 'b--', linewidth=1.5, alpha=0.6)
    ax.text(mid_x + 75, mid_y, SMALL_HEIGHT + 5, f'{int(SMALL_HEIGHT)}m',
            fontsize=8, color='darkblue', ha='left')
    # UAV高度区域
    ax.plot([mid_x + 120, mid_x + 120], [mid_y, mid_y], [UAV_ALT_MIN, UAV_ALT_MAX],
            'g--', linewidth=1.2, alpha=0.4)
    ax.text(mid_x + 130, mid_y, (UAV_ALT_MIN + UAV_ALT_MAX)/2, 'UAV空域',
            fontsize=7, color='darkgreen', ha='left', rotation=90, va='center')

    # ====== 坐标轴设置 ======
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('高度 Z (m)', fontsize=12, labelpad=8)
    ax.set_xlim(0, POS_RANGE)
    ax.set_ylim(0, POS_RANGE)
    ax.set_zlim(0, UAV_ALT_MAX + 20)

    # ====== 视角（从斜上方俯视，能清晰看到高度差）======
    ax.view_init(elev=28, azim=225)

    # ====== 标题（不含"辅助通信网络"）=======
    ax.set_title('异构多层蜂窝无人机网络拓扑结构示意图', fontsize=16, fontweight='bold', pad=18)

    # 图例
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.01, 0.98),
              framealpha=0.92, edgecolor='lightgray', fancybox=True)

    # 网格
    ax.grid(True, alpha=0.25, linestyle='-')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"拓扑图已保存至: {output_path}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "network_topology.png")

    # 使用seed=42但强制混合场景以确保宏/微都有
    # default场景有50%概率出小基站，8个BS期望值是4+4
    # 如果某个seed恰好全为macro，尝试其他seed
    for candidate_seed in [42, 123, 2024, 7, 88]:
        bs_pos, bs_types, uav_pos, biz_types = generate_topology(
            num_bs=8, num_uav=300, scenario='default', seed=candidate_seed
        )
        n_macro = bs_types.count('macro')
        n_small = bs_types.count('small')
        if n_macro >= 3 and n_small >= 2:  # 确保两种类型都足够展示
            print(f"使用 seed={candidate_seed}: 宏基站={n_macro}, 微基站={n_small}")
            break
    else:
        # 兜底：手动指定混合类型
        print("警告: 随机种子未产生足够混合，使用手动分配")
        bs_pos, bs_types, uav_pos, biz_types = generate_topology(
            num_bs=8, num_uav=300, scenario='default', seed=42
        )
        # 手动确保前4个为macro，后4个为small
        bs_types = ['macro'] * 4 + ['small'] * 4
        for i, bs_type in enumerate(bs_types):
            h = MACRO_HEIGHT if bs_type == 'macro' else SMALL_HEIGHT
            bs_pos[i, 2] = h + np.random.uniform(-2, 2)

    plot_topology(bs_pos, bs_types, uav_pos, biz_types, output_path)

    # 打印统计信息
    print(f"\n===== 网络拓扑统计 =====")
    print(f"基站总数: {len(bs_pos)}")
    print(f"  宏基站: {bs_types.count('macro')} (高度 ≈{MACRO_HEIGHT}m)")
    print(f"  微基站: {bs_types.count('small')} (高度 ≈{SMALL_HEIGHT}m)")
    print(f"UAV总数: {len(uav_pos)}")
    print(f"  控制信令: {biz_types.count(0)} (绿色)")
    print(f"  视频回传: {biz_types.count(1)} (橙色)")
    print(f"  环境监测: {biz_types.count(2)} (紫色)")
    print(f"水平范围: {POS_RANGE} × {POS_RANGE} m")
    print(f"UAV高度范围: {UAV_ALT_MIN}-{UAV_ALT_MAX} m")


if __name__ == "__main__":
    main()
