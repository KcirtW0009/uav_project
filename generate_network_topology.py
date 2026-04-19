#!/usr/bin/env python3
"""
生成无人机辅助通信网络拓扑结构示意图
对应第二章图2-3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_topology(num_bs=8, num_uav=300, scenario='default', seed=42):
    """生成基站和UAV的位置"""
    np.random.seed(seed)
    
    # 基站位置参数
    pos_range_map = {
        'urban': 800, 'emergency': 1200, 'agriculture': 1500, 'default': 1000,
        'smart_city': 800, 'industrial_inspection': 600, 'emergency_rescue': 1200,
        'logistics_delivery': 1500
    }
    pos_range = pos_range_map.get(scenario, 1000)
    
    # 生成基站位置
    bs_positions = np.random.rand(num_bs, 3) * pos_range
    # 基站类型：城市场景40%为小基站，其他为宏基站
    bs_types = []
    for i in range(num_bs):
        if scenario == 'urban' and np.random.rand() < 0.4:
            bs_types.append('small')
        else:
            bs_types.append('macro')
    
    # 生成UAV位置（均匀分布）
    uav_positions = np.random.rand(num_uav, 3) * pos_range
    
    # 为UAV分配业务类型（用于颜色区分）
    # 业务比例: [控制信令, 视频回传, 环境监测]
    ratios_map = {
        'emergency': [0.3, 0.5, 0.2],
        'agriculture': [0.15, 0.25, 0.60],
        'default': [0.4, 0.3, 0.3],
        'smart_city': [0.30, 0.60, 0.10],
        'industrial_inspection': [0.15, 0.75, 0.10],
        'emergency_rescue': [0.85, 0.10, 0.05],
        'logistics_delivery': [0.50, 0.40, 0.10],
    }
    ratios = ratios_map.get(scenario, ratios_map['default'])
    business_types = []
    for i in range(num_uav):
        rand = np.random.rand()
        if rand < ratios[0]:
            business_types.append(0)  # 控制信令
        elif rand < ratios[0] + ratios[1]:
            business_types.append(1)  # 视频回传
        else:
            business_types.append(2)  # 环境监测
    
    return bs_positions, bs_types, uav_positions, business_types, pos_range

def plot_topology(bs_positions, bs_types, uav_positions, business_types, pos_range, output_path):
    """绘制3D网络拓扑图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制基站
    macro_x, macro_y, macro_z = [], [], []
    small_x, small_y, small_z = [], [], []
    for i, (pos, bs_type) in enumerate(zip(bs_positions, bs_types)):
        if bs_type == 'macro':
            macro_x.append(pos[0])
            macro_y.append(pos[1])
            macro_z.append(pos[2])
        else:
            small_x.append(pos[0])
            small_y.append(pos[1])
            small_z.append(pos[2])
    
    # 宏基站：红色立方体
    if macro_x:
        ax.scatter(macro_x, macro_y, macro_z, c='red', marker='s', s=120, 
                   label=f'宏基站 ({len(macro_x)}个)', depthshade=False, alpha=0.8)
    # 小基站：蓝色三角
    if small_x:
        ax.scatter(small_x, small_y, small_z, c='blue', marker='^', s=100,
                   label=f'小基站 ({len(small_x)}个)', depthshade=False, alpha=0.8)
    
    # 绘制UAV，按业务类型着色
    control_x, control_y, control_z = [], [], []
    video_x, video_y, video_z = [], [], []
    monitor_x, monitor_y, monitor_z = [], [], []
    
    for i, (pos, biz_type) in enumerate(zip(uav_positions, business_types)):
        if biz_type == 0:  # 控制信令
            control_x.append(pos[0])
            control_y.append(pos[1])
            control_z.append(pos[2])
        elif biz_type == 1:  # 视频回传
            video_x.append(pos[0])
            video_y.append(pos[1])
            video_z.append(pos[2])
        else:  # 环境监测
            monitor_x.append(pos[0])
            monitor_y.append(pos[1])
            monitor_z.append(pos[2])
    
    # 控制信令：绿色圆点
    if control_x:
        ax.scatter(control_x, control_y, control_z, c='green', marker='o', s=40,
                   label=f'控制信令UAV ({len(control_x)}架)', depthshade=False, alpha=0.6)
    # 视频回传：橙色圆点
    if video_x:
        ax.scatter(video_x, video_y, video_z, c='orange', marker='o', s=40,
                   label=f'视频回传UAV ({len(video_x)}架)', depthshade=False, alpha=0.6)
    # 环境监测：紫色圆点
    if monitor_x:
        ax.scatter(monitor_x, monitor_y, monitor_z, c='purple', marker='o', s=40,
                   label=f'环境监测UAV ({len(monitor_x)}架)', depthshade=False, alpha=0.6)
    
    # 设置坐标轴标签和范围
    ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('高度 (m)', fontsize=12, labelpad=10)
    ax.set_xlim(0, pos_range)
    ax.set_ylim(0, pos_range)
    ax.set_zlim(0, pos_range)
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 添加标题和图例
    ax.set_title('无人机辅助通信网络拓扑结构示意图', fontsize=16, pad=20)
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"拓扑图已保存至: {output_path}")

def main():
    # 确保输出目录存在
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "network_topology.png")
    
    # 生成拓扑数据（使用默认场景）
    bs_positions, bs_types, uav_positions, business_types, pos_range = generate_topology(
        num_bs=8, num_uav=300, scenario='default', seed=42
    )
    
    # 绘制并保存
    plot_topology(bs_positions, bs_types, uav_positions, business_types, pos_range, output_path)
    
    # 打印统计信息
    print(f"基站总数: {len(bs_positions)}")
    print(f"  宏基站: {bs_types.count('macro')}")
    print(f"  小基站: {bs_types.count('small')}")
    print(f"UAV总数: {len(uav_positions)}")
    print(f"  控制信令: {business_types.count(0)}")
    print(f"  视频回传: {business_types.count(1)}")
    print(f"  环境监测: {business_types.count(2)}")
    print(f"场景范围: {pos_range} × {pos_range} × {pos_range} m^3")

if __name__ == "__main__":
    main()