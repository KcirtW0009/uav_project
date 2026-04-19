#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成传统A3事件切换算法流程图
对应第二章图2-5
"""

import os
import graphviz

def create_a3_flowchart(output_path="figures/a3_handover_flowchart.png"):
    """创建传统A3事件切换算法流程图"""
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建有向图
    dot = graphviz.Digraph(
        name='A3_Handover_Flowchart',
        format='png',
        graph_attr={
            'rankdir': 'TB',  # 从上到下
            'splines': 'ortho',  # 直线连接
            'nodesep': '0.5',
            'ranksep': '0.8',
            'fontname': 'SimHei',
            'fontsize': '12',
            'label': '传统A3事件切换算法流程图',
            'labelloc': 't',
            'labeljust': 'c',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'lightgray',
            'fontname': 'SimHei',
            'fontsize': '11',
            'height': '0.3',
            'width': '1.5',
        },
        edge_attr={
            'fontname': 'SimHei',
            'fontsize': '10',
            'arrowsize': '0.8',
        }
    )
    
    # 定义节点
    nodes = {
        'start': {'label': '开始\n(每个仿真步)', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'iterate': {'label': '遍历所有已连接无人机', 'fillcolor': 'lightyellow'},
        'scan': {'label': '扫描邻区\n记录SINR最高候选小区', 'fillcolor': 'lightyellow'},
        'check_threshold': {'label': '候选SINR > 服务SINR + 迟滞量?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'emergency_check': {'label': '紧急切换条件?\n(服务SINR < -5dB 或 满意度 < 0.7)', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'initiate_handover': {'label': '发起切换请求\n(先断后连)', 'fillcolor': 'lightgreen'},
        'release_old': {'label': '释放旧基站资源', 'fillcolor': 'lightpink'},
        'request_new': {'label': '向新基站申请资源分配', 'fillcolor': 'lightpink'},
        'check_allocation': {'label': '资源分配成功?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'handover_complete': {'label': '切换完成', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'disconnected': {'label': '无人机进入断连状态', 'shape': 'ellipse', 'fillcolor': 'lightcoral'},
        'find_better': {'label': '跳过迟滞检查\n直接寻找更优基站', 'fillcolor': 'lightyellow'},
        'next_uav': {'label': '处理下一架无人机', 'fillcolor': 'lightyellow'},
        'end': {'label': '结束当前仿真步', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
    }
    
    # 添加节点
    for node_id, attrs in nodes.items():
        dot.node(
            node_id,
            label=attrs['label'],
            shape=attrs.get('shape', 'box'),
            style='rounded,filled',
            fillcolor=attrs['fillcolor']
        )
    
    # 添加边（主流程）
    dot.edge('start', 'iterate', label='')
    dot.edge('iterate', 'scan', label='对每架无人机')
    dot.edge('scan', 'check_threshold', label='')
    
    # 阈值检查分支
    dot.edge('check_threshold', 'initiate_handover', label='是')
    dot.edge('check_threshold', 'emergency_check', label='否')
    
    # 紧急切换分支
    dot.edge('emergency_check', 'find_better', label='是')
    dot.edge('emergency_check', 'next_uav', label='否')
    
    # 寻找更优基站后返回到扫描
    dot.edge('find_better', 'scan', label='重新扫描', style='dashed')
    
    # 切换执行流程
    dot.edge('initiate_handover', 'release_old', label='')
    dot.edge('release_old', 'request_new', label='')
    dot.edge('request_new', 'check_allocation', label='')
    
    # 资源分配检查分支
    dot.edge('check_allocation', 'handover_complete', label='是')
    dot.edge('check_allocation', 'disconnected', label='否')
    
    # 处理下一架无人机
    dot.edge('next_uav', 'iterate', label='继续遍历', style='dashed')
    
    # 切换完成或断连后继续处理下一架
    dot.edge('handover_complete', 'next_uav', label='')
    dot.edge('disconnected', 'next_uav', label='')
    
    # 所有无人机处理完毕后结束
    dot.edge('iterate', 'end', label='所有无人机处理完毕', style='dashed', constraint='false')
    
    # 渲染图表
    print(f"正在生成流程图: {output_path}")
    
    # 保存为PNG
    dot.render(filename=output_path.replace('.png', ''), cleanup=True)
    
    # 检查文件是否生成
    if os.path.exists(output_path):
        print(f"流程图已成功生成: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path):,} bytes")
    else:
        # 尝试其他可能的输出路径
        alt_path = output_path.replace('.png', '.png')
        if os.path.exists(alt_path):
            print(f"流程图已生成: {alt_path}")
        else:
            print("警告: 未找到生成的图片文件")

if __name__ == "__main__":
    # 输出路径
    output_path = os.path.join("figures", "a3_handover_flowchart.png")
    create_a3_flowchart(output_path)