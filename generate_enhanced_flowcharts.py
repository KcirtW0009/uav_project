#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成增强算法相关流程图：
1. 三级保障机制流程图 (three_level_guarantee_flowchart.png)
2. 增强切换算法完整流程图 (enhanced_algorithm_flowchart.png)
"""

import os
import graphviz

def create_three_level_guarantee_flowchart(output_path="figures/three_level_guarantee_flowchart.png"):
    """创建三级保障机制流程图"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dot = graphviz.Digraph(
        name='Three_Level_Guarantee_Flowchart',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'fontname': 'SimHei',
            'fontsize': '12',
            'label': '增强算法三级保障机制流程图',
            'labelloc': 't',
            'labeljust': 'c',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'lightgray',
            'fontname': 'SimHei',
            'fontsize': '11',
        },
        edge_attr={
            'fontname': 'SimHei',
            'fontsize': '10',
            'arrowsize': '0.8',
        }
    )
    
    # 节点定义
    nodes = {
        'start': {'label': '开始切换执行', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'level1': {'label': '第一级: 直接分配\n释放旧资源 → 申请目标基站资源', 'fillcolor': 'lightyellow'},
        'check1': {'label': '目标基站容量充足?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'success1': {'label': '分配成功\n切换正常完成\n计算动态冷却时间', 'shape': 'ellipse', 'fillcolor': 'lightgreen'},
        'level2': {'label': '第二级: 抢占与软迁移\n调用kick_low_priority方法\n释放低优先级非关键无人机资源', 'fillcolor': 'lightpink'},
        'check2': {'label': '抢占释放足够空间?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'soft_migration': {'label': '为被抢占无人机\n搜索替代基站\n执行软迁移', 'fillcolor': 'lightcyan'},
        'success2': {'label': '重新执行分配\n切换成功', 'shape': 'ellipse', 'fillcolor': 'lightgreen'},
        'level3': {'label': '第三级: 回滚与兜底\n尝试夺回旧基站资源', 'fillcolor': 'lightcoral'},
        'check3': {'label': '回滚成功?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'rollback_success': {'label': '恢复原有连接\n切换等效于未发生', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'disconnected': {'label': '无人机进入断连状态\n计入回滚失败和幽灵断连计数', 'shape': 'ellipse', 'fillcolor': 'red'},
        'next_step': {'label': '下一仿真步优先处理\n断连无人机', 'fillcolor': 'lightyellow'},
    }
    
    for node_id, attrs in nodes.items():
        dot.node(
            node_id,
            label=attrs['label'],
            shape=attrs.get('shape', 'box'),
            style='rounded,filled',
            fillcolor=attrs['fillcolor']
        )
    
    # 边
    dot.edge('start', 'level1')
    dot.edge('level1', 'check1')
    dot.edge('check1', 'success1', label='是')
    dot.edge('check1', 'level2', label='否')
    dot.edge('level2', 'check2')
    dot.edge('check2', 'soft_migration', label='是')
    dot.edge('soft_migration', 'success2')
    dot.edge('check2', 'level3', label='否')
    dot.edge('level3', 'check3')
    dot.edge('check3', 'rollback_success', label='是')
    dot.edge('check3', 'disconnected', label='否')
    dot.edge('disconnected', 'next_step')
    dot.edge('success1', 'next_step', style='dashed', constraint='false')
    dot.edge('success2', 'next_step', style='dashed', constraint='false')
    dot.edge('rollback_success', 'next_step', style='dashed', constraint='false')
    
    print(f"正在生成三级保障机制流程图: {output_path}")
    dot.render(filename=output_path.replace('.png', ''), cleanup=True)
    
    if os.path.exists(output_path):
        print(f"三级保障机制流程图已生成: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path):,} bytes")
    else:
        print("警告: 文件可能未生成")

def create_enhanced_algorithm_flowchart(output_path="figures/enhanced_algorithm_flowchart.png"):
    """创建增强切换算法完整流程图"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dot = graphviz.Digraph(
        name='Enhanced_Algorithm_Flowchart',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'fontname': 'SimHei',
            'fontsize': '12',
            'label': '增强切换算法完整流程图',
            'labelloc': 't',
            'labeljust': 'c',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': 'lightgray',
            'fontname': 'SimHei',
            'fontsize': '11',
        },
        edge_attr={
            'fontname': 'SimHei',
            'fontsize': '10',
            'arrowsize': '0.8',
        }
    )
    
    # 节点定义 - 基于描述
    nodes = {
        'start': {'label': '开始决策周期', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'iterate': {'label': '遍历所有无人机', 'fillcolor': 'lightyellow'},
        'business_id': {'label': '业务识别模块\n获取业务类型', 'fillcolor': 'lightcyan'},
        'current_state': {'label': '评估当前状态\n(满意度、SINR等)', 'fillcolor': 'lightyellow'},
        'emergency_check': {'label': '紧急切换条件?\nSINR<-5dB或满意度<0.7', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'emergency_select': {'label': '紧急选择逻辑\n搜索SINR最高且有容量基站\n允许降级', 'fillcolor': 'lightpink'},
        'epsilon_check': {'label': 'ε-greedy探索\n1%概率随机探索', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'random_explore': {'label': '随机选择候选基站\n验证可行性', 'fillcolor': 'lightpink'},
        'utility_calc': {'label': '计算所有邻区基站效用\n(五种降级比率)', 'fillcolor': 'lightcyan'},
        'best_candidate': {'label': '选择最佳候选方案', 'fillcolor': 'lightyellow'},
        'threshold_check': {'label': '效用增益超过动态阈值?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'no_handover': {'label': '保持当前连接', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
        'initiate_handover': {'label': '发出切换指令', 'fillcolor': 'lightgreen'},
        'execute_handover': {'label': '三级保障机制执行切换', 'fillcolor': 'lightpink'},
        'global_balance_check': {'label': '当前步是5的倍数\n且非高负载模式?', 'shape': 'diamond', 'fillcolor': 'lightcoral'},
        'global_balance': {'label': '执行全局负载均衡\n(排序、筛选、迁移)', 'fillcolor': 'lightcyan'},
        'next_uav': {'label': '处理下一架无人机', 'fillcolor': 'lightyellow'},
        'end': {'label': '决策周期结束', 'shape': 'ellipse', 'fillcolor': 'lightblue'},
    }
    
    for node_id, attrs in nodes.items():
        dot.node(
            node_id,
            label=attrs['label'],
            shape=attrs.get('shape', 'box'),
            style='rounded,filled',
            fillcolor=attrs['fillcolor']
        )
    
    # 边 - 主流程
    dot.edge('start', 'iterate')
    dot.edge('iterate', 'business_id')
    dot.edge('business_id', 'current_state')
    dot.edge('current_state', 'emergency_check')
    
    # 紧急切换分支
    dot.edge('emergency_check', 'emergency_select', label='是')
    dot.edge('emergency_select', 'execute_handover')
    dot.edge('emergency_check', 'epsilon_check', label='否')
    
    # ε-greedy分支
    dot.edge('epsilon_check', 'random_explore', label='触发')
    dot.edge('random_explore', 'execute_handover')
    dot.edge('epsilon_check', 'utility_calc', label='未触发')
    
    # 效用计算分支
    dot.edge('utility_calc', 'best_candidate')
    dot.edge('best_candidate', 'threshold_check')
    dot.edge('threshold_check', 'initiate_handover', label='是')
    dot.edge('threshold_check', 'no_handover', label='否')
    dot.edge('no_handover', 'next_uav')
    dot.edge('initiate_handover', 'execute_handover')
    
    # 切换执行后
    dot.edge('execute_handover', 'global_balance_check')
    dot.edge('global_balance_check', 'global_balance', label='是')
    dot.edge('global_balance_check', 'next_uav', label='否')
    dot.edge('global_balance', 'next_uav')
    
    # 遍历下一架无人机
    dot.edge('next_uav', 'iterate', label='继续遍历', style='dashed')
    
    # 所有无人机处理完毕
    dot.edge('iterate', 'end', label='所有无人机处理完毕', style='dashed', constraint='false')
    
    print(f"正在生成增强算法完整流程图: {output_path}")
    dot.render(filename=output_path.replace('.png', ''), cleanup=True)
    
    if os.path.exists(output_path):
        print(f"增强算法完整流程图已生成: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path):,} bytes")
    else:
        print("警告: 文件可能未生成")

if __name__ == "__main__":
    # 生成三级保障机制流程图
    create_three_level_guarantee_flowchart()
    print()
    
    # 生成增强算法完整流程图
    create_enhanced_algorithm_flowchart()
    print()
    
    print("所有增强算法流程图生成完成！")