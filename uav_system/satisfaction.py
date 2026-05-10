"""
=============================================================================
  UAV业务识别与切换决策系统 - 满意度评估模块 (satisfaction.py)
=============================================================================

【模块概述】
本模块实现了层次化的QoS满意度评估系统，是整个系统的"评价层"核心，
负责从多个维度量化UAV用户的服务质量体验。

【设计哲学】

1. **多维度评估模型**:
   不再使用单一的"满意/不满意"二元判断，而是综合考虑:
   - 速率满足度 (Rate Satisfaction)
   - 时延满足度 (Latency Satisfaction)  
   - 丢包率满足度 (Loss Rate Satisfaction)
   
   通过加权组合得到更精细的连续值评分[0, 1]。

2. **业务差异化处理**:
   不同业务类型对QoS维度的敏感度不同:
   - 控制信令: 高时延敏感(ls=0.9)，严格的关键指标判定
   - 视频回传: 中等时延敏感(ls=0.6)，重视带宽保障
   - 环境监测: 低时延敏感(ls=0.3)，容忍度高
   
   权重动态调整以反映真实业务需求。

3. **层次化指标体系**:
   ┌─────────────────────────────────────────────────┐
   │ L1: critical (关键指标满足)                     │ ← 二元(0/1)
   │     仅控制信令: 速率达标 AND 时延达标            │
   │     其他业务:   速率达标即可                    │
   ├─────────────────────────────────────────────────┤
   │ L2: overall (整体满足)                         │ ← 二元(0/1)
   │     基于最低服务要求(min_rate)的硬性判断        │
   ├─────────────────────────────────────────────────┤
   │ L3: weighted (优先级加权)                      │ ← 连续[0, priority]
   │     使用QoS配置中的priority作为满分             │
   ├─────────────────────────────────────────────────┤
   │ L4: latency_met/rate_met/loss_met              │ ← 二元(0/1)
   │     各维度的独立阈值判断                       │
   ├─────────────────────────────────────────────────┤
   │ L5: delay_sat/rate_sat/loss_sat                │ ← 连续[0, 1]
   │     各维度的归一化满意度(用于综合计算)         │
   └─────────────────────────────────────────────────┘

4. **关键业务特殊处理**:
   对于criticality ≥ 0.9的业务（目前仅控制信令）:
   - 一旦关键指标不满足，overall强制设为0
   - weighted分数也强制为0
   - 确保安全关键业务的QoS违规被严厉惩罚

【核心组件】
┌─────────────────────────────────────────────────────────────────────┐
│ 类名                            │ 功能描述                           │
├─────────────────────────────────────────────────────────────────────┤
│ HierarchicalSatisfactionMetric  │ 层次化满意度计算器(静态方法类)    │
│ ├ compute_satisfaction()       │ 单个UAV的多维度满意度评估          │
│ ├ compute_network_metrics()    │ 网络整体统计指标                   │
│ └ compute_business_type_sat..()│ 按业务类型的分组统计               │
└─────────────────────────────────────────────────────────────────────┘

【满意度计算公式详解】

单个UAV评估 (compute_satisfaction):

输入数据:
  - uav.current_allocated_rate: 当前分配速率(Mbps)
  - uav.current_latency: 当前端到端时延(ms)
  - uav.packet_loss_rate: 当前丢包率(0-1)
  - uav.true_business_type: 真实业务类型(ground truth)
  
参考标准(来自QOS_PROFILES):
  - true_qos.min_rate: 最低可接受速率
  - true_qos.ideal_rate: 理想目标速率
  - true_qos.max_delay: 最大容忍时延
  - true_qos.max_loss_rate: 最大容忍丢包率
  - true_qos.latency_sensitivity: 时延敏感系数[0, 1]
  - true_qos.criticality: 关键性等级[0, 1]
  - true_qos.priority: 业务优先级权重

各维度计算:

1. **速率维度** (rate):
   rate_met = (current_rate >= min_rate) ? 1 : 0           [二元]
   rate_ratio = current_rate / ideal_rate                  [连续, 0~∞]
   rate_sat = min(1.0, rate_ratio)                         [截断到0~1]

2. **时延维度** (delay):
   delay_sat = min(1.0, max_delay / estimated_latency)     [反比关系]
   
   特殊情况:
   - latency ≤ 0 或 max_delay ≤ 0 → delay_sat = 1.0 (完美)
   - latency > max_delay → delay_sat < 1.0 (有惩罚)
   
   latency_met = (delay_sat >= 0.8) ? 1 : 0                [阈值判定]

3. **丢包维度** (loss):
   loss_sat = min(1.0, max_loss_rate / (loss_rate + ε))  [反比+平滑]
   
   其中ε=1e-6防止除零错误
   
   loss_met = (loss_sat >= 0.5) ? 1 : 0                   [阈值判定]

综合评分公式 (非关键业务):
  overall = (w_rate × rate_ratio + w_delay × delay_sat + w_loss × loss_sat) / w_total
  
  动态权重(基于latency_sensitivity):
  - w_rate  = max(0.4, 1.0 - ls × 0.4)    [最低40%给速率]
  - w_delay = ls                          [直接使用敏感系数]
  - w_loss  = 0.2 + ls × 0.3             [基础20%+敏感加成]
  - w_total = w_rate + w_delay + w_loss   [归一化分母]
  
  示例(控制信令, ls=0.9):
  - w_rate = max(0.4, 1-0.36) = 0.64
  - w_delay = 0.9
  - w_loss = 0.2 + 0.27 = 0.47
  - w_total = 2.01
  - overall ≈ 0.32×rate_ratio + 0.45×delay_sat + 0.23×loss_sat
  
  示例(环境监测, ls=0.3):
  - w_rate = max(0.4, 1-0.12) = 0.88
  - w_delay = 0.3
  - w_loss = 0.2 + 0.09 = 0.29
  - w_total = 1.47
  - overall ≈ 0.60×rate_ratio + 0.20×delay_sat + 0.20×loss_sat

关键业务特殊逻辑 (criticality ≥ 0.9):
  if NOT (rate_met AND latency < threshold):
      overall = 0.0          # 强制零分
      critical = 0.0         # 关键指标不满足
      weighted = 0.0         # 无优先级得分
  
  设计原因:
  控制信令用于无人机遥控和安全指令，
  一旦时延或速率不达标可能导致飞行事故，
  因此采用"一票否决"的严格策略。

【网络级指标计算】(compute_network_metrics)

对所有UAV取均值:

┌───────────────────────────┬────────────────────────────────────────┐
│ 指标名称                   │ 计算方法                               │
├───────────────────────────┼────────────────────────────────────────┤
│ critical_satisfaction     │ mean(critical_i) 所有UAV的平均         │
│ overall_satisfaction      │ mean(overall_i)                        │
│ weighted_satisfaction     │ mean(weighted_i)                       │
│ latency_satisfaction      │ mean(latency_met_i)                    │
│ rate_satisfaction         │ mean(rate_met_i)                       │
│ loss_satisfaction         │ mean(loss_met_i)                       │
│ avg_delay_sat             │ mean(delay_sat_i) 连续值的平均         │
│ avg_loss_sat              │ mean(loss_sat_i)                       │
│ control_satisfaction      │ 仅控制信令UAV的mean(overall_i)         │
│                           │ 若无控制信令UAV则返回1.0               │
└───────────────────────────┴────────────────────────────────────────┘

【按业务类型分组统计】(compute_business_type_satisfaction)

输入: business_type枚举值

输出字典:
{
  'count': N,                    # 该类型UAV数量
  'satisfaction': mean(overall), # 平均整体满意度
  'rate_met': mean(rate_met),    # 速率满足比例
  'latency_met': mean(lat...),   # 时延满足比例
  'loss_met': mean(loss_met),    # 丢包满足比例
  'avg_rate_sat': mean(rat...),  # 平均速率连续满意度
  'avg_delay_sat': mean(del...), # 平均时延连续满意度
  'avg_loss_sat': mean(los...)   # 平均丢包连续满意度
}

边界情况:
  若该业务类型无UAV → 返回全零字典(count=0)

【典型应用场景】

场景1: 实验结果分析
  >>> from satisfaction import HierarchicalSatisfactionMetric
  >>> metrics = HierarchicalSatisfactionMetric.compute_network_metrics(env)
  >>> print(f"系统满意度: {metrics['overall_satisfaction']:.1%}")
  >>> print(f"关键业务满意度: {metrics['critical_satisfaction']:.1%}")

场景2: 单UAV诊断
  >>> sat = HierarchicalSatisfactionMetric.compute_satisfaction(uav)
  >>> if sat['critical'] == 0:
  ...     print(f"警告! UAV{uav.uav_id}关键指标未满足")
  ...     print(f"  时延: {sat['estimated_latency']:.1f}ms")
  ...     print(f"  速率满足: {'Yes' if sat['rate_met'] else 'No'}")

场景3: 业务类型对比
  >>> for bt in BusinessType:
  ...     stats = HierarchicalSatisfactionMetric.compute_business_type_satisfaction(env, bt)
  ...     print(f"{bt.name}: {stats['count']} UAVs, 满意度={stats['satisfaction']:.1%}")

【与其他模块的关系】

上游依赖:
  - business.py: BusinessType枚举, QOS_PROFILES配置字典
  - environment.py: UAV对象属性(current_allocated_rate等)

下游调用:
  - environment.py: get_state_statistics()调用compute_network_metrics()
  - experiments.py: 收集avg_satisfaction, critical_satisfaction等指标
  - mappo_environment.py: 奖励函数中使用current_satisfaction属性
  - visualization.py: 绘制满意度变化曲线

设计约束:
  - 必须基于true_business_type(而非predicted)进行评估
  - 确保评价的客观性和公平性
  - 识别误差的影响在recognition模块单独统计

【性能特征】

计算复杂度:
  - 单UAV: O(1) (固定数量的算术运算)
  - 网络级: O(N) (N=UAV数量)
  - 分组统计: O(N) (单次遍历+过滤)

数值稳定性:
  - 除法运算添加ε=1e-6防止除零
  - np.clip确保输出在[0, 1]范围
  - 边界条件显式处理(delay≤0, rate≤0等)

【已知限制】
  1. 时延值为估算值(非实际ping测量)，可能存在偏差
  2. 丢包率基于统计模型，不完全反映真实网络状况
  3. 权重参数(latency_sensitivity)来自专家经验，缺乏学习优化
  4. 不考虑历史累积效应(如持续低QoS的惩罚递增)
  5. 各维度独立性假设(实际可能相关，如高负载同时影响延迟和丢包)

【版本历史】
  V1.0: 初始版本，实现基础的三维(速率/时延/丢包)评估
  V1.1: 添加关键业务特殊处理逻辑
  V1.2: 引入层次化指标体系(critical/overall/weighted)
  V1.3: 优化权重动态分配算法(基于latency_sensitivity)
  V1.4: 添加网络级聚合和分组统计接口
"""

import numpy as np
from typing import Dict
from .business import BusinessType, QOS_PROFILES


class HierarchicalSatisfactionMetric:
    """
    层次化满意度评估

    计算维度:
    - critical: 关键指标满足（控制信令速率+时延，其他业务速率）
    - overall: 整体满足
    - weighted: 按业务优先级加权的满足
    - latency_met: 时延指标满足
    - rate_met: 速率指标满足
    """

    @staticmethod
    def compute_satisfaction(uav) -> Dict[str, float]:
        """
        计算单个UAV的满意度（基于真实业务类型的多维度评估）

        所有判断阈值均从真实业务类型的 QoS 配置中动态获取。
        综合速率、时延、丢包率三个维度。
        """
        true_qos = QOS_PROFILES[uav.true_business_type]
        rate_met = uav.current_allocated_rate >= true_qos.min_rate

        # 速率满意度
        rate_ratio = uav.current_allocated_rate / true_qos.ideal_rate if true_qos.ideal_rate > 0 else 0

        # 时延满意度（基于 UAV 实时估算的时延）
        estimated_latency = uav.current_latency
        if estimated_latency > 0 and true_qos.max_delay > 0:
            delay_sat = min(1.0, true_qos.max_delay / estimated_latency)
        else:
            delay_sat = 1.0
        latency_met = 1.0 if delay_sat >= 0.8 else 0.0

        # 丢包率满意度
        if true_qos.max_loss_rate > 0:
            loss_sat = min(1.0, true_qos.max_loss_rate / (uav.packet_loss_rate + 1e-6))
        else:
            loss_sat = 1.0
        loss_met = 1.0 if loss_sat >= 0.5 else 0.0

        # 关键业务判定
        is_critical = (true_qos.criticality >= 0.9)
        critical_latency_threshold = true_qos.max_delay

        if is_critical:
            critical_satisfied = rate_met and (estimated_latency < critical_latency_threshold)
            weighted_score = true_qos.priority
            # 多维度综合
            overall = (max(0.4, 1.0 - true_qos.latency_sensitivity * 0.4) * (1.0 if rate_met else 0.0) +
                       true_qos.latency_sensitivity * delay_sat +
                       (0.2 + true_qos.latency_sensitivity * 0.3) * loss_sat)
            overall = np.clip(overall / 3.0, 0.0, 1.0)
            if not critical_satisfied:
                overall = 0.0
            return {
                'critical': 1.0 if critical_satisfied else 0.0,
                'overall': 1.0 if critical_satisfied else 0.0,
                'weighted': weighted_score if critical_satisfied else 0.0,
                'latency_met': latency_met,
                'rate_met': 1.0 if rate_met else 0.0,
                'loss_met': loss_met,
                'estimated_latency': estimated_latency,
                'rate_sat': 1.0 if rate_met else 0.0,
                'delay_sat': delay_sat,
                'loss_sat': loss_sat,
            }
        else:
            service_satisfied = uav.current_allocated_rate >= true_qos.min_rate
            weighted_score = true_qos.priority
            # 多维度综合
            ls = true_qos.latency_sensitivity
            w_rate = max(0.4, 1.0 - ls * 0.4)
            w_delay = ls
            w_loss = 0.2 + ls * 0.3
            w_total = w_rate + w_delay + w_loss
            overall = (w_rate * rate_ratio + w_delay * delay_sat + w_loss * loss_sat) / w_total
            overall = np.clip(overall, 0.0, 1.0)
            return {
                'critical': 1.0,
                'overall': 1.0 if service_satisfied else 0.0,
                'weighted': weighted_score if service_satisfied else 0.0,
                'latency_met': latency_met,
                'rate_met': 1.0 if rate_met else 0.0,
                'loss_met': loss_met,
                'estimated_latency': estimated_latency,
                'rate_sat': min(1.0, rate_ratio),
                'delay_sat': delay_sat,
                'loss_sat': loss_sat,
            }

    @staticmethod
    def compute_network_metrics(env) -> Dict[str, float]:
        """
        计算网络整体的满意度指标

        Args:
            env: 网络环境对象

        Returns:
            包含各网络级别满意度指标的字典（速率+时延+丢包率多维度）
        """
        all_satisfactions = [HierarchicalSatisfactionMetric.compute_satisfaction(uav)
                             for uav in env.uavs.values()]
        return {
            'critical_satisfaction': np.mean([s['critical'] for s in all_satisfactions]),
            'overall_satisfaction': np.mean([s['overall'] for s in all_satisfactions]),
            'weighted_satisfaction': np.mean([s['weighted'] for s in all_satisfactions]),
            'latency_satisfaction': np.mean([s['latency_met'] for s in all_satisfactions]),
            'rate_satisfaction': np.mean([s['rate_met'] for s in all_satisfactions]),
            'loss_satisfaction': np.mean([s['loss_met'] for s in all_satisfactions]),
            'avg_delay_sat': np.mean([s['delay_sat'] for s in all_satisfactions]),
            'avg_loss_sat': np.mean([s['loss_sat'] for s in all_satisfactions]),
            'control_satisfaction': (lambda uavs, sats: np.mean([
                s['overall'] for u, s in zip(uavs, sats)
                if u.true_business_type == BusinessType.CONTROL_SIGNAL])(
                list(env.uavs.values()), all_satisfactions)
                if any(u.true_business_type == BusinessType.CONTROL_SIGNAL
                       for u in env.uavs.values()) else 1.0),
        }

    @staticmethod
    def compute_business_type_satisfaction(env, business_type: BusinessType) -> Dict[str, float]:
        """
        计算指定业务类型所有UAV的满意度

        Args:
            env: 网络环境对象
            business_type: 要统计的业务类型

        Returns:
            包含该业务类型UAV数量、平均满意度等指标的字典
        """
        uavs_of_type = [uav for uav in env.uavs.values() if uav.true_business_type == business_type]
        if not uavs_of_type:
            return {'count': 0, 'satisfaction': 0.0, 'rate_met': 0.0, 'latency_met': 0.0, 'loss_met': 0.0}
        satisfactions = [HierarchicalSatisfactionMetric.compute_satisfaction(uav) for uav in uavs_of_type]
        return {
            'count': len(uavs_of_type),
            'satisfaction': np.mean([s['overall'] for s in satisfactions]),
            'rate_met': np.mean([s['rate_met'] for s in satisfactions]),
            'latency_met': np.mean([s['latency_met'] for s in satisfactions]),
            'loss_met': np.mean([s['loss_met'] for s in satisfactions]),
            'weighted': np.mean([s['weighted'] for s in satisfactions]),
            'avg_delay_sat': np.mean([s['delay_sat'] for s in satisfactions]),
            'avg_loss_sat': np.mean([s['loss_sat'] for s in satisfactions]),
        }
