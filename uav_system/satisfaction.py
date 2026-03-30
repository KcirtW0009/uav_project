"""
满意度评估模块

提供层次化满意度计算方法，支持按业务类型和网络整体进行评估。
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
