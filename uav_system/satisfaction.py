import numpy as np
from typing import Dict
from .business import BusinessType

class HierarchicalSatisfactionMetric:
    @staticmethod
    def compute_satisfaction(uav) -> Dict[str, float]:
        qos = uav.qos_profile
        rate_met = uav.current_allocated_rate >= qos.min_rate
        estimated_latency = qos.max_delay * (1.5 - min(uav.sinr_db / 20, 1.0))
        latency_met = estimated_latency < qos.max_delay

        if uav.true_business_type == BusinessType.CONTROL_SIGNAL:
            critical_satisfied = rate_met and (estimated_latency < 10)
            return {
                'critical': 1.0 if critical_satisfied else 0.0,
                'overall': 1.0 if critical_satisfied else 0.0,
                'weighted': 0.95 if critical_satisfied else 0.0,
                'latency_met': 1.0 if latency_met else 0.0,
                'rate_met': 1.0 if rate_met else 0.0,
                'estimated_latency': estimated_latency
            }
        elif uav.true_business_type == BusinessType.VIDEO_STREAMING:
            video_satisfied = uav.current_allocated_rate >= 150
            return {
                'critical': 1.0,
                'overall': 1.0 if video_satisfied else 0.0,
                'weighted': 0.65 if video_satisfied else 0.0,
                'latency_met': 1.0 if latency_met else 0.0,
                'rate_met': 1.0 if rate_met else 0.0,
                'estimated_latency': estimated_latency
            }
        else:
            env_satisfied = uav.current_allocated_rate >= 30
            return {
                'critical': 1.0,
                'overall': 1.0 if env_satisfied else 0.0,
                'weighted': 0.35 if env_satisfied else 0.0,
                'latency_met': 1.0 if latency_met else 0.0,
                'rate_met': 1.0 if rate_met else 0.0,
                'estimated_latency': estimated_latency
            }

    @staticmethod
    def compute_network_metrics(env) -> Dict[str, float]:
        all_satisfactions = [HierarchicalSatisfactionMetric.compute_satisfaction(uav)
                             for uav in env.uavs.values()]
        return {
            'critical_satisfaction': np.mean([s['critical'] for s in all_satisfactions]),
            'overall_satisfaction': np.mean([s['overall'] for s in all_satisfactions]),
            'weighted_satisfaction': np.mean([s['weighted'] for s in all_satisfactions]),
            'latency_satisfaction': np.mean([s['latency_met'] for s in all_satisfactions]),
            'rate_satisfaction': np.mean([s['rate_met'] for s in all_satisfactions]),
            'control_satisfaction': np.mean([s['overall'] for s in all_satisfactions
                                             if list(env.uavs.values())[0].true_business_type == BusinessType.CONTROL_SIGNAL]),
        }

    @staticmethod
    def compute_business_type_satisfaction(env, business_type: BusinessType) -> Dict[str, float]:
        uavs_of_type = [uav for uav in env.uavs.values() if uav.true_business_type == business_type]
        if not uavs_of_type:
            return {'count': 0, 'satisfaction': 0.0, 'rate_met': 0.0, 'latency_met': 0.0}
        satisfactions = [HierarchicalSatisfactionMetric.compute_satisfaction(uav) for uav in uavs_of_type]
        return {
            'count': len(uavs_of_type),
            'satisfaction': np.mean([s['overall'] for s in satisfactions]),
            'rate_met': np.mean([s['rate_met'] for s in satisfactions]),
            'latency_met': np.mean([s['latency_met'] for s in satisfactions]),
            'weighted': np.mean([s['weighted'] for s in satisfactions])
        }