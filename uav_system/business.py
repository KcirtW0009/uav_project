from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np

class BusinessType(Enum):
    CONTROL_SIGNAL = 0
    VIDEO_STREAMING = 1
    ENVIRONMENT_MONITORING = 2

@dataclass
class QoSProfile:
    business_type: BusinessType
    min_rate: float
    ideal_rate: float
    max_delay: float
    max_loss_rate: float
    priority: float
    downgrade_tolerance: float
    criticality: float = 0.5
    latency_sensitivity: float = 0.5

    def calculate_satisfaction(self, allocated_rate: float) -> float:
        rate_ratio = allocated_rate / self.ideal_rate
        if self.business_type == BusinessType.CONTROL_SIGNAL:
            if rate_ratio >= 0.95:
                return 1.0
            elif rate_ratio >= 0.85:
                return 0.7 + 0.3 * (rate_ratio - 0.85) / 0.1
            elif rate_ratio >= 0.7:
                return 0.3 + 0.4 * (rate_ratio - 0.7) / 0.15
            else:
                return 0.0
        elif self.business_type == BusinessType.VIDEO_STREAMING:
            if rate_ratio >= 0.9:
                return 1.0
            elif rate_ratio >= 0.7:
                x = (rate_ratio - 0.7) / 0.2
                return 0.5 + 0.5 * (3*x**2 - 2*x**3)
            elif rate_ratio >= 0.5:
                return 0.2 + 0.3 * (rate_ratio - 0.5) / 0.2
            else:
                return max(0.0, rate_ratio / 0.5 * 0.2)
        else:
            if rate_ratio >= 0.8:
                return 1.0
            elif rate_ratio >= 0.3:
                return 0.4 + 0.6 * (rate_ratio - 0.3) / 0.5
            else:
                return max(0.0, rate_ratio / 0.3 * 0.4)

    def get_feasible_downgrade_ratios(self) -> List[float]:
        if self.business_type == BusinessType.CONTROL_SIGNAL:
            return [1.0, 0.95, 0.9]
        elif self.business_type == BusinessType.VIDEO_STREAMING:
            return [1.0, 0.9, 0.8, 0.7, 0.6]
        else:
            return [1.0, 0.8, 0.6, 0.4, 0.3]

QOS_PROFILES = {
    BusinessType.CONTROL_SIGNAL: QoSProfile(
        business_type=BusinessType.CONTROL_SIGNAL,
        min_rate=45, ideal_rate=50, max_delay=10, max_loss_rate=0.001,
        priority=0.95, downgrade_tolerance=0.05,
        criticality=1.0, latency_sensitivity=1.0
    ),
    BusinessType.VIDEO_STREAMING: QoSProfile(
        business_type=BusinessType.VIDEO_STREAMING,
        min_rate=150, ideal_rate=200, max_delay=50, max_loss_rate=0.01,
        priority=0.65, downgrade_tolerance=0.35,
        criticality=0.7, latency_sensitivity=0.6
    ),
    BusinessType.ENVIRONMENT_MONITORING: QoSProfile(
        business_type=BusinessType.ENVIRONMENT_MONITORING,
        min_rate=30, ideal_rate=80, max_delay=200, max_loss_rate=0.05,
        priority=0.35, downgrade_tolerance=0.75,
        criticality=0.4, latency_sensitivity=0.3
    )
}

BUSINESS_FEATURE_PARAMS = {
    BusinessType.CONTROL_SIGNAL: {
        'delay': (5, 2), 'bandwidth': (50, 10),
        'loss_beta': (1, 50), 'loss_scale': 0.5, 'jitter': (1, 0.5)
    },
    BusinessType.VIDEO_STREAMING: {
        'delay': (30, 10), 'bandwidth': (200, 50),
        'loss_beta': (2, 10), 'loss_scale': 1.0, 'jitter': (5, 2)
    },
    BusinessType.ENVIRONMENT_MONITORING: {
        'delay': (10, 3), 'bandwidth': (100, 30),
        'loss_beta': (2, 20), 'loss_scale': 0.5, 'jitter': (2, 1)
    }
}