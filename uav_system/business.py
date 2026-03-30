"""
业务类型与QoS模型定义

定义三类无人机业务（控制信令、视频回传、环境监测）的枚举、
QoS配置文件、以及业务特征生成参数。
"""

from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np


class BusinessType(Enum):
    """无人机业务类型枚举"""
    CONTROL_SIGNAL = 0           # 控制信令：高优先级，低时延，低带宽
    VIDEO_STREAMING = 1          # 视频回传：中优先级，高带宽，中时延
    ENVIRONMENT_MONITORING = 2   # 环境监测：低优先级，低带宽，高时延容忍


@dataclass
class QoSProfile:
    """
    QoS配置文件

    Attributes:
        business_type: 业务类型
        min_rate: 最低保障速率(Mbps)
        ideal_rate: 理想速率(Mbps)
        max_delay: 最大允许时延(ms)
        max_loss_rate: 最大允许丢包率
        priority: 业务优先级(0-1)
        downgrade_tolerance: 降级容忍度(0-1)
        criticality: 关键程度(0-1)
        latency_sensitivity: 时延敏感度(0-1)
    """
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
        """
        根据分配速率计算满意度

        各业务类型有不同的满意度曲线：
        - 控制信令：严格阈值型，低于70%速率满意度直接归零
        - 视频回传：平滑过渡型，使用Smoothstep函数
        - 环境监测：线性型，容忍较大的速率波动
        """
        rate_ratio = allocated_rate / self.ideal_rate

        if self.business_type == BusinessType.CONTROL_SIGNAL:
            # 控制信令：严格阶梯函数
            if rate_ratio >= 0.95:
                return 1.0
            elif rate_ratio >= 0.85:
                return 0.7 + 0.3 * (rate_ratio - 0.85) / 0.1
            elif rate_ratio >= 0.7:
                return 0.3 + 0.4 * (rate_ratio - 0.7) / 0.15
            else:
                return 0.0

        elif self.business_type == BusinessType.VIDEO_STREAMING:
            # 视频回传：Smoothstep平滑过渡
            if rate_ratio >= 0.9:
                return 1.0
            elif rate_ratio >= 0.7:
                x = (rate_ratio - 0.7) / 0.2
                return 0.5 + 0.5 * (3 * x**2 - 2 * x**3)
            elif rate_ratio >= 0.5:
                return 0.2 + 0.3 * (rate_ratio - 0.5) / 0.2
            else:
                return max(0.0, rate_ratio / 0.5 * 0.2)

        else:  # ENVIRONMENT_MONITORING
            # 环境监测：线性衰减
            if rate_ratio >= 0.8:
                return 1.0
            elif rate_ratio >= 0.3:
                return 0.4 + 0.6 * (rate_ratio - 0.3) / 0.5
            else:
                return max(0.0, rate_ratio / 0.3 * 0.4)

    def get_feasible_downgrade_ratios(self) -> List[float]:
        """
        获取可行的降级比例列表

        控制信令容忍最低降级(5%)，环境监测容忍最大降级(70%)。
        """
        if self.business_type == BusinessType.CONTROL_SIGNAL:
            return [1.0, 0.95, 0.9]
        elif self.business_type == BusinessType.VIDEO_STREAMING:
            return [1.0, 0.9, 0.8, 0.7, 0.6]
        else:
            return [1.0, 0.8, 0.6, 0.4, 0.3]


# ==================== 预定义QoS配置 ====================
# 参数来源：华为白皮书、3GPP TS 22.125、论文KPI表格
# - 控制信令(URlLC): ≈200kbps上行, ≤20ms时延, 99.999%可靠性
# - 视频回传(eMBB): 25-100Mbps带宽, ≈20ms时延, 连续传输高可靠
# - 环境监测(mMTC): ≤1Mbps带宽, <1000ms时延, 容忍一定丢包
QOS_PROFILES = {
    BusinessType.CONTROL_SIGNAL: QoSProfile(
        business_type=BusinessType.CONTROL_SIGNAL,
        min_rate=0.15, ideal_rate=0.5, max_delay=20, max_loss_rate=0.00001,
        priority=0.99, downgrade_tolerance=0.05,
        criticality=1.0, latency_sensitivity=1.0
    ),
    BusinessType.VIDEO_STREAMING: QoSProfile(
        business_type=BusinessType.VIDEO_STREAMING,
        min_rate=25, ideal_rate=50, max_delay=20, max_loss_rate=0.001,
        priority=0.75, downgrade_tolerance=0.35,
        criticality=0.7, latency_sensitivity=0.8
    ),
    BusinessType.ENVIRONMENT_MONITORING: QoSProfile(
        business_type=BusinessType.ENVIRONMENT_MONITORING,
        min_rate=0.5, ideal_rate=1.0, max_delay=1000, max_loss_rate=0.05,
        priority=0.30, downgrade_tolerance=0.75,
        criticality=0.3, latency_sensitivity=0.2
    )
}


# ==================== 业务特征生成参数 ====================
# 用于模拟各业务类型的网络流量特征（时延ms、带宽Mbps、丢包率、抖动ms）
# 与QOS_PROFILES对齐：控制信令低带宽低时延、视频高带宽中时延、监测低带宽高时延容忍
BUSINESS_FEATURE_PARAMS = {
    BusinessType.CONTROL_SIGNAL: {
        'delay': (10, 3),                # 10±3ms (≤20ms要求)
        'bandwidth': (0.5, 0.1),         # 500±100kbps
        'loss_beta': (1, 1000),          # 极低丢包率(99.999%可靠性)
        'loss_scale': 0.00001,
        'jitter': (1, 0.5)
    },
    BusinessType.VIDEO_STREAMING: {
        'delay': (15, 5),                # 15±5ms (≈20ms要求)
        'bandwidth': (50, 15),           # 50±15Mbps (25-100Mbps范围)
        'loss_beta': (5, 100),           # 低丢包(连续传输)
        'loss_scale': 0.001,
        'jitter': (3, 1)
    },
    BusinessType.ENVIRONMENT_MONITORING: {
        'delay': (500, 200),             # 500±200ms (<1000ms)
        'bandwidth': (1, 0.3),           # 1±0.3Mbps (≤1Mbps)
        'loss_beta': (2, 20),
        'loss_scale': 0.05,
        'jitter': (50, 20)
    }
}
