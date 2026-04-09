# uav_system/__init__.py
"""
无人机业务识别与切换决策联动系统
"""
__version__ = "1.1"

# 导出优化模块
from .reward_functions import (
    RewardFunctionV2,
    RewardFunctionV3,
    CompositeRewardFunction,
    RewardNormalizer,
    get_reward_function,
)

from .enhanced_observation import (
    EnhancedObservationSpace,
    ObservationNormalizer,
    StateAugmenter,
    create_enhanced_observation,
)

from .communication_metrics import (
    CommunicationMetricsCollector,
    CommunicationMetrics,
    RealTimeMetricsMonitor,
    compute_critical_satisfaction,
    compute_weighted_satisfaction,
)

from .mappo_optimized_config import (
    OPTIMIZED_MAPPO_CONFIG,
    LOAD_SCENARIO_CONFIGS,
    ENHANCED_ALGORITHM_CONFIG,
    EXPERIMENT_CONFIG,
    get_optimized_config,
)

__all__ = [
    # 奖励函数
    'RewardFunctionV2',
    'RewardFunctionV3',
    'CompositeRewardFunction',
    'RewardNormalizer',
    'get_reward_function',
    # 观测空间
    'EnhancedObservationSpace',
    'ObservationNormalizer',
    'StateAugmenter',
    'create_enhanced_observation',
    # 通信指标
    'CommunicationMetricsCollector',
    'CommunicationMetrics',
    'RealTimeMetricsMonitor',
    'compute_critical_satisfaction',
    'compute_weighted_satisfaction',
    # 配置
    'OPTIMIZED_MAPPO_CONFIG',
    'LOAD_SCENARIO_CONFIGS',
    'ENHANCED_ALGORITHM_CONFIG',
    'EXPERIMENT_CONFIG',
    'get_optimized_config',
]