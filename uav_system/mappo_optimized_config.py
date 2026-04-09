# -*- coding: utf-8 -*-
"""
MAPPO 优化配置模块

整合所有优化后的配置参数
"""

from typing import Dict, Any


# 优化后的 MAPPO 配置
OPTIMIZED_MAPPO_CONFIG = {
    # 网络结构
    'hidden_dim': 128,          # 增加网络容量
    'critic_hidden_dim': 256,   # Critic需要更大的容量
    
    # 学习率（基于参数搜索的预期最优值）
    'actor_lr': 3e-5,           # 降低学习率提高稳定性
    'critic_lr': 3e-4,          # Critic学习率保持相对较高
    
    # PPO核心参数（基于参数搜索结果优化）
    'gamma': 0.99,
    'gae_lambda': 0.99,         # 参数搜索: 0.99优于0.95
    'clip_epsilon': 0.2,        # 参数搜索最佳: 0.2
    'entropy_coef': 0.02,       # 参数搜索最佳: 0.02
    'value_coef': 0.5,
    
    # 训练参数
    'rollout_length': 150,
    'num_epochs': 5,
    'batch_size': 256,          # 增大batch size
    
    # 早停设置
    'use_early_stopping': True,
    'early_stop_patience': 100,
    'min_delta': 0.001,
    
    # 采样优化
    'train_sample_agents': 50,
    'attention_sample_agents': 50,
    
    # 功能开关
    'use_biz_heads': True,
    'use_attention_critic': True,
    'use_data_augmentation': True,
    
    # 奖励函数配置
    'reward_function': {
        'version': 'v2',
        'sat_weight': 10.0,
        'switch_penalty_base': 1.0,
        'disconnect_penalty': 50.0,
        'load_balance_weight': 0.1,
        'use_biz_aware_penalty': True,
    },
    
    # 观测空间配置
    'observation': {
        'history_length': 5,
        'use_relative_position': True,
        'use_state_augmentation': True,
    },
    
    # 学习率调度
    'lr_schedule': {
        'type': 'cosine',  # 'cosine', 'step', 'exponential', 'none'
        'T_max': 300,
        'eta_min': 1e-6,
    },
}


# 不同负载率场景的配置
LOAD_SCENARIO_CONFIGS = {
    'low': {
        'num_uav': 32,
        'bs_capacity_range': (800, 1200),
        'description': '低负载场景 (约30%)',
    },
    'medium': {
        'num_uav': 64,
        'bs_capacity_range': (600, 1000),
        'description': '中负载场景 (约60%)',
    },
    'high': {
        'num_uav': 128,
        'bs_capacity_range': (500, 900),
        'description': '高负载场景 (约88%)',
    },
    'extreme': {
        'num_uav': 150,
        'bs_capacity_range': (400, 800),
        'description': '极高负载场景 (约95%)',
    },
}


# 增强算法独立配置（与MAPPO实验严格隔离）
ENHANCED_ALGORITHM_CONFIG = {
    'sinr_threshold': 3.0,
    'hysteresis': 1.0,
    'load_balance_weight': 0.3,
    'business_priority': {
        0: 1.0,  # CONTROL_SIGNAL
        1: 0.8,  # VIDEO_STREAMING
        2: 0.5,  # ENVIRONMENT_MONITORING
    },
    'prediction_horizon': 5,
    'exploration_rate': 0.1,
    'use_multi_objective': True,
    'use_predictive_handover': False,
}


# 实验运行配置
EXPERIMENT_CONFIG = {
    'train_episodes': 300,
    'eval_episodes': 3,
    'eval_interval': 10,
    'save_interval': 50,
    'map_size': 1000,
    'num_bs': 3,
    'seed': 42,
    
    # 可视化配置
    'visualization': {
        'satisfaction_ylim_margin': 0.15,  # 满意度y轴边距比例
        'use_smoothing': True,
        'smoothing_window': 10,
        'create_comprehensive_plots': True,
    },
    
    # 数据记录配置
    'data_logging': {
        'log_level': 'detailed',  # 'basic', 'detailed', 'full'
        'save_episode_data': True,
        'save_network_stats': True,
        'save_communication_metrics': True,
        'log_dir': './experiment_logs',
    },
}


def get_optimized_config(scenario: str = 'high', 
                        custom_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取优化后的配置
    
    Args:
        scenario: 负载场景 ('low', 'medium', 'high', 'extreme')
        custom_overrides: 自定义覆盖参数
        
    Returns:
        完整配置字典
    """
    # 基础配置
    config = OPTIMIZED_MAPPO_CONFIG.copy()
    
    # 场景特定配置
    if scenario in LOAD_SCENARIO_CONFIGS:
        scenario_config = LOAD_SCENARIO_CONFIGS[scenario]
        config['scenario'] = scenario
        config['num_uav'] = scenario_config['num_uav']
        config['bs_capacity_range'] = scenario_config['bs_capacity_range']
    
    # 实验配置
    config.update(EXPERIMENT_CONFIG)
    
    # 应用自定义覆盖
    if custom_overrides:
        _deep_update(config, custom_overrides)
    
    return config


def _deep_update(base_dict: Dict, update_dict: Dict):
    """深度更新字典"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def print_config(config: Dict[str, Any]):
    """打印配置"""
    print("=" * 60)
    print("MAPPO 优化配置")
    print("=" * 60)
    
    def _print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                _print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    _print_dict(config)
    print("=" * 60)


if __name__ == '__main__':
    # 测试配置
    config = get_optimized_config('high')
    print_config(config)
