"""
全局配置模块

定义随机种子、实验参数、颜色方案、matplotlib设置等全局配置。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# ==================== 随机种子 ====================
GLOBAL_SEED = 30042  # 原始=42, 更换种子验证sat天花板非seed相关


def set_global_seed(seed=GLOBAL_SEED):
    """设置所有随机数生成器的种子，确保实验可复现"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass
    print(f"全局随机种子已设置为：{seed}")


# ==================== 目录配置 ====================
RESULT_DIR = "experiment_results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ==================== 中断检测配置 ====================
INTERRUPTION_CONFIG = {
    'threshold': 0.3,                 # 满足率阈值，低于此值视为中断
    'duration': 5,                     # 持续步数，低于阈值且持续N步才计为中断
    'control_signal_threshold': 0.4,   # 控制信令业务的中断阈值（更高要求）
    'control_signal_duration': 3       # 控制信令业务的持续步数（更严格要求）
}


# ==================== 颜色方案 ====================
COLORS = {
    'control': '#FF6B6B',
    'video': '#4ECDC4',
    'environment': '#95E1D3',
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'info': '#60a5fa',
    'neutral': '#9ca3af'
}


def create_gradient_cmap(color1, color2, name='custom'):
    """创建渐变色映射"""
    return LinearSegmentedColormap.from_list(name, [color1, color2])


CMAP_PRIMARY = create_gradient_cmap(COLORS['primary'], COLORS['secondary'], 'primary')
CMAP_SUCCESS = create_gradient_cmap('#86efac', COLORS['success'], 'success')
CMAP_WARNING = create_gradient_cmap('#fcd34d', COLORS['warning'], 'warning')


# ==================== Matplotlib 全局设置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100


# ==================== MAPPO 集中配置类 (V21: 解决P1/P2级硬编码) ====================

class MAPPOConfig:
    """
    MAPPO系统集中式超参数配置
    
    目的: 消除代码中的硬编码值,提升可维护性和可调优性
    使用方式: from .config import MAPPOConfig; cfg = MAPPOConfig.RewardConfig.delta_scale
    
    版本: V21 (2026-05-09)
    覆盖范围:
      - P1级: Reward函数核心参数 (24处)
      - P2级: 业务权重、负载因子、环境参数 (22处)
    """
    
    class RewardConfig:
        """Reward Function 核心参数 (P1级)"""
        
        # ---- 基础信号分量 ----
        delta_scale: float = 5.0              # 连续速率比信号的缩放因子 [3.0, 8.0]
        counterfactual_scale: float = 3.0     # 反事实比较信号缩放因子
        
        # ---- 动作奖励阈值 (V19核心) ----
        excellent_switch_threshold: float = 0.05   # 优秀切换的满意度提升阈值 [0.03, 0.08]
        good_switch_threshold: float = 0.015       # 好切换的提升阈值
        acceptable_switch_threshold: float = -0.03 # 可接受的微负切换损失
        
        # ---- 留守动作奖励 ----
        stay_base_reward: float = 0.80            # 留守基础奖励 (V19: 高价值默认动作)
        stay_bonus_threshold: float = 0.93         # 激活bonus的满意度阈值
        stay_bonus_scale: float = 2.0             # bonus缩放因子
        
        # ---- 切换动作奖励系数 ----
        excellent_switch_base: float = 1.0         # 优秀切换基础奖励
        excellent_switch_coeff: float = 2.0        # 优秀切换满意度系数
        good_switch_base: float = 0.55             # 好切换基础奖励
        good_switch_coeff: float = 3.0             # 好切换满意度系数
        micro_positive_base: float = 0.15          # 微正切换基础
        micro_positive_coeff: float = 5.0          # 微正切换系数
        acceptable_penalty_coeff: float = 4.0      # 可接受惩罚系数
        bad_switch_coeff: float = 6.0              # 坏切换满意度系数
        bad_switch_penalty: float = -0.08          # 坏切换固定惩罚
        non_standard_action_penalty: float = -0.15 # 非标准动作惩罚
        
        # ---- Reward裁剪范围 ----
        reward_clip_min: float = -1.5              # 最小reward值
        reward_clip_max: float = 2.0               # 最大reward值
    
    class BusinessWeightConfig:
        """业务类型权重配置 (P2级: 统一权重来源)"""
        
        # 统一的业务权重 (解决action定义与reward不一致问题)
        weights: dict = {
            0: 2.0,   # 控制信令 (高优先级)
            1: 2.5,   # 视频回传 (最高优先级)
            2: 1.5,   # 环境监测 (普通优先级)
        }
        default_weight: float = 2.0                # 默认权重
        
        # 动作定义中的SINR/Capacity权重 (与biz_weights联动)
        action_sinr_capacity_weights: dict = {
            0: (0.8, 0.2),   # 控制信令: 重SINR轻容量
            1: (0.3, 0.7),   # 视频回传: 轻SINR重容量
            2: (0.5, 0.5),   # 环境监测: 均衡
        }
    
    class LoadAdaptiveConfig:
        """负载自适应因子配置 (P2级)"""
        
        # 负载区间阈值
        low_load_threshold: float = 0.60           # 低负载阈值 (<60%)
        medium_low_threshold: float = 0.75         # 中低负载阈值 (60-75%)
        normal_load_threshold: float = 0.90         # 正常负载阈值 (75-90%)
        
        # 切换动作的负载因子
        low_load_factor: float = 1.8               # 低负载时强力鼓励切换
        medium_low_factor: float = 1.4             # 中低负载时适度增强
        normal_load_factor: float = 1.0            # 正常负载时不调整
        high_load_factor: float = 0.8              # 高负载时保守策略
        
        # 留守动作的负载惩罚/奖励
        stay_low_load_punish_threshold: float = 0.80   # 低负载留守惩罚阈值
        stay_low_load_sat_threshold: float = 0.70      # 低负载下可接受的最低sat
        stay_low_load_punish_max: float = -0.20         # 最大留守惩罚
        stay_high_load_reward_threshold: float = 0.95   # 高负载奖励阈值
        stay_high_load_sat_threshold: float = 0.70      # 高负载下可接受的最低sat
        stay_high_load_reward: float = 0.10             # 高负载留守奖励
        
        # 全局负载均衡惩罚 (V13)
        load_balance_penalty_scale: float = 2.0     # 负载均衡惩罚权重α [1.0, 3.0]
    
    class TargetGapConfig:
        """目标差距惩罚配置 (V13改进)"""
        
        control_signal_weight: float = 2.0         # 控制信令业务目标差距权重
        video_weight: float = 2.5                  # 视频回传业务目标差距权重
        environment_weight: float = 1.5            # 环境监测业务目标差距权重
    
    class ConnectionPenaltyConfig:
        """连接状态惩罚配置"""
        
        disconnect_new_penalty: float = -4.0       # 新断连惩罚
        disconnect_continuous_penalty: float = -1.0 # 持续断连惩罚
    
    class EnvironmentConfig:
        """环境参数配置 (P2级)"""
        
        # 切换延迟模拟
        base_handover_latency_ms: float = 50.0     # 基础切换延迟(毫秒)
        load_latency_coefficient: float = 0.5      # 基站负载对延迟的影响系数
        
        # QoS违规判定
        qos_violation_threshold: float = 0.4       # 触发QoS违规的满意度阈值
        
        # Observation Normalizer初始化
        normalizer_init_mean: float = 0.0          # RunningMeanStd初始均值
        normalizer_init_var: float = 1.0           # RunningMeanStd初始方差
        normalizer_decay: float = 0.99             # RunningMeanStd衰减率
    
    class TrainingConfig:
        """训练参数配置"""
        
        # ---- 早停参数 (V4) ----
        early_stop_window: int = 40                # 综合评分早停窗口大小 (原120)
        early_stop_min_delta: float = 0.001        # 最小改善阈值 (原0.0005)
        warmup_ratio: float = 0.1                 # warmup期占比 (原0.25)
        
        # ---- 综合评分权重 (V4) ----
        composite_weights: dict = {
            'satisfaction': 0.35,                  # 满意度权重
            'connected_ratio': 0.25,               # 连接保持率权重
            'load_balance': 0.15,                  # 负载均衡权重
            'switch_success': 0.15,                # 切换成功率权重
            'critical_sat': 0.10,                  # 关键业务满意度权重
        }
        
        # ---- Seed Randomization (V14) ----
        prime_offset: int = 1009                   # 质数偏移量
        max_jitter: int = 100                      # 最大随机抖动范围
        
        # ---- Domain Randomization (V20) ----
        dr_capacity_low_scale: float = 0.88        # DR容量下限比例
        dr_capacity_high_scale: float = 1.12       # DR容量上限比例
        
        # ---- 模型保存 ----
        save_interval: int = 50                    # latest模型保存间隔(episodes)
    
    @classmethod
    def validate_config(cls):
        """验证配置合理性"""
        errors = []
        
        # 验证Reward不等式: excellent_switch > stay > neutral_switch
        rc = cls.RewardConfig
        excellent_max = rc.excellent_switch_base + rc.excellent_switch_coeff * 0.2
        stay_max = rc.stay_base_reward + rc.stay_bonus_scale * 0.07
        neutral_max = rc.micro_positive_base + rc.micro_positive_coeff * 0.03
        
        if not (excellent_max > stay_max > neutral_max):
            errors.append(f"Reward不等式不满足: excellent({excellent_max:.2f}) > stay({stay_max:.2f}) > neutral({neutral_max:.2f})")
        
        # 验证综合评分权重和为1.0
        tc = cls.TrainingConfig
        weight_sum = sum(tc.composite_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"综合评分权重和不等于1.0: {weight_sum:.3f}")
        
        # 验证负载因子单调性
        lac = cls.LoadAdaptiveConfig
        if not (lac.low_load_factor >= lac.medium_low_factor >= lac.normal_load_factor >= lac.high_load_factor):
            errors.append("负载因子应随负载增加而递减")
        
        if errors:
            print("[ERROR] MAPPOConfig验证失败:")
            for err in errors:
                print(f"  - {err}")
            return False
        else:
            print("[OK] MAPPOConfig验证通过")
            return True


# 创建全局单例便于访问
mappo_config = MAPPOConfig()

# 启动时验证配置
if __name__ != '__main__':  # 避免在导入时打印
    MAPPOConfig.validate_config()
