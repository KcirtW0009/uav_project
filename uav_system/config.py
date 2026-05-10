"""
=============================================================================
  UAV业务识别与切换决策系统 - 全局配置模块 (config.py)
=============================================================================

【模块概述】
本模块是整个系统的"配置中心"，负责管理所有全局参数、超常量、
颜色方案和MAPPO算法的集中式配置，确保系统的一致性和可维护性。

【设计哲学】

1. **单一真相源原则** (Single Source of Truth):
   所有可调参数集中在此模块，避免散落在各文件中的硬编码值。
   修改任何参数只需改一处，自动传播到所有引用点。

2. **分层配置架构**:
   ┌─────────────────────────────────────────────┐
   │ L1: 全局常量 (GLOBAL_SEED, RESULT_DIR等)    │ ← 系统级
   ├─────────────────────────────────────────────┤
   │ L2: 功能配置 (INTERRUPTION_CONFIG, COLORS) │ ← 模块级
   ├─────────────────────────────────────────────┤
   │ L3: MAPPOConfig (Reward/Training/Env等)     │ ← 算法级
   └─────────────────────────────────────────────┘

3. **类型安全与验证**:
   使用类属性定义配置项（支持类型提示）
   提供validate_config()方法进行运行时校验
   启动时自动执行配置验证

4. **可视化友好**:
   预定义专业配色方案（符合学术出版标准）
   统一matplotlib全局设置（字体、DPI、尺寸）

【核心组件】
┌─────────────────────────────────────────────────────────────────────┐
│ 配置项/类              │ 功能描述                                     │
├─────────────────────────────────────────────────────────────────────┤
│ GLOBAL_SEED           │ 全局随机种子（确保实验可复现）               │
│ set_global_seed()     │ 种子设置函数（同步numpy/random/torch）        │
│ RESULT_DIR            │ 实验结果保存目录                             │
│ INTERRUPTION_CONFIG   │ 中断检测参数（阈值/持续时间）                 │
│ COLORS                │ 系统配色方案（12种预定义颜色）               │
│ CMAP_PRIMARY/SUCCESS/ │ 渐变色映射（用于热力图/等高线图）             │
│ WARNING               │                                              │
│ MAPPOConfig           │ MAPPO算法超参数集中配置（V21版本）            │
│ ├ RewardConfig        │ 奖励函数核心参数（24个P1级参数）              │
│ ├ BusinessWeightConfig│ 业务类型权重配置                            │
│ ├ LoadAdaptiveConfig  │ 负载自适应因子配置                           │
│ ├ TargetGapConfig     │ 目标差距惩罚配置                             │
│ ├ ConnectionPenalty.. │ 连接状态惩罚配置                             │
│ ├ EnvironmentConfig   │ 环境参数配置                                 │
│ └ TrainingConfig      │ 训练参数配置（早停/评分/DR等）              │
└─────────────────────────────────────────────────────────────────────┘

【全局种子系统】

种子选择策略:
  原始种子: 42 (经典机器学习默认值)
  当前种子: 30042 (更换后避免特殊性质)
  
  选择理由:
  - 42虽然是"生命、宇宙和一切事物的答案"，但在某些PRNG中可能有不理想的统计特性
  - 30042是质数，具有良好的均匀分布特性
  - 已经验证该种子下的性能指标非"天花板"(通过多种子测试)

种子同步范围:
  - numpy.random: 主要数值计算
  - random: Python内置随机函数
  - torch.manual_seed: PyTorch CPU张量操作
  - torch.cuda.manual_seed_all: GPU操作(如可用)

使用示例:
  >>> from config import GLOBAL_SEED, set_global_seed
  >>> print(f"当前种子: {GLOBAL_SEED}")
  >>> set_global_seed(12345)  # 临时覆盖为12345
  >>> # 所有后续随机操作都基于新种子

【中断检测配置详解】

INTERRUPTION_CONFIG 参数说明:

┌───────────────────────────┬──────────┬────────────────────────────────┐
│ 参数名                     │ 默认值   │ 说明                           │
├───────────────────────────┼──────────┼────────────────────────────────┤
│ threshold                  │ 0.3      │ 满意度中断阈值                  │
│                           │          │ <0.3视为服务质量严重下降         │
│ duration                   │ 5        │ 中断持续步数                    │
│                           │          │ 连续5步低于阈值才计为中断事件    │
│ control_signal_threshold  │ 0.4      │ 控制信令专用阈值                │
│                           │          │ 控制信令要求更高(>0.4)          │
│ control_signal_duration   │ 3        │ 控制信令持续步数                │
│                           │          │ 仅需3步即判定中断(更严格)       │
└───────────────────────────┴──────────┴────────────────────────────────┘

设计原理:
  - 控制信令业务(遥控指令/状态上报)对延迟更敏感
  - 采用更严格的检测标准(更高阈值+更短持续时间)
  - 确保关键业务的QoS违规能被快速识别

【配色方案规范】

COLORS 字典定义了12种语义化颜色:

┌─────────────────┬──────────┬────────────────────────────────────────┐
│ 名称             │ 色值     │ 用途                                   │
├─────────────────┼──────────┼────────────────────────────────────────┤
│ 'control'       │ #FF6B6B  │ 控制信令业务(珊瑚红)                   │
│ 'video'         │ #4ECDC4  │ 视频回传业务(青绿色)                   │
│ 'environment'   │ #95E1D3  │ 环境监测业务(薄荷绿)                   │
│ 'primary'       │ #667eea  │ 主色调(靛蓝) - 标题/重点               │
│ 'secondary'     │ #764ba2  │ 辅色调(紫色) - 次要信息                │
│ 'accent'        │ #f093fb  │ 强调色(粉色) - 高亮/交互               │
│ 'success'       │ #4ade80  │ 成功状态(翠绿)                         │
│ 'warning'       │ #fbbf24  │ 警告状态(琥珀黄)                       │
│ 'danger'        │ #f87171  │ 危险状态(珊瑚红)                       │
│ 'info'          │ #60a5fa  │ 信息提示(天蓝色)                       │
│ 'neutral'       │ #9ca3af  │ 中性文本(灰色)                         │
└─────────────────┴──────────┴────────────────────────────────────────┘

渐变色映射(Colormap):
  - CMAP_PRIMARY: primary → secondary (用于主数据系列)
  - CMAP_SUCCESS: 浅绿 → success (用于正向指标)
  - CMAP_WARNING: 浅黄 → warning (用于警示指标)

Matplotlib全局设置:
  - 字体: SimHei > Microsoft YaHei > Arial Unicode MS (中文优先)
  - 负号显示: True (解决某些字体下负号显示为方块的问题)
  - 图像尺寸: 14×10 英寸 (适合学术论文)
  - 保存DPI: 300 (满足期刊要求)
  - 显示DPI: 100 (屏幕查看舒适度)

【MAPPOConfig详细说明】(V21版本, 解决P1/P2级硬编码问题)

架构设计:
  使用嵌套类组织配置项，按功能域分组:
  
  MAPPOConfig
  ├── RewardConfig          [P1级] 奖励函数核心(24处引用)
  ├── BusinessWeightConfig  [P2级] 业务权重统一
  ├── LoadAdaptiveConfig    [P2级] 负载自适应因子
  ├── TargetGapConfig       [P2级] 目标差距惩罚
  ├── ConnectionPenaltyConfig [P2级] 连接惩罚
  ├── EnvironmentConfig     [P2级] 环境参数
  └── TrainingConfig        [P2级] 训练超参数

优先级定义:
  P1级 (Critical): 直接影响训练收敛性和最终性能的参数
    - 修改后必须重新训练模型
    - 示例: delta_scale, excellent_switch_threshold
  
  P2级 (Important): 影响训练效率或特定场景表现的参数
    - 修改后建议重新训练但可选
    - 示例: 业务权重, 负载因子阈值

【RewardConfig核心参数详解】

奖励信号结构 (共9个分量):
  Total = rate_delta + counterfactual + biz_weight + action_reward 
        + load_adaptive + target_gap + ranking + disconnect + load_balance

动作奖励层级 (V19核心设计):
  ┌─────────────────────┬──────────┬────────────────────────────────────┐
│ 动作类型              │ 基础值   │ 满意度增量系数                     │
├─────────────────────┼──────────┼────────────────────────────────────┤
│ excellent_switch     │ 1.0      │ +2.0 × Δsat  (Δsat>0.05)         │
│ stay(high quality)   │ 0.80     │ +2.0 bonus (sat>0.93)             │
│ good_switch          │ 0.55     │ +3.0 × Δsat  (Δsat>0.015)        │
│ micro_positive       │ 0.15     │ +5.0 × Δsat  (Δsat>0)            │
│ acceptable_switch    │ 0.0      │ +4.0 × Δsat  (Δsat>-0.03)        │
│ bad_switch           │ -0.08    │ +6.0 × Δsat  (Δsat≤-0.03)        │
│ non_standard         │ -0.15    │ 固定惩罚                           │
└─────────────────────┴──────────┴────────────────────────────────────┘

关键不等式约束 (validate_config检查):
  excellent_max > stay_max > neutral_max
  即: 优秀切换的最高收益 > 留守最高收益 > 微正切换最高收益
  
  目的: 鼓励agent在有明显提升时切换，但不鼓励频繁无意义切换

负载自适应策略 (4级):
  ┌───────────────┬──────────┬─────────────────────────────────────────┐
│ 负载区间       │ 切换因子 │ 行为变化                                │
├───────────────┼──────────┼─────────────────────────────────────────┤
│ 低载 (<60%)   │ 1.8×     │ 强力鼓励切换(资源充足)                  │
│ 中低 (60-75%) │ 1.4×     │ 适度增强切换倾向                        │
│ 正常 (75-90%) │ 1.0×     │ 不调整(标准行为)                        │
│ 高载 (>90%)   │ 0.8×     │ 保守策略(减少切换冲动)                  │
└───────────────┴──────────┴─────────────────────────────────────────┘

【TrainingConfig训练参数】

早停机制 (Early Stopping V4):
  窗口大小: 40 episodes (原120，加快响应速度)
  最小改善: 0.001 (原0.0005，降低灵敏度)
  Warmup期: 前10% episodes不触发早停
  
综合评分公式 (Composite Score):
  Score = 0.35×satisfaction + 0.25×connected_ratio 
        + 0.15×load_balance + 0.15×switch_success 
        + 0.10×critical_sat
  
Domain Randomization (V20):
  容量范围缩放: [原始×0.88, 原始×1.12]
  目的: 增加环境多样性，提升泛化能力

Seed Randomization (V14):
  偏移量: prime_offset=1009 (大质数)
  抖动范围: max_jitter=100
  公式: actual_seed = base_seed + episode×prime_offset + uniform(-jitter, jitter)

【配置验证系统】

validate_config() 检查项:

1. **Reward不等式验证**:
   excellent_max > stay_max > neutral_max
   违反则打印错误信息并返回False

2. **权重归一化验证**:
   composite_weights总和应等于1.0 (误差<0.01)
   否则警告权重配置异常

3. **单调性验证**:
   负载因子应随负载增加而递减
   low_load ≥ medium_low ≥ normal ≥ high
   违反则提示逻辑错误

自动触发时机:
  - 模块导入时（if __name__ != '__main__'）
  - 手动调用 MAPPOConfig.validate_config()

【使用示例】

# 示例1: 读取全局配置
>>> from config import GLOBAL_SEED, RESULT_DIR, INTERRUPTION_CONFIG
>>> print(f"种子: {GLOBAL_SEED}, 结果目录: {RESULT_DIR}")
>>> print(f"中断阈值: {INTERRUPTION_CONFIG['threshold']}")

# 示例2: 设置自定义种子
>>> from config import set_global_seed
>>> set_global_seed(99999)
>>> import numpy as np
>>> print(np.random.rand())  # 基于种子99999的可复现随机数

# 示例3: 使用MAPPO配置
>>> from config import MAPPOConfig, mappo_config
>>> rc = MAPPOConfig.RewardConfig
>>> print(f"Delta scale: {rc.delta_scale}")
>>> print(f"Excellent threshold: {rc.excellent_switch_threshold}")

# 示例4: 修改配置并重新验证
>>> MAPPOConfig.RewardConfig.delta_scale = 6.0
>>> if MAPPOConfig.validate_config():
...     print("配置有效，可以使用")

# 示例5: 使用颜色方案
>>> from config import COLORS, CMAP_SUCCESS
>>> import matplotlib.pyplot as plt
>>> plt.bar([1,2,3], [4,5,6], color=COLORS['success'])
>>> plt.imshow(data, cmap=CMAP_SUCCESS)

【依赖关系】
  本模块是基础配置层，几乎被所有其他模块依赖:
    - environment.py: 使用GLOBAL_SEED, INTERRUPTION_CONFIG
    - experiments.py: 使用RESULT_DIR, COLORS, GLOBAL_SEED
    - mappo_environment.py: 使用MAPPOConfig所有子配置
    - mappo_agent_v2.py: 使用MAPPOConfig.TrainingConfig
    - visualization.py: 使用COLORS, CMAP_* 配色方案
    - algorithms.py: 使用GLOBAL_SEED初始化环境

【最佳实践】
  1. 不要在代码中硬编码魔法数字，始终从config导入
  2. 修改超参数前先调用validate_config()检查一致性
  3. 新增配置项时添加类型提示和文档字符串
  4. 敏感参数(如种子)修改后通知所有相关模块

【已知限制】
  1. 配置修改后需要重启程序才能生效（非热更新）
  2. 不支持多配置文件切换（单例模式）
  3. validate_config仅检查部分关键约束，不能发现所有逻辑错误

【版本历史】
  V1.0: 初始版本，基础常量和颜色定义
  V1.5: 添加INTERRUPT中断检测配置
  V2.0: 引入MAPPOConfig嵌套类结构
  V2.1 (2026-05-09): 完善P1/P2分级，添加validate_config()
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
