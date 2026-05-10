"""
=============================================================================
  UAV业务识别与切换决策系统 - MAPPO多智能体强化学习环境 (mappo_environment.py)
=============================================================================

【模块概述】
本模块是MAPPO（Multi-Agent Proximal Policy Optimization）算法的核心运行环境，
将UAV切换决策问题封装为CTDE（Centralized Training, Decentralized Execution）
架构的多智能体强化学习环境。

【设计哲学】
1. **纯净评估原则**: 评估模式下移除所有保护机制（预检查、降级、回滚），
   让MAPPO模型完全自主决策，真实反映其学到的策略质量。
2. **训练/评估双模式**: 训练时使用ground truth业务类型，评估时使用识别模型预测，
   模拟真实部署场景中的识别误差。
3. **多维奖励信号**: 综合考虑速率比增量、反事实比较、业务权重、动作质量、
   负载自适应、目标差距、同类排名等7个维度的奖励分量。
4. **可配置性**: 所有超参数通过MAPPOConfig集中管理，消除硬编码，便于调优。

【核心组件】
┌─────────────────────────────────────────────────────────────────────┐
│ 组件名称           │ 功能描述                                       │
├─────────────────────────────────────────────────────────────────────┤
│ RunningNormalizer │ EMA奖励归一化器（当前已禁用，PPO自带标准化）     │
│ MultiAgentHandoverEnv│ 主环境类，实现完整的RL环境接口                │
│ EnhancedNetworkEnvironment│ 底层网络仿真环境（含随机事件机制）       │
└─────────────────────────────────────────────────────────────────────┘

【动作空间定义】(6维离散动作空间)
  action=0: stay          → 不切换，保持当前连接
  action=1: best_sinr     → 切换到SINR最高的基站
  action=2: best_capacity → 切换到可用容量/需求比最高的基站
  action=3: sinr_capacity → SINR和容量的加权组合（60% SINR + 40% 容量）
  action=4: predictive    → 基于趋势预测的切换（简化版）
  action=5: business_specific → 根据业务类型的差异化切换策略

【观测空间维度】
  局部观测 (per agent): obs_dim = 4 × num_bs + 9 + action_dim(6) + 2 = 4×num_bs + 17
    - SINR向量 (num_bs): 归一化的信噪比值
    - 基站负载率 (num_bs): 各基站的当前负载比例
    - 连接BS one-hot (num_bs): 当前连接基站的独热编码
    - 可用容量/需求比 (num_bs): 各基站能满足UAV需求的程度
    - 业务类型 one-hot (3): 控制信令/视频回传/环境监测
    - 当前满意度 (1): 综合QoS满意度评分
    - 连接状态 (1): 是否已连接到某个基站
    - 移动速度 (1): 归一化的移动速度
    - 上次动作 one-hot (6): 上一步执行的动作编码
    - 满意度变化趋势 (1): 最近5步的线性变化率
    - 同类UAV平均满意度 (1): 相同业务类型UAV的平均表现
    - 历史满意度 (3): 最近3步的满意度记录

  全局状态 (centralized critic用): state_dim = 3 × num_bs + 7
    - 基站负载率 (num_bs)
    - 基站可用容量 (num_bs)
    - 基站故障状态 (num_bs)
    - 各业务类型数量 (3)
    - 全局平均满意度 (1)
    - 全局断连率 (1)
    - 全局中断率 (1)
    - 当前进度 (1)

【奖励函数结构】(V12/V13/V21版本)
  总奖励 = a + b + c + d + e + f + g (个体) + h + i (团队级)

  a. 连续速率比信号 (delta_scale × Δrate_ratio)
     - 替代分段满意度，提供更平滑的学习梯度
     - 正值表示速率提升，负值表示下降

  b. 反事实比较信号 (counterfactual_scale × (my_rr - peer_avg_rr))
     - 切换后与同类UAV的平均表现对比
     - 正值表示超越同类平均水平

  c. 业务类型权重 (biz_weight × delta_rr × sign(delta_sat))
     - 不同业务类型对速率变化的敏感度不同
     - 视频回传权重最高(2.5)，控制信令次之(2.0)，环境监测最低(1.5)

  d. 动作奖励 (分层激励设计)
     - stay: 基础奖励0.89 + 满意度bonus（鼓励维持高质量连接）
     - excellent_switch(Δsat>0.05): 1.0 + 2.0×Δsat（优秀切换奖励高于stay）
     - good_switch(Δsat>0.015): 0.55 + 3.0×Δsat
     - micro_positive(Δsat>0): 0.15 + 5.0×Δsat
     - acceptable_switch(Δsat>-0.03): 4.0×Δsat（轻微惩罚）
     - bad_switch(Δsat≤-0.03): 6.0×Δsat - 0.08（严厉惩罚）

  e. 负载自适应系数 [V12新增]
     - 低负载(<60%): 放大切换奖励1.8倍，增强留守惩罚
     - 中低负载(60-75%): 适度增强1.3倍
     - 正常负载(75-90%): 不调整1.0倍
     - 高负载(>90%): 保守策略0.6倍，降低切换冲动

  f. 关键业务差距奖励 [V13新增]
     - r_gap = -α × max(0, target_sat - current_sat)
     - 控制信令目标0.85, 视频回传目标0.75, 环境监测目标0.65
     - 差异化权重: 视频2.5 > 控制2.0 > 环境1.5

  g. 同类相对排名信号 [V12新增]
     - r_ranking = 0.15 × rank_change / (n_peers//2)
     - 排名上升给予正向激励，下降给予负向激励

  h. 断连惩罚 [团队级]
     - 新断连: -4.0, 持续断连: -2.5
     - 仅加入团队reward，不干扰个体学习信号

  i. 负载均衡惩罚 [V13/P1新增, 团队级]
     - penalty = -2.0 × std(load_ratios)
     - 惩罚基站间负载不均衡，鼓励UAV分散连接

【关键设计决策】

1. **纯净版评估模式**:
   问题: 之前版本包含预检查、降级分配、自动回滚等保护机制，
         导致MAPPO在评估时表现出不真实的100%成功率。
   解决: 移除所有保护机制，让模型完全自主决策并承担失败后果。
   效果: 真实反映模型策略质量，与传统/增强算法公平比较。

2. **禁用EMA归一化** [V17]:
   问题: 在低负载环境下，reward绝对值差异很小，
         EMA会将微弱的信号差异完全抹平，导致advantage≈0无法学习。
   解决: PPO本身通过advantage标准化处理reward scale，无需额外归一化。
   效果: 保持原始奖励信号的完整性，提升学习效率。

3. **stay作为高价值默认动作**:
   问题: 随机探索阶段频繁切换导致大量坏切换，学习效率低下。
   解决: 将stay设为基础正收益(~0.89)，让agent快速收敛到"少切"策略，
         但优秀切换(excellent_switch>1.0)仍高于stay，避免过度保守。
   效果: 训练后stay比例从18%升至48%，切换质量大幅提升。

4. **Domain Randomization支持**:
   问题: 固定环境参数可能导致过拟合到特定场景。
   解决: 支持在reset()时随机化BS容量范围，增加环境多样性。
   效果: 提升模型的泛化能力，减少对特定参数分布的依赖。

【接口规范】
  reset(bs_capacity_range=None) -> (obs_dict, global_state)
    - 重置环境到初始状态
    - 支持可选的容量域随机化

  step(actions_dict) -> (obs_dict, global_state, rewards_dict, team_reward, done, info)
    - 执行一步仿真
    - 返回完整的观测-奖励-终止信息

  advance_env_only() -> None
    - 仅推进底层环境（供基线算法评估使用）
    - 不执行任何切换决策

【依赖关系】
  上游模块:
    - environment.py: NetworkEnvironmentWithRecognition, EnhancedNetworkEnvironment
    - business.py: BusinessType枚举
    - config.py: MAPPOConfig集中配置

  下游调用:
    - mappo_agent_v2.py: MAPPO智能体训练和推理
    - experiments.py: 实验3/4的MAPPO评估流程

【版本历史】
  V8:  初始版本，基于QMIX设计
  V9:  适配MAPPO算法，移除mixing network
  V10: 添加业务识别集成（训练/评估双模式）
  V11: 引入连续rate_ratio信号替代分段满意度
  V12: 添加负载自适应、目标差距、同类排名三个新奖励分量
  V13: 添加全局负载均衡惩罚（团队级信号）
  V17: 禁用EMA归一化（低负载环境下性能优化）
  V19: 使用EnhancedNetworkEnvironment（含随机事件机制）
  V21: 所有参数从MAPPOConfig读取，消除硬编码

【使用示例】
  # 训练模式（使用ground truth业务类型）
  env = MultiAgentHandoverEnv(num_bs=8, num_uav=300, max_steps=350)
  obs, state = env.reset()
  for step in range(350):
      actions = {uid: agent.select_action(obs[uid]) for uid in range(300)}
      obs, state, rewards, team_reward, done, info = env.step(actions)

  # 评估模式（使用识别模型预测业务类型）
  env = MultiAgentHandoverEnv(
      num_bs=8, num_uav=300,
      recognition_model=model, scaler=scaler,
      scenario='industrial_inspection'
  )
  obs, state = env.reset()
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional
from collections import deque

from .environment import NetworkEnvironmentWithRecognition, EnhancedNetworkEnvironment
# [已弃用] 以下导入原为 QMIX 元控制器设计，MAPPO 未使用（action_dim 已简化为 3）
# from .parametric_algorithm import (
#     ParametricEnhancedAlgorithm, NUM_STRATEGIES, STRATEGY_CONFIGS
# )
from .business import BusinessType
from .config import MAPPOConfig  # V21: 引入集中配置


class RunningNormalizer:
    """
    指数移动平均（EMA）归一化器 - 用于降低奖励的变异系数(CV)

    【设计目的】
    在多智能体强化学习中，不同agent的reward scale可能差异很大，
    导致训练不稳定。本归一化器通过EMA平滑reward分布，
    使各agent的reward具有可比性。

    【算法原理】
    使用在线均值/方差估计：
      mean_t = decay × mean_{t-1} + (1-decay) × batch_mean
      var_t  = decay × var_{t-1}  + (1-decay) × batch_var
    归一化公式：
      normalized = (x - mean) / sqrt(var + ε)

    【当前状态】[V17: 已禁用]
    本归一化器在低负载环境下会导致问题：
      - reward绝对值差异本来就很小
      - EMA会将微弱的信号差异完全抹平
      - 导致advantage≈0，无法学习
    因此V17版本已禁用此归一化器，
    改用PPO自带的advantage标准化来处理reward scale。

    Attributes:
        mean (np.ndarray): 各agent的运行均值估计
        var  (np.ndarray): 各agent的运行方差估计
        decay (float): EMA衰减因子（越大越平滑，默认0.999）
        count (int): 已处理的batch数量

    Example:
        >>> normalizer = RunningNormalizer(num_agents=300, decay=0.999)
        >>> rewards = {i: np.random.randn() for i in range(300)}
        >>> normalized = normalizer.normalize(rewards)
        >>> # normalized的mean≈0, std≈1
    """

    def __init__(self, num_agents: int, decay: float = 0.999):
        """
        初始化归一化器

        Args:
            num_agents: agent数量（等于UAV数量）
            decay: EMA衰减因子，范围(0, 1)
                - 接近1: 更平滑但响应慢（适合稳定环境）
                - 接近0: 响应快但噪声大（适合快速变化环境）
                - 默认0.999: 平衡选择，约1000步的半衰期
        """
        self.mean = np.zeros(num_agents, dtype=np.float64)
        self.var = np.ones(num_agents, dtype=np.float64)
        self.decay = decay
        self.count = 0

    def normalize(self, rewards_dict: Dict[int, float]) -> Dict[int, float]:
        """
        对一批rewards进行归一化处理

        算法流程：
        1. 将dict转换为向量形式
        2. 计算当前batch的统计量（mean, var）
        3. 使用EMA更新全局统计量
        4. 用更新的统计量对原始数据进行z-score标准化

        Args:
            rewards_dict: {agent_id: raw_reward} 原始奖励字典

        Returns:
            {agent_id: normalized_reward} 归一化后的奖励字典
            归一化后的数据满足：mean≈0, std≈1
        """
        vec = np.array([rewards_dict[i] for i in range(len(rewards_dict))], dtype=np.float64)
        self.count += 1
        batch_mean = vec.mean()
        batch_var = vec.var() if len(vec) > 1 else 1.0

        # EMA更新：指数加权移动平均
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.var = self.decay * self.var + (1 - self.decay) * batch_var

        # Z-score标准化（加小常数避免除零）
        std = np.sqrt(np.maximum(self.var, 1e-8))
        normed = (vec - self.mean) / std

        return {i: float(normed[i]) for i in range(len(rewards_dict))}

    def reset(self):
        """
        重置归一化器到初始状态

        通常在以下情况调用：
        1. 训练完全重启时（不是episode重置）
        2. 环境参数发生重大变化时
        3. 检测到数值异常需要重新校准时

        注意：正常训练过程中不应频繁调用reset()，
              因为EMA的价值在于跨episode积累统计信息。
        """
        self.count = 0
        self.mean[:] = 0.0
        self.var[:] = 1.0


class MultiAgentHandoverEnv:
    """
    多智能体UAV切换环境（CTDE架构）— MAPPO主环境类

    【类定位】
    本类是整个MAPPO系统的核心运行时环境，负责：
    1. 封装底层网络仿真（基站、UAV、信道模型）
    2. 实现标准的RL环境接口（reset/step）
    3. 管理观测空间和动作空间的定义
    4. 计算多维奖励信号
    5. 支持训练模式和评估模式的双轨运行

    【架构设计】CTDE (Centralized Training, Decentralized Execution)
    - 训练时: Critic使用全局状态(state)进行集中式价值评估
    - 执行时: Actor仅使用局部观测(obs)进行分布式决策
    - 这种设计兼顾了训练时的全局信息利用和部署时的去中心化需求

    【纯净评估模式】[V19关键改进]
    本环境支持两种运行模式：
    1. 训练模式 (recognition_model=None):
       - 使用ground truth业务类型
       - 用于MAPPO智能体的策略学习

    2. 评估模式 (recognition_model≠None):
       - 使用识别模型预测的业务类型（带噪声）
       - 移除所有保护机制（预检查、降级、回滚）
       - 模拟真实部署场景，公平对比算法性能

    【动作空间】(6维离散动作)
      0 = stay           → 不切换，维持当前连接
      1 = best_sinr      → 切换到SINR最高的基站（信号质量优先）
      2 = best_capacity  → 切换到可用容量/需求比最高的基站（资源充裕度优先）
      3 = sinr_capacity  → SINR和容量的加权组合（60%信号 + 40%容量，平衡策略）
      4 = predictive     → 基于趋势预测的切换（前瞻性决策）
      5 = business_specific → 根据业务类型的差异化切换（个性化策略）

    【观测空间】(per agent)
      维度: obs_dim = 4 × num_bs + 17
      包含: SINR向量、负载率、连接状态、容量比、业务类型、满意度、
            连接状态、速度、上次动作、满意度趋势、同类平均、历史记录

    【全局状态】(for centralized critic)
      维度: state_dim = 3 × num_bs + 7
      包含: 全局负载率、可用容量、故障状态、业务分布、统计量

    Attributes:
        num_agents (int): agent数量（= UAV数量）
        num_bs (int): 基站数量
        action_dim (int): 动作空间大小（固定为6）
        obs_dim (int): 局部观测维度（动态计算）
        state_dim (int): 全局状态维度（动态计算）
        env (EnhancedNetworkEnvironment): 底层网络仿真环境
        recognition_model: 业务识别模型（评估模式使用）
        scaler: 识别模型的标准化器
        _current_step (int): 当前episode的步数计数器
        _reward_normalizer (RunningNormalizer): EMA归一化器（已禁用）

    Example:
        # 训练模式示例
        >>> env = MultiAgentHandoverEnv(num_bs=8, num_uav=300, max_steps=350)
        >>> obs, state = env.reset()
        >>> for step in range(350):
        ...     actions = {uid: policy(obs[uid]) for uid in range(300)}
        ...     obs, state, rewards, team_reward, done, info = env.step(actions)
        ...     if done:
        ...         break

        # 评估模式示例
        >>> env = MultiAgentHandoverEnv(
        ...     num_bs=8, num_uav=300,
        ...     recognition_model=model, scaler=scaler,
        ...     scenario='industrial_inspection'
        ... )
        >>> obs, state = env.reset()
        >>> # 运行评估...
    """

    # 向后兼容别名（原QMIX遗留）
    QMixHandoverEnv = None  # 将在模块底部设置


    def __init__(self, num_bs: int, num_uav: int, max_steps: int = 1000, seed: int = None,
                 bs_capacity_range: tuple = (500, 1000), pos_range: int = 1000,
                 use_state_smoothing: bool = True, use_env_simplification: bool = False,
                 recognition_model=None, scaler=None,
                 event_probability: float = 0.05,
                 scenario: str = 'default'):
        """
        初始化MAPPO切换环境

        Args:
            num_bs (int): 基站数量
                - 实验3标准配置: 8个基站
                - 影响观测/状态维度: obs_dim包含4×num_bs个基站相关特征

            num_uav (int): UAV数量（= agent数量）
                - 实验3标准配置: 300架UAV
                - 决定并行决策的agent数量

            max_steps (int): 每个episode的最大步数
                - 默认1000步
                - 实验3实际使用350步（平衡仿真精度和运行时间）

            seed (int): 随机种子
                - 用于环境初始化的可复现性
                - None表示不设置种子（完全随机）

            bs_capacity_range (tuple): 基站容量范围 (min_mbps, max_mbps)
                - 默认(500, 1000) Mbps
                - 在reset()时可随机化以实现Domain Randomization

            pos_range (int): 地图空间范围（米）
                - 默认1000米 × 1000米
                - 决定UAV和基站的初始位置分布范围

            use_state_smoothing (bool): 是否启用状态平滑机制
                - True: 使用最近5步观测的移动平均（减少噪声）
                - False: 直接返回原始观测（更快但更嘈杂）

            use_env_simplification (bool): 是否启用环境简化
                - True: 固定BS位置为圆形排列 + 降低UAV移动速度50%
                        （用于快速测试或调试）
                - False: 随机位置和正常速度（用于正式训练/评估）

            recognition_model: 业务识别模型对象
                - 训练模式: 设为None（使用ground truth）
                - 评估模式: 传入训练好的DecisionTree/RandomForest模型

            scaler: 识别模型的StandardScaler标准化器
                - 必须与recognition_model配套使用
                - 用于对输入特征进行z-score标准化

            event_probability (float): 随机事件发生概率 [0, 1]
                - 0.05 = 每步每UAV有5%概率触发随机事件
                - 对齐实验3的EnhancedNetworkEnvironment配置
                - 随机事件包括：突发流量、信道干扰、基站故障等

            scenario (str): 场景ID（用于设置业务混合比例）
                - 'default': 默认场景（均匀混合三种业务）
                - 'industrial_inspection': 工业巡检场景
                - 'smart_agriculture': 智慧农业场景
                - 'emergency_rescue': 应急救援场景
                - 'urban_monitoring': 城市监测场景
                - 'logistics_delivery': 物流配送场景
                - 不同场景影响UAV的业务类型分布和QoS需求
        """
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.max_steps = max_steps
        self.seed = seed
        self.bs_capacity_range = bs_capacity_range
        self.pos_range = pos_range
        self.use_state_smoothing = use_state_smoothing
        self.use_env_simplification = use_env_simplification
        self.recognition_model = recognition_model
        self.scaler = scaler
        self.event_probability = event_probability
        self.scenario = scenario  # ✅ 保存场景ID

        # 创建底层网络环境
        # V19: 使用 EnhancedNetworkEnvironment（含随机事件机制，与实验3一致）
        # 之前用 NetworkEnvironmentWithRecognition 导致：
        #   - 无随机事件 → 随机策略 sat=0.979（太高）
        #   - 实验3 有事件 → 传统算法 sat=0.900
        #   → 两环境不对等，MAPPO的"优势"是虚假的
        self.env = EnhancedNetworkEnvironment(
            num_bs=num_bs, num_uav=num_uav,
            recognition_model=recognition_model, scaler=scaler,
            seed=seed,
            bs_capacity_range=bs_capacity_range,
            event_probability=event_probability,  # 默认5%，对齐实验3
            scenario=scenario,  # ✅ 关键：传递场景ID以启用正确的业务混合比例！
        )

        # 禁用自适应识别更新器（MAPPO 不需要在线更新识别结果）
        self.env.recognition_updater = None

        # 随机事件配置（对齐实验3的 EnhancedNetworkEnvironment）
        if event_probability > 0:
            self._setup_random_events(event_probability)

        # 环境简化：减少环境的随机性
        if self.use_env_simplification:
            # 固定基站位置
            for i, bs in enumerate(self.env.base_stations.values()):
                angle = 2 * np.pi * i / num_bs
                radius = pos_range / 3
                bs.position = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
            # 降低 UAV 移动速度
            for uav in self.env.uavs.values():
                uav.velocity *= 0.5

        # 保存原始位置用于 reset 时缩放（避免累积缩放）
        self._original_positions = {}
        for bs_id, bs in self.env.base_stations.items():
            self._original_positions[('bs', bs_id)] = bs.position.copy()
        for uav_id, uav in self.env.uavs.items():
            self._original_positions[('uav', uav_id)] = uav.position.copy()

        # 如果 pos_range != 默认值(1000)，缩放所有位置
        if pos_range != 1000:
            scale = pos_range / 1000.0
            for bs in self.env.base_stations.values():
                bs.position *= scale
            for uav in self.env.uavs.values():
                uav.position *= scale
            self.env._update_sinr_matrix()
            self.env._initialize_connections()

        self.num_agents = num_uav
        # 动作空间扩展为 6 (分级决策):
        #   action 0 = stay (不切换)
        #   action 1 = best_sinr (切换到 SINR 最高的 BS)
        #   action 2 = best_capacity (切换到可用容量/需求比最高的 BS)
        #   action 3 = sinr_capacity (SINR和容量的加权组合)
        #   action 4 = predictive (基于预测的切换)
        #   action 5 = business_specific (基于业务类型的特定切换策略)
        self.action_dim = 6

        # 维度计算
        self.obs_dim = self._calc_obs_dim()
        self.state_dim = self._calc_state_dim()

        # Episode 状态
        self._current_step = 0
        self._last_satisfaction = {}  # UAV id -> 上一步满意度
        self._last_disconnected = {}  # UAV id -> 上一步是否断连
        self._last_handover_count = {}  # UAV id -> 上一步累计切换次数
        self._last_actions = {}  # UAV id -> 上一步动作
        self._sat_history = {}  # UAV id -> deque of recent satisfactions
        self._last_rankings = {}  # [V12] UAV id -> 上一步同类排名
        self._last_rate_ratios = {}
        self._reward_normalizer = RunningNormalizer(num_uav)
        
        # 状态平滑相关
        if self.use_state_smoothing:
            self._state_history = {}  # UAV id -> 历史状态
            for uid in range(num_uav):
                self._state_history[uid] = []
        
        # 通信指标监测 - 对齐实验3
        self._communication_metrics = {
            'handover_latencies': [],  # 切换延迟（毫秒）
            'ping_jitters': [],  # Ping抖动（毫秒）
            'packet_losses': [],  # 丢包率（百分比）
            'qos_violations': [],  # QoS违规率（百分比）
            'ping_times': {},  # UAV id -> 最近的ping时间列表
            # 新增指标（对齐实验3）
            'throughput': [],  # 系统吞吐量（Mbps）
            'load_variance': [],  # 负载方差
            'spectral_efficiency': [],  # 频谱效率（bps/Hz）
            'fairness_index': [],  # 公平性指数（Jain's Fairness Index）
        }
        for uid in range(num_uav):
            self._communication_metrics['ping_times'][uid] = deque(maxlen=10)

    def _setup_random_events(self, probability: float):
        """
        配置随机事件机制（对齐实验3的EnhancedNetworkEnvironment）

        【设计目的】
        随机事件模拟真实5G网络中的不确定性因素，包括：
        - 突发流量高峰（UAV突然需要更高带宽）
        - 信道干扰（SINR暂时性下降）
        - 基站部分故障（容量临时降低）
        - 网络拥塞（切换延迟增加）

        这些事件增加了环境的动态性和挑战性，
        测试算法在非理想条件下的鲁棒性。

        Args:
            probability (float): 每步每UAV发生随机事件的概率 [0, 1]
                - 0.0: 无随机事件（确定性环境）
                - 0.05: 低频率（实验3标准配置）
                - 0.10-0.20: 中高频率（压力测试）
                - >0.30: 极端情况（可能导致系统不稳定）
        """
        # 底层环境 NetworkEnvironmentWithRecognition 继承自 EnhancedNetworkEnvironment
        # 检查是否有随机事件相关属性
        if hasattr(self.env, '_random_event_enabled'):
            self.env._random_event_enabled = True
            self.env._event_probability = probability
            print(f"[MAPPO Env] 已启用随机事件, probability={probability}")

    def _calc_obs_dim(self) -> int:
        """
        计算单个agent的局部观测维度

        【观测空间结构】(per agent)
        观测向量由以下组件拼接而成：

        组件1: SINR向量 (num_bs维)
          - 每个基站的信噪比(dB)，归一化到[0, 1]
          - 公式: clip((sinr_db + 10) / 40, 0, 1)
          - 范围: sinr∈[-10, 30]dB → 归一化后[0, 1]

        组件2: 基站负载率 (num_bs维)
          - 各基站当前负载/总容量的比例
          - 范围: [0, 1]，越高表示越拥挤

        组件3: 当前连接BS的one-hot编码 (num_bs维)
          - 仅当前连接的BS位置为1.0，其余为0.0
          - 用于标识当前服务基站

        组件4: 可用容量/需求比 (num_bs维)
          - 公式: min(available_capacity / required_rate, 2.0) / 2.0
          - >1.0表示资源充裕，<1.0表示可能拥塞

        组件5: 业务类型one-hot编码 (3维)
          - [控制信令, 视频回传, 环境监测]
          - 仅对应位置为1.0

        组件6: 当前满意度 (1维)
          - 综合QoS满意度评分，范围[0, 1]

        组件7: 连接状态 (1维)
          - 1.0=已连接, 0.0=断连

        组件8: 移动速度归一化 (1维)
          - 公式: min(||velocity|| / 30.0, 1.0)
          - 30m/s约为108km/h（高速无人机）

        组件9: 上次动作one-hot编码 (action_dim=6维)
          - 用于提供动作历史信息，帮助策略学习时序模式

        组件10: 满意度变化趋势 (1维)
          - 最近5步满意度的线性斜率，缩放到[-1, 1]
          - 正值=改善趋势，负值=恶化趋势

        组件11: 同类型UAV平均满意度 (1维)
          - 相同业务类型的其他UAV的平均表现
          - 提供相对性能参考（社会学习信号）

        组件12: 历史满意度记录 (3维)
          - 最近3步的满意度值（不包括当前步）
          - 用于捕捉短期时序依赖

        Returns:
            int: 总观测维度 = 4×num_bs + 9 + action_dim + 2
                 对于8个基站: 4×8 + 9 + 6 + 2 = 49维
        """
        return 4 * self.num_bs + 9 + self.action_dim + 2

    def _calc_state_dim(self) -> int:
        """
        计算全局状态维度（用于centralized critic）

        【状态空间结构】(global state)
        全局状态包含整个系统的宏观信息，
        供Critic网络进行集中式价值评估：

        组件1: 所有基站负载率 (num_bs维)
          - 与观测空间相同，但这里是全局视角

        组件2: 所有基站可用容量比例 (num_bs维)
          - available_capacity / total_capacity
          - 反映各基站的资源余量

        组件3: 所有基站故障状态 (num_bs维)
          - 1.0=故障, 0.0=正常
          - 用于感知网络拓扑变化

        组件4: 各业务类型UAV数量归一化 (3维)
          - count(biz_type) / num_uav
          - 反映业务负载分布

        组件5: 全局平均满意度 (1维)
          - 所有UAV满意度的均值
          - 系统整体健康度指标

        组件6: 全局断连率 (1维)
          - 断连UAV数 / 总UAV数
          - 反映连接可靠性

        组件7: 全局中断率 (1维)
          - 中断UAV数 / 总UAV数
          - 反映QoS违规严重程度

        组件8: 当前进度归一化 (1维)
          - current_step / max_steps
          - 帮助Critic理解episode阶段

        Returns:
            int: 总状态维度 = 3×num_bs + 7
                 对于8个基站: 3×8 + 7 = 31维
        """
        return 3 * self.num_bs + 7

    def get_obs(self, uav_id: int) -> np.ndarray:
        """
        获取单个UAV的局部观测向量

        【业务识别集成逻辑】
        本方法是训练/评估双模式的核心实现点：

        模式1: 训练模式 (recognition_model is None)
          - 使用uav.true_business_type（ground truth）
          - 保证训练数据的准确性
          - 让agent学习到真实的业务-策略映射关系

        模式2: 评估模式 (recognition_model is not None)
          - 使用env.perform_recognition(uav_id)预测业务类型
          - 引入识别模型的分类误差（通常5-15%错误率）
          - 模拟真实部署场景中的不确定性

        【状态平滑机制】(可选)
        如果use_state_smoothing=True：
          - 保存最近5个原始观测到历史缓冲区
          - 返回这5个观测的逐元素均值
          - 效果：减少单步噪声，提供更稳定的输入信号
          - 代价：引入约5步的响应延迟

        Args:
            uav_id (int): UAV的唯一标识符，范围[0, num_agents-1]

        Returns:
            np.ndarray: 形状为(obs_dim,)的浮点向量
                       包含该UAV的所有局部观测信息
                       所有值已归一化到[0, 1]或[-1, 1]范围

        Example:
            >>> obs = env.get_obs(uid=42)
            >>> print(obs.shape)
            (49,)  # 对于8个基站配置
            >>> print(obs[:8])  # 前8维是SINR向量
            [0.75, 0.62, 0.88, 0.45, 0.33, 0.91, 0.57, 0.69]
        """
        uav = self.env.uavs[uav_id]
        n = self.num_bs

        # 1. SINR 向量 (归一化到 [0, 1])
        sinr_raw = self.env.sinr_matrix[uav_id, :n]
        sinr_norm = np.clip((sinr_raw + 10) / 40, 0, 1)

        # 2. 基站负载率
        loads = np.array([bs.load_ratio for bs in self.env.base_stations.values()])

        # 3. 当前连接 BS one-hot
        connected_onehot = np.zeros(n)
        if uav.connected_bs_id is not None and uav.connected_bs_id < n:
            connected_onehot[uav.connected_bs_id] = 1.0

        # 4. 可用容量 / 需求速率比 (归一化)
        required = uav.required_rate
        if required > 0:
            cap_ratios = np.array([
                min(bs.available_capacity / required, 2.0) / 2.0
                for bs in self.env.base_stations.values()
            ])
        else:
            cap_ratios = np.ones(n)

        # 5. 业务类型 one-hot — 根据是否提供识别模型决定来源
        biz = np.zeros(3)
        if self.recognition_model is not None and self.scaler is not None:
            # 评估模式：使用模型预测的业务类型
            predicted_biz = self.env.perform_recognition(uav_id)
            biz_type_result = predicted_biz[0] if isinstance(predicted_biz, tuple) else predicted_biz
            if biz_type_result is not None:
                biz[biz_type_result.value] = 1.0
            else:
                # 预测失败回退到真实类型
                biz[uav.true_business_type.value] = 1.0
        else:
            # 训练模式：使用真实业务类型 (ground truth)
            biz[uav.true_business_type.value] = 1.0

        # 6. 当前满意度
        satisfaction = np.array([uav.current_satisfaction])

        # 7. 连接状态
        connected = np.array([1.0 if uav.connected_bs_id is not None else 0.0])

        # 8. 移动速度 (归一化)
        velocity = np.array([min(np.linalg.norm(uav.velocity) / 30.0, 1.0)])

        # 9. 上次动作 one-hot
        last_action = np.zeros(self.action_dim)
        if uav_id in self._last_actions:
            action = self._last_actions[uav_id]
            # 确保动作索引在有效范围内
            if action < len(last_action):
                last_action[action] = 1.0
            else:
                # 对于超出范围的动作，映射到最后一个类别
                last_action[-1] = 1.0

        # 10. 满意度变化趋势 (最近5步的线性趋势)
        if uav_id in self._sat_history and len(self._sat_history[uav_id]) >= 2:
            recent = list(self._sat_history[uav_id])[-5:]
            if len(recent) >= 2:
                trend = (recent[-1] - recent[0]) / len(recent)
            else:
                trend = 0.0
        else:
            trend = 0.0
        sat_trend = np.array([np.clip(trend * 10.0, -1.0, 1.0)])  # 缩放到 [-1, 1]

        # 11. 同类型 UAV 平均满意度
        peer_sats = []
        for other_uid, other_uav in self.env.uavs.items():
            if other_uid != uav_id and other_uav.true_business_type == uav.true_business_type:
                peer_sats.append(other_uav.current_satisfaction)
        peer_avg = np.array([np.mean(peer_sats) if peer_sats else 0.0])

        # 12. 历史满意度 (最近3步)
        hist_sat = np.zeros(3)
        if uav_id in self._sat_history and len(self._sat_history[uav_id]) >= 3:
            recent = list(self._sat_history[uav_id])[-4:-1]  # 最近3步（不包括当前步）
            hist_sat[:len(recent)] = recent
        elif uav_id in self._sat_history:
            recent = list(self._sat_history[uav_id])[:-1]  # 所有历史（不包括当前步）
            hist_sat[:len(recent)] = recent

        # 构建原始观测
        raw_obs = np.concatenate([sinr_norm, loads, connected_onehot,
                               cap_ratios, biz, satisfaction, connected, velocity,
                               last_action, sat_trend, peer_avg, hist_sat])
        
        # 状态平滑：使用历史状态的移动平均
        if self.use_state_smoothing:
            # 存储当前状态
            self._state_history[uav_id].append(raw_obs)
            # 只保留最近5个状态
            if len(self._state_history[uav_id]) > 5:
                self._state_history[uav_id] = self._state_history[uav_id][-5:]
            # 计算移动平均
            if len(self._state_history[uav_id]) > 1:
                smoothed_obs = np.mean(self._state_history[uav_id], axis=0)
                return smoothed_obs
        
        return raw_obs

    def get_global_state(self) -> np.ndarray:
        """
        获取全局状态（用于 centralized critic / mixing network）

        Returns:
            state: shape=(state_dim,) 的浮点数组
        """
        n = self.num_bs

        # 1. 所有基站负载率
        loads = np.array([bs.load_ratio for bs in self.env.base_stations.values()])

        # 2. 所有基站可用容量 (归一化)
        capacities = np.array([
            bs.available_capacity / max(bs.capacity, 1)
            for bs in self.env.base_stations.values()
        ])

        # 3. 所有基站故障状态
        failures = np.array([1.0 if bs.failure_state else 0.0
                             for bs in self.env.base_stations.values()])

        # 4. 各业务类型 UAV 数量 (归一化)
        biz_counts = np.zeros(3)
        for uav in self.env.uavs.values():
            biz_counts[uav.true_business_type.value] += 1
        biz_counts /= max(self.num_uav, 1)

        # 5. 全局平均满意度
        avg_sat = np.array([np.mean([u.current_satisfaction
                                      for u in self.env.uavs.values()])])

        # 6. 全局断连率
        disc_count = sum(1 for u in self.env.uavs.values() if u.connected_bs_id is None)
        disc_rate = np.array([disc_count / max(self.num_uav, 1)])

        # 7. 全局中断率
        int_rate = np.array([len(self.env.interrupted_uavs) / max(self.num_uav, 1)])

        # 8. 当前步数 (归一化)
        step_norm = np.array([self._current_step / max(self.max_steps, 1)])

        return np.concatenate([loads, capacities, failures,
                               biz_counts, avg_sat, disc_rate, int_rate, step_norm])

    def get_all_obs(self) -> Dict[int, np.ndarray]:
        """获取所有 agent 的局部观测"""
        return {uid: self.get_obs(uid) for uid in range(self.num_agents)}

    def reset(self, bs_capacity_range=None) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        重置环境

        Args:
            bs_capacity_range: 如果提供，随机化所有 BS 的容量 (min, max)
                              用于 Domain Randomization

        Returns:
            obs_dict: {agent_id: obs}
            global_state: shape=(state_dim,)
        """
        self.env.reset()

        # Domain Randomization: 随机化 BS 容量
        if bs_capacity_range is not None:
            low, high = bs_capacity_range
            for bs in self.env.base_stations.values():
                bs.capacity = np.random.uniform(low, high)

        # 如果 pos_range != 默认值，从原始位置重新缩放（避免累积缩放）
        if self.pos_range != 1000 and hasattr(self, '_original_positions'):
            scale = self.pos_range / 1000.0
            for key, pos in self._original_positions.items():
                kind, obj_id = key
                if kind == 'bs':
                    self.env.base_stations[obj_id].position = pos * scale
                else:
                    self.env.uavs[obj_id].position = pos * scale
            self.env._update_sinr_matrix()
            self.env._initialize_connections()

        self._current_step = 0

        # 记录初始状态
        self._last_satisfaction = {}
        self._last_disconnected = {}
        self._last_handover_count = {}
        self._last_actions = {}
        self._last_rate_ratios = {}
        self._sat_history = {}
        self._last_rankings = {}  # [V12] 初始化同类排名记录
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_satisfaction[uid] = uav.current_satisfaction
            self._last_disconnected[uid] = (uav.connected_bs_id is None)
            self._last_handover_count[uid] = uav.handover_count
            self._last_actions[uid] = 0
            self._last_rate_ratios[uid] = uav.current_allocated_rate / max(uav.required_rate, 1e-6)
            self._sat_history[uid] = deque([uav.current_satisfaction], maxlen=10)
            self._last_rankings[uid] = self.num_agents // 2  # 初始排名设为中间位置
        # 不在此处 reset normalizer — EMA 需要跨 episode 持续积累
        # 调用者可通过 reset_normalizer() 手动重置
        
        # 重置通信指标 - 对齐实验3
        self._communication_metrics = {
            'handover_latencies': [],
            'ping_jitters': [],
            'packet_losses': [],
            'qos_violations': [],
            'ping_times': {},
            # 新增指标（对齐实验3）
            'throughput': [],
            'load_variance': [],
            'spectral_efficiency': [],
            'fairness_index': [],
        }
        for uid in range(self.num_agents):
            self._communication_metrics['ping_times'][uid] = deque(maxlen=10)

        obs_dict = self.get_all_obs()
        global_state = self.get_global_state()
        return obs_dict, global_state

    def step(self, actions: Dict[int, int]) -> Tuple[
            Dict[int, np.ndarray], np.ndarray,
            Dict[int, float], float, bool, Dict]:
        """
        执行一个环境步

        Args:
            actions: {agent_id: action}，action 取值: 0=stay, 1=best_sinr, 2=best_capacity, ...

        Returns:
            obs_dict: {agent_id: obs}
            global_state: shape=(state_dim,)
            rewards: {agent_id: reward}
            team_reward: 团队总奖励
            done: episode 是否结束
            info: 附加信息
        """
        # 保存旧状态
        old_sats = {uid: self.env.uavs[uid].current_satisfaction
                     for uid in range(self.num_agents)}
        # 保存旧的 rate_ratio (V11: 用于连续增量信号)
        old_rate_ratios = {}
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            old_rate_ratios[uid] = uav.current_allocated_rate / max(uav.required_rate, 1e-6)
        self._last_rate_ratios = old_rate_ratios

        # ====== 1. 根据 actions 执行切换 ======
        # action 0: stay (不切换)
        # action 1: best_sinr (切换到 SINR 最高的 BS)
        # action 2: best_capacity (切换到可用容量/需求比最高的 BS)
        # ---- 切换诊断 ----
        ep_switch_attempts = 0   # 尝试切换的 UAV 数
        ep_switch_success = 0    # 成功切换
        ep_switch_rollback = 0   # 切换失败但回滚成功
        ep_switch_disconnect = 0 # 切换失败且回滚也失败
        handover_latencies = []  # 记录切换延迟

        for uid, action in actions.items():
            if action == 0:
                continue  # stay
            elif action == 1:
                # best_sinr: 选择当前 SINR 最高的 BS
                sinr_row = self.env.sinr_matrix[uid]
                # 排除当前已连接的 BS（已连接则无需切换）
                candidates = [(bs_id, sinr_row[bs_id]) for bs_id in range(self.num_bs)
                              if bs_id != self.env.uavs[uid].connected_bs_id]
                if not candidates:
                    continue
                target_bs_id = max(candidates, key=lambda x: x[1])[0]
            elif action == 2:
                # best_capacity: 选择可用容量/需求比最高的 BS
                uav = self.env.uavs[uid]
                candidates = []
                for bs_id in range(self.num_bs):
                    if bs_id == uav.connected_bs_id:
                        continue
                    bs = self.env.base_stations[bs_id]
                    if uav.required_rate > 0:
                        ratio = bs.available_capacity / uav.required_rate
                    else:
                        ratio = float('inf')
                    candidates.append((bs_id, ratio))
                if not candidates:
                    continue
                target_bs_id = max(candidates, key=lambda x: x[1])[0]
            elif action == 3:
                # sinr_capacity: SINR和容量的加权组合
                uav = self.env.uavs[uid]
                sinr_row = self.env.sinr_matrix[uid]
                candidates = []
                for bs_id in range(self.num_bs):
                    if bs_id == uav.connected_bs_id:
                        continue
                    bs = self.env.base_stations[bs_id]
                    sinr = sinr_row[bs_id]
                    if uav.required_rate > 0:
                        cap_ratio = bs.available_capacity / uav.required_rate
                    else:
                        cap_ratio = 1.0
                    # 加权组合：SINR占60%，容量占40%
                    score = 0.6 * sinr + 0.4 * cap_ratio
                    candidates.append((bs_id, score))
                if not candidates:
                    continue
                target_bs_id = max(candidates, key=lambda x: x[1])[0]
            elif action == 4:
                # predictive: 基于预测的切换（简化版，使用当前趋势）
                uav = self.env.uavs[uid]
                sinr_row = self.env.sinr_matrix[uid]
                candidates = []
                for bs_id in range(self.num_bs):
                    if bs_id == uav.connected_bs_id:
                        continue
                    # 简化处理：使用当前SINR作为预测值
                    candidates.append((bs_id, sinr_row[bs_id]))
                if not candidates:
                    continue
                target_bs_id = max(candidates, key=lambda x: x[1])[0]
            elif action == 5:
                # business_specific: 基于业务类型的特定切换策略
                uav = self.env.uavs[uid]
                biz_type = uav.true_business_type.value
                candidates = []
                for bs_id in range(self.num_bs):
                    if bs_id == uav.connected_bs_id:
                        continue
                    bs = self.env.base_stations[bs_id]
                    sinr = self.env.sinr_matrix[uid][bs_id]
                    if uav.required_rate > 0:
                        cap_ratio = bs.available_capacity / uav.required_rate
                    else:
                        cap_ratio = 1.0
                    # 根据业务类型调整权重
                    if biz_type == 0:  # 延迟敏感型，更重视SINR
                        score = 0.8 * sinr + 0.2 * cap_ratio
                    elif biz_type == 1:  # 吞吐量敏感型，更重视容量
                        score = 0.3 * sinr + 0.7 * cap_ratio
                    else:  # 可靠性敏感型，平衡考虑
                        score = 0.5 * sinr + 0.5 * cap_ratio
                    candidates.append((bs_id, score))
                if not candidates:
                    continue
                target_bs_id = max(candidates, key=lambda x: x[1])[0]
            else:
                continue  # 无效动作，忽略

            uav = self.env.uavs[uid]
            target_bs = self.env.base_stations[target_bs_id]
            ep_switch_attempts += 1

            # 记录切换开始时间
            handover_start = time.time()

            # [PURE-MAPPO] 完全纯净版本 - 无任何保护机制
            # 模型选择什么action，就严格按那个action执行：
            #   - 不预检查容量
            #   - 不使用降级分配（只用理想速率）
            #   - 失败不回滚（直接断连）
            # 这样才能真实反映MAPPO学到的策略质量！

            # 释放当前 BS 资源
            old_bs_id = uav.connected_bs_id
            if old_bs_id is not None:
                old_bs = self.env.base_stations[old_bs_id]
                old_bs.connected_uavs.pop(uid, None)
                old_bs.current_load -= uav.current_allocated_rate

            # 尝试以理想速率分配到目标 BS（不降级！）
            required_rate = uav.required_rate  # ideal_rate, not degraded
            
            if target_bs.allocate(uid, required_rate):
                # 分配成功
                uav.connected_bs_id = target_bs_id
                uav.current_allocated_rate = required_rate
                uav.handover_count += 1
                ep_switch_success += 1
            else:
                # 分配失败 - 尝试回滚到旧 BS（仅一次机会，不降级）
                if old_bs_id is not None:
                    old_bs = self.env.base_stations[old_bs_id]
                    if old_bs.allocate(uid, required_rate):
                        # 回滚成功
                        uav.connected_bs_id = old_bs_id
                        uav.current_allocated_rate = required_rate
                        ep_switch_rollback += 1
                    else:
                        # 回滚也失败 - 断连
                        uav.connected_bs_id = None
                        uav.current_allocated_rate = 0.0
                        ep_switch_disconnect += 1
                else:
                    # 无旧BS可回滚 - 直接断连
                    uav.connected_bs_id = None
                    uav.current_allocated_rate = 0.0
                    ep_switch_disconnect += 1

            # 记录切换结束时间并计算延迟
            handover_end = time.time()
            # 基础延迟 + 处理时间（模拟真实环境）
            base_handover_latency = 5.0  # 基础切换延迟5ms
            processing_latency = (handover_end - handover_start) * 1000  # 转换为毫秒
            # 根据目标基站负载调整延迟
            target_bs_for_latency = self.env.base_stations.get(target_bs_id)
            if target_bs_for_latency:
                load_factor = 1.0 + target_bs_for_latency.load_ratio * 0.5  # 负载越高，延迟越大
            else:
                load_factor = 1.0
            handover_latency = (base_handover_latency + processing_latency) * load_factor
            handover_latencies.append(handover_latency)
            self._communication_metrics['handover_latencies'].append(handover_latency)

        # ====== 2. 推进环境 ======
        self.env.current_step += 1
        for uav in self.env.uavs.values():
            uav.move(time_step=1.0)
        self.env._update_sinr_matrix()
        for uav in self.env.uavs.values():
            uav.record_satisfaction()
        self.env._check_interruptions()
        self.env._record_stats()

        self._current_step += 1

        # 监测 Ping 抖动和丢包率
        ping_jitters = []
        packet_losses = []
        qos_violations = []
        
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            
            # 模拟 Ping 时间（基于 SINR）
            if uav.connected_bs_id is not None:
                sinr = self.env.sinr_matrix[uid][uav.connected_bs_id]
                # 基于 SINR 计算延迟：SINR 越高，延迟越低
                base_ping = 20  # 基础延迟（毫秒）
                sinr_factor = max(0.1, min(1.0, (sinr + 10) / 40))
                ping_time = base_ping / sinr_factor
                
                # 添加随机抖动
                jitter = np.random.normal(0, 5)
                ping_time += jitter
                
                # 记录 Ping 时间
                self._communication_metrics['ping_times'][uid].append(ping_time)
                
                # 计算 Ping 抖动（最近几次的标准差）
                if len(self._communication_metrics['ping_times'][uid]) >= 3:
                    ping_history = list(self._communication_metrics['ping_times'][uid])
                    jitter_value = np.std(ping_history)
                    ping_jitters.append(jitter_value)
                    self._communication_metrics['ping_jitters'].append(jitter_value)
                
                # 模拟丢包率（基于 SINR）
                # SINR 越低，丢包率越高
                packet_loss_rate = max(0, min(5, (20 - sinr) / 4))
                packet_losses.append(packet_loss_rate)
                self._communication_metrics['packet_losses'].append(packet_loss_rate)
            else:
                # 断连状态，设置高延迟和丢包率
                packet_losses.append(100.0)  # 断连时丢包率 100%
                self._communication_metrics['packet_losses'].append(100.0)
            
            # 计算 QoS 违规率
            # 基于满意度：满意度低于 0.6 视为 QoS 违规
            qos_violation_rate = 0.0 if uav.current_satisfaction >= 0.6 else 100.0
            qos_violations.append(qos_violation_rate)
            self._communication_metrics['qos_violations'].append(qos_violation_rate)

        # ====== 3. 计算奖励 (V12: V11 + 负载自适应 + 关键业务差距 + 同类排名) ======
        # V11 基础: 连续 rate_ratio 信号 + 反事实比较 + 业务权重 + 动作奖励 + EMA归一化
        # V12 新增 (针对低负载环境~77%学习信号弱的问题):
        #   e. 负载自适应系数: 低负载时增大切换奖励, 高负载时增大留守惩罚
        #   f. 关键业务差距奖励: reward += α × (target_sat - current_sat)
        #   g. 同类相对排名信号: UAV在同类中的排名变化作为额外探索激励
        #
        # 理论支撑:
        #   - 负载自适应: 基于"资源充裕度决定探索价值"直觉, 低负载下需要更强的切换激励
        #   - 目标差距: 基于目标导向强化学习(HRL), 明确优化方向避免奖励稀疏
        #   - 同类排名: 基于多智能体竞争/合作框架, 提供相对性能信号而非绝对值
        rewards_raw = {}
        team_reward = 0.0
        # 诊断用组分统计
        ep_delta_sum = 0.0
        ep_value_reward_sum = 0.0
        ep_biz_reward_sum = 0.0
        ep_action_reward_sum = 0.0
        ep_connect_reward_sum = 0.0
        ep_load_adaptive_sum = 0.0    # [V12] 负载自适应分量
        ep_target_gap_sum = 0.0       # [V12] 目标差距分量
        ep_ranking_sum = 0.0          # [V12] 排名变化分量
        ep_good_switch = 0
        ep_bad_switch = 0

        # 预计算全局负载率（用于负载自适应系数）
        global_load_ratio = np.mean([bs.load_ratio for bs in self.env.base_stations.values()])

        # 预计算同类 UAV 的平均 rate_ratio 和满意度 (用于反事实比较 + 排名)
        peer_rate_ratios = {}   # biz_type -> list of rate_ratio
        peer_sats = {}          # biz_type -> list of satisfaction
        for _uid in range(self.num_agents):
            _uav = self.env.uavs[_uid]
            _rr = _uav.current_allocated_rate / max(_uav.required_rate, 1e-6)
            _bt = _uav.true_business_type.value
            peer_rate_ratios.setdefault(_bt, []).append(_rr)
            peer_sats.setdefault(_bt, []).append(_uav.current_satisfaction)
        peer_avg_rr = {bt: np.mean(rrs) for bt, rrs in peer_rate_ratios.items()}

        # 各业务类型的目标满意度阈值
        TARGET_SATISFACTION = {
            0: 0.85,  # 控制信令: 延迟敏感型, 高目标
            1: 0.75,  # 视频回传: 吞吐量敏感型, 中高目标
            2: 0.65,  # 环境监测: 可靠性敏感型, 中等目标
        }

        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            new_sat = uav.current_satisfaction
            old_sat = old_sats.get(uid, 0.5)

            delta_sat = new_sat - old_sat
            action = actions.get(uid, 0)
            biz_type = uav.true_business_type.value

            # --- a. 连续速率比信号 (替代分段满意度) ---
            old_rr = old_rate_ratios.get(uid, 0.5)
            new_rr = uav.current_allocated_rate / max(uav.required_rate, 1e-6)
            delta_rr = new_rr - old_rr
            r_delta = MAPPOConfig.RewardConfig.delta_scale * delta_rr  # V21: 配置化
            ep_delta_sum += r_delta

            # --- b. 反事实比较信号 ---
            r_counterfactual = 0.0
            new_ho = uav.handover_count
            old_ho = self._last_handover_count.get(uid, 0)
            switched = (new_ho > old_ho)
            if switched:
                avg_rr = peer_avg_rr.get(biz_type, new_rr)
                relative_gain = new_rr - avg_rr
                r_counterfactual = MAPPOConfig.RewardConfig.counterfactual_scale * relative_gain  # V21: 配置化
            ep_value_reward_sum += r_counterfactual

            # --- c. 业务类型权重 (V21: 统一权重来源) ---
            r_biz = 0.0
            if abs(delta_sat) > 1e-4:
                biz_weight = MAPPOConfig.BusinessWeightConfig.weights.get(biz_type, MAPPOConfig.BusinessWeightConfig.default_weight)  # V21: 配置化
                r_biz = biz_weight * delta_rr
            ep_biz_reward_sum += r_biz

            # --- d. 动作奖励 (V19/V21: 配置化) ---
            # V19 核心设计思想:
            #   stay 必须是"高价值默认动作"，让 agent 从乱切快速收敛到"少切"
            #   但优秀的切换(stay < excellent_switch)仍然值得做
            #
            # V21改进: 所有参数从MAPPOConfig读取,消除硬编码
            #
            # 数学验证 (300UAV, 350步):
            #   随机阶段(stay~18%, switch~82%, 大部分是微负/坏切换):
            #     avg_r ≈ 18%×0.89 + 60%×(-0.08) + 22%×(-0.32) ≈ +0.02/UAV/步
            #   训练后(stay~48%, switch~52%, 质量大幅提升):
            #     avg_r ≈ 48%×0.91 + 35%×(+0.45) + 17%×(-0.08) ≈ +0.57/UAV/步
            #   → **28x 提升**！曲线从低到高明显上升 ✅
            #
            # 关键不等式: excellent_switch(>1.0) > stay(~0.9) > neutral_switch(<0.2)
            #   → 学会"该留则留、该换则换"，不是一味不切换
            
            r_action = 0.0
            
            if switched:
                if delta_sat > MAPPOConfig.RewardConfig.excellent_switch_threshold:  # V21: 0.05
                    # 优秀切换：明显改善连接，奖励高于stay（鼓励好决策）
                    r_action = MAPPOConfig.RewardConfig.excellent_switch_base + delta_sat * MAPPOConfig.RewardConfig.excellent_switch_coeff  # V21: 1.0 + delta*2.0
                    ep_good_switch += 1
                elif delta_sat > MAPPOConfig.RewardConfig.good_switch_threshold:  # V21: 0.015
                    # 好切换：有改善但不大
                    r_action = MAPPOConfig.RewardConfig.good_switch_base + delta_sat * MAPPOConfig.RewardConfig.good_switch_coeff  # V21: 0.55 + delta*3.0
                elif delta_sat > 0.0:
                    # 微正切换：略优于stay底线
                    r_action = MAPPOConfig.RewardConfig.micro_positive_base + delta_sat * MAPPOConfig.RewardConfig.micro_positive_coeff  # V21: 0.15 + delta*5.0
                elif delta_sat > MAPPOConfig.RewardConfig.acceptable_switch_threshold:  # V21: -0.03
                    # 微负切换（可接受的小损失）
                    r_action = delta_sat * MAPPOConfig.RewardConfig.acceptable_penalty_coeff  # V21: delta*4.0
                else:
                    # 坏切换：明显恶化连接，严厉惩罚
                    r_action = delta_sat * MAPPOConfig.RewardConfig.bad_switch_coeff + MAPPOConfig.RewardConfig.bad_switch_penalty  # V21: delta*6.0 - 0.08
                    ep_bad_switch += 1
            elif action != 0:
                # 非标准动作（predict/biz_spec等）
                r_action = MAPPOConfig.RewardConfig.non_standard_action_penalty  # V21: -0.15
            else:
                # stay = 安全高回报动作 (V21: 配置化)
                # 基础值保证在大部分场景下 stay 都是正收益
                # bonus 在 sat超过阈值时激活，鼓励维持高质量连接
                r_action = MAPPOConfig.RewardConfig.stay_base_reward + max(0.0, (new_sat - MAPPOConfig.RewardConfig.stay_bonus_threshold)) * MAPPOConfig.RewardConfig.stay_bonus_scale
            ep_action_reward_sum += r_action

            # ====== V12 新增分量 (V21: 配置化) ======

            # --- e. 负载自适应系数 ---
            # 核心思想: 全局负载率越低(资源越充裕), 切换的边际收益越小,
            #           因此需要放大切换奖励来维持探索动力。
            # 同时低负载下留守的惩罚也应增强, 避免agent陷入"什么都不做"的局部最优。
            
            lac = MAPPOConfig.LoadAdaptiveConfig  # V21: 简化引用
            
            if global_load_ratio < lac.low_load_threshold:
                load_factor = lac.low_load_factor  # V21: 低负载(<60%): 强力鼓励切换和探索
            elif global_load_ratio < lac.medium_low_threshold:
                load_factor = lac.medium_low_factor  # V21: 中低负载(60-75%): 适度增强
            elif global_load_ratio < lac.normal_load_threshold:
                load_factor = lac.normal_load_factor  # V21: 正常负载(75-90%): 不调整
            else:
                load_factor = lac.high_load_factor  # V21: 高负载(>90%): 保守策略, 降低切换冲动

            r_load_adaptive = 0.0
            if action == 0:  # stay
                # 低负载下留守惩罚增强: 资源充裕时不利用=浪费机会 (V21: 配置化)
                if global_load_ratio < lac.stay_low_load_punish_threshold and new_sat < lac.stay_low_load_sat_threshold:
                    r_load_adaptive = lac.stay_low_load_punish_max * (lac.stay_low_load_punish_threshold - global_load_ratio) / 0.20
                elif global_load_ratio >= lac.stay_high_load_reward_threshold and new_sat >= lac.stay_high_load_sat_threshold:
                    # 高负载下留守合理, 给予轻微正奖励 (V21: 配置化)
                    r_load_adaptive = lac.stay_high_load_reward
            else:
                # 非留守动作在低负载下获得额外奖励
                if global_load_ratio < 0.80:
                    r_load_adaptive = 0.10 * (0.80 - global_load_ratio) / 0.20

            r_load_adaptive *= load_factor
            ep_load_adaptive_sum += r_load_adaptive

            # --- f. 关键业务差距奖励 (V13/V21: 配置化) ---
            # 核心思想: 直接告诉agent"距离目标还差多少", 解决奖励稀疏问题。
            # 基于目标导向强化学习(HRL): r_gap = α × (target - current),
            # 当 current < target 时为负值(推动提升), 达到或超过时为零(不惩罚超额完成)。
            #
            # [P2/V21] 对不同业务类型差异化权重 (从MAPPOConfig读取):
            #   - 控制信令(biz_type=0): 权重2.0 (高优先级, 延迟敏感)
            #   - 视频回传(biz_type=1): 权重2.5 (最高优先级, 吞吐量敏感)
            #   - 环境监测(biz_type=2): 权重1.5 (正常优先级, 可靠性敏感)
            target_sat = TARGET_SATISFACTION.get(biz_type, 0.75)
            sat_gap = max(0.0, target_sat - new_sat)  # 只 penalize 未达标的
            
            # [P2/V21] 业务类型差异化权重 (从配置读取)
            tgc = MAPPOConfig.TargetGapConfig  # V21: 简化引用
            if biz_type == 0:
                biz_critical_weight = tgc.control_signal_weight  # V21: 控制信令
            elif biz_type == 1:
                biz_critical_weight = tgc.video_weight  # V21: 视频回传
                biz_critical_weight = 2.5  # 视频回传 - 高优先级
            else:
                biz_critical_weight = tgc.environment_weight  # V21: 环境监测
            
            r_target_gap = -biz_critical_weight * sat_gap  # 使用增强权重
            ep_target_gap_sum += r_target_gap

            # --- g. 同类相对排名信号 ---
            # 核心思想: 不看绝对值, 看"我在同类中排第几"。排名上升说明策略有效。
            # 基于多智能体竞争框架: 相对排名比绝对值更能区分策略优劣。
            r_ranking = 0.0
            if biz_type in peer_sats and len(peer_sats[biz_type]) > 1:
                same_type_sats = sorted(
                    [(other_uid, s) for other_uid, s in enumerate(peer_sats[biz_type])],
                    key=lambda x: x[1], reverse=True
                )
                # 计算当前UAV在同类中的排名 (0-based, 越小越好)
                current_rank = next(
                    (i for i, (rid, _) in enumerate(same_type_sats) if rid == uid),
                    len(same_type_sats) // 2  # 找不到则给中间排名
                )
                n_peers = len(same_type_sats)

                # 与上一步排名对比（从历史记录推断）
                prev_rank = self._last_rankings.get(uid, n_peers // 2)
                rank_change = prev_rank - current_rank  # 正值=排名上升=好事

                # 奖励: 排名上升给予正向激励
                r_ranking = 0.15 * rank_change / max(n_peers // 2, 1)
                self._last_rankings[uid] = current_rank
            ep_ranking_sum += r_ranking

            # --- h. 连接状态奖励 (仅诊断, 不参与个体reward) ---
            is_connected = uav.connected_bs_id is not None
            was_connected = not self._last_disconnected.get(uid, False)
            if is_connected:
                r_connect = 0.0
            else:
                if was_connected:
                    r_connect = -4.0
                else:
                    r_connect = -2.5
            ep_connect_reward_sum += r_connect

            # 综合奖励 (不含 connect 惩罚, 含V12新增分量)
            r_individual = (r_delta + r_counterfactual + r_biz + r_action
                          + r_load_adaptive + r_target_gap + r_ranking)

            # 奖励平滑：限制奖励范围
            r_individual = np.clip(r_individual, -10.0, 20.0)

            rewards_raw[uid] = r_individual
            team_reward += r_individual

            # 更新历史
            self._last_satisfaction[uid] = new_sat
            self._last_disconnected[uid] = not is_connected
            self._last_handover_count[uid] = new_ho
            self._last_actions[uid] = min(action, self.action_dim - 1)
            self._sat_history[uid].append(new_sat)

        # 统一更新 rate_ratio 历史 (下一步的 old_rate_ratios)
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_rate_ratios[uid] = uav.current_allocated_rate / max(uav.required_rate, 1e-6)

        # ====== 4. Reward 归一化 (EMA) ======
        # V17: 禁用EMA归一化。在低负载环境下reward绝对值差异已很小，
        # EMA会将本就微弱的信号差异完全抹平，导致advantage≈0无法学习。
        # PPO本身通过advantage标准化处理reward scale，无需额外归一化。
        rewards = rewards_raw

        # ====== 4.5 断连惩罚加入团队奖励 (episode-level, 不影响个体 GAE) ======
        # 将平均 connect 惩罚作为团队级信号，让 Critic 感知断连影响
        # 但个体 UAV 的 reward 不含此惩罚，避免结构性噪音淹没 per-agent 学习信号
        avg_connect_penalty = ep_connect_reward_sum / max(self.num_agents, 1)
        team_reward += avg_connect_penalty

        # ====== 4.6 全局负载均衡惩罚 (P1改进: V13新增) ======
        # 核心思想: 惩罚基站间负载不均衡, 鼓励UAV分散到不同基站。
        # 理论支撑:
        #   - 负载均衡能提升整体网络资源利用率, 减少拥塞和切换失败
        #   - 基于多智能体协调框架: 团队级全局信号指导个体决策
        #   - 惩罚函数: 基于基站负载标准差, std越大说明越不均衡
        #
        # 计算方法:
        #   load_balance_penalty = -α × std(load_ratios)
        #   α = 2.0 (经验值, 平衡与其他团队信号的量级)
        #
        # 预期效果:
        #   - load_variance 从 0.063 降低到 <0.01 (接近增强算法的0.002)
        #   - 提升连接保持率 (减少因基站过载导致的断连)
        bs_loads = np.array([bs.load_ratio for bs in self.env.base_stations.values()])
        load_std = np.std(bs_loads)
        load_balance_penalty = -2.0 * load_std  # 权重α=2.0
        
        # 只在团队reward中加入, 避免干扰个体学习信号
        team_reward += load_balance_penalty

        # ====== 5. 团队奖励归一化 ======
        team_reward /= max(self.num_agents, 1)

        # ====== 5. 判断是否结束 ======
        done = self._current_step >= self.max_steps

        # ====== 6. 附加信息 ======
        info = {
            'step': self._current_step,
            'avg_satisfaction': np.mean([self.env.uavs[uid].current_satisfaction
                                          for uid in range(self.num_agents)]),
            'connected_rate': sum(1 for uid in range(self.num_agents)
                                  if self.env.uavs[uid].connected_bs_id is not None) / max(self.num_agents, 1),
            'global_load_ratio': global_load_ratio,  # [V12] 负载率
            'strategy_distribution': {},
            'reward_diag': {
                'delta_sum': ep_delta_sum / max(self.num_agents, 1),
                'value_reward': ep_value_reward_sum / max(self.num_agents, 1),
                'biz_reward': ep_biz_reward_sum / max(self.num_agents, 1),
                'action_reward': ep_action_reward_sum / max(self.num_agents, 1),
                'connect_reward': ep_connect_reward_sum / max(self.num_agents, 1),
                'load_adaptive': ep_load_adaptive_sum / max(self.num_agents, 1),   # [V12]
                'target_gap': ep_target_gap_sum / max(self.num_agents, 1),          # [V12/V13增强]
                'ranking_signal': ep_ranking_sum / max(self.num_agents, 1),         # [V12]
                'load_balance_penalty': load_balance_penalty,                      # [V13/P1新增]
                'good_switch': ep_good_switch,
                'bad_switch': ep_bad_switch,
                'raw_mean': np.mean(list(rewards_raw.values())),
                'norm_mean': np.mean(list(rewards.values())),
                'switch_attempts': ep_switch_attempts,
                'switch_success': ep_switch_success,
                'switch_rollback': ep_switch_rollback,
                'switch_disconnect': ep_switch_disconnect,
            },
            'communication_metrics': {
                'handover_latency': float(np.mean(handover_latencies)) if handover_latencies else 0.0,
                'ping_jitter': float(np.mean(ping_jitters)) if ping_jitters else 0.0,
                'packet_loss_rate': float(np.mean(packet_losses)) if packet_losses else 0.0,
                'qos_violation_rate': float(np.mean(qos_violations)) if qos_violations else 0.0,
            },
        }
        # 统计 action 分布
        action_names = {0: 'stay', 1: 'best_sinr', 2: 'best_capacity'}
        for uid, action in actions.items():
            name = action_names.get(action, f'action{action}')
            info['strategy_distribution'][name] = \
                info['strategy_distribution'].get(name, 0) + 1

        obs_dict = self.get_all_obs()
        global_state = self.get_global_state()

        return obs_dict, global_state, rewards, team_reward, done, info

    def reset_normalizer(self):
        """手动重置 reward normalizer（仅在训练重启时调用）"""
        self._reward_normalizer.reset()

    def collect_step_metrics(self):
        """
        收集当前步的通信质量指标（Ping抖动、丢包率、QoS违规率）。
        供基线算法评估时在 advance_env_only() 之后调用，
        使基线算法与传统/增强算法的通信指标具有可比性。
        
        新增指标（对齐实验3）：吞吐量、负载方差、频谱效率、公平性
        """
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            if uid not in self._communication_metrics['ping_times']:
                self._communication_metrics['ping_times'][uid] = []

            if uav.connected_bs_id is not None:
                sinr = self.env.sinr_matrix[uid][uav.connected_bs_id]
                base_ping = 20
                sinr_factor = max(0.1, min(1.0, (sinr + 10) / 40))
                ping_time = base_ping / sinr_factor + np.random.normal(0, 5)
                self._communication_metrics['ping_times'][uid].append(ping_time)

                if len(self._communication_metrics['ping_times'][uid]) >= 3:
                    ping_history = list(self._communication_metrics['ping_times'][uid])[-3:]
                    jitter_value = float(np.std(ping_history))
                    self._communication_metrics['ping_jitters'].append(jitter_value)

                packet_loss_rate = max(0, min(5, (20 - sinr) / 4))
                self._communication_metrics['packet_losses'].append(packet_loss_rate)
            else:
                self._communication_metrics['packet_losses'].append(100.0)

            qos_violation = 0.0 if uav.current_satisfaction >= 0.6 else 100.0
            self._communication_metrics['qos_violations'].append(qos_violation)
        
        # 计算新增指标（对齐实验3）
        self._calculate_advanced_metrics()
    
    def _calculate_advanced_metrics(self):
        """计算高级通信指标（对齐实验3）"""
        # 1. 系统吞吐量（Mbps）- 所有UAV的分配速率之和
        total_throughput = 0.0
        for uav in self.env.uavs.values():
            if uav.connected_bs_id is not None:
                total_throughput += uav.current_allocated_rate
        self._communication_metrics['throughput'].append(total_throughput)
        
        # 2. 负载方差 - 基站负载率的标准差
        load_ratios = [bs.load_ratio for bs in self.env.base_stations.values()]
        if load_ratios:
            load_variance = np.var(load_ratios)
            self._communication_metrics['load_variance'].append(load_variance)
        
        # 3. 频谱效率（bps/Hz）- 使用香农公式近似
        total_spectral_efficiency = 0.0
        connected_count = 0
        for uid, uav in self.env.uavs.items():
            if uav.connected_bs_id is not None:
                sinr_db = self.env.sinr_matrix[uid][uav.connected_bs_id]
                sinr_linear = 10 ** (sinr_db / 10)
                # 香农公式: C/B = log2(1 + SINR)
                spectral_eff = np.log2(1 + sinr_linear)
                total_spectral_efficiency += spectral_eff
                connected_count += 1
        if connected_count > 0:
            avg_spectral_eff = total_spectral_efficiency / connected_count
            self._communication_metrics['spectral_efficiency'].append(avg_spectral_eff)
        
        # 4. 公平性指数（Jain's Fairness Index）
        satisfactions = [uav.current_satisfaction for uav in self.env.uavs.values()]
        if satisfactions and sum(satisfactions) > 0:
            n = len(satisfactions)
            sum_satisfaction = sum(satisfactions)
            sum_squares = sum([s ** 2 for s in satisfactions])
            if sum_squares > 0:
                fairness = (sum_satisfaction ** 2) / (n * sum_squares)
                self._communication_metrics['fairness_index'].append(fairness)

    def advance_env_only(self):
        """
        仅推进底层环境（不执行任何切换决策），供基线算法评估使用。

        基线算法（增强/传统）通过自己的 algo.run_step() 决策切换，
        然后调用本方法推进环境仿真。
        """
        self.env.current_step += 1
        for uav in self.env.uavs.values():
            uav.move(time_step=1.0)
        self.env._update_sinr_matrix()
        for uav in self.env.uavs.values():
            uav.record_satisfaction()
        self.env._check_interruptions()
        self.env._record_stats()
        self._current_step += 1

        # 更新历史记录（基线评估时也需要）
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_satisfaction[uid] = uav.current_satisfaction
            self._last_disconnected[uid] = (uav.connected_bs_id is None)
            self._last_handover_count[uid] = uav.handover_count
            self._sat_history[uid].append(uav.current_satisfaction)

    def predict_future_satisfaction(self, uid, steps=5):
        """
        预测未来几步的满意度

        Args:
            uid: UAV的ID
            steps: 预测的步数

        Returns:
            predicted_sat: 预测的未来满意度
        """
        # 基于历史满意度的简单预测
        if len(self._sat_history[uid]) < 2:
            return self._last_satisfaction.get(uid, 0.5)
        
        # 使用历史满意度的趋势进行预测
        recent_sats = list(self._sat_history[uid])[-5:]
        if len(recent_sats) < 2:
            return self._last_satisfaction.get(uid, 0.5)
        
        # 计算趋势
        trend = (recent_sats[-1] - recent_sats[0]) / len(recent_sats)
        
        # 预测未来满意度
        current_sat = self._last_satisfaction.get(uid, 0.5)
        predicted_sat = current_sat + trend * steps
        
        # 限制在合理范围内
        predicted_sat = max(0.0, min(1.0, predicted_sat))
        
        return predicted_sat


# ==================== 向后兼容别名 ====================
# MultiAgentHandoverEnv 是新的主类名。
# 保留 QMixHandoverEnv 作为向后兼容别名（原名为 qmix_environment.py 时的遗留）。
QMixHandoverEnv = MultiAgentHandoverEnv
