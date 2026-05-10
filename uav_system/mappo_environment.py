"""
MAPPO 多智能体切换环境 (MultiAgentHandoverEnv / MAPPOHandoverEnv)

将 UAV 切换决策封装为 CTDE (Centralized Training Decentralized Execution) 多智能体 RL 环境。
支持 MAPPO 算法的训练与评估。

核心设计:
- 每个 UAV agent 独立选择切换策略（6种动作）
- 训练时使用全局状态（state），执行时仅使用局部观测（obs）
- 团队奖励函数：综合 rate_ratio 增量 + 反事实比较 + 业务权重 + 动作奖励 + EMA归一化

动作空间 (6维):
  0 = stay (不切换)
  1 = best_sinr (切换到 SINR 最高的 BS)
  2 = best_capacity (切换到可用容量/需求比最高的 BS)
  3 = sinr_capacity (SINR 和容量加权组合)
  4 = predictive (基于预测的切换)
  5 = business_specific (基于业务类型的差异化切换)

观测维度: obs_dim = 4 * num_bs + 9 + action_dim(6) + 2 = 4*num_bs + 17
全局状态: state_dim = 3 * num_bs + 7

接口:
- reset() -> (obs_dict, global_state)
- step(actions_dict) -> (obs_dict, global_state, rewards_dict, team_reward, done, info)
- advance_env_only() -> 仅推进底层环境（供基线算法评估使用）

业务识别集成:
- 无 recognition_model: 使用真实业务类型（训练模式，ground truth）
- 有 recognition_model + scaler: 使用模型预测结果（评估模式，带识别噪声）

依赖:
- 底层环境: NetworkEnvironmentWithRecognition (uav_system/environment.py)
- 基站/实体: uav_system/entities.py, business.py

注意:
- 本模块原名 qmix_environment.py (QMIX 设计遗留)，已重命名为 mappo_environment.py
- QMixHandoverEnv 类名保留向后兼容，推荐用 MultiAgentHandoverEnv 别名
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
    """指数移动平均归一化器，用于降低 reward 的 CV"""

    def __init__(self, num_agents: int, decay: float = 0.999):
        self.mean = np.zeros(num_agents, dtype=np.float64)
        self.var = np.ones(num_agents, dtype=np.float64)
        self.decay = decay
        self.count = 0

    def normalize(self, rewards_dict: Dict[int, float]) -> Dict[int, float]:
        vec = np.array([rewards_dict[i] for i in range(len(rewards_dict))], dtype=np.float64)
        self.count += 1
        batch_mean = vec.mean()
        batch_var = vec.var() if len(vec) > 1 else 1.0
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.var = self.decay * self.var + (1 - self.decay) * batch_var
        std = np.sqrt(np.maximum(self.var, 1e-8))
        normed = (vec - self.mean) / std
        return {i: float(normed[i]) for i in range(len(rewards_dict))}

    def reset(self):
        self.count = 0
        self.mean[:] = 0.0
        self.var[:] = 1.0


class MultiAgentHandoverEnv:
    """
    多智能体 UAV 切换环境（CTDE 架构）— MAPPO 主环境类

    通用多智能体强化学习环境，支持 MAPPO / QMIX / IPPO 等算法。
    每个 UAV 作为一个独立 agent，每个 step 选择一种切换策略。

    动作空间 (6维):
      0 = stay (不切换)
      1 = best_sinr (切换到 SINR 最高的 BS)
      2 = best_capacity (切换到可用容量/需求比最高的 BS)
      3 = sinr_capacity (SINR 和容量加权组合)
      4 = predictive (基于预测的切换)
      5 = business_specific (基于业务类型的差异化切换)

    Attributes:
        num_agents: agent 数量（等于 UAV 数量）
        num_bs: 基站数量
        action_dim: 每个 agent 的动作空间大小 (= 6)
        obs_dim: 每个 agent 的局部观测维度 (= 4*num_bs + 17)
        state_dim: 全局状态维度 (= 3*num_bs + 7)
    """

    # 向后兼容别名
    QMixHandoverEnv = None  # 将在模块底部设置


    def __init__(self, num_bs: int, num_uav: int, max_steps: int = 1000, seed: int = None,
                 bs_capacity_range: tuple = (500, 1000), pos_range: int = 1000,
                 use_state_smoothing: bool = True, use_env_simplification: bool = False,
                 recognition_model=None, scaler=None,
                 event_probability: float = 0.05,
                 scenario: str = 'default'):  # ✅ 新增：场景ID（用于设置业务混合比例）
        """
        初始化 MAPPO 切换环境

        Args:
            num_bs: 基站数量
            num_uav: UAV 数量
            max_steps: 每个 episode 最大步数
            seed: 随机种子
            bs_capacity_range: 基站容量范围 (min, max) Mbps
            pos_range: 地图空间范围 (米)，默认 1000
            use_state_smoothing: 是否使用状态平滑机制（5步移动平均）
            use_env_simplification: 是否使用环境简化机制（固定BS位置、降速）
            recognition_model: 业务识别模型（评估模式传入，训练模式为None）
            scaler: 识别模型的标准化器（与recognition_model配套）
            event_probability: 随机事件概率（对齐实验3，默认0=无事件）
            scenario: 场景ID（用于设置业务混合比例和基站容量，如'industrial_inspection'等）
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
        """配置随机事件机制（对齐实验3的 event_probability）

        Args:
            probability: 每步每UAV发生随机事件的概率 [0, 1]
        """
        # 底层环境 NetworkEnvironmentWithRecognition 继承自 EnhancedNetworkEnvironment
        # 检查是否有随机事件相关属性
        if hasattr(self.env, '_random_event_enabled'):
            self.env._random_event_enabled = True
            self.env._event_probability = probability
            print(f"[MAPPO Env] 已启用随机事件, probability={probability}")

    def _calc_obs_dim(self) -> int:
        """
        局部观测维度 (per agent):
        - SINR 向量: num_bs
        - 基站负载率: num_bs
        - 当前连接 BS one-hot: num_bs
        - 可用容量/需求比: num_bs
        - 业务类型 one-hot: 3
        - 当前满意度: 1
        - 连接状态: 1
        - 移动速度: 1
        - 上次动作 one-hot: action_dim (6)
        - 满意度变化趋势: 1
        - 同类型 UAV 平均满意度: 1
        - 历史满意度 (最近3步): 3
        总计: 4 * num_bs + 9 + action_dim + 2
        """
        return 4 * self.num_bs + 9 + self.action_dim + 2

    def _calc_state_dim(self) -> int:
        """
        全局状态维度:
        - 所有基站负载率: num_bs
        - 所有基站可用容量: num_bs
        - 所有基站故障状态: num_bs
        - 各业务类型数量: 3
        - 全局平均满意度: 1
        - 全局断连率: 1
        - 全局中断率: 1
        - 当前步数 (归一化): 1
        总计: 3 * num_bs + 7
        """
        return 3 * self.num_bs + 7

    def get_obs(self, uav_id: int) -> np.ndarray:
        """
        获取单个 UAV 的局部观测

        业务识别集成逻辑:
        - 如果提供了 recognition_model（评估模式）：使用模型预测的业务类型
        - 如果未提供 recognition_model（训练模式）：使用真实业务类型 (ground truth)

        Returns:
            obs: shape=(obs_dim,) 的浮点数组
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
