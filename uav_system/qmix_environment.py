"""
QMIX 多智能体环境

将 UAV 切换决策封装为 CTDE (Centralized Training Decentralized Execution) 多智能体 RL 环境。

核心设计:
- 每个 UAV agent 独立选择策略配置（θ）来控制自己的切换行为
- 训练时使用全局状态（所有 UAV 观测聚合），执行时仅使用局部观测
- 团队奖励函数：综合满意度 + 切换惩罚 + 资源利用

接口:
- reset() -> (obs_dict, global_state)
- step(actions_dict) -> (obs_dict, global_state, rewards_dict, team_reward, done, info)

与 MAPPO 的对接:
- N 个 agent，每个 agent 动作空间 = 3 (stay / best_sinr / best_capacity)
- 局部观测维度 = obs_dim（每 UAV 独立）
- 全局状态维度 = state_dim（全局信息聚合）
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Optional
from collections import deque

from .environment import NetworkEnvironmentWithRecognition
# [已弃用] 以下导入原为 QMIX 元控制器设计，MAPPO 未使用（action_dim 已简化为 3）
# from .parametric_algorithm import (
#     ParametricEnhancedAlgorithm, NUM_STRATEGIES, STRATEGY_CONFIGS
# )
from .business import BusinessType


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


class QMixHandoverEnv:
    """
    多智能体 UAV 切换环境（CTDE 架构）

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
        obs_dim: 每个 agent 的局部观测维度
        state_dim: 全局状态维度
    """

    # 类别名：提高可读性，MAPPO 实验使用此名称
    # 用法: from .qmix_environment import MultiAgentHandoverEnv as MAPPOHandoverEnv


    def __init__(self, num_bs: int, num_uav: int, max_steps: int = 1000, seed: int = None,
                 bs_capacity_range: tuple = (500, 1000), pos_range: int = 1000,
                 use_state_smoothing: bool = True, use_env_simplification: bool = False):
        """
        初始化 QMIX 切换环境 (无识别噪声)

        Args:
            num_bs: 基站数量
            num_uav: UAV 数量
            max_steps: 每个 episode 最大步数
            seed: 随机种子
            bs_capacity_range: 基站容量范围 (min, max)
            pos_range: 地图空间范围 (米)，默认 1000
            use_state_smoothing: 是否使用状态平滑机制
            use_env_simplification: 是否使用环境简化机制
        """
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.max_steps = max_steps
        self.seed = seed
        self.bs_capacity_range = bs_capacity_range
        self.pos_range = pos_range
        self.use_state_smoothing = use_state_smoothing
        self.use_env_simplification = use_env_simplification

        # 创建底层网络环境（无识别模型，QMIX 使用真实业务类型）
        self.env = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav,
            recognition_model=None, scaler=None,
            seed=seed,
            bs_capacity_range=bs_capacity_range,
        )

        # 禁用自适应识别更新器（QMIX 不需要）
        self.env.recognition_updater = None

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
        - 上次动作 one-hot: action_dim (3)
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

        # 5. 业务类型 one-hot (真实类型)
        biz = np.zeros(3)
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
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_satisfaction[uid] = uav.current_satisfaction
            self._last_disconnected[uid] = (uav.connected_bs_id is None)
            self._last_handover_count[uid] = uav.handover_count
            self._last_actions[uid] = 0
            self._last_rate_ratios[uid] = uav.current_allocated_rate / max(uav.required_rate, 1e-6)
            self._sat_history[uid] = deque([uav.current_satisfaction], maxlen=10)
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
            actions: {agent_id: action}，action 取值: 0=stay, 1=best_sinr, 2=best_capacity

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

            # ====== 预检查: 在释放旧资源前，先确认目标 BS 是否有足够容量 ======
            # 计算释放旧资源后 UAV 自身可贡献的容量
            # 跨BS切换时也计入自身释放容量，因为切换流程是先释放旧BS再分配新BS
            uav_self_free = uav.current_allocated_rate  # UAV 自身释放的容量
            effective_target_cap = target_bs.available_capacity + uav_self_free

            # 检查目标 BS 能否至少以最低降级比率接受
            min_ratio = uav.qos_profile.get_feasible_downgrade_ratios()[-1]  # 最低降级比率
            min_required = uav.required_rate * min_ratio


            if effective_target_cap < min_required and target_bs_id != uav.connected_bs_id:
                # 目标 BS 容量不足，跳过切换（保持当前连接）
                ep_switch_rollback += 1  # 计为回滚（切换未实际发生）
                handover_end = time.time()
                handover_latency = (handover_end - handover_start) * 1000
                handover_latencies.append(handover_latency)
                continue

            # 释放当前 BS 资源
            old_bs_id = uav.connected_bs_id
            if old_bs_id is not None:
                old_bs = self.env.base_stations[old_bs_id]
                old_bs.connected_uavs.pop(uid, None)
                old_bs.current_load -= uav.current_allocated_rate

            # 尝试切换到目标 BS（按业务类型降级，与增强算法一致）
            allocated = False
            for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                if target_bs.allocate(uid, uav.required_rate * ratio):
                    uav.connected_bs_id = target_bs_id
                    uav.current_allocated_rate = uav.required_rate * ratio
                    uav.handover_count += 1
                    allocated = True
                    ep_switch_success += 1
                    break
            if not allocated:
                # 切换失败，尝试回滚到旧 BS（也用降级比率）
                if old_bs_id is not None:
                    old_bs = self.env.base_stations[old_bs_id]
                    for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                        if old_bs.allocate(uid, uav.required_rate * ratio):
                            uav.connected_bs_id = old_bs_id
                            uav.current_allocated_rate = uav.required_rate * ratio
                            allocated = True  # 回滚成功
                            ep_switch_rollback += 1
                            break
                if not allocated:
                    # 回滚也失败，断连
                    uav.connected_bs_id = None
                    uav.current_allocated_rate = 0.0
                    ep_switch_disconnect += 1

            # 记录切换结束时间并计算延迟
            handover_end = time.time()
            # 基础延迟 + 处理时间（模拟真实环境）
            base_handover_latency = 5.0  # 基础切换延迟5ms
            processing_latency = (handover_end - handover_start) * 1000  # 转换为毫秒
            # 根据目标基站负载调整延迟
            target_bs = self.env.base_stations.get(action)
            if target_bs:
                load_factor = 1.0 + target_bs.load_ratio * 0.5  # 负载越高，延迟越大
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

        # ====== 3. 计算奖励 (V11: 连续信号 + 反事实比较) ======
        # V11 相对 V9 的改进:
        #   a. 核心信号从分段满意度→连续 rate_ratio，解决 advantage 趋零问题
        #   b. 加入反事实比较: 切换后的表现 vs 同类 UAV 基线
        #   c. 留守策略: 高负载下取消留守正信号，低 sat 留守加强惩罚
        #   d. 切换奖励: 去除硬阈值，用连续的满意度增量
        rewards_raw = {}
        team_reward = 0.0
        # 诊断用组分统计
        ep_delta_sum = 0.0
        ep_value_reward_sum = 0.0
        ep_biz_reward_sum = 0.0
        ep_action_reward_sum = 0.0
        ep_connect_reward_sum = 0.0
        ep_good_switch = 0
        ep_bad_switch = 0
        # 预计算同类 UAV 的平均 rate_ratio (用于反事实比较)
        peer_rate_ratios = {}  # biz_type -> list of rate_ratio
        for _uid in range(self.num_agents):
            _uav = self.env.uavs[_uid]
            _rr = _uav.current_allocated_rate / max(_uav.required_rate, 1e-6)
            _bt = _uav.true_business_type.value
            peer_rate_ratios.setdefault(_bt, []).append(_rr)
        peer_avg_rr = {bt: np.mean(rrs) for bt, rrs in peer_rate_ratios.items()}

        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            new_sat = uav.current_satisfaction
            old_sat = old_sats.get(uid, 0.5)

            delta_sat = new_sat - old_sat
            action = actions.get(uid, 0)
            biz_type = uav.true_business_type.value

            # --- a. 连续速率比信号 (替代分段满意度) ---
            # rate_ratio = allocated / ideal，连续且有自然梯度
            # reward = 当前 rate_ratio - 旧 rate_ratio（增量式，消除常量）
            old_rr = old_rate_ratios.get(uid, 0.5)  # 需在上方保存
            new_rr = uav.current_allocated_rate / max(uav.required_rate, 1e-6)
            delta_rr = new_rr - old_rr
            r_delta = 5.0 * delta_rr  # 适度权重
            ep_delta_sum += r_delta

            # --- b. 反事实比较信号 (方案3核心) ---
            # 如果切换了，比较: 我的 rate_ratio vs 同类 UAV 平均 rate_ratio
            # 正值 = 我比同类做得好（切换有效），负值 = 切换后反而更差
            r_counterfactual = 0.0
            new_ho = uav.handover_count
            old_ho = self._last_handover_count.get(uid, 0)
            switched = (new_ho > old_ho)
            if switched:
                avg_rr = peer_avg_rr.get(biz_type, new_rr)
                relative_gain = new_rr - avg_rr  # >0 表示优于同类平均
                r_counterfactual = 3.0 * relative_gain
            ep_value_reward_sum += r_counterfactual

            # --- c. 业务类型权重 ---
            r_biz = 0.0
            if abs(delta_sat) > 1e-4:
                biz_weight = {0: 2.0, 1: 2.5, 2: 1.5}.get(biz_type, 2.0)
                r_biz = biz_weight * delta_rr  # 用连续信号替代分段满意度
            ep_biz_reward_sum += r_biz

            # --- d. 动作奖励 (V14: 增强信号强度，明确区分好坏动作) ---
            # 添加切换惩罚以降低Ping抖动（频繁切换会导致抖动增加）
            r_action = 0.0
            
            # 切换惩罚：无论切换是否成功，都给予轻微惩罚以抑制频繁切换
            switch_penalty = -0.05 if switched else 0.0
            
            if switched:
                # 增强切换奖励信号，让有益切换有更明显的正收益
                if delta_sat > 0.05:
                    # 成功切换：大幅奖励（但减去切换惩罚）
                    r_action = 8.0 * delta_sat + 0.5 + switch_penalty  # 基础奖励+比例奖励+切换惩罚
                    ep_good_switch += 1
                elif delta_sat > 0.0:
                    # 轻微改善：小奖励（但减去切换惩罚）
                    r_action = 4.0 * delta_sat + 0.1 + switch_penalty
                elif delta_sat > -0.05:
                    # 轻微恶化：小惩罚（加上切换惩罚）
                    r_action = 4.0 * delta_sat - 0.05 + switch_penalty
                else:
                    # 严重恶化：大惩罚（加上切换惩罚）
                    r_action = 6.0 * delta_sat - 0.2 + switch_penalty
                    ep_bad_switch += 1
            elif action != 0:
                # 尝试切换但未成功：轻微惩罚（加上切换惩罚）
                r_action = -0.15 + switch_penalty
            else:
                # 留守: 强化分层信号，明确区分好坏
                if new_sat < 0.3:
                    r_action = -0.60   # 极低 sat 留守: 极强惩罚
                elif new_sat < 0.5:
                    r_action = -0.35   # 低 sat 留守: 强惩罚
                elif new_sat < 0.7:
                    r_action = -0.10   # 中等 sat: 轻微惩罚，鼓励探索
                elif new_sat < 0.85:
                    r_action = 0.08    # 较高 sat: 适度奖励
                else:
                    r_action = 0.15    # 高 sat 留守: 明确奖励
            ep_action_reward_sum += r_action

            # --- e. 连接状态奖励 (仅诊断，不参与个体 reward) ---
            # 断连惩罚从个体 reward 中分离，避免结构性噪音淹没学习信号。
            # 原因: 约 30% UAV 因容量不足始终断连，每步 r_connect=-2.5
            # 导致 reward 方差的 90%+ 来自断连（与 action 无关），GAE 无法区分好坏动作。
            # 改为 episode 级别团队惩罚（见下方 team_reward 调整）。
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

            # 综合奖励 (不含 connect 惩罚，避免结构性噪音干扰 GAE)
            r_individual = r_delta + r_counterfactual + r_biz + r_action

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
        rewards = self._reward_normalizer.normalize(rewards_raw)

        # ====== 4.5 断连惩罚加入团队奖励 (episode-level, 不影响个体 GAE) ======
        # 将平均 connect 惩罚作为团队级信号，让 Critic 感知断连影响
        # 但个体 UAV 的 reward 不含此惩罚，避免结构性噪音淹没 per-agent 学习信号
        avg_connect_penalty = ep_connect_reward_sum / max(self.num_agents, 1)
        team_reward += avg_connect_penalty

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
            'strategy_distribution': {},
            'reward_diag': {
                'delta_sum': ep_delta_sum / max(self.num_agents, 1),
                'value_reward': ep_value_reward_sum / max(self.num_agents, 1),
                'biz_reward': ep_biz_reward_sum / max(self.num_agents, 1),
                'action_reward': ep_action_reward_sum / max(self.num_agents, 1),
                'connect_reward': ep_connect_reward_sum / max(self.num_agents, 1),
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


# ==================== 类别名 ====================
# QMixHandoverEnv 是通用多智能体环境，QMIX 和 MAPPO 共用。
# 提供别名以增强代码可读性。
MultiAgentHandoverEnv = QMixHandoverEnv
