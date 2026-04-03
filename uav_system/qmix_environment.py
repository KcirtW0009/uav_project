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

与 QMIX 的对接:
- N 个 agent，每个 agent 动作空间 = NUM_STRATEGIES (5)
- 局部观测维度 = obs_dim（每 UAV 独立）
- 全局状态维度 = state_dim（全局信息聚合）
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

from .environment import NetworkEnvironmentWithRecognition
from .parametric_algorithm import (
    ParametricEnhancedAlgorithm, NUM_STRATEGIES, STRATEGY_CONFIGS
)
from .business import BusinessType


class QMixHandoverEnv:
    """
    UAV 切换的 QMIX 多智能体环境

    每个 UAV 作为一个独立 agent，每个 step 选择一种策略配置来指导切换行为。
    环境负责：执行所有 agent 的切换决策、推进仿真、计算奖励。

    Attributes:
        num_agents: agent 数量（等于 UAV 数量）
        num_bs: 基站数量
        action_dim: 每个 agent 的动作空间大小 (= NUM_STRATEGIES = 5)
        obs_dim: 每个 agent 的局部观测维度
        state_dim: 全局状态维度
    """

    def __init__(self, num_bs: int = 8, num_uav: int = 20,
                 max_steps: int = 150, seed: int = 42,
                 bs_capacity_range: Optional[Tuple[float, float]] = None):
        """
        Args:
            num_bs: 基站数量
            num_uav: UAV 数量
            max_steps: 每个 episode 最大步数
            seed: 随机种子
            bs_capacity_range: 基站容量范围 (min, max)
        """
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.max_steps = max_steps
        self.seed = seed
        self.bs_capacity_range = bs_capacity_range

        # 创建底层网络环境（无识别模型，QMIX 使用真实业务类型）
        self.env = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav,
            recognition_model=None, scaler=None,
            seed=seed,
            bs_capacity_range=bs_capacity_range,
        )

        # 禁用自适应识别更新器（QMIX 不需要）
        self.env.recognition_updater = None

        self.num_agents = num_uav
        self.action_dim = NUM_STRATEGIES

        # 维度计算
        self.obs_dim = self._calc_obs_dim()
        self.state_dim = self._calc_state_dim()

        # Episode 状态
        self._current_step = 0
        self._last_satisfaction = {}  # UAV id -> 上一步满意度
        self._last_disconnected = {}  # UAV id -> 上一步是否断连
        self._last_handover_count = {}  # UAV id -> 上一步累计切换次数

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
        总计: 4 * num_bs + 6
        """
        return 4 * self.num_bs + 6

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

        return np.concatenate([sinr_norm, loads, connected_onehot,
                               cap_ratios, biz, satisfaction, connected, velocity])

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

    def reset(self) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        重置环境

        Returns:
            obs_dict: {agent_id: obs}
            global_state: shape=(state_dim,)
        """
        self.env.reset()
        self._current_step = 0

        # 记录初始状态
        self._last_satisfaction = {}
        self._last_disconnected = {}
        self._last_handover_count = {}
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_satisfaction[uid] = uav.current_satisfaction
            self._last_disconnected[uid] = (uav.connected_bs_id is None)
            self._last_handover_count[uid] = uav.handover_count

        obs_dict = self.get_all_obs()
        global_state = self.get_global_state()
        return obs_dict, global_state

    def step(self, actions: Dict[int, int]) -> Tuple[
            Dict[int, np.ndarray], np.ndarray,
            Dict[int, float], float, bool, Dict]:
        """
        执行一个环境步

        Args:
            actions: {agent_id: action}，action 为策略索引 (0 ~ NUM_STRATEGIES-1)

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

        # ====== 1. 根据 actions 创建参数化算法并执行切换 ======
        # 将 UAV 按选择的策略分组，同一策略组共享一个算法实例
        strategy_groups: Dict[str, List[int]] = {}
        for uid, action in actions.items():
            strat_name = STRATEGY_CONFIGS.get(
                list(STRATEGY_CONFIGS.keys())[action],
                list(STRATEGY_CONFIGS.keys())[1]
            ) if action < len(STRATEGY_CONFIGS) else list(STRATEGY_CONFIGS.keys())[1]
            # 直接通过索引获取策略名
            strat_idx = min(action, len(STRATEGY_CONFIGS) - 1)
            strat_name = list(STRATEGY_CONFIGS.keys())[strat_idx]
            if strat_name not in strategy_groups:
                strategy_groups[strat_name] = []
            strategy_groups[strat_name].append(uid)

        # 为每个策略组创建算法实例并执行
        for strat_name, uav_ids in strategy_groups.items():
            algo = ParametricEnhancedAlgorithm.from_strategy_name(self.env, strat_name)
            # 执行该组 UAV 的切换
            for uid in uav_ids:
                decision = algo.make_intelligent_decision(uid)
                if decision is not None:
                    algo.execute_handover(uid, decision[0], decision[1])

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

        # ====== 3. 计算奖励 ======
        rewards = {}
        team_reward = 0.0

        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            new_sat = uav.current_satisfaction
            old_sat = old_sats.get(uid, 0.5)

            # a. 满意度变化奖励（个体）
            delta_sat = new_sat - old_sat
            r_individual = 3.0 * delta_sat

            # b. 满意度绝对值奖励
            if new_sat >= 0.3:
                r_individual += new_sat * 0.5

            # c. 切换惩罚
            new_ho = uav.handover_count
            old_ho = self._last_handover_count.get(uid, 0)
            if new_ho > old_ho:
                r_individual -= 0.8

            # d. 断连惩罚
            is_connected = uav.connected_bs_id is not None
            was_connected = not self._last_disconnected.get(uid, False)
            if not is_connected and was_connected:
                r_individual -= 3.0

            # e. 保持连接奖励
            if is_connected:
                r_individual += 0.05

            rewards[uid] = r_individual
            team_reward += r_individual

            # 更新历史
            self._last_satisfaction[uid] = new_sat
            self._last_disconnected[uid] = not is_connected
            self._last_handover_count[uid] = new_ho

        # ====== 4. 团队奖励归一化 ======
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
        }
        # 统计策略分布
        for uid, action in actions.items():
            strat_idx = min(action, len(STRATEGY_CONFIGS) - 1)
            strat_name = list(STRATEGY_CONFIGS.keys())[strat_idx]
            info['strategy_distribution'][strat_name] = \
                info['strategy_distribution'].get(strat_name, 0) + 1

        obs_dict = self.get_all_obs()
        global_state = self.get_global_state()

        return obs_dict, global_state, rewards, team_reward, done, info
