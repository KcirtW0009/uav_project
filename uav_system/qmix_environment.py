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
    UAV 切换的 QMIX 多智能体环境

    每个 UAV 作为一个独立 agent，每个 step 选择一种策略配置来指导切换行为。
    环境负责：执行所有 agent 的切换决策、推进仿真、计算奖励。

    Attributes:
        num_agents: agent 数量（等于 UAV 数量）
        num_bs: 基站数量
        action_dim: 每个 agent 的动作空间大小 (= 3: stay/best_sinr/best_capacity)
        obs_dim: 每个 agent 的局部观测维度
        state_dim: 全局状态维度
    """

    def __init__(self, num_bs: int = 8, num_uav: int = 20,
                 max_steps: int = 150, seed: int = 42,
                 bs_capacity_range: Optional[Tuple[float, float]] = None,
                 pos_range: float = 1000):
        """
        Args:
            num_bs: 基站数量
            num_uav: UAV 数量
            max_steps: 每个 episode 最大步数
            seed: 随机种子
            bs_capacity_range: 基站容量范围 (min, max)
            pos_range: 地图空间范围 (米)，默认 1000
        """
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.max_steps = max_steps
        self.seed = seed
        self.bs_capacity_range = bs_capacity_range
        self.pos_range = pos_range

        # 创建底层网络环境（无识别模型，QMIX 使用真实业务类型）
        self.env = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav,
            recognition_model=None, scaler=None,
            seed=seed,
            bs_capacity_range=bs_capacity_range,
        )

        # 禁用自适应识别更新器（QMIX 不需要）
        self.env.recognition_updater = None

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
        # 动作空间简化为 3 (高层决策，而非低级 BS 选择):
        #   action 0 = stay (不切换)
        #   action 1 = best_sinr (切换到 SINR 最高的 BS)
        #   action 2 = best_capacity (切换到可用容量/需求比最高的 BS)
        self.action_dim = 3

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
        总计: 4 * num_bs + 6 + action_dim + 2
        """
        return 4 * self.num_bs + 6 + self.action_dim + 2

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
            last_action[self._last_actions[uav_id]] = 1.0

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

        return np.concatenate([sinr_norm, loads, connected_onehot,
                               cap_ratios, biz, satisfaction, connected, velocity,
                               last_action, sat_trend, peer_avg])

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

        # 如果 pos_range != 默认值，重新缩放位置
        if self.pos_range != 1000:
            scale = self.pos_range / 1000.0
            for bs in self.env.base_stations.values():
                bs.position *= scale
            for uav in self.env.uavs.values():
                uav.position *= scale
            self.env._update_sinr_matrix()
            self.env._initialize_connections()

        self._current_step = 0

        # 记录初始状态
        self._last_satisfaction = {}
        self._last_disconnected = {}
        self._last_handover_count = {}
        self._last_actions = {}
        self._sat_history = {}
        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            self._last_satisfaction[uid] = uav.current_satisfaction
            self._last_disconnected[uid] = (uav.connected_bs_id is None)
            self._last_handover_count[uid] = uav.handover_count
            self._last_actions[uid] = 0
            self._sat_history[uid] = deque([uav.current_satisfaction], maxlen=10)
        # 不在此处 reset normalizer — EMA 需要跨 episode 持续积累
        # 调用者可通过 reset_normalizer() 手动重置

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

        # ====== 1. 根据 actions 执行切换 ======
        # action 0: stay (不切换)
        # action 1: best_sinr (切换到 SINR 最高的 BS)
        # action 2: best_capacity (切换到可用容量/需求比最高的 BS)
        # ---- 切换诊断 ----
        ep_switch_attempts = 0   # 尝试切换的 UAV 数
        ep_switch_success = 0    # 成功切换
        ep_switch_rollback = 0   # 切换失败但回滚成功
        ep_switch_disconnect = 0 # 切换失败且回滚也失败

        for uid, action in actions.items():
            if action == 0:
                continue  # stay
            if action == 1:
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
            else:
                continue  # 无效动作，忽略

            uav = self.env.uavs[uid]
            target_bs = self.env.base_stations[target_bs_id]
            ep_switch_attempts += 1

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

        # ====== 3. 计算奖励 (V6: 基于切换诊断 V5 的改进) ======
        # V5 问题诊断:
        #   - 训练中零实际切换 (good=0, bad=0)
        #   - stay_bonus=0.2 过高，agent 学到纯 stay 策略
        #   - switch_cost=-0.3 抑制了探索，即使切换可能有益
        #   - random_bs (33% stay) 打败 MAPPO (75% stay)
        # V6 设计:
        #   - stay_bonus 降至 0.05 (仅保持微弱正向信号)
        #   - switch_cost 降至 0.1 (降低切换门槛)
        #   - 成功切换(alloc成功): +0.3 奖励 (不依赖 delta_sat 的噪声)
        #   - 回滚: -0.05 轻微惩罚 (尝试了但没成功)
        #   - 断连: -1.0 (保持强惩罚)
        #   - delta_sat 权重保持 2.0
        rewards_raw = {}
        team_reward = 0.0
        # 诊断用组分统计
        ep_delta_sum = 0.0
        ep_stay_bonus_sum = 0.0
        ep_switch_cost_sum = 0.0
        ep_disconnect_sum = 0.0
        ep_good_switch = 0
        ep_bad_switch = 0

        for uid in range(self.num_agents):
            uav = self.env.uavs[uid]
            new_sat = uav.current_satisfaction
            old_sat = old_sats.get(uid, 0.5)

            delta_sat = new_sat - old_sat
            action = actions.get(uid, 0)

            # a. 满意度变化 (核心信号)
            r_delta = 2.0 * delta_sat
            ep_delta_sum += r_delta

            # b. 判断是否发生切换 (通过 handover_count)
            new_ho = uav.handover_count
            old_ho = self._last_handover_count.get(uid, 0)
            switched = (new_ho > old_ho)

            if switched:
                # ---- 切换成功 ----
                r_switch = -0.1   # 基础切换成本 (降低)
                ep_switch_cost_sum += 0.1
                if delta_sat > 0.05:
                    r_switch += 0.5   # 好的切换: net = +0.4
                    ep_good_switch += 1
                elif delta_sat < -0.05:
                    r_switch -= 0.3   # 坏的切换: net = -0.4
                    ep_bad_switch += 1
                else:
                    r_switch += 0.2   # 中性切换: net = +0.1 (鼓励探索)
                r_individual = r_delta + r_switch
            elif action != 0:
                # ---- 尝试切换但未成功 (allocation 失败) ----
                # 轻微惩罚 (浪费了一次切换机会)
                r_individual = r_delta - 0.05
            else:
                # ---- 留守 ----
                r_stay = 0.0
                if new_sat > 0.8:
                    r_stay = 0.05   # 维持高满意度: 微弱正向信号
                ep_stay_bonus_sum += r_stay
                r_individual = r_delta + r_stay

            # c. 断连惩罚 (适中，避免极端负值干扰学习)
            is_connected = uav.connected_bs_id is not None
            was_connected = not self._last_disconnected.get(uid, False)
            if not is_connected and was_connected:
                r_individual -= 1.5
                ep_disconnect_sum += 1.5

            rewards_raw[uid] = r_individual
            team_reward += r_individual

            # 更新历史
            self._last_satisfaction[uid] = new_sat
            self._last_disconnected[uid] = not is_connected
            self._last_handover_count[uid] = new_ho
            self._last_actions[uid] = action
            self._sat_history[uid].append(new_sat)

        # ====== 4. Reward 归一化 (EMA) ======
        rewards = self._reward_normalizer.normalize(rewards_raw)

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
                'stay_bonus': ep_stay_bonus_sum / max(self.num_agents, 1),
                'switch_cost': ep_switch_cost_sum / max(self.num_agents, 1),
                'disconnect_pen': ep_disconnect_sum / max(self.num_agents, 1),
                'good_switch': ep_good_switch,
                'bad_switch': ep_bad_switch,
                'raw_mean': np.mean(list(rewards_raw.values())),
                'norm_mean': np.mean(list(rewards.values())),
                'switch_attempts': ep_switch_attempts,
                'switch_success': ep_switch_success,
                'switch_rollback': ep_switch_rollback,
                'switch_disconnect': ep_switch_disconnect,
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
