"""
RL 环境包装器

将 NetworkEnvironmentWithRecognition 包装为标准 RL 接口 (reset/step)，
供 DQN/PPO 等 RL Agent 使用。仅控制单个目标 UAV，其余 UAV 由增强启发式算法管理。
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from .environment import NetworkEnvironmentWithRecognition
from .algorithms import EnhancedHandoverAlgorithm
from .business import BusinessType


class RLHandoverEnv:
    """
    UAV 切换决策的 RL 环境

    将网络环境包装为标准的 reset / step 接口：
    - 状态 (state): 目标 UAV 的 SINR 向量、基站负载率、业务类型、速度、满意度、当前连接 BS 等
    - 动作 (action): 离散动作，编码为 (保持 / 切换到基站X + 降级比例Y)
    - 奖励 (reward): 基于满意度变化、切换惩罚、吞吐量增益的组合奖励

    设计原则：
    - 零侵入：不修改任何现有模块
    - 单 UAV 控制：避免多智能体耦合
    - 其余 UAV 由 EnhancedHandoverAlgorithm 自动管理
    """

    # 动作编码：动作0=保持，动作1~N = 切换到 bs_id（全速率）
    # 简化动作空间：原 1+8*3=25 → 现 1+8=9，大幅提高 stay 的探索概率 (4%→11%)
    DOWNGRADE_RATIOS = [1.0]

    def __init__(self, env: NetworkEnvironmentWithRecognition, target_uav_id: int = 0,
                 max_steps: int = 150, reward_config: Optional[Dict] = None,
                 skip_recognition: bool = True):
        """
        Args:
            env: 底层网络环境实例
            target_uav_id: RL 控制的目标 UAV 编号
            max_steps: 每个 episode 的最大步数
            reward_config: 奖励函数权重配置，默认 None 使用默认值
            skip_recognition: 是否跳过业务识别（RL训练时建议 True 以加速仿真）
        """
        self.env = env
        self.target_uav_id = target_uav_id
        self.max_steps = max_steps
        self.skip_recognition = skip_recognition
        self._current_step = 0
        self._last_satisfaction = 0.0
        self._last_allocated_rate = 0.0
        self._last_connected = True
        self._total_handovers = 0

        # 切换时间追踪（防止乒乓振荡）
        self._steps_since_last_switch = 0
        self._last_switched_from_bs = None   # 上次切换前连接的 BS ID
        self._consecutive_stays = 0           # 连续 stay 步数
        self._pre_switch_cooldown = 0         # 动作前冷却步数（用于奖励计算）
        self._cooldown_steps = 10             # 冷却窗口（步数）

        # 奖励权重配置
        # 设计原则：让"保持连接"成为默认最优策略，仅在满意度显著下降时才值得切换
        # 切换一次总成本最高: 1.5 + 0.3 + 0.5 = 2.3
        # 需要delta_sat > 1.5/3.0 = 0.5 才值得切换（满意度大幅改善时）
        # stay每步收益: 0.1 + 0.05 + sat*0.5 ≈ 0.65
        self.reward_config = reward_config or {
            'satisfaction_weight': 3.0,        # 满意度变化权重（从10降低，防止delta_sat主导）
            'satisfaction_baseline': 0.3,      # 满意度绝对值奖励(当高于此阈值时)
            'handover_penalty': 1.5,           # 每次切换惩罚（从0.5提高，大幅增加切换成本）
            'invalid_switch_penalty': 0.5,     # 无效切换惩罚
            'disconnect_penalty': 3.0,         # 断连惩罚
            'throughput_bonus': 0.01,          # 吞吐量增益权重
            'stay_bonus': 0.1,                 # 维持连接的奖励（从0.05提高）
            'cooldown_penalty': 0.3,           # 冷却期内切换的额外惩罚
            'ping_pong_penalty': 0.5,          # 切回上一个 BS 的额外惩罚
            'consecutive_stay_bonus': 0.05,    # 连续 stay >5 步的额外奖励
        }

        # 为非目标 UAV 创建增强算法（自动管理）
        self._heuristic_algo = EnhancedHandoverAlgorithm(env)

        # 动作映射表（延迟构建，依赖 num_bs）
        self._action_map: List[Tuple] = []
        self._build_action_map()

    def _build_action_map(self):
        """构建动作映射表：动作索引 -> (类型, bs_id, downgrade_ratio)"""
        self._action_map = [('stay', None, 1.0)]  # 动作0: 保持
        for bs_id in range(self.env.num_bs):
            for ratio in self.DOWNGRADE_RATIOS:
                self._action_map.append(('switch', bs_id, ratio))

    @property
    def state_dim(self) -> int:
        """
        状态空间维度

        状态组成 (共 num_bs*4 + 7 维):
        - [0, num_bs): 对每个基站的 SINR 值 (归一化到 [0, 1])
        - [num_bs, 2*num_bs): 各基站负载率
        - [2*num_bs, 3*num_bs): 当前连接 BS one-hot
        - [3*num_bs, 3*num_bs+3): 业务类型 one-hot 编码
        - [3*num_bs+3, 3*num_bs+4): 移动速度 (归一化)
        - [3*num_bs+4, 3*num_bs+5): 当前满意度
        - [3*num_bs+5, 3*num_bs+6): 连接状态 (1=已连接, 0=断连)
        - [3*num_bs+6, 3*num_bs+7): 切换冷却计时 (归一化，防止乒乓)
        - [3*num_bs+7, 4*num_bs+7): 上次切换来源 BS one-hot
        """
        return self.env.num_bs * 4 + 7

    @property
    def action_dim(self) -> int:
        """动作空间维度"""
        return len(self._action_map)  # 1 + num_bs * len(DOWNGRADE_RATIOS)

    @property
    def action_map(self) -> List[Tuple]:
        """返回动作映射表"""
        return self._action_map

    def get_state(self) -> np.ndarray:
        """
        获取当前状态向量

        状态组成 (共 num_bs*4 + 7 维):
        - [0, num_bs): 对每个基站的 SINR 值 (归一化到 [0, 1])
        - [num_bs, 2*num_bs): 各基站负载率
        - [2*num_bs, 3*num_bs): 当前连接 BS one-hot 编码（让 DQN 知道自己连着哪个）
        - [3*num_bs, 3*num_bs+3): 业务类型 one-hot 编码
        - [3*num_bs+3, 3*num_bs+4): 移动速度 (归一化)
        - [3*num_bs+4, 3*num_bs+5): 当前满意度
        - [3*num_bs+5, 3*num_bs+6): 连接状态 (1=已连接, 0=断连)
        - [3*num_bs+6, 3*num_bs+7): 切换冷却计时 (归一化)
        - [3*num_bs+7, 4*num_bs+7): 上次切换来源 BS one-hot
        """
        uav = self.env.uavs[self.target_uav_id]
        n_bs = self.env.num_bs

        # 1. SINR 向量 (归一化)
        sinr_raw = self.env.sinr_matrix[self.target_uav_id, :n_bs]
        sinr_norm = np.clip((sinr_raw + 10) / 40, 0, 1)

        # 2. 基站负载率
        loads = np.array([bs.load_ratio for bs in self.env.base_stations.values()])

        # 3. 当前连接 BS one-hot（关键新特征：让 DQN 知道当前连接）
        connected_bs_onehot = np.zeros(n_bs)
        if uav.connected_bs_id is not None and uav.connected_bs_id < n_bs:
            connected_bs_onehot[uav.connected_bs_id] = 1.0

        # 4. 业务类型 one-hot (使用真实类型，RL 不需要识别过程)
        biz = np.zeros(3)
        biz[uav.true_business_type.value] = 1

        # 5. 移动速度 (归一化到 [0, 1]，假设最大 30 m/step)
        velocity_norm = min(np.linalg.norm(uav.velocity) / 30.0, 1.0)

        # 6. 当前满意度
        satisfaction = uav.current_satisfaction

        # 7. 连接状态
        connected = 1.0 if uav.connected_bs_id is not None else 0.0

        # 8. 切换冷却计时（归一化，让 DQN 感知"刚切换过"）
        cooldown_norm = min(self._steps_since_last_switch / 20.0, 1.0)

        # 9. 上次切换来源 BS one-hot（让 DQN 知道从哪个 BS 切过来的）
        last_bs_onehot = np.zeros(n_bs)
        if self._last_switched_from_bs is not None and self._last_switched_from_bs < n_bs:
            last_bs_onehot[self._last_switched_from_bs] = 1.0

        return np.concatenate([sinr_norm, loads, connected_bs_onehot, biz,
                              [velocity_norm, satisfaction, connected, cooldown_norm],
                              last_bs_onehot])

    def get_invalid_actions(self) -> List[int]:
        """
        获取当前状态下应屏蔽的无效动作索引

        Returns:
            无效动作索引列表（切换到当前已连接基站的所有降级档位）
        """
        uav = self.env.uavs[self.target_uav_id]
        n_bs = self.env.num_bs
        num_ratios = len(self.DOWNGRADE_RATIOS)
        invalid = []

        if uav.connected_bs_id is not None:
            # 屏蔽"切换到当前基站"的所有降级档位
            for r in range(num_ratios):
                action_idx = 1 + uav.connected_bs_id * num_ratios + r
                invalid.append(action_idx)

        return invalid

    def _decode_action(self, action: int) -> Tuple[str, Optional[int], float]:
        """
        解码离散动作为具体的切换决策

        Args:
            action: 离散动作索引 (0 ~ action_dim-1)

        Returns:
            (动作类型, 目标基站ID, 降级比例)
        """
        if action < 0 or action >= self.action_dim:
            return ('stay', None, 1.0)
        return self._action_map[action]

    def _compute_reward(self, action: int, actual_switch: bool = False,
                        invalid_action: bool = False) -> float:
        """
        计算即时奖励

        奖励组成:
        1. 满意度变化 (主奖励)
        2. 满意度绝对值奖励 (鼓励维持高满意度)
        3. 降级惩罚 (分配速率低于理想时扣分)
        4. 切换惩罚 (抑制乒乓效应，仅实际切换时)
        5. 无效切换惩罚 (切换到同基站/故障基站，作为额外安全网)
        6. 断连惩罚
        7. 维持连接奖励
        """
        uav = self.env.uavs[self.target_uav_id]
        rc = self.reward_config

        current_satisfaction = uav.current_satisfaction
        current_rate = uav.current_allocated_rate
        ideal_rate = uav.required_rate
        is_connected = uav.connected_bs_id is not None

        # 1. 满意度变化（核心奖励）
        delta_sat = current_satisfaction - self._last_satisfaction
        reward = rc['satisfaction_weight'] * delta_sat

        # 2. 满意度绝对值奖励（让RL有动力维持高满意度，而非仅仅避免惩罚）
        baseline = rc.get('satisfaction_baseline', 0.3)
        if current_satisfaction >= baseline:
            reward += current_satisfaction * 0.5  # 高满意度时的正向激励

        # 3. 降级惩罚（分配速率低于理想速率时扣分，引导RL争取全速率）
        if ideal_rate > 0 and current_rate < ideal_rate:
            rate_deficit = (ideal_rate - current_rate) / ideal_rate
            reward -= rate_deficit * 0.3

        # 4. 切换惩罚（仅在 _apply_action 中实际切换了才惩罚）
        if actual_switch:
            reward -= rc['handover_penalty']

            # 4a. 冷却惩罚：距上次切换越近，惩罚越大
            if self._pre_switch_cooldown < self._cooldown_steps:
                cooldown_ratio = 1 - self._pre_switch_cooldown / self._cooldown_steps
                reward -= rc['cooldown_penalty'] * cooldown_ratio

            # 4b. 乒乓惩罚：切回上一个 BS 时额外惩罚（仅在冷却期内）
            _, target_bs_id, _ = self._decode_action(action)
            if (self._last_switched_from_bs is not None and
                    target_bs_id == self._last_switched_from_bs and
                    self._pre_switch_cooldown < self._cooldown_steps):
                reward -= rc['ping_pong_penalty']

        # 5. 无效切换惩罚（安全网：如果动作掩码没生效，这里也会惩罚）
        if invalid_action:
            reward -= rc.get('invalid_switch_penalty', 0.3)

        # 6. 断连惩罚
        if not is_connected and self._last_connected:
            reward -= rc['disconnect_penalty']

        # 7. 维持连接奖励 + 连续 stay 奖励（鼓励稳定性）
        action_type = self._decode_action(action)[0]
        if is_connected and action_type == 'stay':
            reward += rc['stay_bonus']
            if self._consecutive_stays > 5:
                reward += rc['consecutive_stay_bonus']

        return reward

    def _apply_action(self, action: int) -> Tuple[bool, bool]:
        """
        将 RL 动作应用到目标 UAV

        Returns:
            (actual_switch, invalid_action):
            - actual_switch: 是否实际执行了有效切换
            - invalid_action: 是否为无效动作（切到同基站/故障基站）
        """
        action_type, target_bs_id, downgrade_ratio = self._decode_action(action)

        if action_type == 'stay':
            return False, False

        # 执行切换
        uav = self.env.uavs[self.target_uav_id]
        target_bs = self.env.base_stations[target_bs_id]

        # 跳过无效切换：切换到当前基站
        if uav.connected_bs_id == target_bs_id:
            return False, True

        # 跳过故障基站
        if target_bs.failure_state:
            return False, True

        required_rate = uav.required_rate * downgrade_ratio
        old_bs_id = uav.connected_bs_id
        old_bs = self.env.base_stations[old_bs_id] if old_bs_id is not None else None
        old_allocated_rate = uav.current_allocated_rate

        # 释放旧基站
        if old_bs_id is not None and old_bs_id != target_bs_id:
            old_bs.release(self.target_uav_id)
            self.env.connection_matrix[self.target_uav_id, old_bs_id] = 0

        # 直接分配
        if target_bs.allocate(self.target_uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[self.target_uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self._last_switched_from_bs = old_bs_id
            return True, False

        # 分配失败，尝试抢占
        freed, kicked_ids = target_bs.kick_low_priority(uav, self.env.uavs)
        if freed >= required_rate and target_bs.allocate(self.target_uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[self.target_uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self._last_switched_from_bs = old_bs_id
            # 为被抢占的 UAV 尝试软迁移
            self._soft_migrate_kicked(kicked_ids, target_bs_id)
            return True, False

        # 抢占也失败，尝试回滚
        if old_bs_id is not None and old_bs_id != target_bs_id:
            if old_bs.available_capacity >= old_allocated_rate:
                old_bs.allocate(self.target_uav_id, old_allocated_rate)
                self.env.connection_matrix[self.target_uav_id, old_bs_id] = 1
                uav.connected_bs_id = old_bs_id
                return False, False

        # 回滚失败，断连
        uav.connected_bs_id = None
        uav.current_allocated_rate = 0.0
        return False, False

    def _soft_migrate_kicked(self, kicked_ids: list, exclude_bs_id: int):
        """为被抢占的 UAV 尝试软迁移"""
        for kicked_id in kicked_ids:
            kicked_uav = self.env.uavs.get(kicked_id)
            if kicked_uav is None or kicked_uav.connected_bs_id is not None:
                continue
            best_alt_bs, best_alt_score, best_ratio = None, -1, 1.0
            for bs_id, bs in self.env.base_stations.items():
                if bs_id == exclude_bs_id:
                    continue
                for r in kicked_uav.qos_profile.get_feasible_downgrade_ratios():
                    needed = kicked_uav.required_rate * r
                    if bs.available_capacity >= needed * 0.9:
                        score = self.env.sinr_matrix[kicked_id, bs_id] + bs.available_capacity * 0.01
                        if score > best_alt_score:
                            best_alt_score, best_alt_bs, best_ratio = score, bs_id, r
                        break
            if best_alt_bs is not None:
                needed = kicked_uav.required_rate * best_ratio
                if self.env.base_stations[best_alt_bs].allocate(kicked_id, needed):
                    kicked_uav.connected_bs_id = best_alt_bs
                    kicked_uav.current_allocated_rate = needed
                    self.env.connection_matrix[kicked_id, best_alt_bs] = 1

    def _manage_other_uavs(self):
        """用启发式算法管理所有非目标 UAV"""
        for uav_id in self.env.uavs.keys():
            if uav_id == self.target_uav_id:
                continue
            uav = self.env.uavs[uav_id]

            # 断连 UAV 优先处理
            if uav.connected_bs_id is None:
                decision = self._heuristic_algo.make_intelligent_decision(uav_id)
                if decision is not None:
                    self._heuristic_algo.execute_handover(uav_id, decision[0], decision[1])
                continue

            # 已连接 UAV 的常规决策
            decision = self._heuristic_algo.make_intelligent_decision(uav_id)
            if decision is not None:
                self._heuristic_algo.execute_handover(uav_id, decision[0], decision[1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一个 RL 步

        Args:
            action: RL Agent 选择的离散动作索引

        Returns:
            (next_state, reward, done, info)
            - next_state: 下一步状态向量 (np.ndarray, shape=(state_dim,))
            - reward: 即时奖励 (float)
            - done: episode 是否结束 (bool)
            - info: 附加信息字典
        """
        # 保存动作前的状态
        self._last_satisfaction = self.env.uavs[self.target_uav_id].current_satisfaction
        self._last_allocated_rate = self.env.uavs[self.target_uav_id].current_allocated_rate
        self._last_connected = self.env.uavs[self.target_uav_id].connected_bs_id is not None
        self._pre_switch_cooldown = self._steps_since_last_switch

        # 1. 将 RL 动作应用到目标 UAV
        actual_switch, invalid_action = self._apply_action(action)

        # 更新切换追踪计数器
        action_decoded = self._decode_action(action)
        if actual_switch:
            self._total_handovers += 1
            self._steps_since_last_switch = 0
            self._consecutive_stays = 0
        elif action_decoded[0] == 'stay':
            self._consecutive_stays += 1
            self._steps_since_last_switch += 1
        else:
            self._consecutive_stays = 0
            self._steps_since_last_switch += 1

        # 2. 用启发式算法管理其余 UAV
        self._manage_other_uavs()

        # 3. 环境步进（移动、SINR 更新、满意度记录等）
        # 跳过识别过程以加速训练
        if self.skip_recognition:
            self.env.current_step += 1
            for uav in self.env.uavs.values():
                uav.move(time_step=1.0)
            self.env._update_sinr_matrix()
            for uav in self.env.uavs.values():
                uav.record_satisfaction()
            self.env._check_interruptions()
            self.env._record_stats()
        else:
            self.env.step()

        self._current_step += 1

        # 4. 计算奖励
        reward = self._compute_reward(action, actual_switch, invalid_action)

        # 5. 获取下一步状态
        next_state = self.get_state()

        # 6. 判断是否结束
        done = self._current_step >= self.max_steps

        # 附加信息
        uav = self.env.uavs[self.target_uav_id]
        info = {
            'step': self._current_step,
            'satisfaction': uav.current_satisfaction,
            'connected_bs': uav.connected_bs_id,
            'allocated_rate': uav.current_allocated_rate,
            'handover_count': uav.handover_count,
            'total_handovers': self._total_handovers,
            'action_type': action_decoded[0],
            'action_target_bs': action_decoded[1],
            'actual_switch': actual_switch,
            'invalid_action': invalid_action,
        }

        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        """
        重置环境

        Returns:
            初始状态向量
        """
        self.env.reset()
        self._current_step = 0
        self._total_handovers = 0
        self._steps_since_last_switch = 0
        self._last_switched_from_bs = None
        self._consecutive_stays = 0
        self._pre_switch_cooldown = 0

        # 重建动作映射表（reset 后 num_bs 可能变化）
        self._build_action_map()

        # 重建启发式算法实例
        self._heuristic_algo = EnhancedHandoverAlgorithm(self.env)

        uav = self.env.uavs[self.target_uav_id]
        self._last_satisfaction = uav.current_satisfaction
        self._last_allocated_rate = uav.current_allocated_rate
        self._last_connected = uav.connected_bs_id is not None

        return self.get_state()

    def render(self):
        """打印当前环境状态摘要"""
        uav = self.env.uavs[self.target_uav_id]
        print(f"[Step {self._current_step}] UAV[{self.target_uav_id}] "
              f"Type={uav.true_business_type.name} "
              f"BS={uav.connected_bs_id} "
              f"Rate={uav.current_allocated_rate:.1f}/{uav.required_rate:.1f} "
              f"Sat={uav.current_satisfaction:.3f} "
              f"Handovers={uav.handover_count}")


def create_rl_env(num_bs: int = 8, num_uav: int = 20, target_uav_id: int = 0,
                  max_steps: int = 150, seed: int = 42, skip_recognition: bool = True,
                  **env_kwargs) -> RLHandoverEnv:
    """
    工厂函数：创建 RL 环境实例

    Args:
        num_bs: 基站数量
        num_uav: UAV 总数（包括 RL 控制的 1 个）
        target_uav_id: RL 控制的目标 UAV 编号
        max_steps: 每个 episode 最大步数
        seed: 随机种子
        skip_recognition: 是否跳过业务识别（加速仿真）
        **env_kwargs: 传递给 NetworkEnvironmentWithRecognition 的额外参数

    Returns:
        RLHandoverEnv 实例
    """
    env = NetworkEnvironmentWithRecognition(
        num_bs=num_bs, num_uav=num_uav, seed=seed, **env_kwargs
    )
    return RLHandoverEnv(
        env, target_uav_id=target_uav_id, max_steps=max_steps,
        skip_recognition=skip_recognition
    )
