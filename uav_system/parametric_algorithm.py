"""
参数化增强切换算法

将 EnhancedHandoverAlgorithm 的 7 个关键可调参数提取为外部参数向量 θ，
使 QMIX 元控制器可以通过调整 θ 来适配不同的网络状态。

可学习参数 (θ):
  θ[0]: w_sinr       — SINR 权重 (业务感知效用函数)
  θ[1]: w_load       — 负载权重 (业务感知效用函数)
  θ[2]: w_rate       — 速率权重 (业务感知效用函数)
  θ[3]: threshold    — 切换阈值 (动态阈值偏移)
  θ[4]: epsilon      — ε-greedy 探索率
  θ[5]: lb_interval  — 负载均衡触发间隔 (步数)
  θ[6]: lb_threshold — 负载均衡触发阈值 (负载标准差)

参数通过动作编码映射到连续参数空间:
  每个 agent 的离散动作 → 对应一组预定义的策略配置 (θ)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from .business import BusinessType, QOS_PROFILES
from .environment import NetworkEnvironmentWithRecognition


# ==============================================================================
# 预定义策略配置表 (5 种离散策略)
# ==============================================================================

# 策略名称 -> 参数向量 θ (7 维)
#   [w_sinr, w_load, w_rate, threshold, epsilon, lb_interval, lb_threshold]
STRATEGY_CONFIGS = {
    'conservative': np.array([0.55, 0.30, 0.15, 0.015, 0.00, 15, 0.12], dtype=np.float32),
    'balanced':     np.array([0.45, 0.25, 0.30, 0.005, 0.03, 10, 0.08], dtype=np.float32),
    'aggressive':   np.array([0.30, 0.20, 0.50, -0.005, 0.05, 5, 0.05], dtype=np.float32),
    'rate_focus':   np.array([0.25, 0.15, 0.60, -0.010, 0.04, 5, 0.05], dtype=np.float32),
    'stability':    np.array([0.60, 0.30, 0.10, 0.020, 0.00, 20, 0.15], dtype=np.float32),
}

# 动作索引 -> 策略名称
ACTION_TO_STRATEGY = {0: 'conservative', 1: 'balanced', 2: 'aggressive',
                      3: 'rate_focus', 4: 'stability'}

# 策略数量
NUM_STRATEGIES = len(STRATEGY_CONFIGS)


# ==============================================================================
# 参数化增强切换算法
# ==============================================================================

class ParametricEnhancedAlgorithm:
    """
    参数化增强切换算法

    与 EnhancedHandoverAlgorithm 功能等价，但将关键参数外部化为参数向量 θ，
    供 QMIX 元控制器在运行时动态选择。

    保持所有核心执行机制不变：
    - 降级比例搜索
    - 抢占机制（高优先级抢占低优先级）
    - 回滚机制（切换失败回退到旧基站）
    - 软迁移（被抢占 UAV 尝试迁移到其他基站）
    - 负载均衡（周期性迁移高负载基站的低优先级 UAV）
    """

    def __init__(self, env: NetworkEnvironmentWithRecognition,
                 w_sinr: float = 0.45, w_load: float = 0.25,
                 w_rate: float = 0.30, threshold: float = 0.005,
                 epsilon: float = 0.03, lb_interval: int = 10,
                 lb_threshold: float = 0.08):
        """
        Args:
            env: 网络环境实例
            w_sinr: 效用函数中 SINR 权重
            w_load: 效用函数中负载权重
            w_rate: 效用函数中速率权重
            threshold: 切换阈值偏移量
            epsilon: ε-greedy 探索率
            lb_interval: 负载均衡触发间隔 (步数)
            lb_threshold: 负载均衡触发阈值 (负载标准差)
        """
        self.env = env

        # ====== QMIX 可控参数 ======
        self.w_sinr = w_sinr
        self.w_load = w_load
        self.w_rate = w_rate
        self.base_threshold = threshold
        self.epsilon = epsilon
        self.lb_interval = lb_interval
        self.lb_threshold = lb_threshold

        # 紧急切换阈值（固定，不参与学习）
        self.emergency_sinr_threshold = -5
        self.emergency_satisfaction_threshold = 0.7
        self.confidence_factor_coeff = 0.002
        self.mobility_factor_coeff = 0.003
        self.priority_factor_control = 0.003
        self.threshold_lower_bound = 0.005

        # 统计指标
        self.handover_attempts = 0
        self.handover_successes = 0
        self.decision_calls = 0
        self.missed_opportunity = 0
        self.migration_attempts = 0
        self.migration_successes = 0
        self.decision_log = []
        self.switching_latency_history = []
        self.decision_time_history = []
        self.failure_reasons = defaultdict(int)
        self.utility_history = []
        self.threshold_history = []
        self.rollback_fail_count = 0
        self.ghost_disconnect_count = 0
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.reconnect_cooldown = {}
        self.disconnect_timer = {}
        self.emergency_count = 0
        self.current_step_emergency = 0
        self.handover_by_business = {bt: {'attempts': 0, 'successes': 0} for bt in BusinessType}

    @classmethod
    def from_strategy_name(cls, env: NetworkEnvironmentWithRecognition,
                           strategy_name: str) -> 'ParametricEnhancedAlgorithm':
        """从策略名称创建算法实例"""
        if strategy_name not in STRATEGY_CONFIGS:
            raise ValueError(f"未知策略: {strategy_name}, 可选: {list(STRATEGY_CONFIGS.keys())}")
        params = STRATEGY_CONFIGS[strategy_name]
        return cls(
            env=env,
            w_sinr=float(params[0]),
            w_load=float(params[1]),
            w_rate=float(params[2]),
            threshold=float(params[3]),
            epsilon=float(params[4]),
            lb_interval=int(params[5]),
            lb_threshold=float(params[6]),
        )

    @classmethod
    def from_action(cls, env: NetworkEnvironmentWithRecognition,
                    action: int) -> 'ParametricEnhancedAlgorithm':
        """从 QMIX 离散动作创建算法实例"""
        strategy_name = ACTION_TO_STRATEGY.get(action, 'balanced')
        return cls.from_strategy_name(env, strategy_name)

    def get_params_vector(self) -> np.ndarray:
        """获取当前参数向量 θ"""
        return np.array([self.w_sinr, self.w_load, self.w_rate,
                         self.base_threshold, self.epsilon,
                         self.lb_interval, self.lb_threshold], dtype=np.float32)

    def set_params_vector(self, theta: np.ndarray):
        """设置参数向量 θ"""
        self.w_sinr = float(theta[0])
        self.w_load = float(theta[1])
        self.w_rate = float(theta[2])
        self.base_threshold = float(theta[3])
        self.epsilon = float(theta[4])
        self.lb_interval = int(np.clip(theta[5], 1, 30))
        self.lb_threshold = float(theta[6])

    @staticmethod
    def get_action_from_strategy(strategy_name: str) -> int:
        """从策略名称获取动作索引"""
        for action, name in ACTION_TO_STRATEGY.items():
            if name == strategy_name:
                return action
        return 1  # 默认 balanced

    # ==================== 效用函数 ====================

    def calculate_utility_with_downgrade(self, uav, bs_id: int,
                                         downgrade_ratio: float) -> Tuple[float, bool]:
        """计算带降级的业务感知效用值（使用参数化权重）"""
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        sinr_norm = np.clip((sinr + 10) / 40, 0, 1)
        bs = self.env.base_stations[bs_id]
        load_ratio = bs.load_ratio
        required = uav.required_rate * downgrade_ratio
        available = bs.available_capacity

        is_feasible = (available >= required * (0.6 if downgrade_ratio >= 0.8 else 0.7))

        rate_match = 0.0
        if required > 0:
            rate_ratio = available / required
            rate_match = 1 - np.exp(-3 * min(rate_ratio, 1.5))

        business_bonus = 0.05 * (downgrade_ratio - 0.8) / 0.2 if downgrade_ratio >= 0.8 else 0.0

        # 使用参数化权重
        w_total = self.w_sinr + self.w_load + self.w_rate
        ws, wl, wr = self.w_sinr / w_total, self.w_load / w_total, self.w_rate / w_total
        utility = (ws * sinr_norm + wl * (1 - load_ratio) +
                   wr * rate_match + business_bonus)
        return utility, is_feasible

    def calculate_dynamic_threshold(self, uav) -> float:
        """计算动态切换阈值（使用参数化阈值）"""
        base = self.base_threshold
        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            base *= 0.5

        if uav.connected_bs_id is not None:
            load_factor = self.env.base_stations[uav.connected_bs_id].load_ratio
            adjustment = -0.005 * min(load_factor, 1.0)
            if uav.business_type == BusinessType.CONTROL_SIGNAL and load_factor > 0.7:
                adjustment -= 0.01
        else:
            adjustment = 0

        confidence_factor = (1 - uav.recognition_confidence) * self.confidence_factor_coeff
        velocity_norm = np.linalg.norm(uav.velocity)
        mobility_factor = -self.mobility_factor_coeff * min(velocity_norm / 10, 1.0)
        priority_factor = -self.priority_factor_control * 1.5 if uav.business_type == BusinessType.CONTROL_SIGNAL else 0

        dynamic_threshold = base + adjustment + confidence_factor + mobility_factor + priority_factor
        lower_bound = self.threshold_lower_bound * (0.5 if uav.business_type == BusinessType.CONTROL_SIGNAL else 1.0)
        return max(lower_bound, dynamic_threshold)

    # ==================== 决策 ====================

    def _emergency_select(self, uav) -> Tuple[Optional[int], float]:
        """紧急切换：选择SINR最高且有足够容量的基站"""
        best_bs, best_sinr, best_ratio = None, -999, 1.0
        for bs_id in self.env.base_stations.keys():
            sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
            if sinr > best_sinr:
                bs = self.env.base_stations[bs_id]
                for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                    if bs.available_capacity >= uav.required_rate * ratio * 0.9:
                        best_bs, best_sinr, best_ratio = bs_id, sinr, ratio
                        break
        return best_bs, best_ratio

    def make_intelligent_decision(self, uav_id: int) -> Optional[Tuple[int, float]]:
        """参数化增强决策"""
        from time import time
        t_start = time()
        self.decision_calls += 1
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id

        # 未连接：使用宽松策略
        if current_bs_id is None:
            best_bs, best_utility, best_ratio = None, -1, 1.0
            for bs_id in self.env.base_stations.keys():
                for ratio in [1.0, 0.8, 0.6, 0.4, 0.2]:
                    utility, _ = self.calculate_utility_with_downgrade(uav, bs_id, ratio)
                    if utility > best_utility:
                        best_utility, best_bs, best_ratio = utility, bs_id, ratio
            self.decision_time_history.append((time() - t_start) * 1000)
            if best_bs is not None:
                self.reconnect_attempts += 1
                return (best_bs, best_ratio)
            return None

        # 紧急切换判定
        emergency = False
        if current_bs_id is not None:
            current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
            sinr_thresh = 0 if uav.business_type == BusinessType.CONTROL_SIGNAL else self.emergency_sinr_threshold
            if current_sinr < sinr_thresh:
                emergency = True
            if uav.business_type == BusinessType.CONTROL_SIGNAL and current_sinr < 5 and uav.current_satisfaction < 0.85:
                emergency = True

        if emergency:
            self.emergency_count += 1
            self.current_step_emergency += 1
            best_bs, best_ratio = self._emergency_select(uav)
            self.decision_time_history.append((time() - t_start) * 1000)
            return (best_bs, best_ratio) if best_bs is not None else None

        # ε-greedy 探索
        if self.epsilon > 0 and np.random.rand() < self.epsilon:
            candidate_bs_ids = [bs_id for bs_id in self.env.base_stations.keys() if bs_id != current_bs_id]
            if candidate_bs_ids:
                random_bs = np.random.choice(candidate_bs_ids)
                for ratio in [1.0, 0.8, 0.6]:
                    _, feasible = self.calculate_utility_with_downgrade(uav, random_bs, ratio)
                    if feasible:
                        self.decision_log.append({
                            'uav_id': uav.uav_id, 'step': self.env.current_step,
                            'current_bs': current_bs_id, 'target_bs': random_bs,
                            'downgrade_ratio': ratio, 'filter_reason': 'epsilon_greedy'
                        })
                        self.decision_time_history.append((time() - t_start) * 1000)
                        return (random_bs, ratio)

        # 核心决策：降级比例搜索 + 效用比较
        all_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
        current_utility, _ = self.calculate_utility_with_downgrade(uav, current_bs_id, 1.0)
        best_bs, best_utility, best_ratio = None, current_utility, 1.0
        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            for ratio in all_ratios:
                utility, _ = self.calculate_utility_with_downgrade(uav, bs_id, ratio)
                if utility > best_utility:
                    best_utility, best_bs, best_ratio = utility, bs_id, ratio

        if best_bs is not None:
            dynamic_threshold = self.calculate_dynamic_threshold(uav)
            self.utility_history.append({'current': current_utility, 'best': best_utility})
            self.threshold_history.append(dynamic_threshold)
            if best_utility > current_utility + dynamic_threshold:
                self.decision_log.append({
                    'uav_id': uav.uav_id, 'step': self.env.current_step,
                    'current_bs': current_bs_id, 'target_bs': best_bs,
                    'downgrade_ratio': best_ratio, 'filter_reason': None
                })
                self.decision_time_history.append((time() - t_start) * 1000)
                return (best_bs, best_ratio)

        self.decision_time_history.append((time() - t_start) * 1000)
        return None

    # ==================== 执行 ====================

    def _soft_migrate_kicked_uavs(self, kicked_ids: list, exclude_bs_id: int):
        """为被抢占的 UAV 尝试软迁移到其他基站"""
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
                    self.disconnect_timer.pop(kicked_id, None)

    def execute_handover(self, uav_id: int, target_bs_id: int, downgrade_ratio: float) -> bool:
        """执行切换：释放旧基站 -> 分配新基站 -> 抢占 -> 回滚 -> 断连"""
        from time import time
        t_start = time()
        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio
        is_reconnect = (uav.connected_bs_id is None)
        self.handover_attempts += 1

        biz_type = uav.business_type
        if biz_type in self.handover_by_business:
            self.handover_by_business[biz_type]['attempts'] += 1

        old_bs_id = uav.connected_bs_id
        old_bs = self.env.base_stations[old_bs_id] if old_bs_id is not None else None
        old_allocated_rate = uav.current_allocated_rate

        # 1. 释放旧基站
        if old_bs_id is not None and old_bs_id != target_bs_id:
            old_bs.release(uav_id)
            self.env.connection_matrix[uav_id, old_bs_id] = 0

        # 2. 直接分配
        if target_bs.allocate(uav_id, required_rate):
            return self._complete_handover(uav_id, target_bs_id, required_rate, is_reconnect, t_start)

        # 3. 抢占低优先级
        self.failure_reasons['allocation_failed'] += 1
        freed, kicked_ids = target_bs.kick_low_priority(uav, self.env.uavs)
        if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
            self._soft_migrate_kicked_uavs(kicked_ids, target_bs_id)
            return self._complete_handover(uav_id, target_bs_id, required_rate, is_reconnect, t_start)

        # 4. 回滚到旧基站
        self.failure_reasons['preemption_failed'] += 1
        rollback_ok = False
        if old_bs_id is not None and old_bs_id != target_bs_id:
            rollback_ok = self._try_rollback(uav_id, old_bs_id, old_bs, old_allocated_rate)

        # 5. 回滚失败，断连
        if not rollback_ok:
            if not is_reconnect:
                uav.connected_bs_id = None
                uav.current_allocated_rate = 0.0
                self.rollback_fail_count += 1
                self.ghost_disconnect_count += 1
            self.reconnect_cooldown[uav_id] = 0
            if not is_reconnect:
                self.disconnect_timer[uav_id] = 0
            else:
                self.disconnect_timer[uav_id] = self.disconnect_timer.get(uav_id, 0) + 1

        self.switching_latency_history.append((time() - t_start) * 1000)
        return False

    def _complete_handover(self, uav_id, target_bs_id, required_rate, is_reconnect, t_start) -> bool:
        """完成切换分配"""
        from time import time
        uav = self.env.uavs[uav_id]
        uav.connected_bs_id = target_bs_id
        uav.current_allocated_rate = required_rate
        self.env.connection_matrix[uav_id, target_bs_id] = 1
        uav.handover_count += 1
        self.handover_successes += 1
        biz_type = uav.business_type
        if biz_type in self.handover_by_business:
            self.handover_by_business[biz_type]['successes'] += 1
        if is_reconnect:
            self.reconnect_successes += 1
            self.reconnect_cooldown.pop(uav_id, None)
            self.disconnect_timer.pop(uav_id, None)
        self.switching_latency_history.append((time() - t_start) * 1000)
        return True

    def _try_rollback(self, uav_id, old_bs_id, old_bs, old_allocated_rate) -> bool:
        """尝试回滚到旧基站"""
        if old_bs.available_capacity >= old_allocated_rate:
            old_bs.allocate(uav_id, old_allocated_rate)
            self.env.connection_matrix[uav_id, old_bs_id] = 1
            self.env.uavs[uav_id].connected_bs_id = old_bs_id
            return True
        rollback_freed, rollback_kicked = old_bs.kick_low_priority(self.env.uavs[uav_id], self.env.uavs)
        if rollback_freed >= old_allocated_rate and old_bs.allocate(uav_id, old_allocated_rate):
            self.env.connection_matrix[uav_id, old_bs_id] = 1
            self.env.uavs[uav_id].connected_bs_id = old_bs_id
            self._soft_migrate_kicked_uavs(rollback_kicked, old_bs_id)
            return True
        self.failure_reasons['rollback_failed'] += 1
        return False

    # ==================== 负载均衡 ====================

    def global_load_balancing_v2(self) -> int:
        """全局负载均衡（使用参数化间隔和阈值）"""
        load_ratios = [bs.load_ratio for bs in self.env.base_stations.values()]
        if np.std(load_ratios) < self.lb_threshold:
            return 0

        load_with_id = sorted(self.env.base_stations.items(), key=lambda x: x[1].load_ratio, reverse=True)
        high_bs_id, high_load = load_with_id[0][0], load_with_id[0][1].load_ratio
        low_bs_id, low_load = load_with_id[-1][0], load_with_id[-1][1].load_ratio
        if high_load - low_load < 0.1:
            return 0

        high_bs = self.env.base_stations[high_bs_id]
        low_bs = self.env.base_stations[low_bs_id]

        candidates = []
        for uav_id in list(high_bs.connected_uavs.keys()):
            uav = self.env.uavs[uav_id]
            sinr_loss = self.env.sinr_matrix[uav_id, high_bs_id] - self.env.sinr_matrix[uav_id, low_bs_id]
            if sinr_loss > 5:
                continue
            required = uav.current_allocated_rate
            if low_bs.available_capacity <= required * 0.3:
                continue
            current_utility, _ = self.calculate_utility_with_downgrade(uav, high_bs_id, 1.0)
            target_utility, _ = self.calculate_utility_with_downgrade(uav, low_bs_id, 1.0)
            if target_utility - current_utility < -0.05:
                continue
            score = (required / max(uav.qos_profile.priority, 0.1)) * (1 - uav.qos_profile.criticality)
            candidates.append((uav_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        migrations = 0
        for uav_id, _ in candidates[:5]:
            self.migration_attempts += 1
            if self.execute_handover(uav_id, low_bs_id, 1.0):
                migrations += 1
                self.migration_successes += 1
        return migrations

    def get_disconnected_count(self) -> int:
        """统计当前处于断连状态的 UAV 数量"""
        return sum(1 for uav in self.env.uavs.values() if uav.connected_bs_id is None)

    # ==================== 主循环 ====================

    def run_step(self, enable_load_balancing=True) -> Tuple[int, int]:
        """执行一个仿真步，返回 (切换次数, 迁移次数)"""
        handover_count = 0
        self.current_step_emergency = 0

        # 优先处理断连 UAV
        disconnected_ids = [uid for uid in self.env.uavs.keys()
                            if self.env.uavs[uid].connected_bs_id is None]
        if disconnected_ids:
            for uid in disconnected_ids:
                self.disconnect_timer[uid] = self.disconnect_timer.get(uid, 0) + 1
            disconnected_ids.sort(key=lambda uid: self.disconnect_timer.get(uid, 0), reverse=True)
            for uav_id in disconnected_ids:
                decision = self.make_intelligent_decision(uav_id)
                if decision is not None:
                    if self.execute_handover(uav_id, decision[0], decision[1]):
                        handover_count += 1
                        self.disconnect_timer.pop(uav_id, None)

        # 处理已连接 UAV
        for uav_id in self.env.uavs.keys():
            if self.env.uavs[uav_id].connected_bs_id is not None:
                decision = self.make_intelligent_decision(uav_id)
                if decision is not None:
                    if self.execute_handover(uav_id, decision[0], decision[1]):
                        handover_count += 1

        # 周期性负载均衡（使用参数化间隔）
        migration_count = 0
        if enable_load_balancing and self.env.current_step % self.lb_interval == 0:
            migration_count = self.global_load_balancing_v2()

        return handover_count, migration_count

    def reset_stats(self):
        """重置统计指标"""
        self.handover_attempts = 0
        self.handover_successes = 0
        self.decision_calls = 0
        self.missed_opportunity = 0
        self.migration_attempts = 0
        self.migration_successes = 0
        self.decision_log.clear()
        self.switching_latency_history.clear()
        self.decision_time_history.clear()
        self.failure_reasons.clear()
        self.utility_history.clear()
        self.threshold_history.clear()
        self.rollback_fail_count = 0
        self.ghost_disconnect_count = 0
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.reconnect_cooldown.clear()
        self.disconnect_timer.clear()
        self.emergency_count = 0
        self.current_step_emergency = 0
        self.handover_by_business = {bt: {'attempts': 0, 'successes': 0} for bt in BusinessType}

    def get_detailed_stats(self) -> Dict:
        """获取算法详细统计"""
        normal_attempts = max(self.handover_attempts - self.reconnect_attempts, 1)
        normal_success_rate = (self.handover_successes - self.reconnect_successes) / normal_attempts
        reconnect_success_rate = self.reconnect_successes / max(self.reconnect_attempts, 1)
        return {
            'avg_decision_time_ms': np.mean(self.decision_time_history) if self.decision_time_history else 0,
            'max_decision_time_ms': max(self.decision_time_history) if self.decision_time_history else 0,
            'avg_switching_latency_ms': np.mean(self.switching_latency_history) if self.switching_latency_history else 0,
            'max_switching_latency_ms': max(self.switching_latency_history) if self.switching_latency_history else 0,
            'failure_reasons': dict(self.failure_reasons),
            'handover_success_rate': normal_success_rate,
            'reconnect_success_rate': reconnect_success_rate,
            'reconnect_attempts': self.reconnect_attempts,
            'reconnect_successes': self.reconnect_successes,
            'missed_opportunity_rate': self.missed_opportunity / max(self.decision_calls, 1),
            'migration_success_rate': self.migration_successes / max(self.migration_attempts, 1) if self.migration_attempts > 0 else 0,
            'avg_utility_improvement': np.mean([u['best'] - u['current'] for u in self.utility_history]) if self.utility_history else 0,
            'avg_dynamic_threshold': np.mean(self.threshold_history) if self.threshold_history else 0,
            'rollback_fail_count': self.rollback_fail_count,
            'ghost_disconnect_count': self.ghost_disconnect_count,
            'disconnected_count': self.get_disconnected_count(),
            'emergency_count': self.emergency_count,
            'handover_by_business': {bt.name: data for bt, data in self.handover_by_business.items()},
            'weighted_success_rate': self._compute_weighted_success_rate(),
            'params': {
                'w_sinr': self.w_sinr, 'w_load': self.w_load, 'w_rate': self.w_rate,
                'threshold': self.base_threshold, 'epsilon': self.epsilon,
                'lb_interval': self.lb_interval, 'lb_threshold': self.lb_threshold,
            },
        }

    def _compute_weighted_success_rate(self) -> float:
        """计算按业务类型优先级加权的切换成功率"""
        total_weighted = 0.0
        total_weight = 0.0
        for bt in BusinessType:
            data = self.handover_by_business[bt]
            if data['attempts'] > 0:
                weight = QOS_PROFILES[bt].priority
                rate = data['successes'] / data['attempts']
                total_weighted += weight * rate
                total_weight += weight
        return total_weighted / total_weight if total_weight > 0 else 0.0
