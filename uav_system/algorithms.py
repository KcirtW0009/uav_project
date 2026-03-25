import numpy as np
from typing import Optional, Tuple, Dict
from collections import defaultdict
from .business import BusinessType
from .environment import NetworkEnvironmentWithRecognition

class IntegratedHandoverAlgorithm:
    def __init__(self, env: NetworkEnvironmentWithRecognition):
        self.env = env
        self.w_sinr = 0.4
        self.w_load = 0.3
        self.w_rate = 0.3
        self.handover_threshold = 0.005
        self.downgrade_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
        self.handover_attempts = 0
        self.handover_successes = 0
        self.decision_calls = 0
        self.missed_opportunity = 0
        self.emergency_sinr_threshold = -5
        self.emergency_satisfaction_threshold = 0.7
        self.sat_control = []
        self.sat_video = []
        self.sat_env = []
        self.switching_latency_history = []
        self.decision_time_history = []
        self.failure_reasons = defaultdict(int)
        self.reconnect_attempts = 0
        self.reconnect_successes = 0

    def calculate_utility(self, uav, bs_id: int, downgrade_ratio: float = 1.0) -> float:
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        sinr_norm = np.clip((sinr + 10) / 40, 0, 1)
        bs = self.env.base_stations[bs_id]
        load_ratio = bs.load_ratio
        required = uav.required_rate * downgrade_ratio
        available = bs.available_capacity
        rate_match = min(available / required, 1.0) if required > 0 else 0
        utility = (self.w_sinr * sinr_norm +
                   self.w_load * (1 - load_ratio) +
                   self.w_rate * rate_match)
        return utility

    def make_decision(self, uav_id: int) -> Optional[Tuple[int, float]]:
        from time import time
        t_start = time()
        self.decision_calls += 1
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id
        emergency = False
        if current_bs_id is not None:
            current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
            if current_sinr < self.emergency_sinr_threshold or uav.current_satisfaction < self.emergency_satisfaction_threshold:
                emergency = True
        if current_bs_id is None:
            best_bs = None
            best_utility = -1
            best_ratio = 1.0
            for bs_id in self.env.base_stations.keys():
                for ratio in self.downgrade_ratios:
                    utility = self.calculate_utility(uav, bs_id, ratio)
                    if utility > best_utility:
                        best_utility = utility
                        best_bs = bs_id
                        best_ratio = ratio
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            return (best_bs, best_ratio) if best_bs is not None else None
        current_utility = self.calculate_utility(uav, current_bs_id, 1.0)
        best_bs = current_bs_id
        best_utility = current_utility
        best_ratio = 1.0
        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            for ratio in self.downgrade_ratios:
                utility = self.calculate_utility(uav, bs_id, ratio)
                if utility > best_utility + self.handover_threshold:
                    best_utility = utility
                    best_bs = bs_id
                    best_ratio = ratio
        if emergency and best_bs == current_bs_id:
            self.missed_opportunity += 1
        t_end = time()
        self.decision_time_history.append((t_end - t_start) * 1000)
        if best_bs != current_bs_id:
            return (best_bs, best_ratio)
        return None

    def execute_handover(self, uav_id: int, target_bs_id: int, downgrade_ratio: float) -> bool:
        from time import time
        t_start = time()
        self.handover_attempts += 1
        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio

        # 区分重连和正常切换
        is_reconnect = (uav.connected_bs_id is None)
        if is_reconnect:
            self.reconnect_attempts += 1

        if uav.connected_bs_id is not None:
            old_bs = self.env.base_stations[uav.connected_bs_id]
            old_bs.release(uav_id)
            self.env.connection_matrix[uav_id, uav.connected_bs_id] = 0
        if target_bs.allocate(uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self.handover_successes += 1
            if is_reconnect:
                self.reconnect_successes += 1
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return True
        else:
            self.failure_reasons['allocation_failed'] += 1
            freed, _ = target_bs.kick_low_priority(uav, self.env.uavs)
            if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
                uav.connected_bs_id = target_bs_id
                uav.current_allocated_rate = required_rate
                self.env.connection_matrix[uav_id, target_bs_id] = 1
                uav.handover_count += 1
                self.handover_successes += 1
                if is_reconnect:
                    self.reconnect_successes += 1
                t_end = time()
                self.switching_latency_history.append((t_end - t_start) * 1000)
                return True
            self.failure_reasons['preemption_failed'] += 1
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return False

    def run_step(self) -> int:
        handover_count = 0
        for uav_id in self.env.uavs.keys():
            decision = self.make_decision(uav_id)
            if decision is not None:
                target_bs_id, ratio = decision
                success = self.execute_handover(uav_id, target_bs_id, ratio)
                if success:
                    handover_count += 1
        return handover_count

    def get_detailed_stats(self) -> Dict:
        # 切换成功率：仅计正常切换（排除重连），与增强算法对齐
        normal_attempts = max(self.handover_attempts - self.reconnect_attempts, 1)
        normal_success_rate = (self.handover_successes - self.reconnect_successes) / normal_attempts if normal_attempts > 0 else 0
        reconnect_success_rate = self.reconnect_successes / max(self.reconnect_attempts, 1)
        return {
            'avg_decision_time_ms': np.mean(self.decision_time_history) if self.decision_time_history else 0,
            'avg_switching_latency_ms': np.mean(self.switching_latency_history) if self.switching_latency_history else 0,
            'max_switching_latency_ms': max(self.switching_latency_history) if self.switching_latency_history else 0,
            'failure_reasons': dict(self.failure_reasons),
            'handover_success_rate': normal_success_rate,
            'reconnect_success_rate': reconnect_success_rate,
            'reconnect_attempts': self.reconnect_attempts,
            'reconnect_successes': self.reconnect_successes,
            'missed_opportunity_rate': self.missed_opportunity / max(self.decision_calls, 1)
        }


class EnhancedHandoverAlgorithm:
    def __init__(self, env: NetworkEnvironmentWithRecognition, weight_config='optimized'):
        self.env = env
        self.w_sinr = 0.45
        self.w_load = 0.25
        self.w_rate = 0.30
        self.base_threshold = -0.002  # 负阈值：效用有任何正向增益即切换，比传统更积极
        self.confidence_factor_coeff = 0.002
        self.mobility_factor_coeff = 0.003
        self.priority_factor_control = 0.003
        self.threshold_lower_bound = 0.005  # 进一步提高下限，减少不必要的切换，提升成功率稳定性
        self.epsilon = 0.05
        self.emergency_sinr_threshold = -5
        self.emergency_satisfaction_threshold = 0.7
        self.business_weights = {
            BusinessType.CONTROL_SIGNAL: {'sinr': 0.5, 'load': 0.2, 'rate': 0.3},
            BusinessType.VIDEO_STREAMING: {'sinr': 0.3, 'load': 0.25, 'rate': 0.45},
            BusinessType.ENVIRONMENT_MONITORING: {'sinr': 0.25, 'load': 0.25, 'rate': 0.5}
        }
        self.handover_attempts = 0
        self.handover_successes = 0
        self.decision_calls = 0
        self.missed_opportunity = 0
        self.sat_control = []
        self.sat_video = []
        self.sat_env = []
        self.migration_attempts = 0
        self.migration_successes = 0
        self.decision_log = []
        self.switching_latency_history = []
        self.decision_time_history = []
        self.failure_reasons = defaultdict(int)
        self.utility_history = []
        self.threshold_history = []
        self.execution_filter_stats = defaultdict(int)
        self.rollback_fail_count = 0
        self.ghost_disconnect_count = 0
        self.reconnect_cooldown = {}  # {uav_id: cooldown_remaining_steps}
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.RECONNECT_COOLDOWN_STEPS = 0   # 禁用重连冷却，与传统算法一致的恢复速度
        self.disconnect_timer = {}  # {uav_id: 断连步数}
        self.emergency_count = 0
        self.emergency_cooldown = {}  # 保留结构但不使用
        self.EMERGENCY_COOLDOWN_STEPS = 0  # 禁用紧急冷却
        self.MAX_EMERGENCY_PER_STEP = 999  # 不限制
        self.current_step_emergency = 0

    def get_disconnected_count(self) -> int:
        """统计当前处于断连状态的UAV数量（connected_bs_id为None）"""
        return sum(1 for uav in self.env.uavs.values() if uav.connected_bs_id is None)

    def calculate_utility_with_downgrade(self, uav, bs_id: int, downgrade_ratio: float) -> Tuple[float, bool]:
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        sinr_norm = np.clip((sinr + 10) / 40, 0, 1)
        bs = self.env.base_stations[bs_id]
        load_ratio = bs.load_ratio
        required = uav.required_rate * downgrade_ratio
        available = bs.available_capacity
        if downgrade_ratio >= 0.8:
            is_feasible = (available >= required * 0.6)  # 从0.9降至0.6，允许更多候选
        else:
            is_feasible = (available >= required * 0.7)  # 从0.95降至0.7
        if required > 0:
            rate_ratio = available / required
            rate_match = 1 - np.exp(-3 * min(rate_ratio, 1.5))
        else:
            rate_match = 0
        business_bonus = 0.0
        if downgrade_ratio >= 0.8:
            business_bonus = 0.05 * (downgrade_ratio - 0.8) / 0.2
        weights = self.business_weights.get(uav.business_type, {'sinr': 0.4, 'load': 0.3, 'rate': 0.3})
        utility = (weights['sinr'] * sinr_norm +
                   weights['load'] * (1 - load_ratio) +
                   weights['rate'] * rate_match +
                   business_bonus)
        return utility, is_feasible

    def calculate_dynamic_threshold(self, uav) -> float:
        base_threshold = self.base_threshold
        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            base_threshold *= 0.5
        if uav.connected_bs_id is not None:
            current_bs = self.env.base_stations[uav.connected_bs_id]
            load_factor = current_bs.load_ratio
            threshold_adjustment = -0.005 * min(load_factor, 1.0)
            if uav.business_type == BusinessType.CONTROL_SIGNAL and load_factor > 0.7:
                threshold_adjustment -= 0.01
        else:
            threshold_adjustment = 0
        confidence_factor = (1 - uav.recognition_confidence) * self.confidence_factor_coeff
        velocity_norm = np.linalg.norm(uav.velocity)
        mobility_factor = -self.mobility_factor_coeff * min(velocity_norm / 10, 1.0)
        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            priority_factor = -self.priority_factor_control * 1.5
        else:
            priority_factor = 0
        dynamic_threshold = (base_threshold + threshold_adjustment +
                             confidence_factor + mobility_factor + priority_factor)
        lower_bound = self.threshold_lower_bound * 0.5 if uav.business_type == BusinessType.CONTROL_SIGNAL else self.threshold_lower_bound
        return max(lower_bound, dynamic_threshold)

    def predict_handover_success(self, uav, bs_id: int, downgrade_ratio: float) -> float:
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        bs = self.env.base_stations[bs_id]
        sinr_success = 1 / (1 + np.exp(-0.5 * (sinr + 5)))

        # 优化：放松负载预测模型，提高预测成功率
        # 原系数：-3.0 → 负载80%时预测：exp(-2.4) = 0.090
        # 新系数：-2.0 → 负载80%时预测：exp(-1.6) = 0.202 (提升124%)
        load_success = np.exp(-2.0 * bs.load_ratio)

        # 实际可用容量检查
        required_rate = uav.required_rate * downgrade_ratio
        if bs.available_capacity < required_rate * 0.8:  # 需要80%的可用容量才考虑成功
            load_success = min(load_success, 0.1)  # 大幅降低预测成功率

        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            business_weight = 0.7
        elif uav.business_type == BusinessType.VIDEO_STREAMING:
            business_weight = 0.5
        else:
            business_weight = 0.3
        confidence_factor = 0.5 + 0.5 * uav.recognition_confidence
        success_prob = (business_weight * sinr_success +
                        (1 - business_weight) * load_success) * confidence_factor
        return np.clip(success_prob, 0, 1)

    def _emergency_select(self, uav) -> Tuple[Optional[int], float]:
        best_bs = None
        best_sinr = -999
        best_ratio = 1.0
        for bs_id in self.env.base_stations.keys():
            sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
            if sinr > best_sinr:
                bs = self.env.base_stations[bs_id]
                for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                    required = uav.required_rate * ratio
                    if bs.available_capacity >= required * 0.9:
                        best_bs = bs_id
                        best_sinr = sinr
                        best_ratio = ratio
                        break
        return best_bs, best_ratio

    def make_intelligent_decision(self, uav_id: int) -> Optional[Tuple[int, float]]:
        from time import time
        t_start = time()
        self.decision_calls += 1
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id

        # 调试信息：记录决策阶段的详细状态
        debug_info = {
            'uav_id': uav_id,
            'step': self.env.current_step,
            'business_type': uav.business_type.name if hasattr(uav, 'business_type') else 'unknown',
            'current_bs_id': current_bs_id,
            'current_sinr': None,
            'current_satisfaction': uav.current_satisfaction,
            'emergency': False,
            'num_candidates': 0,
            'num_high_success': 0,
            'best_success_prob': None,
            'best_utility': None,
            'best_bs_id': None,
            'filter_reason': None,
            'filter_details': {}
        }

        # 重连逻辑：如果UAV未连接，使用与传统一致的宽松策略
        if current_bs_id is None:
            best_bs = None
            best_utility = -1
            best_ratio = 1.0
            all_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]
            for bs_id in self.env.base_stations.keys():
                for ratio in all_ratios:
                    utility, _ = self.calculate_utility_with_downgrade(uav, bs_id, ratio)
                    if utility > best_utility:
                        best_utility = utility
                        best_bs = bs_id
                        best_ratio = ratio
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            if best_bs is not None:
                self.reconnect_attempts += 1
                return (best_bs, best_ratio)
            return None

        emergency = False
        if current_bs_id is not None:
            current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
            # SINR极低时触发紧急
            control_signal_sinr_threshold = 0 if uav.business_type == BusinessType.CONTROL_SIGNAL else self.emergency_sinr_threshold
            if current_sinr < control_signal_sinr_threshold:
                emergency = True
            # 控制信号UAV的紧急判定保持不变
            if uav.business_type == BusinessType.CONTROL_SIGNAL and current_sinr < 5 and uav.current_satisfaction < 0.85:
                emergency = True
        if emergency:
            self.emergency_count += 1
            self.current_step_emergency += 1
            best_bs, best_ratio = self._emergency_select(uav)
            if best_bs is not None:
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                return (best_bs, best_ratio)
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            return None
        # 核心决策：使用与传统一致的降级比例，但用业务感知效用函数
        all_ratios = [1.0, 0.8, 0.6, 0.4, 0.2]  # 与传统算法一致
        current_utility, _ = self.calculate_utility_with_downgrade(uav, current_bs_id, 1.0)
        best_bs = None
        best_utility = current_utility
        best_ratio = 1.0
        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            for ratio in all_ratios:
                utility, _ = self.calculate_utility_with_downgrade(uav, bs_id, ratio)
                if utility > best_utility:
                    best_utility = utility
                    best_bs = bs_id
                    best_ratio = ratio
        if best_bs is not None:
            dynamic_threshold = self.calculate_dynamic_threshold(uav)
            self.utility_history.append({'current': current_utility, 'best': best_utility})
            self.threshold_history.append(dynamic_threshold)
            if best_utility > current_utility + dynamic_threshold:
                self.decision_log.append({
                    'uav_id': uav.uav_id,
                    'step': self.env.current_step,
                    'current_bs': current_bs_id,
                    'target_bs': best_bs,
                    'best_success_prob': None,
                    'current_satisfaction': uav.current_satisfaction,
                    'downgrade_ratio': best_ratio,
                    'filter_reason': None
                })
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                return (best_bs, best_ratio)
        t_end = time()
        self.decision_time_history.append((t_end - t_start) * 1000)
        return None

    def _soft_migrate_kicked_uavs(self, kicked_ids: list, exclude_bs_id: int):
        """为被抢占的UAV尝试软迁移到其他BS，减少级联断连"""
        for kicked_id in kicked_ids:
            kicked_uav = self.env.uavs.get(kicked_id)
            if kicked_uav is None or kicked_uav.connected_bs_id is not None:
                continue  # 已经被迁移或连接
            best_alt_bs = None
            best_alt_score = -1
            best_ratio = 1.0
            for bs_id, bs in self.env.base_stations.items():
                if bs_id == exclude_bs_id:
                    continue
                for r in kicked_uav.qos_profile.get_feasible_downgrade_ratios():
                    needed = kicked_uav.required_rate * r
                    if bs.available_capacity >= needed * 0.9:
                        sinr = self.env.sinr_matrix[kicked_id, bs_id]
                        score = sinr + bs.available_capacity * 0.01
                        if score > best_alt_score:
                            best_alt_score = score
                            best_alt_bs = bs_id
                            best_ratio = r
                        break
            if best_alt_bs is not None:
                alt_bs = self.env.base_stations[best_alt_bs]
                needed = kicked_uav.required_rate * best_ratio
                if alt_bs.allocate(kicked_id, needed):
                    kicked_uav.connected_bs_id = best_alt_bs
                    kicked_uav.current_allocated_rate = needed
                    self.env.connection_matrix[kicked_id, best_alt_bs] = 1
                    self.disconnect_timer.pop(kicked_id, None)

    def execute_handover(self, uav_id: int, target_bs_id: int, downgrade_ratio: float) -> bool:
        """
        切换执行：先释放旧基站，再分配新基站（与传统算法一致）
        失败时回滚到旧基站，避免UAV断连
        回滚失败时清空connected_bs_id，使UAV进入重连路径
        重连失败时设置冷却期，避免级联重连尝试
        """
        from time import time
        t_start = time()

        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio

        # 区分重连和正常切换
        is_reconnect = (uav.connected_bs_id is None)

        # 所有调用都计入尝试
        self.handover_attempts += 1

        # 记录旧基站信息以便回滚
        old_bs_id = uav.connected_bs_id
        old_bs = self.env.base_stations[old_bs_id] if old_bs_id is not None else None
        old_allocated_rate = uav.current_allocated_rate

        # 先释放旧基站资源（减少资源竞争）
        if old_bs_id is not None and old_bs_id != target_bs_id:
            old_bs.release(uav_id)
            self.env.connection_matrix[uav_id, old_bs_id] = 0

        # 再尝试分配到新基站
        if target_bs.allocate(uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self.handover_successes += 1
            if is_reconnect:
                self.reconnect_successes += 1
                self.reconnect_cooldown.pop(uav_id, None)  # 清除冷却
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return True

        # 直接分配失败，尝试抢占低优先级UAV
        self.failure_reasons['allocation_failed'] += 1
        freed, kicked_ids = target_bs.kick_low_priority(uav, self.env.uavs)
        if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self.handover_successes += 1
            if is_reconnect:
                self.reconnect_successes += 1
                self.reconnect_cooldown.pop(uav_id, None)
                self.disconnect_timer.pop(uav_id, None)
            self._soft_migrate_kicked_uavs(kicked_ids, target_bs_id)
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return True

        # 抢占也失败，回滚到旧基站（避免断连）
        self.failure_reasons['preemption_failed'] += 1
        rollback_ok = False
        if old_bs_id is not None and old_bs_id != target_bs_id:
            if old_bs.available_capacity >= old_allocated_rate:
                old_bs.allocate(uav_id, old_allocated_rate)
                self.env.connection_matrix[uav_id, old_bs_id] = 1
                uav.connected_bs_id = old_bs_id
                rollback_ok = True
            else:
                # 尝试通过抢占旧基站低优先级UAV来回滚
                rollback_freed, rollback_kicked = old_bs.kick_low_priority(uav, self.env.uavs)
                if rollback_freed >= old_allocated_rate and old_bs.allocate(uav_id, old_allocated_rate):
                    self.env.connection_matrix[uav_id, old_bs_id] = 1
                    uav.connected_bs_id = old_bs_id
                    rollback_ok = True
                    self._soft_migrate_kicked_uavs(rollback_kicked, old_bs_id)
                else:
                    self.failure_reasons['rollback_failed'] += 1

        if not rollback_ok:
            # 回滚失败（有旧基站）或重连失败（无旧基站）
            if is_reconnect:
                # 重连失败：设置冷却期，防止反复尝试
                self.reconnect_cooldown[uav_id] = self.RECONNECT_COOLDOWN_STEPS
                self.disconnect_timer[uav_id] = self.disconnect_timer.get(uav_id, 0) + 1
            else:
                # 正常切换回滚失败：清空连接，进入重连路径
                uav.connected_bs_id = None
                uav.current_allocated_rate = 0.0
                self.rollback_fail_count += 1
                self.ghost_disconnect_count += 1
                self.reconnect_cooldown[uav_id] = self.RECONNECT_COOLDOWN_STEPS
                self.disconnect_timer[uav_id] = 0

        t_end = time()
        self.switching_latency_history.append((t_end - t_start) * 1000)
        return False

    def global_load_balancing_v2(self) -> int:
        migrations = 0
        load_ratios = [bs.load_ratio for bs in self.env.base_stations.values()]
        load_std = np.std(load_ratios)
        if load_std < 0.05:  # 降低阈值，更积极地均衡
            return 0
        load_with_id = [(bs_id, bs.load_ratio) for bs_id, bs in self.env.base_stations.items()]
        load_with_id.sort(key=lambda x: x[1], reverse=True)
        high_bs_id, high_load = load_with_id[0]
        low_bs_id, low_load = load_with_id[-1]
        if high_load - low_load < 0.1:  # 降低差距阈值
            return 0
        high_bs = self.env.base_stations[high_bs_id]
        low_bs = self.env.base_stations[low_bs_id]
        candidates = []
        for uav_id in list(high_bs.connected_uavs.keys()):
            uav = self.env.uavs[uav_id]
            current_sinr = self.env.sinr_matrix[uav_id, high_bs_id]
            target_sinr = self.env.sinr_matrix[uav_id, low_bs_id]
            sinr_loss = current_sinr - target_sinr
            if sinr_loss > 5:  # 放宽SINR损失容忍(从3到5)
                continue
            required = uav.current_allocated_rate
            if low_bs.available_capacity <= required * 0.3:  # 降低容量要求(从1.2到0.3)
                continue
            # 移除is_feasible和success_prob检查，直接用效用比较
            current_utility, _ = self.calculate_utility_with_downgrade(uav, high_bs_id, 1.0)
            target_utility, _ = self.calculate_utility_with_downgrade(uav, low_bs_id, 1.0)
            if target_utility - current_utility < -0.05:
                continue
            score = (required / max(uav.qos_profile.priority, 0.1)) * (1 - uav.qos_profile.criticality)
            candidates.append((uav_id, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        max_migrations = min(5, len(candidates))  # 从3增加到5
        for uav_id, _ in candidates[:max_migrations]:
            self.migration_attempts += 1
            if self.execute_handover(uav_id, low_bs_id, 1.0):
                migrations += 1
                self.migration_successes += 1
        return migrations

    def run_step(self, enable_load_balancing=True) -> Tuple[int, int]:
        handover_count = 0
        self.current_step_emergency = 0  # 每步重置

        # 优先处理断连UAV（按断连时长降序，断连越久优先级越高）
        disconnected_ids = [uid for uid in self.env.uavs.keys()
                            if self.env.uavs[uid].connected_bs_id is None]
        if disconnected_ids:
            # 递增断连计时器
            for uid in disconnected_ids:
                self.disconnect_timer[uid] = self.disconnect_timer.get(uid, 0) + 1
            # 按断连时长排序，最久的先处理
            disconnected_ids.sort(key=lambda uid: self.disconnect_timer.get(uid, 0), reverse=True)
            # 每步最多尝试重连所有断连UAV（冷却已禁用）
            for uav_id in disconnected_ids:
                decision = self.make_intelligent_decision(uav_id)
                if decision is not None:
                    target_bs_id, ratio = decision
                    if self.execute_handover(uav_id, target_bs_id, ratio):
                        handover_count += 1
                        self.disconnect_timer.pop(uav_id, None)

        # 再处理已连接UAV
        connected_ids = [uid for uid in self.env.uavs.keys()
                         if self.env.uavs[uid].connected_bs_id is not None]
        for uav_id in connected_ids:
            decision = self.make_intelligent_decision(uav_id)
            if decision is not None:
                target_bs_id, ratio = decision
                if self.execute_handover(uav_id, target_bs_id, ratio):
                    handover_count += 1

        migration_count = 0
        if enable_load_balancing and self.env.current_step % 5 == 0:
            migration_count = self.global_load_balancing_v2()
        return handover_count, migration_count

    def get_detailed_stats(self) -> Dict:
        # 切换成功率：仅计正常切换（排除重连），与传统算法对齐
        normal_attempts = max(self.handover_attempts - self.reconnect_attempts, 1)
        normal_success_rate = (self.handover_successes - self.reconnect_successes) / normal_attempts if normal_attempts > 0 else 0
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
            'execution_filter_stats': dict(self.execution_filter_stats),
            'emergency_count': self.emergency_count,
        }