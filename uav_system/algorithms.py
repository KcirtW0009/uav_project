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
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return True
        else:
            self.failure_reasons['allocation_failed'] += 1
            freed = target_bs.kick_low_priority(uav, self.env.uavs)
            if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
                uav.connected_bs_id = target_bs_id
                uav.current_allocated_rate = required_rate
                self.env.connection_matrix[uav_id, target_bs_id] = 1
                uav.handover_count += 1
                self.handover_successes += 1
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
        return {
            'avg_decision_time_ms': np.mean(self.decision_time_history) if self.decision_time_history else 0,
            'avg_switching_latency_ms': np.mean(self.switching_latency_history) if self.switching_latency_history else 0,
            'max_switching_latency_ms': max(self.switching_latency_history) if self.switching_latency_history else 0,
            'failure_reasons': dict(self.failure_reasons),
            'handover_success_rate': self.handover_successes / max(self.handover_attempts, 1),
            'missed_opportunity_rate': self.missed_opportunity / max(self.decision_calls, 1)
        }


class EnhancedHandoverAlgorithm:
    def __init__(self, env: NetworkEnvironmentWithRecognition, weight_config='optimized'):
        self.env = env
        self.w_sinr = 0.45
        self.w_load = 0.25
        self.w_rate = 0.30
        self.base_threshold = 0.005
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

    def calculate_utility_with_downgrade(self, uav, bs_id: int, downgrade_ratio: float) -> Tuple[float, bool]:
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        sinr_norm = np.clip((sinr + 10) / 40, 0, 1)
        bs = self.env.base_stations[bs_id]
        load_ratio = bs.load_ratio
        required = uav.required_rate * downgrade_ratio
        available = bs.available_capacity
        if downgrade_ratio >= 0.8:
            is_feasible = (available >= required * 0.9)
        else:
            is_feasible = (available >= required * 0.95)
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

        # 重连逻辑：如果UAV未连接，优先尝试重连到最佳基站
        if current_bs_id is None:
            best_bs, best_ratio = self._emergency_select(uav)
            if best_bs is not None:
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                return (best_bs, best_ratio)
            # 无可用基站，记录时间后返回
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            return None

        emergency = False
        if current_bs_id is not None:
            current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
            control_signal_sinr_threshold = 0 if uav.business_type == BusinessType.CONTROL_SIGNAL else self.emergency_sinr_threshold
            if current_sinr < control_signal_sinr_threshold or uav.current_satisfaction < self.emergency_satisfaction_threshold:
                emergency = True
            if uav.business_type == BusinessType.CONTROL_SIGNAL and current_sinr < 5 and uav.current_satisfaction < 0.85:
                emergency = True
        if emergency:
            best_bs, best_ratio = self._emergency_select(uav)
            if best_bs is not None:
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                return (best_bs, best_ratio)
            # emergency分支下但无可用基站,也记录时间
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            return None
        feasible_ratios = uav.qos_profile.get_feasible_downgrade_ratios()
        candidates = []
        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            for ratio in feasible_ratios:
                utility, is_feasible = self.calculate_utility_with_downgrade(uav, bs_id, ratio)
                # 优化：放宽ratio限制，从0.6降至0.4，增加候选数量
                if is_feasible and ratio >= 0.4:
                    success_prob = self.predict_handover_success(uav, bs_id, ratio)
                    candidates.append((bs_id, ratio, utility, success_prob))
        if not candidates:
            t_end = time()
            self.decision_time_history.append((t_end - t_start) * 1000)
            return None

        # 统一预测阈值（优化：从0.55放松至0.52）
        UNIFIED_THRESHOLD = 0.52

        # 筛选候选（不再分层，使用统一阈值）
        high_success_candidates = [c for c in candidates if c[3] >= UNIFIED_THRESHOLD]

        # 如果仍然没有候选（极端情况），才使用兜底策略
        if not high_success_candidates:
            # 兜底：选择预测最高的候选（但预测都很低）
            high_success_candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
            # 只取前3个最优候选
            high_success_candidates = high_success_candidates[:3]
        if np.random.rand() < self.epsilon:
            choice = high_success_candidates[np.random.randint(len(high_success_candidates))]
            best_bs, best_ratio = choice[0], choice[1]
        else:
            # 优化：调整high_ratio阈值，从0.8降至0.6
            high_ratio_candidates = [c for c in high_success_candidates if c[1] >= 0.6]
            if high_ratio_candidates:
                candidates_to_use = high_ratio_candidates
            else:
                candidates_to_use = high_success_candidates
            candidates_to_use.sort(key=lambda x: (x[2], x[3]), reverse=True)
            best_bs, best_ratio = candidates_to_use[0][0], candidates_to_use[0][1]
        if current_bs_id is not None:
            current_utility, _ = self.calculate_utility_with_downgrade(uav, current_bs_id, 1.0)
            best_candidate = next(c for c in candidates if c[0]==best_bs and c[1]==best_ratio)
            best_utility = best_candidate[2]
            best_success_prob = best_candidate[3]
            dynamic_threshold = self.calculate_dynamic_threshold(uav)
            self.utility_history.append({'current': current_utility, 'best': best_utility})
            self.threshold_history.append(dynamic_threshold)
            if best_utility <= current_utility + dynamic_threshold:
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                return None

            # 简化过滤：只保留最基本的保护机制，提高连接稳定性
            current_bs = self.env.base_stations[current_bs_id] if current_bs_id is not None else None

            # 调试日志：记录决策信息
            self.decision_log.append({
                'uav_id': uav.uav_id,
                'current_load': current_bs.load_ratio if current_bs else None,
                'best_success_prob': best_success_prob,
                'current_satisfaction': uav.current_satisfaction
            })

            # 简化过滤：综合考虑多个因素
            current_bs = self.env.base_stations[current_bs_id] if current_bs_id is not None else None
            target_bs = self.env.base_stations[best_bs]

            # 调试日志：记录决策信息
            self.decision_log.append({
                'uav_id': uav.uav_id,
                'current_load': current_bs.load_ratio if current_bs else None,
                'best_success_prob': best_success_prob,
                'current_satisfaction': uav.current_satisfaction
            })

            # 定义统一阈值常量（优化：与决策阶段对齐）
            DECISION_THRESHOLD = 0.52  # 决策阶段的统一预测阈值
            HIGH_SUCCESS_THRESHOLD = 0.70  # 高成功率阈值（用于稳定性检查）

            # 过滤条件1：当前基站负载较低时不切换（避免不必要的切换）
            # 优化：阈值对齐至0.52，与决策阶段保持一致
            if current_bs is not None and current_bs.load_ratio < 0.3:
                if best_success_prob < DECISION_THRESHOLD:
                    # 记录过滤原因
                    self.decision_log[-1]['filter_reason'] = 'low_load'
                    self.decision_log[-1]['load_threshold'] = 0.3
                    t_end = time()
                    self.decision_time_history.append((t_end - t_start) * 1000)
                    self.missed_opportunity += 1
                    return None

            # 过滤条件2：目标基站预测成功率较低时谨慎切换
            # 优化：阈值从0.55降至0.48，与决策阶段保持一致
            if best_success_prob < DECISION_THRESHOLD and uav.current_satisfaction >= 0.7:
                # 记录过滤原因
                self.decision_log[-1]['filter_reason'] = 'low_success_prob'
                self.decision_log[-1]['prob_threshold'] = DECISION_THRESHOLD
                t_end = time()
                self.decision_time_history.append((t_end - t_start) * 1000)
                self.missed_opportunity += 1
                return None

            # 过滤条件3：连接稳定性检查
            # 优化：保持高阈值0.70，确保连接稳定性
            if current_bs is not None and current_bs.load_ratio < 0.7:
                current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
                target_sinr = self.env.sinr_matrix[uav_id, best_bs]
                sinr_gain = target_sinr - current_sinr
                # 如果SINR提升很小且当前满足率较高，过滤切换
                if sinr_gain < 2.0 and uav.current_satisfaction >= 0.8 and best_success_prob < HIGH_SUCCESS_THRESHOLD:
                    self.decision_log[-1]['filter_reason'] = 'low_gain'
                    self.decision_log[-1]['sinr_gain'] = sinr_gain
                    t_end = time()
                    self.decision_time_history.append((t_end - t_start) * 1000)
                    self.missed_opportunity += 1
                    return None
        t_end = time()
        self.decision_time_history.append((t_end - t_start) * 1000)
        return (best_bs, best_ratio)

    def execute_handover(self, uav_id: int, target_bs_id: int, downgrade_ratio: float) -> bool:
        """
        优化：移除执行阶段的预测过滤和统计过滤

        核心改进：
        1. 所有进入execute的切换都计入handover_attempts（移除统计偏差）
        2. 不再重新计算预测，直接执行切换
        3. 保持决策阶段的筛选逻辑，执行阶段只负责执行
        """
        from time import time
        t_start = time()

        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio

        # 优化：所有调用都计入尝试（移除统计过滤）
        self.handover_attempts += 1

        # 记录旧基站信息以便回滚
        old_bs_id = uav.connected_bs_id
        old_bs = self.env.base_stations[old_bs_id] if old_bs_id is not None else None
        old_allocated_rate = uav.current_allocated_rate

        # 先尝试分配到新基站
        if target_bs.allocate(uav_id, required_rate):
            # 分配成功，再释放旧连接（先分配后释放）
            if old_bs_id is not None and old_bs_id != target_bs_id:
                old_bs.release(uav_id)
                self.env.connection_matrix[uav_id, old_bs_id] = 0
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self.handover_successes += 1  # 所有成功都计入
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return True
        else:
            self.failure_reasons['allocation_failed'] += 1
            # 尝试抢占低优先级UAV
            freed = target_bs.kick_low_priority(uav, self.env.uavs)
            if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
                # 抢占成功，再释放旧连接
                if old_bs_id is not None and old_bs_id != target_bs_id:
                    old_bs.release(uav_id)
                    self.env.connection_matrix[uav_id, old_bs_id] = 0
                uav.connected_bs_id = target_bs_id
                uav.current_allocated_rate = required_rate
                self.env.connection_matrix[uav_id, target_bs_id] = 1
                uav.handover_count += 1
                self.handover_successes += 1  # 所有成功都计入
                t_end = time()
                self.switching_latency_history.append((t_end - t_start) * 1000)
                return True
            # 抢占失败，UAV保持原连接
            self.failure_reasons['preemption_failed'] += 1
            t_end = time()
            self.switching_latency_history.append((t_end - t_start) * 1000)
            return False

    def global_load_balancing_v2(self) -> int:
        migrations = 0
        load_ratios = [bs.load_ratio for bs in self.env.base_stations.values()]
        load_std = np.std(load_ratios)
        if load_std < 0.15:
            return 0
        load_with_id = [(bs_id, bs.load_ratio) for bs_id, bs in self.env.base_stations.items()]
        load_with_id.sort(key=lambda x: x[1], reverse=True)
        high_bs_id, high_load = load_with_id[0]
        low_bs_id, low_load = load_with_id[-1]
        if high_load - low_load < 0.2:
            return 0
        high_bs = self.env.base_stations[high_bs_id]
        low_bs = self.env.base_stations[low_bs_id]
        candidates = []
        for uav_id in list(high_bs.connected_uavs.keys()):
            uav = self.env.uavs[uav_id]
            current_sinr = self.env.sinr_matrix[uav_id, high_bs_id]
            target_sinr = self.env.sinr_matrix[uav_id, low_bs_id]
            sinr_loss = current_sinr - target_sinr
            if sinr_loss > 3:
                continue
            required = uav.current_allocated_rate
            if low_bs.available_capacity <= required * 1.2:
                continue
            current_utility, _ = self.calculate_utility_with_downgrade(uav, high_bs_id, 1.0)
            target_utility, is_feasible = self.calculate_utility_with_downgrade(uav, low_bs_id, 1.0)
            if not is_feasible or target_utility - current_utility < -0.02:
                continue
            success_prob = self.predict_handover_success(uav, low_bs_id, 1.0)
            if success_prob < 0.6:
                continue
            score = (required / max(uav.qos_profile.priority, 0.1)) * (1 - uav.qos_profile.criticality)
            candidates.append((uav_id, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        max_migrations = min(3, len(candidates))
        for uav_id, _ in candidates[:max_migrations]:
            self.migration_attempts += 1
            if self.execute_handover(uav_id, low_bs_id, 1.0):
                migrations += 1
                self.migration_successes += 1
        return migrations

    def run_step(self, enable_load_balancing=True) -> Tuple[int, int]:
        handover_count = 0
        for uav_id in self.env.uavs.keys():
            decision = self.make_intelligent_decision(uav_id)
            if decision is not None:
                target_bs_id, ratio = decision
                if self.execute_handover(uav_id, target_bs_id, ratio):
                    handover_count += 1
        migration_count = 0
        if enable_load_balancing and self.env.current_step % 20 == 0:
            migration_count = self.global_load_balancing_v2()
        return handover_count, migration_count

    def get_detailed_stats(self) -> Dict:
        return {
            'avg_decision_time_ms': np.mean(self.decision_time_history) if self.decision_time_history else 0,
            'max_decision_time_ms': max(self.decision_time_history) if self.decision_time_history else 0,
            'avg_switching_latency_ms': np.mean(self.switching_latency_history) if self.switching_latency_history else 0,
            'max_switching_latency_ms': max(self.switching_latency_history) if self.switching_latency_history else 0,
            'failure_reasons': dict(self.failure_reasons),
            'handover_success_rate': self.handover_successes / max(self.handover_attempts, 1),
            'missed_opportunity_rate': self.missed_opportunity / max(self.decision_calls, 1),
            'migration_success_rate': self.migration_successes / max(self.migration_attempts, 1) if self.migration_attempts > 0 else 0,
            'avg_utility_improvement': np.mean([u['best'] - u['current'] for u in self.utility_history]) if self.utility_history else 0,
            'avg_dynamic_threshold': np.mean(self.threshold_history) if self.threshold_history else 0
        }