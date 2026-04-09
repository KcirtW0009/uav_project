"""
切换算法模块

包含两种切换算法：
1. IntegratedHandoverAlgorithm — 传统切换算法（3GPP LTE/5G 标准基线）
2. EnhancedHandoverAlgorithm — 增强切换算法（业务感知 + 多机制协同）
"""

import numpy as np
from typing import Optional, Tuple, Dict
from collections import defaultdict
from time import time
from .business import BusinessType, QOS_PROFILES
from .environment import NetworkEnvironmentWithRecognition


# =============================================================================
# 传统切换算法（3GPP标准基线）
# =============================================================================

class IntegratedHandoverAlgorithm:
    """
    传统切换算法（3GPP LTE/5G 标准基线）

    核心特征（与真实3GPP协议一致）：
    - A3事件触发：邻区SINR > 服务小区SINR + Hys + Offset
    - 纯SINR目标选择：不考虑负载、业务类型
    - 单次分配尝试：不降级、不抢占，资源不足即失败
    - 无回滚机制：先断后连，失败则断连
    - 无负载均衡
    """

    def __init__(self, env: NetworkEnvironmentWithRecognition):
        self.env = env
        # 3GPP A3事件参数
        self.hysteresis = 2.0       # Hys: 迟滞参数(dB)
        self.offset = 0.0            # Ofn: 频率偏移(dB)
        self.emergency_sinr_threshold = -5
        self.emergency_satisfaction_threshold = 0.7
        # 统计指标
        self.handover_attempts = 0
        self.handover_successes = 0
        self.decision_calls = 0
        self.missed_opportunity = 0
        self.switching_latency_history = []
        self.decision_time_history = []
        self.failure_reasons = defaultdict(int)
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        # 按业务类型统计切换成功/失败
        self.handover_by_business = {bt: {'attempts': 0, 'successes': 0} for bt in BusinessType}

    def make_decision(self, uav_id: int) -> Optional[Tuple[int, float]]:
        """基于纯SINR的切换决策（3GPP A3事件）"""
        from time import time
        t_start = time()
        self.decision_calls += 1
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id

        # 紧急切换判定
        emergency = False
        if current_bs_id is not None:
            current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
            if current_sinr < self.emergency_sinr_threshold or uav.current_satisfaction < self.emergency_satisfaction_threshold:
                emergency = True

        # 未连接：选择SINR最高的基站，以完整速率接入
        if current_bs_id is None:
            best_bs, best_sinr = None, -999
            for bs_id in self.env.base_stations.keys():
                sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
                if sinr > best_sinr:
                    best_sinr, best_bs = sinr, bs_id
            self.decision_time_history.append((time() - t_start) * 1000)
            return (best_bs, 1.0) if best_bs is not None else None

        # 已连接：A3事件判定
        current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
        a3_threshold = current_sinr + self.hysteresis + self.offset
        best_bs, best_sinr = None, -999
        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
            if sinr > best_sinr:
                best_sinr, best_bs = sinr, bs_id

        self.decision_time_history.append((time() - t_start) * 1000)

        if best_bs is not None:
            if emergency:
                # 紧急模式：无迟滞
                if best_sinr > current_sinr:
                    return (best_bs, 1.0)
                self.missed_opportunity += 1
                return None
            # 正常A3事件
            if best_sinr > a3_threshold:
                return (best_bs, 1.0)

        if emergency:
            self.missed_opportunity += 1
        return None

    def execute_handover(self, uav_id: int, target_bs_id: int, downgrade_ratio: float) -> bool:
        """执行切换：先断后连，无回滚，无抢占"""
        from time import time
        t_start = time()
        self.handover_attempts += 1
        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio

        is_reconnect = (uav.connected_bs_id is None)
        if is_reconnect:
            self.reconnect_attempts += 1

        # 按业务类型统计
        biz_type = uav.business_type
        if biz_type in self.handover_by_business:
            self.handover_by_business[biz_type]['attempts'] += 1

        # 先释放旧基站
        if uav.connected_bs_id is not None:
            old_bs = self.env.base_stations[uav.connected_bs_id]
            old_bs.release(uav_id)
            self.env.connection_matrix[uav_id, uav.connected_bs_id] = 0

        # 单次分配尝试
        if target_bs.allocate(uav_id, required_rate):
            uav.connected_bs_id = target_bs_id
            uav.current_allocated_rate = required_rate
            self.env.connection_matrix[uav_id, target_bs_id] = 1
            uav.handover_count += 1
            self.handover_successes += 1
            if biz_type in self.handover_by_business:
                self.handover_by_business[biz_type]['successes'] += 1
            if is_reconnect:
                self.reconnect_successes += 1
            self.switching_latency_history.append((time() - t_start) * 1000)
            return True
        else:
            # 分配失败：UAV断连
            self.failure_reasons['allocation_failed'] += 1
            uav.connected_bs_id = None
            uav.current_allocated_rate = 0.0
            self.switching_latency_history.append((time() - t_start) * 1000)
            return False

    def run_step(self) -> int:
        """执行一个仿真步，返回成功切换次数"""
        handover_count = 0
        for uav_id in self.env.uavs.keys():
            decision = self.make_decision(uav_id)
            if decision is not None:
                target_bs_id, ratio = decision
                if self.execute_handover(uav_id, target_bs_id, ratio):
                    handover_count += 1
        return handover_count

    def get_detailed_stats(self) -> Dict:
        """获取算法详细统计"""
        normal_attempts = max(self.handover_attempts - self.reconnect_attempts, 1)
        normal_success_rate = (self.handover_successes - self.reconnect_successes) / normal_attempts
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
            'missed_opportunity_rate': self.missed_opportunity / max(self.decision_calls, 1),
            'handover_by_business': {bt.name: data for bt, data in self.handover_by_business.items()},
            'weighted_success_rate': self._compute_weighted_success_rate(),
        }

    def _compute_weighted_success_rate(self) -> float:
        """
        计算按业务类型优先级加权的切换成功率。

        关键业务（如控制信令）失败权重更高，
        公式: Σ(priority_i × success_rate_i) / Σ(priority_i)
        """
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


# =============================================================================
# 增强切换算法（业务感知 + 多机制协同）
# =============================================================================

class EnhancedHandoverAlgorithm:
    """
    增强切换算法

    在传统算法基础上引入多种增强机制：
    - 业务感知效用函数：根据业务类型调整SINR/负载/速率的权重
    - 动态切换阈值：根据业务优先级、负载、移动性动态调整
    - 降级比例搜索：资源不足时尝试以降级速率接入
    - 抢占机制：高优先级UAV可抢占低优先级资源
    - 回滚机制：切换失败时回滚到旧基站
    - 软迁移：被抢占的UAV尝试迁移到其他基站
    - 全局负载均衡：周期性迁移高负载基站的部分UAV
    """

    def __init__(self, env: NetworkEnvironmentWithRecognition, weight_config='optimized'):
        self.env = env
        self.weight_config = weight_config  # 保存配置类型
        # 效用函数默认权重
        self.w_sinr, self.w_load, self.w_rate = 0.45, 0.25, 0.30
        # 切换阈值参数 - 使用调优后的最佳参数
        self.base_threshold = 0.01
        self.confidence_factor_coeff = 0.002
        self.mobility_factor_coeff = 0.003
        self.priority_factor_control = 0.003
        self.threshold_lower_bound = 0.005
        self.epsilon = 0.01  # epsilon-greedy探索率 - 使用调优后的最佳参数
        self.handover_cooldown = 5  # 切换冷却时间 - 使用调优后的最佳参数
        self.use_load_mode = True  # 负载模式 - 使用调优后的最佳参数
        # 紧急切换阈值
        self.emergency_sinr_threshold = -5
        self.emergency_satisfaction_threshold = 0.7
        # 业务特化权重
        if weight_config == 'optimized':
            # 方案A：进一步优化的权重配置，用于MAPPO实验
            # 控制信令：进一步提高sinr权重到0.65，确保可靠性
            # 视频回传：进一步提高rate权重到0.60，确保带宽需求
            # 环境监测：降低sinr权重，提高rate权重，优化资源使用
            self.business_weights = {
                BusinessType.CONTROL_SIGNAL: {'sinr': 0.65, 'load': 0.10, 'rate': 0.25},
                BusinessType.VIDEO_STREAMING: {'sinr': 0.25, 'load': 0.15, 'rate': 0.60},
                BusinessType.ENVIRONMENT_MONITORING: {'sinr': 0.25, 'load': 0.15, 'rate': 0.60}
            }
        else:
            # 默认权重配置，保持与原有实验一致
            self.business_weights = {
                BusinessType.CONTROL_SIGNAL: {'sinr': 0.5, 'load': 0.2, 'rate': 0.3},
                BusinessType.VIDEO_STREAMING: {'sinr': 0.3, 'load': 0.25, 'rate': 0.45},
                BusinessType.ENVIRONMENT_MONITORING: {'sinr': 0.25, 'load': 0.25, 'rate': 0.5}
            }
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
        self.execution_filter_stats = defaultdict(int)
        self.rollback_fail_count = 0
        self.ghost_disconnect_count = 0
        self.reconnect_attempts = 0
        self.reconnect_successes = 0
        self.reconnect_cooldown = {}
        self.disconnect_timer = {}
        self.emergency_count = 0
        self.current_step_emergency = 0
        # 切换冷却时间记录
        self.handover_cooldown_timer = {}
        # 按业务类型统计切换成功/失败
        self.handover_by_business = {bt: {'attempts': 0, 'successes': 0} for bt in BusinessType}
        # 动态冷却时间调整相关
        self.cooling_history = []
        self.cooling_effect_analysis = {}
        self.optimal_cooling_times = {bt: 5 for bt in BusinessType}  # 初始最佳冷却时间

    # ==================== 效用函数 ====================

    def calculate_utility_with_downgrade(self, uav, bs_id: int, downgrade_ratio: float) -> Tuple[float, bool]:
        """
        计算带降级的业务感知效用值

        Args:
            uav: UAV实体
            bs_id: 基站ID
            downgrade_ratio: 降级比例(0-1)

        Returns:
            (utility, is_feasible): 效用值和是否可行
        """
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        sinr_norm = np.clip((sinr + 10) / 40, 0, 1)
        bs = self.env.base_stations[bs_id]
        load_ratio = bs.load_ratio
        required = uav.required_rate * downgrade_ratio
        available = bs.available_capacity

        # 可行性判断
        is_feasible = (available >= required * (0.6 if downgrade_ratio >= 0.8 else 0.7))

        # 速率匹配度
        rate_match = 0.0
        if required > 0:
            rate_ratio = available / required
            rate_match = 1 - np.exp(-3 * min(rate_ratio, 1.5))

        # 高速率奖励
        business_bonus = 0.05 * (downgrade_ratio - 0.8) / 0.2 if downgrade_ratio >= 0.8 else 0.0

        # 业务特化权重
        weights = self.business_weights.get(uav.business_type, {'sinr': 0.4, 'load': 0.3, 'rate': 0.3})
        utility = (weights['sinr'] * sinr_norm +
                   weights['load'] * (1 - load_ratio) +
                   weights['rate'] * rate_match +
                   business_bonus)
        return utility, is_feasible

    def calculate_dynamic_threshold(self, uav) -> float:
        """计算动态切换阈值（根据业务优先级、负载、移动性等调整）"""
        base = self.base_threshold
        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            base *= 0.5

        # 负载因子
        if uav.connected_bs_id is not None:
            load_factor = self.env.base_stations[uav.connected_bs_id].load_ratio
            adjustment = -0.005 * min(load_factor, 1.0)
            if uav.business_type == BusinessType.CONTROL_SIGNAL and load_factor > 0.7:
                adjustment -= 0.01
        else:
            adjustment = 0

        # 识别置信度因子
        confidence_factor = (1 - uav.recognition_confidence) * self.confidence_factor_coeff
        # 移动性因子
        velocity_norm = np.linalg.norm(uav.velocity)
        mobility_factor = -self.mobility_factor_coeff * min(velocity_norm / 10, 1.0)
        # 优先级因子
        priority_factor = -self.priority_factor_control * 1.5 if uav.business_type == BusinessType.CONTROL_SIGNAL else 0

        # 方案C：环境负载自适应 - 仅在optimized模式下启用
        load_adaptive_factor = 0.0
        if self.weight_config == 'optimized':
            global_load = self._get_global_load_ratio()
            if global_load > 0.85:  # 高负载时更保守（降低阈值从0.9到0.85）
                load_adaptive_factor = -0.02  # 提高阈值，减少切换（增加因子从0.01到0.02）
            elif global_load < 0.7:  # 低负载时更积极
                load_adaptive_factor = 0.01  # 降低阈值，增加切换（增加因子从0.005到0.01）

        dynamic_threshold = base + adjustment + confidence_factor + mobility_factor + priority_factor + load_adaptive_factor
        lower_bound = self.threshold_lower_bound * (0.5 if uav.business_type == BusinessType.CONTROL_SIGNAL else 1.0)
        return max(lower_bound, dynamic_threshold)

    def _get_global_load_ratio(self) -> float:
        """计算全局负载率"""
        total_load = 0.0
        for bs in self.env.base_stations.values():
            total_load += bs.load_ratio
        return total_load / len(self.env.base_stations) if self.env.base_stations else 0.0

    def predict_handover_success(self, uav, bs_id: int, downgrade_ratio: float) -> float:
        """预测切换成功概率（基于SINR和负载的联合模型）"""
        sinr = self.env.sinr_matrix[uav.uav_id, bs_id]
        bs = self.env.base_stations[bs_id]
        sinr_success = 1 / (1 + np.exp(-0.5 * (sinr + 5)))
        load_success = np.exp(-2.0 * bs.load_ratio)

        # 实际容量检查
        required_rate = uav.required_rate * downgrade_ratio
        if bs.available_capacity < required_rate * 0.8:
            load_success = min(load_success, 0.1)

        # 业务权重
        bw_map = {BusinessType.CONTROL_SIGNAL: 0.7, BusinessType.VIDEO_STREAMING: 0.5, BusinessType.ENVIRONMENT_MONITORING: 0.3}
        business_weight = bw_map.get(uav.business_type, 0.3)
        confidence_factor = 0.5 + 0.5 * uav.recognition_confidence

        success_prob = (business_weight * sinr_success + (1 - business_weight) * load_success) * confidence_factor
        return np.clip(success_prob, 0, 1)

    # ==================== 决策 ====================

    def _is_high_load_mode(self) -> bool:
        """判断系统是否处于高负载模式（需要保守策略）"""
        if not hasattr(self.env, 'base_stations') or not self.env.base_stations:
            return False
        total_capacity = 0
        used_capacity = 0
        for bs in self.env.base_stations.values():
            if hasattr(bs, 'capacity') and hasattr(bs, 'available_capacity'):
                total_cap = bs.capacity
                avail = bs.available_capacity
                if total_cap > 0:
                    total_capacity += total_cap
                    used_capacity += (total_cap - avail)
        if total_capacity == 0:
            return False
        load_ratio = used_capacity / total_capacity
        return load_ratio > 0.85

    def calculate_dynamic_cooling_time(self, uav) -> int:
        """
        计算动态冷却时间
        
        根据系统负载、网络状态和任务优先级自动调节冷却时长
        
        Args:
            uav: UAV实体
            
        Returns:
            冷却时间（步数）
        """
        # 基础冷却时间
        base_cooling = self.handover_cooldown
        
        # 1. 根据业务优先级调整
        priority_factor = uav.qos_profile.priority
        priority_adjustment = int((1 - priority_factor) * 3)  # 高优先级业务冷却时间更短
        
        # 2. 根据系统负载调整
        avg_load = np.mean([bs.load_ratio for bs in self.env.base_stations.values()])
        if avg_load > 0.8:
            load_adjustment = 2  # 高负载时增加冷却时间
        elif avg_load < 0.4:
            load_adjustment = -1  # 低负载时减少冷却时间
        else:
            load_adjustment = 0
        
        # 3. 根据网络状态调整
        sinr = self.env.sinr_matrix[uav.uav_id, uav.connected_bs_id] if uav.connected_bs_id is not None else -10
        if sinr < -5:
            sinr_adjustment = -1  # 信号质量差时减少冷却时间
        elif sinr > 15:
            sinr_adjustment = 1  # 信号质量好时增加冷却时间
        else:
            sinr_adjustment = 0
        
        # 4. 根据业务类型调整
        biz_type = uav.true_business_type
        business_adjustment = {
            BusinessType.CONTROL_SIGNAL: -2,  # 控制信令需要更快的响应
            BusinessType.VIDEO_STREAMING: 1,  # 视频回传可以容忍更长的冷却时间
            BusinessType.ENVIRONMENT_MONITORING: 0,  # 环境监测保持默认
        }.get(biz_type, 0)
        
        # 计算最终冷却时间
        cooling_time = base_cooling + priority_adjustment + load_adjustment + sinr_adjustment + business_adjustment
        
        # 确保冷却时间在合理范围内
        cooling_time = max(1, min(10, cooling_time))
        
        # 记录冷却时间历史
        self.cooling_history.append({
            'uav_id': uav.uav_id,
            'business_type': biz_type.name,
            'cooling_time': cooling_time,
            'load': avg_load,
            'sinr': sinr,
            'priority': priority_factor,
            'step': self.env.current_step
        })
        
        return cooling_time

    def _check_cooling_period(self, uav_id: int) -> bool:
        """
        检查UAV是否处于冷却期
        
        Args:
            uav_id: UAV ID
            
        Returns:
            是否处于冷却期
        """
        if uav_id in self.handover_cooldown_timer:
            cooling_end = self.handover_cooldown_timer[uav_id]
            if self.env.current_step < cooling_end:
                return True
            else:
                del self.handover_cooldown_timer[uav_id]
        return False

    def evaluate_cooling_effect(self) -> Dict:
        """
        评估冷却效果
        
        Returns:
            冷却效果分析结果
        """
        if not self.cooling_history:
            return {}
        
        # 按业务类型分析
        biz_analysis = {}
        for biz_type in BusinessType:
            biz_history = [h for h in self.cooling_history if h['business_type'] == biz_type.name]
            if biz_history:
                avg_cooling = np.mean([h['cooling_time'] for h in biz_history])
                success_rate = self.handover_by_business.get(biz_type, {'attempts': 1, 'successes': 0})['successes'] / \
                    self.handover_by_business.get(biz_type, {'attempts': 1, 'successes': 0})['attempts']
                
                biz_analysis[biz_type.name] = {
                    'avg_cooling_time': avg_cooling,
                    'success_rate': success_rate,
                    'sample_size': len(biz_history)
                }
        
        # 按负载水平分析
        load_analysis = {}
        load_buckets = [0.0, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(load_buckets) - 1):
            min_load, max_load = load_buckets[i], load_buckets[i+1]
            load_history = [h for h in self.cooling_history if min_load <= h['load'] < max_load]
            if load_history:
                avg_cooling = np.mean([h['cooling_time'] for h in load_history])
                load_analysis[f'{min_load:.1f}-{max_load:.1f}'] = {
                    'avg_cooling_time': avg_cooling,
                    'sample_size': len(load_history)
                }
        
        self.cooling_effect_analysis = {
            'business_analysis': biz_analysis,
            'load_analysis': load_analysis,
            'total_samples': len(self.cooling_history)
        }
        
        return self.cooling_effect_analysis

    def _conservative_decision(self, uav_id: int, t_start: float) -> Optional[Tuple[int, float]]:
        """
        高负载模式下的保守决策（类 A3 策略）：
        - 仅在 SINR 显著改善时切换（>3dB 迟滞）
        - 不使用 ε-greedy 探索
        - 不执行降级搜索（避免分配失败）
        - 不触发抢占/负载均衡副作用
        """
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id

        # 未连接：仍用宽松策略
        if current_bs_id is None:
            best_bs, best_sinr = None, -999
            for bs_id in self.env.base_stations.keys():
                sinr = self.env.sinr_matrix[uav_id, bs_id]
                if sinr > best_sinr:
                    bs = self.env.base_stations[bs_id]
                    if bs.available_capacity >= uav.required_rate * 0.9:
                        best_bs, best_sinr = bs_id, sinr
            self.decision_time_history.append((time() - t_start) * 1000)
            if best_bs is not None:
                return (best_bs, 1.0)
            return None

        current_sinr = self.env.sinr_matrix[uav_id, current_bs_id]
        hysteresis = 3.0  # 高负载下增大迟滞到 3dB（比标准 A3 的 2dB 更保守）

        for bs_id in self.env.base_stations.keys():
            if bs_id == current_bs_id:
                continue
            neighbor_sinr = self.env.sinr_matrix[uav_id, bs_id]
            if neighbor_sinr > current_sinr + hysteresis:
                bs = self.env.base_stations[bs_id]
                if bs.available_capacity >= uav.required_rate * 0.9:
                    self.decision_time_history.append((time() - t_start) * 1000)
                    return (bs_id, 1.0)

        self.decision_time_history.append((time() - t_start) * 1000)
        return None

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
        """增强决策：业务感知效用函数 + 动态阈值 + 降级搜索 + 负载自适应"""
        from time import time
        t_start = time()
        self.decision_calls += 1
        uav = self.env.uavs[uav_id]
        current_bs_id = uav.connected_bs_id

        # 检查冷却期（紧急切换和重连除外）
        if current_bs_id is not None:
            if self._check_cooling_period(uav_id):
                # 处于冷却期，记录决策
                self.decision_log.append({
                    'uav_id': uav.uav_id, 'step': self.env.current_step,
                    'current_bs': current_bs_id, 'target_bs': None,
                    'downgrade_ratio': 1.0, 'filter_reason': 'cooling_period'
                })
                self.decision_time_history.append((time() - t_start) * 1000)
                return None

        # [新增] 负载感知自适应机制
        # 当系统负载过高时，退化为保守的类 A3 策略，避免过度切换导致断连
        if self._is_high_load_mode():
            result = self._conservative_decision(uav_id, t_start)
            if result is not None:
                return result

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

        # ε-greedy探索：以epsilon概率随机选择基站
        if self.epsilon > 0 and np.random.rand() < self.epsilon:
            candidate_bs_ids = [bs_id for bs_id in self.env.base_stations.keys() if bs_id != current_bs_id]
            if candidate_bs_ids:
                random_bs = np.random.choice(candidate_bs_ids)
                # 随机选择时仍需满足最低可行性
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
        """为被抢占的UAV尝试软迁移到其他基站"""
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
        """
        执行切换：先释放旧基站 -> 分配新基站 -> 失败则抢占 -> 失败则回滚 -> 回滚失败则断连

        Args:
            uav_id: 无人机ID
            target_bs_id: 目标基站ID
            downgrade_ratio: 降级比例

        Returns:
            是否成功
        """
        from time import time
        t_start = time()
        uav = self.env.uavs[uav_id]
        target_bs = self.env.base_stations[target_bs_id]
        required_rate = uav.required_rate * downgrade_ratio
        is_reconnect = (uav.connected_bs_id is None)
        self.handover_attempts += 1

        # 按业务类型统计
        biz_type = uav.business_type
        if biz_type in self.handover_by_business:
            self.handover_by_business[biz_type]['attempts'] += 1

        # 记录旧基站信息
        old_bs_id = uav.connected_bs_id
        old_bs = self.env.base_stations[old_bs_id] if old_bs_id is not None else None
        old_allocated_rate = uav.current_allocated_rate

        # 1. 释放旧基站
        if old_bs_id is not None and old_bs_id != target_bs_id:
            old_bs.release(uav_id)
            self.env.connection_matrix[uav_id, old_bs_id] = 0

        # 2. 直接分配
        if target_bs.allocate(uav_id, required_rate):
            # 计算并设置动态冷却时间
            if not is_reconnect:
                cooling_time = self.calculate_dynamic_cooling_time(uav)
                self.handover_cooldown_timer[uav_id] = self.env.current_step + cooling_time
            return self._complete_handover(uav_id, target_bs_id, required_rate, is_reconnect, t_start)

        # 3. 抢占低优先级
        self.failure_reasons['allocation_failed'] += 1
        freed, kicked_ids = target_bs.kick_low_priority(uav, self.env.uavs)
        if freed >= required_rate and target_bs.allocate(uav_id, required_rate):
            self._soft_migrate_kicked_uavs(kicked_ids, target_bs_id)
            # 计算并设置动态冷却时间
            if not is_reconnect:
                cooling_time = self.calculate_dynamic_cooling_time(uav)
                self.handover_cooldown_timer[uav_id] = self.env.current_step + cooling_time
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
        # 按业务类型记录成功
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
        # 尝试通过抢占回滚
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
        """全局负载均衡：将高负载基站的低优先级UAV迁移到低负载基站"""
        load_ratios = [bs.load_ratio for bs in self.env.base_stations.values()]
        if np.std(load_ratios) < 0.05:
            return 0

        load_with_id = sorted(self.env.base_stations.items(), key=lambda x: x[1].load_ratio, reverse=True)
        high_bs_id, high_load = load_with_id[0][0], load_with_id[0][1].load_ratio
        low_bs_id, low_load = load_with_id[-1][0], load_with_id[-1][1].load_ratio
        if high_load - low_load < 0.1:
            return 0

        high_bs = self.env.base_stations[high_bs_id]
        low_bs = self.env.base_stations[low_bs_id]

        # 选择迁移候选
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
        """统计当前处于断连状态的UAV数量"""
        return sum(1 for uav in self.env.uavs.values() if uav.connected_bs_id is None)

    # ==================== 主循环 ====================

    def run_step(self, enable_load_balancing=True) -> Tuple[int, int]:
        """
        执行一个仿真步

        Args:
            enable_load_balancing: 是否启用负载均衡

        Returns:
            (切换次数, 迁移次数)
        """
        handover_count = 0
        self.current_step_emergency = 0

        # 优先处理断连UAV
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

        # 处理已连接UAV
        for uav_id in self.env.uavs.keys():
            if self.env.uavs[uav_id].connected_bs_id is not None:
                decision = self.make_intelligent_decision(uav_id)
                if decision is not None:
                    if self.execute_handover(uav_id, decision[0], decision[1]):
                        handover_count += 1

        # 周期性负载均衡（高负载模式下禁用，避免适得其反）
        migration_count = 0
        if enable_load_balancing and self.env.current_step % 5 == 0:
            if not self._is_high_load_mode():
                migration_count = self.global_load_balancing_v2()

        return handover_count, migration_count

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
            'execution_filter_stats': dict(self.execution_filter_stats),
            'emergency_count': self.emergency_count,
            'handover_by_business': {bt.name: data for bt, data in self.handover_by_business.items()},
            'weighted_success_rate': self._compute_weighted_success_rate(),
        }

    def _compute_weighted_success_rate(self) -> float:
        """
        计算按业务类型优先级加权的切换成功率。

        关键业务（如控制信令）失败权重更高，
        公式: Σ(priority_i × success_rate_i) / Σ(priority_i)
        """
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
