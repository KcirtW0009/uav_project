import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
from .business import BusinessType, QoSProfile, QOS_PROFILES, BUSINESS_FEATURE_PARAMS

class BaseStation:
    def __init__(self, bs_id: int, capacity: float, position: np.ndarray = None,
                 bs_type: str = 'macro'):
        self.bs_id = bs_id
        self.capacity = capacity
        self.position = position if position is not None else np.random.rand(3) * 1000
        self.bs_type = bs_type
        self.connected_uavs: Dict[int, float] = {}
        self.current_load = 0.0
        self.failure_state = False
        self.coverage_radius = 500.0 if bs_type == 'macro' else 200.0

    @property
    def available_capacity(self) -> float:
        if self.failure_state:
            return 0.0
        return max(0, self.capacity - self.current_load)

    @property
    def load_ratio(self) -> float:
        if self.failure_state:
            return 1.0
        return min(1.0, self.current_load / self.capacity) if self.capacity > 0 else 1.0

    def can_accept(self, required_rate: float, allow_preemption: bool = True) -> bool:
        if self.failure_state:
            return False
        if self.available_capacity >= required_rate:
            return True
        if allow_preemption:
            return required_rate <= self.capacity
        return False

    def allocate(self, uav_id: int, rate: float) -> bool:
        if self.failure_state:
            return False
        if rate > self.available_capacity:
            return False
        self.connected_uavs[uav_id] = rate
        self.current_load += rate
        return True

    def release(self, uav_id: int):
        if uav_id in self.connected_uavs:
            rate = self.connected_uavs.pop(uav_id)
            self.current_load -= rate
            self.current_load = max(0, self.current_load)

    def kick_low_priority(self, target_uav, uav_pool: Dict[int, 'UAV']) -> float:
        target_priority = target_uav.qos_profile.priority
        freed_space = 0.0
        to_kick = []
        for uav_id, rate in self.connected_uavs.items():
            if uav_id in uav_pool:
                uav = uav_pool[uav_id]
                if (uav.qos_profile.priority < target_priority and
                        uav.business_type != BusinessType.CONTROL_SIGNAL):
                    to_kick.append((uav_id, rate, uav.qos_profile.priority))
        to_kick.sort(key=lambda x: x[2])
        for uav_id, rate, _ in to_kick:
            self.release(uav_id)
            freed_space += rate
            if uav_id in uav_pool:
                uav_pool[uav_id].current_allocated_rate = 0.0
                # 关键修复：清空被抢占UAV的连接状态
                # 避免被抢占UAV持有指向已释放基站的幽灵连接
                uav_pool[uav_id].connected_bs_id = None
            if freed_space > target_uav.qos_profile.ideal_rate:
                break
        return freed_space

    def set_failure(self, failed: bool):
        self.failure_state = failed
        if failed:
            self.current_load = 0.0
            self.connected_uavs.clear()

    def __repr__(self):
        status = "[故障]" if self.failure_state else ""
        return (f"BS[{self.bs_id}]{status} "
                f"(Load:{self.current_load:.1f}/{self.capacity:.1f} "
                f"Ratio:{self.load_ratio:.2f}, Type:{self.bs_type}, "
                f"Connections:{len(self.connected_uavs)})")


class UAV:
    def __init__(self, uav_id: int, business_type: BusinessType,
                 position: np.ndarray = None, velocity: np.ndarray = None,
                 mission_priority: float = 0.5):
        self.uav_id = uav_id
        self.true_business_type = business_type
        self.business_type = business_type
        self.position = position if position is not None else np.random.rand(3) * 1000
        self.qos_profile = QOS_PROFILES[business_type]
        self.connected_bs_id: Optional[int] = None
        self.current_allocated_rate = 0.0
        self.sinr_db = 0.0
        self.recognition_confidence = 1.0
        self.satisfaction_history = deque(maxlen=100)
        self.handover_count = 0
        self.velocity = velocity if velocity is not None else (np.random.rand(3) - 0.5) * 20
        self.mission_priority = mission_priority
        self.current_latency = 0.0
        self.packet_loss_rate = 0.0
        self.recognition_history = deque(maxlen=50)
        self.is_emergency = False

    @property
    def required_rate(self) -> float:
        return self.qos_profile.ideal_rate

    @property
    def min_required_rate(self) -> float:
        return self.qos_profile.min_rate

    @property
    def current_satisfaction(self) -> float:
        # 使用真实业务类型的QoS配置计算满意率,反映真实用户体验
        # 这确保了识别准确率的下降会正确地反映在性能指标上
        true_qos = QOS_PROFILES[self.true_business_type]
        return true_qos.calculate_satisfaction(self.current_allocated_rate)

    def update_recognition(self, recognized_type: BusinessType, confidence: float):
        old_type = self.business_type
        self.business_type = recognized_type
        self.recognition_confidence = confidence
        self.qos_profile = QOS_PROFILES[recognized_type]
        self.recognition_history.append({
            'old_type': old_type,
            'new_type': recognized_type,
            'confidence': confidence,
            'correct': recognized_type == self.true_business_type
        })

    def move(self, time_step: float = 1.0, boundary: Tuple[float, float] = (0, 1000)):
        self.position += self.velocity * time_step
        for i in range(3):
            if self.position[i] < boundary[0]:
                self.position[i] = boundary[0]
                self.velocity[i] = abs(self.velocity[i])
            elif self.position[i] > boundary[1]:
                self.position[i] = boundary[1]
                self.velocity[i] = -abs(self.velocity[i])

    def generate_features(self) -> np.ndarray:
        params = BUSINESS_FEATURE_PARAMS[self.true_business_type]
        delay = np.random.normal(params['delay'][0], params['delay'][1])
        delay = np.clip(delay, 0, 300)
        bandwidth = np.random.normal(params['bandwidth'][0], params['bandwidth'][1])
        bandwidth = np.clip(bandwidth, 10, 500)
        loss_rate = np.random.beta(params['loss_beta'][0], params['loss_beta'][1])
        loss_rate = loss_rate * params['loss_scale']
        jitter = np.random.normal(params['jitter'][0], params['jitter'][1])
        jitter = np.clip(jitter, 0, 20)
        return np.array([delay, bandwidth, loss_rate, jitter])

    def record_satisfaction(self):
        self.satisfaction_history.append(self.current_satisfaction)

    def update_latency_estimate(self, sinr_db: float):
        base_latency = self.qos_profile.max_delay * 0.5
        sinr_factor = max(0.1, 1 - (sinr_db + 10) / 30)
        self.current_latency = base_latency * sinr_factor
        self.packet_loss_rate = max(0, 0.1 - (sinr_db + 10) / 200)

    def __repr__(self):
        return (f"UAV[{self.uav_id}] (TrueType: {self.true_business_type.name}, "
                f"RecogType: {self.business_type.name}, "
                f"Rate: {self.current_allocated_rate:.1f}/{self.required_rate:.1f}, "
                f"Sat: {self.current_satisfaction:.2f}, BS: {self.connected_bs_id}, "
                f"Conf: {self.recognition_confidence:.2f})")