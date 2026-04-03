"""
网络实体定义

定义基站(BaseStation)和无人机(UAV)两个核心实体类。
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
from .business import BusinessType, QoSProfile, QOS_PROFILES, BUSINESS_FEATURE_PARAMS


class BaseStation:
    """
    基站实体

    管理连接的UAV、资源分配、负载统计、故障状态等功能。
    支持资源抢占机制（用于增强算法）。

    Attributes:
        bs_id: 基站唯一标识
        capacity: 总容量(Mbps)
        position: 三维坐标位置(m)
        bs_type: 基站类型('macro'或'small')
        connected_uavs: 已连接UAV及其分配速率的字典
        current_load: 当前总负载(Mbps)
        failure_state: 是否处于故障状态
        coverage_radius: 覆盖半径(m)
    """

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
        """可用容量(Mbps)，故障时为0"""
        if self.failure_state:
            return 0.0
        return max(0, self.capacity - self.current_load)

    @property
    def load_ratio(self) -> float:
        """负载率(0-1)，故障时为1.0"""
        if self.failure_state:
            return 1.0
        return min(1.0, self.current_load / self.capacity) if self.capacity > 0 else 1.0

    def can_accept(self, required_rate: float, allow_preemption: bool = True) -> bool:
        """检查是否可以接受指定速率的UAV连接"""
        if self.failure_state:
            return False
        if self.available_capacity >= required_rate:
            return True
        if allow_preemption:
            return required_rate <= self.capacity
        return False

    def allocate(self, uav_id: int, rate: float) -> bool:
        """
        为UAV分配资源

        Args:
            uav_id: 无人机ID
            rate: 分配速率(Mbps)

        Returns:
            是否分配成功
        """
        if self.failure_state:
            return False
        if rate > self.available_capacity:
            return False
        self.connected_uavs[uav_id] = rate
        self.current_load += rate
        return True

    def release(self, uav_id: int):
        """释放指定UAV占用的资源"""
        if uav_id in self.connected_uavs:
            rate = self.connected_uavs.pop(uav_id)
            self.current_load -= rate
            self.current_load = max(0, self.current_load)

    def kick_low_priority(self, target_uav, uav_pool: Dict[int, 'UAV']) -> Tuple[float, List[int]]:
        """
        抢占低优先级UAV的资源

        按优先级从低到高踢出UAV，直到释放足够空间或遍历完所有候选。
        关键业务（criticality >= 0.9）的UAV不会被踢出。

        Args:
            target_uav: 发起抢占的目标UAV
            uav_pool: 所有UAV的字典

        Returns:
            (释放的空间, 被踢UAV的ID列表)
        """
        target_priority = target_uav.qos_profile.priority
        freed_space = 0.0
        to_kick = []
        for uav_id, rate in self.connected_uavs.items():
            if uav_id in uav_pool:
                uav = uav_pool[uav_id]
                if (uav.qos_profile.priority < target_priority and
                        uav.qos_profile.criticality < 0.9):
                    to_kick.append((uav_id, rate, uav.qos_profile.priority))
        to_kick.sort(key=lambda x: x[2])  # 按优先级升序
        kicked_uav_ids = []
        for uav_id, rate, _ in to_kick:
            self.release(uav_id)
            freed_space += rate
            if uav_id in uav_pool:
                uav_pool[uav_id].current_allocated_rate = 0.0
                uav_pool[uav_id].connected_bs_id = None
                kicked_uav_ids.append(uav_id)
            if freed_space > target_uav.qos_profile.ideal_rate:
                break
        return freed_space, kicked_uav_ids

    def set_failure(self, failed: bool):
        """设置/恢复基站故障状态"""
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
    """
    无人机实体

    模拟UAV的移动、业务特征生成、识别状态更新、满意度计算等功能。

    Attributes:
        uav_id: 无人机唯一标识
        true_business_type: 真实业务类型（不随识别结果变化）
        business_type: 当前识别的业务类型（由识别模型更新）
        position: 三维坐标位置(m)
        velocity: 三维速度向量(m/step)
        connected_bs_id: 当前连接的基站ID，None表示断连
        current_allocated_rate: 当前分配速率(Mbps)
        recognition_confidence: 识别置信度(0-1)
    """

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
        """当前业务类型的理想速率需求(Mbps)"""
        return self.qos_profile.ideal_rate

    @property
    def min_required_rate(self) -> float:
        """当前业务类型的最低速率需求(Mbps)"""
        return self.qos_profile.min_rate

    @property
    def current_satisfaction(self) -> float:
        """
        当前满意度（基于真实业务类型的多维度评估）

        使用真实业务类型的QoS配置，综合速率、时延、丢包率三个维度计算满意度，
        确保识别错误会正确反映为性能下降。
        """
        true_qos = QOS_PROFILES[self.true_business_type]
        return true_qos.calculate_satisfaction(
            self.current_allocated_rate,
            estimated_delay=self.current_latency,
            loss_rate=self.packet_loss_rate
        )

    def update_recognition(self, recognized_type: BusinessType, confidence: float):
        """
        更新业务识别结果

        Args:
            recognized_type: 识别出的业务类型
            confidence: 识别置信度
        """
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
        """更新位置，碰到边界时反弹"""
        self.position += self.velocity * time_step
        for i in range(3):
            if self.position[i] < boundary[0]:
                self.position[i] = boundary[0]
                self.velocity[i] = abs(self.velocity[i])
            elif self.position[i] > boundary[1]:
                self.position[i] = boundary[1]
                self.velocity[i] = -abs(self.velocity[i])

    def generate_features(self) -> np.ndarray:
        """
        生成业务特征向量（用于识别模型输入）

        特征维度: [时延(ms), 带宽(Mbps), 丢包率, 抖动(ms)]
        各特征的分布参数由 BUSINESS_FEATURE_PARAMS 定义。
        """
        params = BUSINESS_FEATURE_PARAMS[self.true_business_type]
        delay = np.clip(np.random.normal(params['delay'][0], params['delay'][1]), 0, 300)
        bandwidth = np.clip(np.random.normal(params['bandwidth'][0], params['bandwidth'][1]), 10, 500)
        loss_rate = np.random.beta(params['loss_beta'][0], params['loss_beta'][1]) * params['loss_scale']
        jitter = np.clip(np.random.normal(params['jitter'][0], params['jitter'][1]), 0, 20)
        return np.array([delay, bandwidth, loss_rate, jitter])

    def record_satisfaction(self):
        """记录当前满意度到历史"""
        self.satisfaction_history.append(self.current_satisfaction)

    def update_latency_estimate(self, sinr_db: float):
        """根据SINR估算时延和丢包率

        使用更敏感的模型，使低 SINR 下满意度明显下降，
        确保满意度指标有足够的区分度来区分不同算法的性能差异。
        """
        # 时延模型: SINR 越低时延越高，接近 max_delay 时满意度才下降
        # SINR >= 10dB: delay ≈ 0.3 * max_delay (良好)
        # SINR = 0dB:   delay ≈ 0.8 * max_delay (一般)
        # SINR = -5dB:  delay ≈ 1.5 * max_delay (差)
        sinr_factor = max(0.05, 1.5 - (sinr_db + 10) / 20)
        base_latency = self.qos_profile.max_delay * 0.8
        self.current_latency = base_latency * sinr_factor

        # 丢包率模型: SINR 越低丢包率越高
        # SINR >= 10dB: loss ≈ 0.001 (极低)
        # SINR = 0dB:   loss ≈ 0.05
        # SINR = -5dB:  loss ≈ 0.15
        self.packet_loss_rate = max(0.0, 0.2 * np.exp(-0.15 * sinr_db))

    def __repr__(self):
        return (f"UAV[{self.uav_id}] (TrueType: {self.true_business_type.name}, "
                f"RecogType: {self.business_type.name}, "
                f"Rate: {self.current_allocated_rate:.1f}/{self.required_rate:.1f}, "
                f"Sat: {self.current_satisfaction:.2f}, BS: {self.connected_bs_id}, "
                f"Conf: {self.recognition_confidence:.2f})")
