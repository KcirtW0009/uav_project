"""
网络环境模块

定义仿真网络环境，包括基站部署、UAV初始化、SINR计算、
业务识别更新、中断检测等功能。提供基础环境和增强环境（含随机事件）两个版本。
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
from .config import GLOBAL_SEED, INTERRUPTION_CONFIG
from .business import BusinessType, QOS_PROFILES
from .entities import BaseStation, UAV
from .satisfaction import HierarchicalSatisfactionMetric
from .recognition import AdaptiveRecognitionUpdater


class NetworkEnvironmentWithRecognition:
    """
    网络环境（含业务识别）

    管理基站、UAV的初始化和仿真步进，包括：
    - SINR矩阵计算（基于路径损耗模型）
    - 业务识别模型集成
    - 中断检测与统计
    - 各类状态统计
    """

    def __init__(self, num_bs=8, num_uav=50, recognition_model=None, scaler=None,
                 seed=GLOBAL_SEED, scenario='default', bs_capacity_range=None):
        np.random.seed(seed)
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.recognition_model = recognition_model
        self.scaler = scaler
        self.scenario = scenario
        self.bs_capacity_range = bs_capacity_range
        self.current_step = 0
        self.recognition_updater = AdaptiveRecognitionUpdater(min_update_interval=5, drift_threshold=0.25)
        self.feedback_buffer = deque(maxlen=100)

        self.base_stations: Dict[int, BaseStation] = {}
        self._init_base_stations(scenario)
        self.uavs: Dict[int, UAV] = {}
        self._init_uavs(scenario)
        self.connection_matrix = np.zeros((num_uav, num_bs), dtype=int)
        self.sinr_matrix = np.zeros((num_uav, num_bs))
        self._update_sinr_matrix()
        self._initialize_connections()

        # 统计历史
        self.stats_history = {
            'step': [], 'avg_satisfaction': [], 'recognition_accuracy': [],
            'total_throughput': [], 'load_variance': [], 'critical_satisfaction': [],
            'weighted_satisfaction': [], 'interruption_rate': [], 'avg_interruption_duration': []
        }

        # 中断率统计
        self.interruption_threshold = INTERRUPTION_CONFIG['threshold']
        self.interruption_duration = INTERRUPTION_CONFIG['duration']
        self.control_signal_threshold = INTERRUPTION_CONFIG['control_signal_threshold']
        self.control_signal_duration = INTERRUPTION_CONFIG['control_signal_duration']
        self.uav_interruption_counters = {uav_id: 0 for uav_id in range(num_uav)}
        self.interruption_events = []
        self.active_interruptions = {}
        self.total_interruptions = 0
        self.interrupted_uavs = set()

    # ==================== 初始化 ====================

    def _init_base_stations(self, scenario: str):
        """
        根据场景初始化基站

        基站部署参数参照 3GPP TR 38.901 / TR 36.777 标准及中国5G实际部署数据:
        - 宏基站(Macro): 高度 25m(楼顶)或 35m(铁塔)，对应3GPP UMa模型 h_BS=25m
        - 小基站(Small): 高度 6~10m(灯杆/墙面)，对应3GPP UMi模型 h_BS=10m
        - 宏微比例: 城市密集区小基站占主体(60%~80%)，其他场景至少40%

        UAV飞行高度: 低空域 60~300m（符合民航规章）
        """
        if self.bs_capacity_range is not None:
            low, high = self.bs_capacity_range
        else:
            # 5G基站容量参考: 宏站100MHz带宽≈1Gbps, 微站≈500Mbps
            # 载波聚合/高阶MIMO可达2Gbps+
            # 容量按各场景UAV数量×业务比例×理想速率的 1/0.77 设计，保持~77%负载率
            capacity_map = {
                'urban': (400, 800), 'emergency': (700, 1000),
                'agriculture': (600, 900), 'default': (500, 1000),
                'smart_city': (1500, 2400),           # 400UAV×60%视频→12100Mbps需求→8×avg1950≈77%
                'industrial_inspection': (1400, 2300), # 300UAV×75%视频→11303Mbps需求→8×avg1850≈77%
                'emergency_rescue': (900, 1200),       # 300UAV×10%视频→1643Mbps需求→~20%负载(URLLC低负载场景)
                'logistics_delivery': (1200, 2100),    # 500UAV×40%视频→10175Mbps需求→8×avg1650≈77%
            }
            low, high = capacity_map.get(scenario, capacity_map['default'])

        # 水平部署范围(m): x,y坐标的覆盖范围
        pos_range_map = {
            'urban': 800, 'emergency': 1200, 'agriculture': 1500, 'default': 1000,
            'smart_city': 800, 'industrial_inspection': 600, 'emergency_rescue': 1200,
            'logistics_delivery': 1500
        }
        self.pos_range_xy = pos_range_map.get(scenario, 1000)  # 实例变量，供UAV初始化使用

        # ====== 基站高度配置（3GPP标准） ======
        # 参考: 3GPP TR 38.901 Table 7.4.1-1
        #   UMa (Urban Macro):   BS天线高度 h_BS = 25m (平均建筑高度/楼顶)
        #   UMi (Urban Micro):    BS天线高度 h_BS = 10m (低于屋顶，街灯/杆)
        #   RMa (Rural Macro):   BS天线高度 h_BS = 35m
        macro_height = 25.0   # 宏基站: 楼顶部署
        small_height = 8.0    # 小基站: 灯杆/墙面/室内
        rural_macro_height = 35.0  # 农村宏基站可稍高

        # 场景特定的高度配置
        height_config = {
            'macro': {'smart_city': macro_height, 'industrial_inspection': macro_height,
                      'logistics_delivery': macro_height, 'emergency_rescue': macro_height},
            'small': {'smart_city': small_height, 'industrial_inspection': small_height,
                      'logistics_delivery': small_height, 'emergency_rescue': small_height}
        }

        # ====== 宏微比例配置（中国5G实际部署） ======
        # 数据来源: 信通院2024-2025报告, 运营商集采公告
        #   城市密集区: 小基站占比 60%~80%（补盲+室分+热点）
        #   工业园区: 小基站占比 ~50%（厂区覆盖）
        #   农业地区: 以宏基站为主, 小基站 ~30%
        #   应急救援: 宏基站为主(可靠性优先), 小基站 ~25%
        #   物流配送: 混合型, 小基站 ~45%
        small_cell_ratios = {
            'smart_city': 0.70,              # 城市监控: 70%小基站(密集商业区)
            'industrial_inspection': 0.50,  # 工业巡检: 50%小基站(厂房内部署)
            'agriculture': 0.30,             # 农业植保: 30%小基站(广域覆盖为主)
            'emergency_rescue': 0.25,         # 应急救援: 25%小基站(可靠性优先)
            'logistics_delivery': 0.45,       # 物流配送: 45%小基站(混合部署)
            'urban': 0.70,                     # 默认城区
            'default': 0.50,                   # 默认混合
            'emergency': 0.40,
        }
        small_ratio = small_cell_ratios.get(scenario, 0.50)

        for i in range(self.num_bs):
            capacity = np.random.uniform(low, high)

            # 确定基站类型（基于场景比例）
            if np.random.rand() < small_ratio:
                bs_type = 'small'
                bs_h = height_config['small'].get(scenario, small_height)
                # 小基站容量通常较低（微功率节点）
                capacity *= np.random.uniform(0.4, 0.7)
            else:
                bs_type = 'macro'
                bs_h = height_config['macro'].get(scenario, macro_height)
                # 农村场景宏基站可能更高
                if scenario == 'agriculture':
                    bs_h = rural_macro_height

            # 构建三维位置: x,y在水平范围随机, z为固定高度加微小偏移(±2m模拟不同楼层)
            x = np.random.uniform(0, self.pos_range_xy)
            y = np.random.uniform(0, self.pos_range_xy)
            z = bs_h + np.random.uniform(-2, 2)  # 微小高度差异

            position = np.array([x, y, z])
            self.base_stations[i] = BaseStation(
                i, capacity=capacity,
                position=position, bs_type=bs_type)

    def _init_uavs(self, scenario: str):
        """根据场景初始化UAV，分配业务类型"""
        # 业务比例: [控制信令, 视频回传, 环境监测]
        # 与论文KPI表格对齐:
        #   - 应急救援: 85%控制(≤20ms,99.999%)+10%视频+5%监测 → 期望时延≈57ms, 带宽≈5.5Mbps (URLLC)
        #   - 物流配送: 50%控制+40%视频+10%监测 → 期望时延≈118ms, 带宽≈20.4Mbps (eMBB为主)
        #   - 城市监控: 30%控制+60%视频+10%监测 → 视频流为主(eMBB)
        #   - 工业巡检: 15%控制+75%视频+10%监测 → 4K视频主导(eMBB)
        #   - 农业监测: 15%控制+25%视频+60%监测 → 海量传感器(mMTC)
        ratios_map = {
            'emergency': [0.3, 0.5, 0.2],
            'agriculture': [0.15, 0.25, 0.60],      # 农田: mMTC大幅量监测
            'default': [0.4, 0.3, 0.3],
            'smart_city': [0.30, 0.60, 0.10],       # 城市监控: 视频为主(eMBB)
            'industrial_inspection': [0.15, 0.75, 0.10], # 工业巡检: 4K视频为主(eMBB)
            'emergency_rescue': [0.85, 0.10, 0.05],  # 应急救援: URLLC, 期望时延≈57ms
            'logistics_delivery': [0.50, 0.40, 0.10], # 物流配送: 均衡型, 期望时延≈118ms
        }
        ratios = ratios_map.get(scenario, ratios_map['default'])
        vel_map = {'urban': 15, 'emergency': 30, 'default': 20,
                   'smart_city': 10, 'industrial_inspection': 15,
                   'emergency_rescue': 30, 'logistics_delivery': 25}
        vel = vel_map.get(scenario, 20)

        business_types = [BusinessType.CONTROL_SIGNAL, BusinessType.VIDEO_STREAMING, BusinessType.ENVIRONMENT_MONITORING]

        # ====== UAV飞行高度配置（低空域） ======
        # 参考: 民航规章/3GPP TR 36.777 UAV-UE场景
        #   民用无人机典型作业高度: 60~300m
        #   城市监控/巡检: 80~150m
        #   农业植保: 50~120m (低空喷洒)
        #   应急救援/物流: 100~200m
        uav_altitude_range = {
            'smart_city': (80, 150),            # 城市监控: 中低空
            'industrial_inspection': (80, 180), # 工业巡检: 厂区上空
            'agriculture': (50, 120),           # 农业植保: 低空作业
            'emergency_rescue': (100, 250),      # 应急救援: 中空
            'logistics_delivery': (80, 200),       # 物流配送: 城市低空
            'urban': (80, 180),
            'default': (80, 200),
            'emergency': (100, 250),
        }
        alt_min, alt_max = uav_altitude_range.get(scenario, (80, 200))

        for i in range(self.num_uav):
            rand = np.random.rand()
            biz_type = business_types[0] if rand < ratios[0] else (
                business_types[1] if rand < ratios[0] + ratios[1] else business_types[2])

            # UAV位置: x,y在水平范围随机, z在低空域高度范围随机
            x = np.random.uniform(0, self.pos_range_xy)
            y = np.random.uniform(0, self.pos_range_xy)
            z = np.random.uniform(alt_min, alt_max)  # 合理的低空域高度
            position = np.array([x, y, z])

            self.uavs[i] = UAV(i, business_type=biz_type,
                             position=position,
                             velocity=(np.random.rand(3) - 0.5) * vel)

    def _update_sinr_matrix(self):
        """更新SINR矩阵（基于距离相关的路径损耗模型）"""
        for uav_id, uav in self.uavs.items():
            for bs_id, bs in self.base_stations.items():
                distance = np.linalg.norm(uav.position - bs.position)
                if self.scenario == 'urban':
                    path_loss = 140 + 38 * np.log10(max(distance / 1000, 0.001))
                else:
                    path_loss = 128.1 + 37.6 * np.log10(max(distance / 1000, 0.001))
                fading = np.random.rayleigh(scale=1.0)
                sinr_db = 40 - path_loss + 10 * np.log10(fading) - (-100)
                self.sinr_matrix[uav_id, bs_id] = sinr_db
                if uav.connected_bs_id == bs_id:
                    uav.sinr_db = sinr_db
                    uav.update_latency_estimate(sinr_db)

    def _initialize_connections(self):
        """为所有UAV初始化到SINR最高基站的连接"""
        for uav_id, uav in self.uavs.items():
            best_bs_id = np.argmax(self.sinr_matrix[uav_id, :])
            bs = self.base_stations[best_bs_id]
            for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                if bs.allocate(uav_id, uav.required_rate * ratio):
                    uav.connected_bs_id = best_bs_id
                    uav.current_allocated_rate = uav.required_rate * ratio
                    self.connection_matrix[uav_id, best_bs_id] = 1
                    break

    # ==================== 仿真步进 ====================

    def perform_recognition(self, uav_id: int) -> Tuple[BusinessType, float]:
        """对指定UAV执行业务识别"""
        uav = self.uavs[uav_id]
        if self.recognition_model is None:
            return uav.true_business_type, 1.0
        return self.recognition_model.predict(uav.generate_features())

    def step(self):
        """执行一个仿真步"""
        self.current_step += 1
        for uav in self.uavs.values():
            uav.move(time_step=1.0)
        self._update_sinr_matrix()

        # 更新业务识别
        if self.recognition_updater is not None:
            for uav_id, uav in self.uavs.items():
                if self.recognition_updater.should_update(uav_id, self.current_step, uav.recognition_confidence):
                    recognized_type, confidence = self.perform_recognition(uav_id)
                    feedback = self.recognition_updater.record_feedback(
                        uav_id, recognized_type, uav.true_business_type, confidence, self.current_step)
                    self.feedback_buffer.append(feedback)
                    if confidence > 0.7:
                        uav.update_recognition(recognized_type, confidence)
            self.recognition_updater.detect_drift(self.feedback_buffer)

        for uav in self.uavs.values():
            uav.record_satisfaction()
        self._check_interruptions()
        self._record_stats()

    # ==================== 中断检测 ====================

    def _check_interruptions(self):
        """检测UAV中断状态（满足率低于阈值且持续N步）"""
        self.interrupted_uavs.clear()
        for uav_id, uav in self.uavs.items():
            satisfaction = uav.current_satisfaction
            is_control = uav.true_business_type == BusinessType.CONTROL_SIGNAL
            threshold = self.control_signal_threshold if is_control else self.interruption_threshold
            duration_threshold = self.control_signal_duration if is_control else self.interruption_duration

            if satisfaction < threshold:
                if uav_id not in self.active_interruptions:
                    self.active_interruptions[uav_id] = self.current_step
                    self.uav_interruption_counters[uav_id] = 1
                else:
                    self.uav_interruption_counters[uav_id] += 1
                if self.uav_interruption_counters[uav_id] == duration_threshold:
                    self.total_interruptions += 1
                    self.interruption_events.append({
                        'uav_id': uav_id, 'start_step': self.active_interruptions[uav_id],
                        'end_step': self.current_step, 'duration': duration_threshold,
                        'business_type': uav.true_business_type.name, 'satisfaction': satisfaction
                    })
                self.interrupted_uavs.add(uav_id)
            else:
                if uav_id in self.active_interruptions:
                    duration = self.current_step - self.active_interruptions[uav_id]
                    if duration >= duration_threshold:
                        for event in reversed(self.interruption_events):
                            if event['uav_id'] == uav_id and event['start_step'] == self.active_interruptions[uav_id]:
                                event['end_step'] = self.current_step
                                event['duration'] = duration
                                break
                    del self.active_interruptions[uav_id]
                    self.uav_interruption_counters[uav_id] = 0

    def _record_stats(self):
        """记录当前步的状态统计"""
        stats = self.get_state_statistics()
        hier = HierarchicalSatisfactionMetric.compute_network_metrics(self)
        self.stats_history['step'].append(self.current_step)
        self.stats_history['avg_satisfaction'].append(stats['avg_satisfaction'])
        self.stats_history['recognition_accuracy'].append(stats['recognition_accuracy'])
        self.stats_history['total_throughput'].append(stats['total_load'])
        self.stats_history['load_variance'].append(stats['load_variance'])
        self.stats_history['critical_satisfaction'].append(hier['critical_satisfaction'])
        self.stats_history['weighted_satisfaction'].append(hier['weighted_satisfaction'])
        self.stats_history['interruption_rate'].append(stats['interruption_rate'])
        self.stats_history['avg_interruption_duration'].append(stats['avg_interruption_duration'])

    # ==================== 统计接口 ====================

    def get_state_statistics(self) -> Dict:
        """获取当前网络状态统计"""
        total_load = sum(bs.current_load for bs in self.base_stations.values())
        load_ratios = [bs.load_ratio for bs in self.base_stations.values()]
        hier = HierarchicalSatisfactionMetric.compute_network_metrics(self)

        true_sats, res_ratios = [], []
        for uav in self.uavs.values():
            true_qos = QOS_PROFILES[uav.true_business_type]
            true_sats.append(true_qos.calculate_satisfaction(
                uav.current_allocated_rate,
                estimated_delay=uav.current_latency,
                loss_rate=uav.packet_loss_rate
            ))
            ideal = true_qos.ideal_rate
            res_ratios.append(uav.current_allocated_rate / ideal if ideal > 0 else 0)

        return {
            'total_load': total_load,
            'load_ratio': np.mean(load_ratios),
            'avg_satisfaction': np.mean([uav.current_satisfaction for uav in self.uavs.values()]),
            'satisfaction_rate': sum(1 for uav in self.uavs.values()
                                     if uav.current_allocated_rate >= uav.min_required_rate) / self.num_uav * 100,
            'avg_sinr': np.mean([uav.sinr_db for uav in self.uavs.values()]),
            'load_variance': np.var(load_ratios),
            'recognition_accuracy': sum(1 for uav in self.uavs.values()
                                        if uav.business_type == uav.true_business_type) / self.num_uav * 100,
            'critical_satisfaction': hier['critical_satisfaction'],
            'weighted_satisfaction': hier['weighted_satisfaction'],
            'latency_satisfaction': hier['latency_satisfaction'],
            'rate_satisfaction': hier['rate_satisfaction'],
            'handover_count': sum(uav.handover_count for uav in self.uavs.values()),
            'connected_count': sum(1 for uav in self.uavs.values() if uav.connected_bs_id is not None),
            'interruption_rate': len(self.interrupted_uavs) / max(self.num_uav, 1),
            'avg_interruption_duration': np.mean([e['duration'] for e in self.interruption_events]) if self.interruption_events else 0.0,
            'total_interruptions': self.total_interruptions,
            'active_interruptions_count': len(self.active_interruptions),
            'avg_true_satisfaction': np.mean(true_sats),
            'resource_match_ratio': np.mean(res_ratios),
        }

    def get_business_type_stats(self) -> Dict:
        """获取各业务类型的满意度统计"""
        return {bt: HierarchicalSatisfactionMetric.compute_business_type_satisfaction(self, bt)
                for bt in BusinessType}

    def reset(self):
        """重置仿真状态"""
        self.current_step = 0
        # 清理 BS 和 UAV 的旧分配状态，防止跨 episode load 泄漏
        for bs in self.base_stations.values():
            bs.connected_uavs.clear()
            bs.current_load = 0.0
        for uav in self.uavs.values():
            uav.connected_bs_id = None
            uav.current_allocated_rate = 0.0
            uav.handover_count = 0
        self.connection_matrix = np.zeros((self.num_uav, self.num_bs), dtype=int)
        self.stats_history = {k: [] for k in self.stats_history.keys()}
        self.feedback_buffer.clear()
        self.uav_interruption_counters = {uav_id: 0 for uav_id in range(self.num_uav)}
        self.interruption_events.clear()
        self.active_interruptions.clear()
        self.total_interruptions = 0
        self.interrupted_uavs.clear()
        self._update_sinr_matrix()
        self._initialize_connections()


class EnhancedNetworkEnvironment(NetworkEnvironmentWithRecognition):
    """
    增强网络环境

    在基础环境上叠加随机事件（基站故障/恢复、信道突发、UAV到达），
    用于测试算法的鲁棒性和故障恢复能力。
    """

    def __init__(self, *args, event_probability=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        # 场景事件概率: 应急场景高动态(0.08), 物流相对稳定(0.03), 农业低频(0.02)
        scenario_event_probs = {
            'emergency_rescue': 0.08,  # 灾害环境高动态
            'smart_city': 0.05,         # 城市中等波动
            'industrial_inspection': 0.05,
            'logistics_delivery': 0.03, # 物流航线相对稳定
            'agriculture': 0.02,        # 农田环境低频变化
            'default': 0.05,
        }
        self.event_probability = scenario_event_probs.get(self.scenario, event_probability)
        self.event_history = []
        self.event_stats = {'bs_failure': 0, 'channel_burst': 0, 'uav_arrival': 0, 'bs_recovery': 0}
        self.recovery_events = []
        self.active_failures = {}
        self.event_id_counter = 0
        self.recovery_stats = {
            'avg_recovery_time': 0.0, 'max_recovery_time': 0.0,
            'min_recovery_time': float('inf'), 'recovery_time_history': []
        }

    def _trigger_random_event(self):
        """以一定概率触发随机事件"""
        if np.random.rand() > self.event_probability:
            return None
        # 场景事件分布: [基站故障, 信道突发, UAV到达, 基站恢复]
        event_dist_map = {
            'emergency_rescue': [0.15, 0.35, 0.20, 0.30],  # 应急: 多恢复事件(快速抢修)
            'logistics_delivery': [0.10, 0.55, 0.25, 0.10], # 物流: 多信道突发(长航线)
            'agriculture': [0.10, 0.50, 0.30, 0.10],        # 农业: 多UAV到达(大批传感器)
            'smart_city': [0.12, 0.53, 0.25, 0.10],
            'industrial_inspection': [0.15, 0.50, 0.20, 0.15],
            'urban': [0.15, 0.6, 0.15, 0.1],
            'emergency': [0.1, 0.5, 0.3, 0.1],
        }
        event_probs = event_dist_map.get(self.scenario, [0.1, 0.7, 0.15, 0.05])
        event_type = np.random.choice(['bs_failure', 'channel_burst', 'uav_arrival', 'bs_recovery'], p=event_probs)
        return self._execute_event(event_type)

    def _execute_event(self, event_type: str):
        """执行指定类型的随机事件"""
        if event_type == 'bs_failure':
            return self._event_bs_failure()
        elif event_type == 'bs_recovery':
            return self._event_bs_recovery()
        elif event_type == 'channel_burst':
            return self._event_channel_burst()
        else:
            return self._event_uav_arrival()

    def _event_bs_failure(self):
        """基站故障事件：随机使一个基站故障，所有连接的UAV断连"""
        bs_id = np.random.choice(list(self.base_stations.keys()))
        bs = self.base_stations[bs_id]
        if not bs.failure_state:
            bs.set_failure(True)
            self.event_id_counter += 1
            event_id = f"failure_{self.event_id_counter}"
            affected = [uav_id for uav_id, uav in self.uavs.items() if uav.connected_bs_id == bs_id]
            self.event_stats['bs_failure'] += 1
            event = {'type': 'bs_failure', 'bs_id': bs_id, 'step': self.current_step,
                     'event_id': event_id, 'affected_uavs': affected}
            self.event_history.append(event)
            self.active_failures[event_id] = {
                'type': 'bs_failure', 'step': self.current_step,
                'affected_uavs': affected, 'bs_id': bs_id
            }
            return event
        return None

    def _event_bs_recovery(self):
        """基站恢复事件：随机恢复一个故障基站"""
        failed_bs = [bs for bs in self.base_stations.values() if bs.failure_state]
        if not failed_bs:
            return None
        bs = np.random.choice(failed_bs)
        bs.set_failure(False)
        event_id = None
        for eid, info in self.active_failures.items():
            if info['type'] == 'bs_failure' and info['bs_id'] == bs.bs_id:
                event_id = eid
                break
        if event_id:
            recovery_duration = self.current_step - self.active_failures[event_id]['step']
            self.recovery_events.append({
                'event_id': event_id, 'event_type': 'bs_recovery',
                'start_step': self.active_failures[event_id]['step'],
                'end_step': self.current_step, 'recovery_duration': recovery_duration,
                'affected_uavs': self.active_failures[event_id]['affected_uavs']
            })
            self.recovery_stats['recovery_time_history'].append(recovery_duration)
            del self.active_failures[event_id]
        self.event_stats['bs_recovery'] += 1
        event = {'type': 'bs_recovery', 'bs_id': bs.bs_id, 'step': self.current_step, 'event_id': event_id}
        self.event_history.append(event)
        return event

    def _event_channel_burst(self):
        """信道突发衰落事件：随机选择一个UAV，临时降低其所有SINR"""
        uav_id = np.random.choice(list(self.uavs.keys()))
        sinr_drop = np.random.uniform(5, 15)
        self.sinr_matrix[uav_id, :] -= sinr_drop
        self.event_stats['channel_burst'] += 1
        event = {'type': 'channel_burst', 'uav_id': uav_id, 'sinr_drop': sinr_drop, 'step': self.current_step}
        self.event_history.append(event)
        return event

    def _event_uav_arrival(self):
        """新UAV到达事件：添加一个新的随机业务类型UAV"""
        new_id = max(self.uavs.keys()) + 1 if self.uavs else 0
        biz_type = np.random.choice(list(BusinessType))
        self.uavs[new_id] = UAV(new_id, business_type=biz_type)
        self.num_uav += 1
        self.connection_matrix = np.vstack([self.connection_matrix, np.zeros(self.num_bs)])
        self.sinr_matrix = np.vstack([self.sinr_matrix, np.zeros(self.num_bs)])
        self.event_stats['uav_arrival'] += 1
        event = {'type': 'uav_arrival', 'uav_id': new_id, 'business_type': biz_type.name, 'step': self.current_step}
        self.event_history.append(event)
        return event

    def step(self):
        """执行一个仿真步（包含随机事件触发）"""
        super().step()
        self._trigger_random_event()
