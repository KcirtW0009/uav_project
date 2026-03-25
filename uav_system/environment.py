import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
from .config import GLOBAL_SEED, INTERRUPTION_CONFIG
from .business import BusinessType, QOS_PROFILES
from .entities import BaseStation, UAV
from .satisfaction import HierarchicalSatisfactionMetric
from .recognition import AdaptiveRecognitionUpdater

class NetworkEnvironmentWithRecognition:
    def __init__(self, num_bs=8, num_uav=50, recognition_model=None, scaler=None,
                 seed=GLOBAL_SEED, scenario='default'):
        np.random.seed(seed)
        self.num_bs = num_bs
        self.num_uav = num_uav
        self.recognition_model = recognition_model
        self.scaler = scaler
        self.scenario = scenario
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

        self.stats_history = {
            'step': [],
            'avg_satisfaction': [],
            'recognition_accuracy': [],
            'total_throughput': [],
            'load_variance': [],
            'critical_satisfaction': [],
            'weighted_satisfaction': [],
            'interruption_rate': [],
            'avg_interruption_duration': []
        }

        # 中断率统计相关
        # 定义: 当UAV满足率低于阈值且持续N步时计为一次中断
        self.interruption_threshold = INTERRUPTION_CONFIG['threshold']  # 满足率阈值
        self.interruption_duration = INTERRUPTION_CONFIG['duration']      # 持续步数
        self.control_signal_threshold = INTERRUPTION_CONFIG['control_signal_threshold']  # 控制信令的中断阈值
        self.control_signal_duration = INTERRUPTION_CONFIG['control_signal_duration']      # 控制信令的持续步数
        self.uav_interruption_counters = {uav_id: 0 for uav_id in range(num_uav)}
        self.interruption_events = []       # 记录中断事件: {'uav_id', 'start_step', 'end_step', 'duration'}
        self.active_interruptions = {}      # 当前活跃的中断: {uav_id: start_step}
        self.total_interruptions = 0
        self.interrupted_uavs = set()       # 当前步处于中断状态的UAV集合

    def _init_base_stations(self, scenario: str):
        if scenario == 'urban':
            for i in range(self.num_bs):
                capacity = np.random.uniform(1500, 2500)  # 方案D: 平衡容量与竞争
                bs_type = 'small' if np.random.rand() < 0.4 else 'macro'
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 800,
                    bs_type=bs_type
                )
        elif scenario == 'emergency':
            for i in range(self.num_bs):
                capacity = np.random.uniform(2000, 3000)  # 方案D: 应急场景高容量
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 1200,
                    bs_type='macro'
                )
        elif scenario == 'agriculture':
            for i in range(self.num_bs):
                capacity = np.random.uniform(1200, 1800)  # 方案D: 农业场景低容量
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 1500,
                    bs_type='macro'
                )
        else:
            for i in range(self.num_bs):
                capacity = np.random.uniform(1500, 2500)  # 方案D: 默认场景
                self.base_stations[i] = BaseStation(i, capacity=capacity)

    def _init_uavs(self, scenario: str):
        if scenario == 'emergency':
            ratios = [0.3, 0.5, 0.2]
        elif scenario == 'agriculture':
            ratios = [0.2, 0.3, 0.5]
        else:
            ratios = [0.4, 0.3, 0.3]
        business_types = [BusinessType.CONTROL_SIGNAL,
                          BusinessType.VIDEO_STREAMING,
                          BusinessType.ENVIRONMENT_MONITORING]
        for i in range(self.num_uav):
            rand = np.random.rand()
            if rand < ratios[0]:
                biz_type = BusinessType.CONTROL_SIGNAL
            elif rand < ratios[0] + ratios[1]:
                biz_type = BusinessType.VIDEO_STREAMING
            else:
                biz_type = BusinessType.ENVIRONMENT_MONITORING
            if scenario == 'urban':
                velocity = (np.random.rand(3) - 0.5) * 15
            elif scenario == 'emergency':
                velocity = (np.random.rand(3) - 0.5) * 30
            else:
                velocity = (np.random.rand(3) - 0.5) * 20
            self.uavs[i] = UAV(i, business_type=biz_type, velocity=velocity)

    def _update_sinr_matrix(self):
        for uav_id, uav in self.uavs.items():
            for bs_id, bs in self.base_stations.items():
                distance = np.linalg.norm(uav.position - bs.position)
                if self.scenario == 'urban':
                    path_loss = 140 + 38 * np.log10(max(distance/1000, 0.001))
                elif self.scenario == 'emergency':
                    path_loss = 128.1 + 37.6 * np.log10(max(distance/1000, 0.001))
                else:
                    path_loss = 128.1 + 37.6 * np.log10(max(distance/1000, 0.001))
                fading = np.random.rayleigh(scale=1.0)
                tx_power = 40
                noise_power = -100
                sinr_db = tx_power - path_loss + 10*np.log10(fading) - noise_power
                self.sinr_matrix[uav_id, bs_id] = sinr_db
                if uav.connected_bs_id == bs_id:
                    uav.sinr_db = sinr_db
                    uav.update_latency_estimate(sinr_db)

    def _initialize_connections(self):
        for uav_id, uav in self.uavs.items():
            best_bs_id = np.argmax(self.sinr_matrix[uav_id, :])
            bs = self.base_stations[best_bs_id]
            for ratio in uav.qos_profile.get_feasible_downgrade_ratios():
                required = uav.required_rate * ratio
                if bs.allocate(uav_id, required):
                    uav.connected_bs_id = best_bs_id
                    uav.current_allocated_rate = required
                    self.connection_matrix[uav_id, best_bs_id] = 1
                    break

    def perform_recognition(self, uav_id: int) -> Tuple[BusinessType, float]:
        uav = self.uavs[uav_id]
        features = uav.generate_features()
        if self.recognition_model is None:
            return uav.true_business_type, 1.0
        recognized_type, confidence = self.recognition_model.predict(features)
        return recognized_type, confidence

    def _check_interruptions(self):
        """
        检测UAV是否处于中断状态

        中断定义: UAV满足率低于阈值且持续N步
        - 控制信令业务: 阈值0.4, 持续3步 (更严格)
        - 其他业务: 阈值0.3, 持续5步 (通用)
        """
        self.interrupted_uavs.clear()

        for uav_id, uav in self.uavs.items():
            satisfaction = uav.current_satisfaction
            business_type = uav.true_business_type

            # 根据业务类型选择不同的中断阈值
            if business_type == BusinessType.CONTROL_SIGNAL:
                threshold = self.control_signal_threshold
                duration_threshold = self.control_signal_duration
            else:
                threshold = self.interruption_threshold
                duration_threshold = self.interruption_duration

            # 检查满足率是否低于阈值
            if satisfaction < threshold:
                if uav_id not in self.active_interruptions:
                    # 开始新的中断
                    self.active_interruptions[uav_id] = self.current_step
                    self.uav_interruption_counters[uav_id] = 1
                else:
                    # 持续中断,增加计数
                    self.uav_interruption_counters[uav_id] += 1

                # 如果持续步数达到阈值,计为一次中断事件
                if self.uav_interruption_counters[uav_id] == duration_threshold:
                    self.total_interruptions += 1
                    event = {
                        'uav_id': uav_id,
                        'start_step': self.active_interruptions[uav_id],
                        'end_step': self.current_step,
                        'duration': duration_threshold,
                        'business_type': business_type.name,
                        'satisfaction': satisfaction,
                        'threshold': threshold
                    }
                    self.interruption_events.append(event)

                self.interrupted_uavs.add(uav_id)
            else:
                # 满足率恢复,记录中断结束
                if uav_id in self.active_interruptions:
                    start_step = self.active_interruptions[uav_id]
                    duration = self.current_step - start_step

                    # 根据业务类型获取duration_threshold
                    if business_type == BusinessType.CONTROL_SIGNAL:
                        duration_threshold = self.control_signal_duration
                    else:
                        duration_threshold = self.interruption_duration

                    if duration >= duration_threshold:
                        # 更新已记录的中断事件
                        for event in reversed(self.interruption_events):
                            if (event['uav_id'] == uav_id and
                                event['start_step'] == start_step and
                                event['end_step'] == start_step + duration_threshold - 1):
                                event['end_step'] = self.current_step
                                event['duration'] = duration
                                break

                    del self.active_interruptions[uav_id]
                    self.uav_interruption_counters[uav_id] = 0

    def step(self):
        self.current_step += 1
        for uav in self.uavs.values():
            uav.move(time_step=1.0)
        self._update_sinr_matrix()
        if self.recognition_updater is not None:
            for uav_id, uav in self.uavs.items():
                if self.recognition_updater.should_update(uav_id, self.current_step,
                                                          uav.recognition_confidence):
                    recognized_type, confidence = self.perform_recognition(uav_id)
                    feedback = self.recognition_updater.record_feedback(
                        uav_id, recognized_type, uav.true_business_type,
                        confidence, self.current_step)
                    self.feedback_buffer.append(feedback)
                    if confidence > 0.7:
                        uav.update_recognition(recognized_type, confidence)
            self.recognition_updater.detect_drift(self.feedback_buffer)
        for uav in self.uavs.values():
            uav.record_satisfaction()
        self._check_interruptions()  # 检测中断
        self._record_stats()

    def _record_stats(self):
        stats = self.get_state_statistics()
        self.stats_history['step'].append(self.current_step)
        self.stats_history['avg_satisfaction'].append(stats['avg_satisfaction'])
        self.stats_history['recognition_accuracy'].append(stats['recognition_accuracy'])
        self.stats_history['total_throughput'].append(stats['total_load'])
        self.stats_history['load_variance'].append(stats['load_variance'])
        hier_metrics = HierarchicalSatisfactionMetric.compute_network_metrics(self)
        self.stats_history['critical_satisfaction'].append(hier_metrics['critical_satisfaction'])
        self.stats_history['weighted_satisfaction'].append(hier_metrics['weighted_satisfaction'])
        # 记录中断率
        self.stats_history['interruption_rate'].append(stats['interruption_rate'])
        self.stats_history['avg_interruption_duration'].append(stats['avg_interruption_duration'])

    def get_state_statistics(self) -> Dict:
        total_load = sum(bs.current_load for bs in self.base_stations.values())
        avg_load_ratio = np.mean([bs.load_ratio for bs in self.base_stations.values()])
        satisfactions = [uav.current_satisfaction for uav in self.uavs.values()]
        avg_satisfaction = np.mean(satisfactions)
        satisfied_count = sum(1 for uav in self.uavs.values()
                              if uav.current_allocated_rate >= uav.min_required_rate)
        satisfaction_rate = satisfied_count / self.num_uav * 100
        avg_sinr = np.mean([uav.sinr_db for uav in self.uavs.values()])
        load_ratios = [bs.load_ratio for bs in self.base_stations.values()]
        load_variance = np.var(load_ratios)
        correct_recognition = sum(1 for uav in self.uavs.values()
                                  if uav.business_type == uav.true_business_type)
        recognition_accuracy = correct_recognition / self.num_uav * 100
        hier_metrics = HierarchicalSatisfactionMetric.compute_network_metrics(self)

        # 新增: 真实业务类型满意率(基于真实需求的满意率)
        true_satisfactions = []
        for uav in self.uavs.values():
            true_qos = QOS_PROFILES[uav.true_business_type]
            true_sat = true_qos.calculate_satisfaction(uav.current_allocated_rate)
            true_satisfactions.append(true_sat)
        avg_true_satisfaction = np.mean(true_satisfactions)

        # 新增: 资源匹配度(分配资源与真实理想需求的比例)
        resource_match_ratios = []
        for uav in self.uavs.values():
            true_ideal = QOS_PROFILES[uav.true_business_type].ideal_rate
            ratio = uav.current_allocated_rate / true_ideal if true_ideal > 0 else 0
            resource_match_ratios.append(ratio)
        avg_resource_match = np.mean(resource_match_ratios)

        # 中断率统计
        interrupted_count = len(self.interrupted_uavs)
        interruption_rate = interrupted_count / max(self.num_uav, 1)  # 当前处于中断状态的UAV比例
        avg_interruption_duration = 0.0
        if self.interruption_events:
            avg_interruption_duration = np.mean([e['duration'] for e in self.interruption_events])

        return {
            'total_load': total_load,
            'load_ratio': avg_load_ratio,
            'avg_satisfaction': avg_satisfaction,
            'satisfaction_rate': satisfaction_rate,
            'avg_sinr': avg_sinr,
            'load_variance': load_variance,
            'recognition_accuracy': recognition_accuracy,
            'critical_satisfaction': hier_metrics['critical_satisfaction'],
            'weighted_satisfaction': hier_metrics['weighted_satisfaction'],
            'latency_satisfaction': hier_metrics['latency_satisfaction'],
            'rate_satisfaction': hier_metrics['rate_satisfaction'],
            'handover_count': sum(uav.handover_count for uav in self.uavs.values()),
            'connected_count': sum(1 for uav in self.uavs.values() if uav.connected_bs_id is not None),
            'interruption_rate': interruption_rate,
            'avg_interruption_duration': avg_interruption_duration,
            'total_interruptions': self.total_interruptions,
            'active_interruptions_count': len(self.active_interruptions),
            'avg_true_satisfaction': avg_true_satisfaction,  # 新增
            'resource_match_ratio': avg_resource_match,       # 新增
        }

    def get_business_type_stats(self) -> Dict:
        stats = {}
        for bt in BusinessType:
            stats[bt] = HierarchicalSatisfactionMetric.compute_business_type_satisfaction(self, bt)
        return stats

    def get_interruption_statistics(self) -> Dict:
        """
        获取详细的中断统计信息

        Returns:
            Dict: 包含中断率、平均持续时间、按业务类型分类的中断统计等
        """
        if not self.interruption_events:
            return {
                'total_interruptions': 0,
                'interruption_rate': 0.0,
                'avg_interruption_duration': 0.0,
                'max_interruption_duration': 0.0,
                'min_interruption_duration': 0.0,
                'by_business_type': {},
                'interrupted_uavs_ratio': 0.0
            }

        durations = [e['duration'] for e in self.interruption_events]

        # 按业务类型统计中断次数
        by_business_type = {}
        for event in self.interruption_events:
            bt = event['business_type']
            if bt not in by_business_type:
                by_business_type[bt] = {'count': 0, 'avg_duration': 0.0}
            by_business_type[bt]['count'] += 1

        # 计算各业务类型的平均中断持续时间
        for bt in by_business_type:
            bt_events = [e for e in self.interruption_events if e['business_type'] == bt]
            by_business_type[bt]['avg_duration'] = np.mean([e['duration'] for e in bt_events])

        return {
            'total_interruptions': self.total_interruptions,
            'interruption_rate': len(self.interrupted_uavs) / max(self.num_uav, 1),
            'avg_interruption_duration': np.mean(durations),
            'max_interruption_duration': max(durations),
            'min_interruption_duration': min(durations),
            'by_business_type': by_business_type,
            'interrupted_uavs_ratio': len(self.interrupted_uavs) / max(self.num_uav, 1),
            'active_interruptions': len(self.active_interruptions)
        }

    def get_recovery_statistics(self) -> Dict:
        """
        获取故障恢复时间统计信息

        Returns:
            Dict: 包含平均恢复时间、最大/最小恢复时间、恢复事件详情等
        """
        if not self.recovery_events:
            return {
                'total_recoveries': 0,
                'avg_recovery_time': 0.0,
                'max_recovery_time': 0.0,
                'min_recovery_time': 0.0,
                'std_recovery_time': 0.0,
                'recovery_events': [],
                'active_failures': len(self.active_failures)
            }

        recovery_times = [e['recovery_duration'] for e in self.recovery_events]
        self.recovery_stats['avg_recovery_time'] = np.mean(recovery_times)
        self.recovery_stats['max_recovery_time'] = max(recovery_times)
        self.recovery_stats['min_recovery_time'] = min(recovery_times)

        return {
            'total_recoveries': len(self.recovery_events),
            'avg_recovery_time': np.mean(recovery_times),
            'max_recovery_time': max(recovery_times),
            'min_recovery_time': min(recovery_times),
            'std_recovery_time': np.std(recovery_times),
            'recovery_events': self.recovery_events,
            'active_failures': len(self.active_failures)
        }

    def check_target_metrics(self, targets: Dict[str, float]) -> Dict:
        """
        检查目标指标是否达标

        Args:
            targets: 目标指标字典,例如 {'satisfaction_rate': 0.9, 'handover_success_rate': 0.95}

        Returns:
            Dict: 包含每个指标的达标状态、实际值和是否达标
        """
        stats = self.get_state_statistics()
        results = {}

        # 定义指标名称到stats字段的映射
        metric_mapping = {
            'satisfaction_rate': 'avg_satisfaction',
            'handover_success_rate': 'handover_success_rate',
            'recognition_accuracy': 'recognition_accuracy',
            'throughput': 'total_load',
            'interruption_rate': 'interruption_rate'
        }

        for metric_name, target_value in targets.items():
            if metric_name in metric_mapping:
                field_name = metric_mapping[metric_name]
                actual_value = stats.get(field_name, 0)

                # 对于中断率,越低越好;对于其他指标,越高越好
                if metric_name == 'interruption_rate':
                    achieved = actual_value <= target_value
                else:
                    achieved = actual_value >= target_value

                results[metric_name] = {
                    'target': target_value,
                    'actual': actual_value,
                    'achieved': achieved,
                    'difference': actual_value - target_value,
                    'relative_difference': (actual_value - target_value) / target_value * 100 if target_value != 0 else 0
                }
            else:
                results[metric_name] = {
                    'target': target_value,
                    'actual': None,
                    'achieved': None,
                    'difference': None,
                    'relative_difference': None,
                    'error': f'Unknown metric: {metric_name}'
                }

        # 计算整体达标情况
        total_metrics = len([r for r in results.values() if r.get('achieved') is not None])
        achieved_metrics = len([r for r in results.values() if r.get('achieved') is True])
        overall_achievement = achieved_metrics / total_metrics if total_metrics > 0 else 0

        results['summary'] = {
            'total_metrics': total_metrics,
            'achieved_metrics': achieved_metrics,
            'achievement_rate': overall_achievement,
            'all_achieved': total_metrics > 0 and achieved_metrics == total_metrics
        }

        return results

    def reset(self):
        self.current_step = 0
        self.stats_history = {k: [] for k in self.stats_history.keys()}
        self.feedback_buffer.clear()
        # 重置中断统计
        self.uav_interruption_counters = {uav_id: 0 for uav_id in range(self.num_uav)}
        self.interruption_events.clear()
        self.active_interruptions.clear()
        self.total_interruptions = 0
        self.interrupted_uavs.clear()
        # 重置故障恢复统计
        self.recovery_events.clear()
        self.active_failures.clear()
        self.event_id_counter = 0
        self.recovery_stats = {
            'avg_recovery_time': 0.0,
            'max_recovery_time': 0.0,
            'min_recovery_time': float('inf'),
            'recovery_time_history': []
        }
        # 重置事件历史
        self.event_history = []
        for key in self.event_stats:
            self.event_stats[key] = 0
        self._update_sinr_matrix()
        self._initialize_connections()


class EnhancedNetworkEnvironment(NetworkEnvironmentWithRecognition):
    def __init__(self, *args, event_probability=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_probability = event_probability
        self.event_history = []
        self.event_stats = {
            'bs_failure': 0,
            'channel_burst': 0,
            'uav_arrival': 0,
            'bs_recovery': 0
        }

        # 故障恢复时间统计
        self.recovery_events = []  # 记录恢复事件: {'event_id', 'event_type', 'start_step', 'end_step', 'recovery_duration', 'affected_uavs'}
        self.active_failures = {}  # 当前活跃的故障: {event_id: {'type', 'step', 'affected_uavs'}}
        self.event_id_counter = 0
        self.recovery_stats = {
            'avg_recovery_time': 0.0,
            'max_recovery_time': 0.0,
            'min_recovery_time': float('inf'),
            'recovery_time_history': []
        }

    def _trigger_random_event(self):
        if np.random.rand() > self.event_probability:
            return None
        if self.scenario == 'urban':
            event_probs = [0.15, 0.6, 0.15, 0.1]
        elif self.scenario == 'emergency':
            event_probs = [0.1, 0.5, 0.3, 0.1]
        else:
            event_probs = [0.1, 0.7, 0.15, 0.05]
        event_type = np.random.choice(['bs_failure', 'channel_burst', 'uav_arrival', 'bs_recovery'],
                                      p=event_probs)
        if event_type == 'bs_failure':
            bs_id = np.random.choice(list(self.base_stations.keys()))
            bs = self.base_stations[bs_id]
            if not bs.failure_state:
                old_capacity = bs.capacity
                bs.set_failure(True)
                self.event_id_counter += 1
                event_id = f"failure_{self.event_id_counter}"
                # 记录受影响的UAV
                affected_uavs = [uav_id for uav_id, uav in self.uavs.items() if uav.connected_bs_id == bs_id]
                event = {'type': 'bs_failure', 'bs_id': bs_id, 'old_capacity': old_capacity, 'step': self.current_step,
                         'event_id': event_id, 'affected_uavs': affected_uavs}
                self.event_stats['bs_failure'] += 1
                self.event_history.append(event)
                # 记录活跃故障
                self.active_failures[event_id] = {
                    'type': 'bs_failure',
                    'step': self.current_step,
                    'affected_uavs': affected_uavs,
                    'bs_id': bs_id
                }
                return event
        elif event_type == 'bs_recovery':
            failed_bs = [bs for bs in self.base_stations.values() if bs.failure_state]
            if failed_bs:
                bs = np.random.choice(failed_bs)
                bs.set_failure(False)
                # 查找对应的故障事件ID
                event_id = None
                for eid, info in self.active_failures.items():
                    if info['type'] == 'bs_failure' and info['bs_id'] == bs.bs_id:
                        event_id = eid
                        break
                if event_id:
                    recovery_duration = self.current_step - self.active_failures[event_id]['step']
                    recovery_event = {
                        'event_id': event_id,
                        'event_type': 'bs_recovery',
                        'start_step': self.active_failures[event_id]['step'],
                        'end_step': self.current_step,
                        'recovery_duration': recovery_duration,
                        'affected_uavs': self.active_failures[event_id]['affected_uavs']
                    }
                    self.recovery_events.append(recovery_event)
                    self.recovery_stats['recovery_time_history'].append(recovery_duration)
                    del self.active_failures[event_id]
                event = {'type': 'bs_recovery', 'bs_id': bs.bs_id, 'step': self.current_step, 'event_id': event_id}
                self.event_stats['bs_recovery'] += 1
                self.event_history.append(event)
                return event
        elif event_type == 'channel_burst':
            uav_id = np.random.choice(list(self.uavs.keys()))
            sinr_drop = np.random.uniform(5, 15)
            self.sinr_matrix[uav_id, :] -= sinr_drop
            self.event_id_counter += 1
            event_id = f"channel_{self.event_id_counter}"
            # 信道突发故障会在下一个step自动恢复(这里简化处理,不作为故障恢复统计)
            event = {'type': 'channel_burst', 'uav_id': uav_id, 'sinr_drop': sinr_drop, 'step': self.current_step}
            self.event_stats['channel_burst'] += 1
            self.event_history.append(event)
            return event
        else:  # uav_arrival
            new_id = max(self.uavs.keys()) + 1 if self.uavs else 0
            biz_type = np.random.choice(list(BusinessType))
            self.uavs[new_id] = UAV(new_id, business_type=biz_type)
            self.num_uav += 1
            self.connection_matrix = np.vstack([self.connection_matrix, np.zeros(self.num_bs)])
            self.sinr_matrix = np.vstack([self.sinr_matrix, np.zeros(self.num_bs)])
            event = {'type': 'uav_arrival', 'uav_id': new_id, 'business_type': biz_type.name, 'step': self.current_step}
            self.event_stats['uav_arrival'] += 1
            self.event_history.append(event)
            return event

    def step(self):
        super().step()
        self._trigger_random_event()