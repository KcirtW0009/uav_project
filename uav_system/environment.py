import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
from .config import GLOBAL_SEED
from .business import BusinessType
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
            'weighted_satisfaction': []
        }

    def _init_base_stations(self, scenario: str):
        if scenario == 'urban':
            for i in range(self.num_bs):
                capacity = np.random.uniform(300, 500)
                bs_type = 'small' if np.random.rand() < 0.4 else 'macro'
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 800,
                    bs_type=bs_type
                )
        elif scenario == 'emergency':
            for i in range(self.num_bs):
                capacity = np.random.uniform(500, 800)
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 1200,
                    bs_type='macro'
                )
        elif scenario == 'agriculture':
            for i in range(self.num_bs):
                capacity = np.random.uniform(400, 600)
                self.base_stations[i] = BaseStation(
                    i, capacity=capacity,
                    position=np.random.rand(3) * 1500,
                    bs_type='macro'
                )
        else:
            for i in range(self.num_bs):
                capacity = np.random.uniform(400, 600)
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
            'connected_count': sum(1 for uav in self.uavs.values() if uav.connected_bs_id is not None)
        }

    def get_business_type_stats(self) -> Dict:
        stats = {}
        for bt in BusinessType:
            stats[bt] = HierarchicalSatisfactionMetric.compute_business_type_satisfaction(self, bt)
        return stats

    def reset(self):
        self.current_step = 0
        self.stats_history = {k: [] for k in self.stats_history.keys()}
        self.feedback_buffer.clear()
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
                event = {'type': 'bs_failure', 'bs_id': bs_id, 'old_capacity': old_capacity, 'step': self.current_step}
                self.event_stats['bs_failure'] += 1
                self.event_history.append(event)
                return event
        elif event_type == 'bs_recovery':
            failed_bs = [bs for bs in self.base_stations.values() if bs.failure_state]
            if failed_bs:
                bs = np.random.choice(failed_bs)
                bs.set_failure(False)
                event = {'type': 'bs_recovery', 'bs_id': bs.bs_id, 'step': self.current_step}
                self.event_stats['bs_recovery'] += 1
                self.event_history.append(event)
                return event
        elif event_type == 'channel_burst':
            uav_id = np.random.choice(list(self.uavs.keys()))
            sinr_drop = np.random.uniform(5, 15)
            self.sinr_matrix[uav_id, :] -= sinr_drop
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