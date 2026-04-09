# -*- coding: utf-8 -*-
"""
通信指标采集模块

根据 exp3_data.json 定义的评估标准，实现所有通信指标的采集与计算
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class HandoverDecision:
    """切换决策记录"""
    uav_id: int
    timestamp: float
    action: int  # 0=stay, 1+=switch to BS
    last_bs: int
    target_bs: int
    success: bool = True
    latency_ms: float = 0.0
    decision_time_ms: float = 0.0
    missed_opportunity: bool = False
    sinr_before: float = 0.0
    sinr_after: float = 0.0


@dataclass
class CommunicationMetrics:
    """通信指标数据类"""
    # 切换相关
    handover_success_rate: float = 0.0
    avg_switching_latency_ms: float = 0.0
    max_switching_latency_ms: float = 0.0
    avg_decision_time_ms: float = 0.0
    missed_opportunity_rate: float = 0.0
    
    # 满意度相关
    avg_satisfaction: float = 0.0
    critical_satisfaction: float = 0.0
    weighted_satisfaction: float = 0.0
    latency_satisfaction: float = 0.0
    rate_satisfaction: float = 0.0
    
    # 网络质量
    load_variance: float = 0.0
    avg_sinr: float = 0.0
    
    # 其他
    recognition_accuracy: float = 0.0
    migration_success_rate: float = 0.0
    connected_ratio: float = 0.0
    
    # 原始数据存储
    raw_data: Dict = field(default_factory=dict)


class CommunicationMetricsCollector:
    """
    通信指标采集器
    
    采集所有 exp3_data.json 中定义的指标
    """
    
    def __init__(self):
        """初始化采集器"""
        self.decisions: List[HandoverDecision] = []
        self.episode_metrics: List[Dict[str, float]] = []
        
        # 按episode存储
        self.current_episode_decisions: List[HandoverDecision] = []
        self.current_episode_stats: Dict[str, List[float]] = defaultdict(list)
        
    def start_episode(self):
        """开始新的episode记录"""
        self.current_episode_decisions = []
        self.current_episode_stats = defaultdict(list)
    
    def record_decision(self, 
                       uav_id: int,
                       action: int,
                       last_bs: int,
                       target_bs: int,
                       decision_time_ms: float,
                       uav=None,
                       env=None):
        """
        记录一个切换决策
        
        Args:
            uav_id: UAV ID
            action: 执行的动作
            last_bs: 上一基站
            target_bs: 目标基站
            decision_time_ms: 决策时间(ms)
            uav: UAV对象（可选）
            env: 环境对象（可选）
        """
        decision = HandoverDecision(
            uav_id=uav_id,
            timestamp=time.time(),
            action=action,
            last_bs=last_bs,
            target_bs=target_bs,
            decision_time_ms=decision_time_ms
        )
        
        # 获取SINR信息
        if uav is not None:
            decision.sinr_before = getattr(uav, 'sinr_before_handover', 0)
            decision.sinr_after = getattr(uav, 'current_sinr', 0)
        
        self.current_episode_decisions.append(decision)
    
    def record_step_stats(self, 
                         env,
                         info: Dict[str, Any],
                         recognition_accuracy: float = 0.0):
        """
        记录一个step的统计信息
        
        Args:
            env: 环境对象
            info: 环境返回的info
            recognition_accuracy: 业务识别准确率
        """
        if not hasattr(env, 'uavs'):
            return
        
        # 满意度统计
        satisfactions = []
        latency_sats = []
        rate_sats = []
        sinrs = []
        
        for uav in env.uavs.values():
            # 基础满意度
            sat = getattr(uav, 'current_satisfaction', 0.5)
            satisfactions.append(sat)
            
            # SINR
            sinr = getattr(uav, 'current_sinr', 0)
            sinrs.append(sinr)
            
            # 延迟满意度
            latency = getattr(uav, 'current_latency', 0)
            latency_sat = max(0, 1 - latency / 100)
            latency_sats.append(latency_sat)
            
            # 速率满意度
            throughput = getattr(uav, 'current_throughput', 0)
            required = getattr(uav, 'required_rate', 1)
            rate_sat = min(throughput / required, 1.0)
            rate_sats.append(rate_sat)
        
        # 存储统计
        if satisfactions:
            self.current_episode_stats['satisfaction'].append(np.mean(satisfactions))
            self.current_episode_stats['latency_satisfaction'].append(np.mean(latency_sats))
            self.current_episode_stats['rate_satisfaction'].append(np.mean(rate_sats))
            self.current_episode_stats['sinr'].append(np.mean(sinrs))
        
        # 连接率
        connected = sum(1 for uav in env.uavs.values() if getattr(uav, 'is_connected', True))
        connected_ratio = connected / len(env.uavs)
        self.current_episode_stats['connected_ratio'].append(connected_ratio)
        
        # 负载方差
        if hasattr(env, 'base_stations'):
            loads = [bs.current_load / bs.capacity for bs in env.base_stations.values()]
            self.current_episode_stats['load_variance'].append(np.var(loads))
        
        # 识别准确率
        self.current_episode_stats['recognition_accuracy'].append(recognition_accuracy)
    
    def end_episode(self, 
                   switch_results: Optional[Dict[int, Dict]] = None,
                   latencies: Optional[Dict[int, float]] = None):
        """
        结束当前episode并计算指标
        
        Args:
            switch_results: 切换结果字典 {uav_id: {'success': bool, 'latency': float}}
            latencies: 切换延迟字典
        """
        # 更新决策结果
        if switch_results:
            for decision in self.current_episode_decisions:
                if decision.uav_id in switch_results:
                    result = switch_results[decision.uav_id]
                    decision.success = result.get('success', True)
                    decision.latency_ms = result.get('latency', 0)
        
        # 计算episode指标
        metrics = self._compute_episode_metrics()
        self.episode_metrics.append(metrics)
        
        # 保存决策
        self.decisions.extend(self.current_episode_decisions)
    
    def _compute_episode_metrics(self) -> Dict[str, float]:
        """计算单个episode的指标"""
        metrics = {}
        
        # 切换相关指标
        handover_decisions = [d for d in self.current_episode_decisions if d.action != 0]
        
        if handover_decisions:
            # 切换成功率
            successful = sum(1 for d in handover_decisions if d.success)
            metrics['handover_success_rate'] = successful / len(handover_decisions)
            
            # 切换延迟
            latencies = [d.latency_ms for d in handover_decisions]
            metrics['avg_switching_latency_ms'] = np.mean(latencies)
            metrics['max_switching_latency_ms'] = np.max(latencies)
        else:
            metrics['handover_success_rate'] = 1.0
            metrics['avg_switching_latency_ms'] = 0.0
            metrics['max_switching_latency_ms'] = 0.0
        
        # 决策时间
        if self.current_episode_decisions:
            decision_times = [d.decision_time_ms for d in self.current_episode_decisions]
            metrics['avg_decision_time_ms'] = np.mean(decision_times)
        
        # 错失机会率
        if handover_decisions:
            missed = sum(1 for d in handover_decisions if d.missed_opportunity)
            metrics['missed_opportunity_rate'] = missed / len(handover_decisions)
        else:
            metrics['missed_opportunity_rate'] = 0.0
        
        # 满意度指标
        if self.current_episode_stats['satisfaction']:
            metrics['avg_satisfaction'] = np.mean(self.current_episode_stats['satisfaction'])
            metrics['latency_satisfaction'] = np.mean(self.current_episode_stats['latency_satisfaction'])
            metrics['rate_satisfaction'] = np.mean(self.current_episode_stats['rate_satisfaction'])
        
        # 关键业务满意度（假设0和1是关键业务）
        # 这里需要根据实际业务类型过滤
        
        # 网络质量
        if self.current_episode_stats['sinr']:
            metrics['avg_sinr'] = np.mean(self.current_episode_stats['sinr'])
        
        if self.current_episode_stats['load_variance']:
            metrics['load_variance'] = np.mean(self.current_episode_stats['load_variance'])
        
        # 连接率
        if self.current_episode_stats['connected_ratio']:
            metrics['connected_ratio'] = np.mean(self.current_episode_stats['connected_ratio'])
        
        # 识别准确率
        if self.current_episode_stats['recognition_accuracy']:
            metrics['recognition_accuracy'] = np.mean(self.current_episode_stats['recognition_accuracy'])
        
        return metrics
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        获取汇总统计
        
        Args:
            last_n: 只统计最近n个episode，None表示全部
            
        Returns:
            指标汇总字典
        """
        if not self.episode_metrics:
            return {}
        
        metrics_to_use = self.episode_metrics[-last_n:] if last_n else self.episode_metrics
        
        summary = {}
        
        # 计算每个指标的统计值
        metric_names = [
            'handover_success_rate',
            'avg_switching_latency_ms',
            'max_switching_latency_ms',
            'avg_decision_time_ms',
            'missed_opportunity_rate',
            'avg_satisfaction',
            'latency_satisfaction',
            'rate_satisfaction',
            'load_variance',
            'avg_sinr',
            'recognition_accuracy',
            'connected_ratio',
        ]
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in metrics_to_use if metric_name in m]
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return summary
    
    def get_comparison_with_baseline(self, 
                                    baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        与基线对比
        
        Args:
            baseline_metrics: 基线指标字典
            
        Returns:
            对比结果
        """
        current = self.get_summary(last_n=10)  # 最近10个episode
        
        comparison = {}
        
        for metric_name in current.keys():
            if metric_name in baseline_metrics:
                current_mean = current[metric_name]['mean']
                baseline_mean = baseline_metrics[metric_name]
                
                # 计算改进百分比
                if baseline_mean != 0:
                    improvement = (current_mean - baseline_mean) / baseline_mean * 100
                else:
                    improvement = 0
                
                comparison[metric_name] = {
                    'current': current_mean,
                    'baseline': baseline_mean,
                    'improvement_percent': improvement,
                    'better': improvement > 0
                }
        
        return comparison
    
    def reset(self):
        """重置所有数据"""
        self.decisions = []
        self.episode_metrics = []
        self.current_episode_decisions = []
        self.current_episode_stats = defaultdict(list)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出为字典格式（兼容exp3_data.json格式）
        
        Returns:
            指标字典
        """
        summary = self.get_summary()
        
        # 转换为exp3_data.json格式
        export = {}
        for metric_name, stats in summary.items():
            export[metric_name] = [
                stats['mean'],
                stats['std']
            ]
        
        return export


class RealTimeMetricsMonitor:
    """
    实时指标监控器
    
    用于训练过程中的实时监控
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化监控器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.metrics_buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            self.metrics_buffers[key].append(value)
    
    def get_current_stats(self) -> Dict[str, Dict[str, float]]:
        """获取当前统计"""
        stats = {}
        
        for key, buffer in self.metrics_buffers.items():
            if buffer:
                values = list(buffer)
                stats[key] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': values[-1] - values[0] if len(values) > 1 else 0
                }
        
        return stats
    
    def get_smoothed_value(self, key: str, window: int = 10) -> float:
        """获取平滑后的值"""
        if key not in self.metrics_buffers:
            return 0.0
        
        buffer = self.metrics_buffers[key]
        if len(buffer) < window:
            return np.mean(buffer)
        
        return np.mean(list(buffer)[-window:])


# 便捷函数
def create_metrics_collector() -> CommunicationMetricsCollector:
    """创建指标采集器"""
    return CommunicationMetricsCollector()


def compute_critical_satisfaction(env, 
                                 critical_biz_types=[0, 1]) -> float:
    """
    计算关键业务满意度
    
    Args:
        env: 环境对象
        critical_biz_types: 关键业务类型列表
        
    Returns:
        关键业务平均满意度
    """
    if not hasattr(env, 'uavs'):
        return 0.0
    
    critical_sats = []
    for uav in env.uavs.values():
        biz_type = getattr(uav, 'business_type', None)
        if biz_type is not None and hasattr(biz_type, 'value'):
            if biz_type.value in critical_biz_types:
                sat = getattr(uav, 'current_satisfaction', 0.5)
                critical_sats.append(sat)
    
    return np.mean(critical_sats) if critical_sats else 0.0


def compute_weighted_satisfaction(env,
                                 weights={0: 0.5, 1: 0.3, 2: 0.2}) -> float:
    """
    计算加权满意度
    
    Args:
        env: 环境对象
        weights: 业务类型权重
        
    Returns:
        加权平均满意度
    """
    if not hasattr(env, 'uavs'):
        return 0.0
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for uav in env.uavs.values():
        biz_type = getattr(uav, 'business_type', None)
        if biz_type is not None and hasattr(biz_type, 'value'):
            weight = weights.get(biz_type.value, 0.2)
            sat = getattr(uav, 'current_satisfaction', 0.5)
            weighted_sum += sat * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0
