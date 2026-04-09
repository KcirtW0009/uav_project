# -*- coding: utf-8 -*-
"""
奖励函数改进模块

提供多种奖励函数实现，支持不同优化目标
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque
from .business import BusinessType


class RewardNormalizer:
    """奖励归一化器 - 使用running statistics"""
    
    def __init__(self, decay: float = 0.99, clip_val: float = 10.0):
        self.decay = decay
        self.clip_val = clip_val
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
    
    def update(self, reward: float):
        """更新running statistics"""
        self.count += 1
        # 使用指数移动平均
        self.mean = self.decay * self.mean + (1 - self.decay) * reward
        self.var = self.decay * self.var + (1 - self.decay) * (reward - self.mean) ** 2
    
    def normalize(self, reward: float) -> float:
        """归一化奖励"""
        if self.count < 10:  # 前期不归一化
            return reward
        std = np.sqrt(self.var) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip_val, self.clip_val)
    
    def reset(self):
        """重置统计"""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


class RewardFunctionV2:
    """
    简化且目标明确的奖励函数 V2
    
    核心目标：
    1. 最大化满意度（主要）
    2. 最小化切换次数（次要）
    3. 避免断连（严重惩罚）
    """
    
    def __init__(self, 
                 sat_weight: float = 10.0,
                 switch_penalty_base: float = 1.0,
                 disconnect_penalty: float = 50.0,
                 load_balance_weight: float = 0.1,
                 use_biz_aware_penalty: bool = True):
        """
        初始化奖励函数
        
        Args:
            sat_weight: 满意度奖励权重
            switch_penalty_base: 基础切换惩罚
            disconnect_penalty: 断连惩罚
            load_balance_weight: 负载均衡权重
            use_biz_aware_penalty: 是否使用业务感知切换惩罚
        """
        self.sat_weight = sat_weight
        self.switch_penalty_base = switch_penalty_base
        self.disconnect_penalty = disconnect_penalty
        self.load_balance_weight = load_balance_weight
        self.use_biz_aware_penalty = use_biz_aware_penalty
        
        # 业务类型切换敏感度
        self.biz_switch_sensitivity = {
            BusinessType.CONTROL_SIGNAL: 2.0,      # 控制信令对切换最敏感
            BusinessType.VIDEO_STREAMING: 1.0,     # 视频中等敏感
            BusinessType.ENVIRONMENT_MONITORING: 0.5,  # 环境监测最不敏感
        }
    
    def compute_reward(self, uav, env, action: int, info: Dict) -> float:
        """
        计算单步奖励
        
        Args:
            uav: UAV对象
            env: 环境对象
            action: 执行的动作
            info: 额外信息
            
        Returns:
            奖励值
        """
        # 1. 基础满意度奖励
        satisfaction = getattr(uav, 'current_satisfaction', 0.5)
        sat_reward = satisfaction * self.sat_weight
        
        # 2. 切换惩罚
        last_bs = getattr(uav, 'last_bs_id', None)
        current_bs = getattr(uav, 'connected_bs_id', None)
        
        switch_penalty = 0.0
        if action != 0 and last_bs is not None:  # 发生切换
            if self.use_biz_aware_penalty:
                biz_type = getattr(uav, 'business_type', BusinessType.ENVIRONMENT_MONITORING)
                sensitivity = self.biz_switch_sensitivity.get(biz_type, 1.0)
                switch_penalty = -self.switch_penalty_base * sensitivity
            else:
                switch_penalty = -self.switch_penalty_base
        
        # 3. 断连惩罚（严重）
        is_connected = getattr(uav, 'is_connected', True)
        disconnect_penalty = -self.disconnect_penalty if not is_connected else 0.0
        
        # 4. 负载均衡奖励
        load_reward = 0.0
        if is_connected and current_bs is not None:
            bs_load = getattr(env, 'bs_loads', {}).get(current_bs, 0)
            # 负载超过80%时给予惩罚
            if bs_load > 0.8:
                load_reward = -self.load_balance_weight * (bs_load - 0.8) * 10
        
        # 5. 额外奖励：如果满意度有提升
        satisfaction_improvement = 0.0
        if hasattr(uav, 'previous_satisfaction'):
            improvement = satisfaction - uav.previous_satisfaction
            if improvement > 0:
                satisfaction_improvement = improvement * 2.0  # 提升奖励
        
        # 总奖励
        total_reward = (
            sat_reward + 
            switch_penalty + 
            disconnect_penalty + 
            load_reward +
            satisfaction_improvement
        )
        
        return total_reward
    
    def compute_team_reward(self, env, actions: Dict[int, int], info: Dict) -> float:
        """
        计算团队奖励（所有UAV的平均）
        
        Args:
            env: 环境对象
            actions: UAV动作字典
            info: 额外信息
            
        Returns:
            平均奖励
        """
        rewards = []
        for uav_id, action in actions.items():
            if hasattr(env, 'uavs') and uav_id in env.uavs:
                uav = env.uavs[uav_id]
                reward = self.compute_reward(uav, env, action, info)
                rewards.append(reward)
        
        return np.mean(rewards) if rewards else 0.0


class RewardFunctionV3:
    """
    奖励函数 V3 - 基于差分的奖励设计
    
    重点奖励相对改进而非绝对值
    """
    
    def __init__(self):
        self.satisfaction_history = {}
        self.sinr_history = {}
    
    def compute_reward(self, uav, env, action: int, info: Dict) -> float:
        """基于差分的奖励计算"""
        uav_id = getattr(uav, 'id', 0)
        
        # 获取当前状态
        current_sat = getattr(uav, 'current_satisfaction', 0.5)
        current_sinr = getattr(uav, 'current_sinr', 0)
        
        # 初始化历史
        if uav_id not in self.satisfaction_history:
            self.satisfaction_history[uav_id] = deque(maxlen=5)
            self.sinr_history[uav_id] = deque(maxlen=5)
        
        # 计算相对改进
        sat_reward = 0
        if len(self.satisfaction_history[uav_id]) > 0:
            prev_avg_sat = np.mean(self.satisfaction_history[uav_id])
            sat_improvement = current_sat - prev_avg_sat
            sat_reward = sat_improvement * 20  # 放大改进信号
        
        # 计算SINR改进
        sinr_reward = 0
        if len(self.sinr_history[uav_id]) > 0:
            prev_avg_sinr = np.mean(self.sinr_history[uav_id])
            sinr_improvement = current_sinr - prev_avg_sinr
            sinr_reward = sinr_improvement * 0.5
        
        # 更新历史
        self.satisfaction_history[uav_id].append(current_sat)
        self.sinr_history[uav_id].append(current_sinr)
        
        # 切换惩罚
        switch_penalty = 0
        last_bs = getattr(uav, 'last_bs_id', None)
        if action != 0 and last_bs is not None:
            switch_penalty = -1.0
        
        # 断连惩罚
        disconnect_penalty = 0
        if not getattr(uav, 'is_connected', True):
            disconnect_penalty = -30
        
        return sat_reward + sinr_reward + switch_penalty + disconnect_penalty


class CompositeRewardFunction:
    """
    组合奖励函数 - 可配置多种奖励组件
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'satisfaction_weight': 8.0,
            'switch_penalty': 1.0,
            'disconnect_penalty': 40.0,
            'throughput_weight': 0.5,
            'latency_weight': 0.5,
            'load_balance_weight': 0.2,
        }
    
    def compute_reward(self, uav, env, action: int, info: Dict) -> float:
        """组合奖励计算"""
        reward = 0.0
        
        # 满意度组件
        satisfaction = getattr(uav, 'current_satisfaction', 0.5)
        reward += satisfaction * self.config['satisfaction_weight']
        
        # 吞吐量组件
        throughput = getattr(uav, 'current_throughput', 0)
        required_rate = getattr(uav, 'required_rate', 1)
        throughput_ratio = min(throughput / required_rate, 2.0)  # 上限2倍
        reward += throughput_ratio * self.config['throughput_weight']
        
        # 延迟组件
        latency = getattr(uav, 'current_latency', 0)
        latency_penalty = -latency / 100 * self.config['latency_weight']
        reward += latency_penalty
        
        # 切换惩罚
        last_bs = getattr(uav, 'last_bs_id', None)
        if action != 0 and last_bs is not None:
            reward -= self.config['switch_penalty']
        
        # 断连惩罚
        if not getattr(uav, 'is_connected', True):
            reward -= self.config['disconnect_penalty']
        
        # 负载均衡
        current_bs = getattr(uav, 'connected_bs_id', None)
        if current_bs is not None:
            bs_load = getattr(env, 'bs_loads', {}).get(current_bs, 0)
            if bs_load > 0.9:  # 高负载惩罚
                reward -= self.config['load_balance_weight'] * (bs_load - 0.9) * 10
        
        return reward


class CooperativeRewardFunction:
    """
    合作奖励函数 - 增强对长期合作行为的激励
    
    特点：
    1. 团队合作奖励：基于整个网络的负载均衡
    2. 长期满意度趋势奖励：奖励稳定或上升的满意度
    3. 全局网络状态奖励：基于整个网络的性能指标
    4. 业务感知的合作奖励：根据业务类型调整奖励策略
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'satisfaction_weight': 10.0,
            'switch_penalty': 1.0,
            'disconnect_penalty': 50.0,
            'team_cooperation_weight': 2.0,
            'long_term_trend_weight': 1.5,
            'global_network_weight': 2.0,
            'throughput_weight': 0.5,
            'latency_weight': 0.5,
        }
        
        # 业务类型权重
        self.biz_weights = {
            BusinessType.CONTROL_SIGNAL: 1.5,
            BusinessType.VIDEO_STREAMING: 1.0,
            BusinessType.ENVIRONMENT_MONITORING: 0.8,
        }
        
        # 历史记录
        self.satisfaction_history = {}
        self.team_performance_history = deque(maxlen=10)  # 团队性能历史
    
    def compute_reward(self, uav, env, action: int, info: Dict) -> float:
        """计算合作奖励"""
        uav_id = getattr(uav, 'id', 0)
        biz_type = getattr(uav, 'business_type', BusinessType.ENVIRONMENT_MONITORING)
        biz_weight = self.biz_weights.get(biz_type, 1.0)
        
        reward = 0.0
        
        # 1. 基础满意度奖励
        satisfaction = getattr(uav, 'current_satisfaction', 0.5)
        reward += satisfaction * self.config['satisfaction_weight'] * biz_weight
        
        # 2. 长期满意度趋势奖励
        if uav_id not in self.satisfaction_history:
            self.satisfaction_history[uav_id] = deque(maxlen=5)
        
        long_term_reward = 0.0
        if len(self.satisfaction_history[uav_id]) >= 3:
            # 计算满意度趋势
            history = list(self.satisfaction_history[uav_id])
            trend = np.polyfit(range(len(history)), history, 1)[0]
            if trend > 0:  # 满意度上升趋势
                long_term_reward = trend * 10 * self.config['long_term_trend_weight']
            elif trend >= -0.01:  # 满意度稳定
                long_term_reward = 0.5 * self.config['long_term_trend_weight']
        reward += long_term_reward
        
        # 3. 切换惩罚
        last_bs = getattr(uav, 'last_bs_id', None)
        if action != 0 and last_bs is not None:
            # 业务感知的切换惩罚
            if biz_type == BusinessType.CONTROL_SIGNAL:
                penalty = self.config['switch_penalty'] * 1.5
            else:
                penalty = self.config['switch_penalty']
            reward -= penalty
        
        # 4. 断连惩罚
        if not getattr(uav, 'is_connected', True):
            reward -= self.config['disconnect_penalty'] * biz_weight
        
        # 5. 吞吐量和延迟奖励
        throughput = getattr(uav, 'current_throughput', 0)
        required_rate = getattr(uav, 'required_rate', 1)
        throughput_ratio = min(throughput / required_rate, 2.0)
        reward += throughput_ratio * self.config['throughput_weight']
        
        latency = getattr(uav, 'current_latency', 0)
        latency_penalty = -latency / 100 * self.config['latency_weight']
        reward += latency_penalty
        
        # 更新历史
        self.satisfaction_history[uav_id].append(satisfaction)
        
        return reward
    
    def compute_team_reward(self, env, actions: Dict[int, int], info: Dict) -> float:
        """
        计算团队合作奖励
        
        基于整个网络的性能指标
        """
        # 计算平均满意度
        avg_satisfaction = np.mean([getattr(uav, 'current_satisfaction', 0.5) for uav in env.uavs.values()])
        
        # 计算负载均衡度（负载方差的倒数）
        if hasattr(env, 'base_stations'):
            load_ratios = [bs.load_ratio for bs in env.base_stations.values()]
            load_balance = 1.0 / (np.var(load_ratios) + 0.01) if load_ratios else 1.0
        else:
            load_balance = 1.0
        
        # 计算网络性能指标
        if hasattr(env, 'uavs'):
            avg_throughput = np.mean([getattr(uav, 'current_throughput', 0) for uav in env.uavs.values()])
            avg_latency = np.mean([getattr(uav, 'current_latency', 0) for uav in env.uavs.values()])
            avg_packet_loss = np.mean([getattr(uav, 'packet_loss_rate', 0) for uav in env.uavs.values()])
        else:
            avg_throughput = 0
            avg_latency = 0
            avg_packet_loss = 0
        
        # 计算团队奖励
        team_reward = 0.0
        
        # 满意度奖励
        team_reward += avg_satisfaction * 5.0
        
        # 负载均衡奖励
        team_reward += load_balance * self.config['team_cooperation_weight']
        
        # 网络性能奖励
        team_reward += (avg_throughput / 10) * 0.5  # 吞吐量奖励
        team_reward -= (avg_latency / 100) * 0.5  # 延迟惩罚
        team_reward -= avg_packet_loss * 10  # 丢包率惩罚
        
        # 记录团队性能
        self.team_performance_history.append({
            'avg_satisfaction': avg_satisfaction,
            'load_balance': load_balance,
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_packet_loss': avg_packet_loss
        })
        
        # 长期团队性能趋势奖励
        if len(self.team_performance_history) >= 5:
            histories = list(self.team_performance_history)
            avg_sat_trend = np.polyfit(range(len(histories)), [h['avg_satisfaction'] for h in histories], 1)[0]
            if avg_sat_trend > 0:
                team_reward += avg_sat_trend * 10 * self.config['long_term_trend_weight']
        
        return team_reward


def get_reward_function(version: str = 'v2', **kwargs):
    """
    获取指定版本的奖励函数
    
    Args:
        version: 版本 ('v1', 'v2', 'v3', 'composite', 'cooperative')
        **kwargs: 额外参数
        
    Returns:
        奖励函数实例
    """
    if version == 'v2':
        return RewardFunctionV2(**kwargs)
    elif version == 'v3':
        return RewardFunctionV3()
    elif version == 'composite':
        return CompositeRewardFunction(kwargs.get('config'))
    elif version == 'cooperative':
        return CooperativeRewardFunction(kwargs.get('config'))
    else:
        return RewardFunctionV2()  # 默认V2
