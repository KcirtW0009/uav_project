# -*- coding: utf-8 -*-
"""
增强观测空间模块

提供更丰富的状态表示，包括历史信息、业务类型编码等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class EnhancedObservationSpace:
    """
    增强观测空间
    
    包含以下信息：
    1. 当前连接状态（SINR、吞吐量、延迟、满意度）
    2. 各基站状态（SINR、负载、连接数）
    3. 历史切换结果
    4. 业务类型编码
    5. 相对位置信息
    """
    
    def __init__(self, 
                 num_bs: int = 3,
                 history_length: int = 5,
                 use_relative_position: bool = True):
        """
        初始化增强观测空间
        
        Args:
            num_bs: 基站数量
            history_length: 历史记录长度
            use_relative_position: 是否使用相对位置信息
        """
        self.num_bs = num_bs
        self.history_length = history_length
        self.use_relative_position = use_relative_position
        
        # 计算观测维度
        self.obs_dim = self._compute_obs_dim()
        
    def _compute_obs_dim(self) -> int:
        """计算观测空间维度"""
        dim = 0
        
        # 当前连接状态 (4维)
        dim += 4
        
        # 各基站状态 (每基站4维)
        dim += self.num_bs * 4
        
        # 历史切换结果
        dim += self.history_length
        
        # 业务类型one-hot编码
        dim += 3
        
        # 相对位置信息（可选）
        if self.use_relative_position:
            dim += 4  # x, y, dx, dy
        
        return dim
    
    def get_observation(self, uav, env) -> np.ndarray:
        """
        获取增强观测向量
        
        Args:
            uav: UAV对象
            env: 环境对象
            
        Returns:
            观测向量
        """
        obs = []
        
        # 1. 当前连接状态
        obs.extend(self._get_connection_state(uav))
        
        # 2. 各基站状态
        obs.extend(self._get_bs_states(uav, env))
        
        # 3. 历史切换结果
        obs.extend(self._get_switch_history(uav))
        
        # 4. 业务类型编码
        obs.extend(self._get_biz_type_encoding(uav))
        
        # 5. 相对位置信息（可选）
        if self.use_relative_position:
            obs.extend(self._get_relative_position(uav, env))
        
        return np.array(obs, dtype=np.float32)
    
    def _get_connection_state(self, uav) -> List[float]:
        """获取当前连接状态"""
        return [
            getattr(uav, 'current_sinr', 0) / 50.0,  # 归一化到[0, 2]
            getattr(uav, 'current_throughput', 0) / (getattr(uav, 'max_throughput', 100) + 1e-8),
            getattr(uav, 'current_latency', 0) / 100.0,  # 归一化到[0, 1]
            getattr(uav, 'current_satisfaction', 0.5),
        ]
    
    def _get_bs_states(self, uav, env) -> List[float]:
        """获取各基站状态"""
        states = []
        
        for bs_id in range(self.num_bs):
            # SINR
            sinr = uav.sinr_to_bs.get(bs_id, -100) if hasattr(uav, 'sinr_to_bs') else -100
            states.append(sinr / 50.0)
            
            # 基站负载
            if hasattr(env, 'base_stations') and bs_id in env.base_stations:
                bs = env.base_stations[bs_id]
                load = bs.current_load / bs.capacity if hasattr(bs, 'capacity') else 0
                states.append(load)
                
                # 连接数比例
                conn_ratio = bs.num_connected_uavs / bs.max_connections if hasattr(bs, 'max_connections') else 0
                states.append(conn_ratio)
            else:
                states.extend([0.0, 0.0])
            
            # 是否当前连接
            current_bs = getattr(uav, 'connected_bs_id', None)
            states.append(1.0 if current_bs == bs_id else 0.0)
        
        return states
    
    def _get_switch_history(self, uav) -> List[float]:
        """获取历史切换结果"""
        history = []
        
        # 获取历史记录
        if hasattr(uav, 'switch_history'):
            recent_history = list(uav.switch_history)[-self.history_length:]
        else:
            recent_history = []
        
        # 填充历史
        for result in recent_history:
            # 1表示成功切换，0表示失败或stay，-1表示失败
            if result == 'success':
                history.append(1.0)
            elif result == 'failed':
                history.append(-1.0)
            else:
                history.append(0.0)
        
        # 填充剩余位置
        while len(history) < self.history_length:
            history.append(0.0)
        
        return history
    
    def _get_biz_type_encoding(self, uav) -> List[float]:
        """获取业务类型one-hot编码"""
        encoding = [0.0, 0.0, 0.0]
        
        biz_type = getattr(uav, 'business_type', None)
        if biz_type is not None and hasattr(biz_type, 'value'):
            idx = biz_type.value
            if 0 <= idx < 3:
                encoding[idx] = 1.0
        
        return encoding
    
    def _get_relative_position(self, uav, env) -> List[float]:
        """获取相对位置信息"""
        # UAV位置
        uav_pos = getattr(uav, 'position', (0, 0))
        uav_x, uav_y = uav_pos[0] / 1000.0, uav_pos[1] / 1000.0  # 归一化到[0, 1]
        
        # 速度/方向（如果有）
        if hasattr(uav, 'velocity'):
            vel = uav.velocity
            dx, dy = vel[0] / 10.0, vel[1] / 10.0  # 归一化
        else:
            dx, dy = 0.0, 0.0
        
        return [uav_x, uav_y, dx, dy]


class ObservationNormalizer:
    """
    观测归一化器
    
    支持多种归一化方法：
    - running_mean: 使用running statistics
    - batch_norm: 批归一化
    - min_max: 最小最大归一化
    """
    
    def __init__(self, obs_dim: int, method: str = 'running_mean'):
        """
        初始化归一化器
        
        Args:
            obs_dim: 观测维度
            method: 归一化方法
        """
        self.obs_dim = obs_dim
        self.method = method
        
        if method == 'running_mean':
            self.mean = np.zeros(obs_dim)
            self.var = np.ones(obs_dim)
            self.count = 0
        elif method == 'min_max':
            self.min_vals = np.zeros(obs_dim)
            self.max_vals = np.ones(obs_dim)
    
    def update(self, obs: np.ndarray):
        """更新统计信息"""
        if self.method == 'running_mean':
            self.count += 1
            delta = obs - self.mean
            self.mean += delta / self.count
            self.var += delta * (obs - self.mean)
    
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        if self.method == 'running_mean':
            if self.count < 10:
                return obs
            std = np.sqrt(self.var / self.count + 1e-8)
            return (obs - self.mean) / std
        elif self.method == 'min_max':
            range_vals = self.max_vals - self.min_vals + 1e-8
            return (obs - self.min_vals) / range_vals
        else:
            return obs
    
    def reset(self):
        """重置统计"""
        if self.method == 'running_mean':
            self.mean = np.zeros(self.obs_dim)
            self.var = np.ones(self.obs_dim)
            self.count = 0


class StateAugmenter:
    """
    状态增强器
    
    添加额外的状态特征以提高学习效果
    """
    
    def __init__(self):
        self.uav_states = {}
    
    def augment(self, uav, env) -> Dict[str, float]:
        """
        增强状态特征
        
        Returns:
            额外的状态特征字典
        """
        features = {}
        
        # 1. 时间特征
        if hasattr(env, 'current_step'):
            step = env.current_step
            features['time_sin'] = np.sin(2 * np.pi * step / 100)
            features['time_cos'] = np.cos(2 * np.pi * step / 100)
        
        # 2. 邻居UAV统计
        if hasattr(uav, 'position') and hasattr(env, 'uavs'):
            neighbor_count = 0
            neighbor_same_bs = 0
            
            current_bs = getattr(uav, 'connected_bs_id', None)
            
            for other_id, other_uav in env.uavs.items():
                if other_id == getattr(uav, 'id', None):
                    continue
                
                if hasattr(other_uav, 'position'):
                    dist = np.linalg.norm(
                        np.array(uav.position) - np.array(other_uav.position)
                    )
                    if dist < 100:  # 100m内认为是邻居
                        neighbor_count += 1
                        if getattr(other_uav, 'connected_bs_id', None) == current_bs:
                            neighbor_same_bs += 1
            
            features['neighbor_count'] = neighbor_count / 10.0  # 归一化
            features['neighbor_same_bs_ratio'] = neighbor_same_bs / max(neighbor_count, 1)
        
        # 3. 趋势特征
        uav_id = getattr(uav, 'id', 0)
        if uav_id not in self.uav_states:
            self.uav_states[uav_id] = {
                'sinr_history': deque(maxlen=5),
                'sat_history': deque(maxlen=5),
            }
        
        sinr = getattr(uav, 'current_sinr', 0)
        sat = getattr(uav, 'current_satisfaction', 0.5)
        
        self.uav_states[uav_id]['sinr_history'].append(sinr)
        self.uav_states[uav_id]['sat_history'].append(sat)
        
        # 计算趋势
        if len(self.uav_states[uav_id]['sinr_history']) >= 2:
            sinr_trend = sinr - self.uav_states[uav_id]['sinr_history'][0]
            sat_trend = sat - self.uav_states[uav_id]['sat_history'][0]
            
            features['sinr_trend'] = sinr_trend / 10.0  # 归一化
            features['sat_trend'] = sat_trend
        else:
            features['sinr_trend'] = 0.0
            features['sat_trend'] = 0.0
        
        # 4. 业务类型特定特征
        biz_type = getattr(uav, 'business_type', None)
        if biz_type is not None:
            # 不同业务类型的QoS需求满足度
            required_rate = getattr(uav, 'required_rate', 1)
            current_rate = getattr(uav, 'current_throughput', 0)
            qos_satisfaction = min(current_rate / required_rate, 2.0) / 2.0
            features['qos_satisfaction'] = qos_satisfaction
        
        return features


def create_enhanced_observation(uav, env, config: Optional[Dict] = None) -> np.ndarray:
    """
    创建增强观测向量的便捷函数
    
    Args:
        uav: UAV对象
        env: 环境对象
        config: 配置字典
        
    Returns:
        增强观测向量
    """
    config = config or {}
    
    obs_space = EnhancedObservationSpace(
        num_bs=config.get('num_bs', 3),
        history_length=config.get('history_length', 5),
        use_relative_position=config.get('use_relative_position', True)
    )
    
    return obs_space.get_observation(uav, env)
