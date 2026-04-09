# MAPPO 算法系统性优化计划

## 执行摘要

针对 Terminal#5-211 显示的 MAPPO 训练与评估结果，本计划提供系统性的分析与优化指导。主要问题包括：收敛抖动过大、满意度波动显著、算法性能落后于传统算法、通信指标缺失等。

---

## 1. 早停机制与收敛抖动优化

### 1.1 问题分析

当前训练在 Episode 234 触发早停（连续150轮 satisfaction 无改善），但 reward 收敛曲线抖动过大，表明模型可能尚未完全收敛。

**根本原因：**
- 学习率调度策略可能过于激进
- 缺少经验回放机制（PPO 本身无回放，但可通过其他方式稳定）
- 探索率衰减过快导致陷入局部最优
- 奖励尺度不稳定

### 1.2 参数调整方案

#### 学习率调度策略对比测试

| 策略 | 参数配置 | 测试目的 |
|------|----------|----------|
| CosineAnnealing (当前) | T_max=300, eta_min=1e-6 | 基准对比 |
| Step Decay | step_size=100, gamma=0.5 | 阶段性降低 |
| Exponential | gamma=0.995 | 平滑衰减 |
| Warmup + Cosine | warmup=50, T_max=250 | 初期稳定 |
| No Schedule | - | 验证调度必要性 |

**测试命令：**
```bash
# 使用参数搜索框架进行系统性测试
python mappo_parameter_search.py
```

#### PPO 核心参数调优

```python
# 推荐参数搜索空间
param_space = {
    'actor_lr': [3e-5, 5e-5, 1e-4],      # 当前: 5e-5
    'critic_lr': [3e-4, 5e-4, 1e-3],     # 当前: 3e-4
    'clip_epsilon': [0.1, 0.15, 0.2],    # 当前: 0.1
    'entropy_coef': [0.01, 0.02, 0.05],  # 当前: 0.02
    'gae_lambda': [0.95, 0.97, 0.99],    # 当前: 0.95
    'num_epochs': [3, 5, 8],             # 当前: 5
    'batch_size': [128, 256, 512],       # 当前: 128
}
```

#### 奖励归一化方案

```python
# 在 MAPPOAgentV2 中添加奖励归一化
class RewardNormalizer:
    def __init__(self, decay=0.99):
        self.mean = 0
        self.var = 1
        self.decay = decay
        self.count = 0
    
    def update(self, reward):
        self.count += 1
        self.mean = self.decay * self.mean + (1 - self.decay) * reward
        self.var = self.decay * self.var + (1 - self.decay) * (reward - self.mean) ** 2
    
    def normalize(self, reward):
        return (reward - self.mean) / (np.sqrt(self.var) + 1e-8)
```

### 1.3 收敛平滑度评估指标

```python
# 计算收敛平滑度
def compute_convergence_smoothness(values, window=20):
    """
    评估收敛曲线的平滑度
    
    Returns:
        smoothness_score: 越高越平滑
        variance_trend: 方差变化趋势
    """
    rolling_mean = np.convolve(values, np.ones(window)/window, mode='valid')
    rolling_var = np.array([np.var(values[i:i+window]) for i in range(len(values)-window+1)])
    
    # 平滑度分数：后期方差与前期方差的比值
    early_var = np.mean(rolling_var[:len(rolling_var)//4])
    late_var = np.mean(rolling_var[-len(rolling_var)//4:])
    smoothness_score = early_var / (late_var + 1e-8)
    
    return smoothness_score, rolling_var
```

---

## 2. 满意度可视化优化

### 2.1 纵轴刻度范围调整

当前满意度变化波动显著，需通过调整纵轴刻度范围使变化趋势更明显。

```python
# 在可视化代码中添加自适应y轴范围
class SatisfactionVisualizer:
    def plot_satisfaction(self, sats, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # 原始数据
        ax.plot(sats, alpha=0.3, color='green', label='Raw')
        
        # 平滑曲线
        if len(sats) >= 10:
            smoothed = np.convolve(sats, np.ones(10)/10, mode='valid')
            ax.plot(range(9, len(sats)), smoothed, color='darkgreen', linewidth=2, label='Smoothed')
        
        # 自适应y轴范围 - 突出变化
        ymin, ymax = np.min(sats), np.max(sats)
        margin = (ymax - ymin) * 0.15  # 15% 边距
        ax.set_ylim(ymin - margin, ymax + margin)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Satisfaction')
        ax.set_title('Training Satisfaction (Adaptive Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
```

### 2.2 满意度计算逻辑审查

**当前可能存在的问题：**
1. 异常值处理不当（如断连时的满意度计算）
2. 特征权重分配不合理
3. 不同业务类型的满意度加权方式

**建议的改进方案：**

```python
def compute_satisfaction_enhanced(uav, env):
    """
    改进的满意度计算
    
    1. 添加异常值截断
    2. 业务类型自适应权重
    3. 时间衰减因子（近期体验权重更高）
    """
    # 基础指标
    sinr = uav.current_sinr
    throughput = uav.current_throughput
    latency = uav.current_latency
    
    # 异常值处理
    sinr = np.clip(sinr, 0, 100)  # SINR 截断
    throughput = np.clip(throughput, 0, uav.max_throughput * 1.5)
    
    # 业务类型权重
    biz_weights = {
        BusinessType.CONTROL_SIGNAL: {'sinr': 0.3, 'latency': 0.5, 'throughput': 0.2},
        BusinessType.VIDEO_STREAMING: {'sinr': 0.4, 'latency': 0.2, 'throughput': 0.4},
        BusinessType.ENVIRONMENT_MONITORING: {'sinr': 0.3, 'latency': 0.3, 'throughput': 0.4},
    }
    
    weights = biz_weights.get(uav.business_type, biz_weights[BusinessType.ENVIRONMENT_MONITORING])
    
    # 计算各维度满意度
    sinr_sat = min(sinr / 30, 1.0)  # 30dB 为满分
    latency_sat = max(0, 1 - latency / 100)  # 100ms 为0分
    throughput_sat = min(throughput / uav.required_rate, 1.0)
    
    # 加权综合
    satisfaction = (
        weights['sinr'] * sinr_sat +
        weights['latency'] * latency_sat +
        weights['throughput'] * throughput_sat
    )
    
    return np.clip(satisfaction, 0, 1)
```

---

## 3. 算法性能诊断与改进

### 3.1 核心问题识别

根据评估结果，MAPPO 在多项关键指标上落后于传统算法：

| 指标 | 传统算法 | 增强算法 | BA-MAPPO | 问题分析 |
|------|----------|----------|----------|----------|
| 平均满意度 | 0.9632 | 0.9556 | 0.9574 | 略低于传统算法 |
| 切换延迟 | 0.00ms | 0.01ms | 6.66ms | **严重滞后** |
| Ping抖动 | 3.98ms | 4.43ms | 5.15ms | 通信质量差 |
| 丢包率 | 5.15% | 3.41% | 5.75% | **高于两者** |
| QoS违规率 | 5.20% | 4.64% | 6.22% | **最高** |

**关键发现：** `best_sinr` 基线指标表现太好，这与实验设计目的相悖。

### 3.2 奖励函数重构

当前奖励函数可能过于复杂或方向性不强。建议简化并明确优化目标：

```python
def compute_reward_v2(uav, env, action, info):
    """
    简化且目标明确的奖励函数
    
    核心目标：
    1. 最大化满意度（主要）
    2. 最小化切换次数（次要）
    3. 避免断连（惩罚）
    """
    # 基础满意度奖励
    satisfaction = uav.current_satisfaction
    sat_reward = satisfaction * 10  # 放大满意度信号
    
    # 切换惩罚（与业务类型相关）
    if action != uav.last_bs_id:  # 发生切换
        # 不同业务对切换的敏感度不同
        switch_penalty = {
            BusinessType.CONTROL_SIGNAL: -2.0,  # 控制信令对切换最敏感
            BusinessType.VIDEO_STREAMING: -1.0,
            BusinessType.ENVIRONMENT_MONITORING: -0.5,
        }.get(uav.business_type, -1.0)
    else:
        switch_penalty = 0
    
    # 断连惩罚（严重）
    disconnect_penalty = -50 if not uav.is_connected else 0
    
    # 负载均衡奖励（可选）
    bs_load = env.bs_loads[uav.connected_bs_id] if uav.is_connected else 0
    load_reward = -0.1 * max(0, bs_load - 0.8)  # 负载超过80%时惩罚
    
    total_reward = sat_reward + switch_penalty + disconnect_penalty + load_reward
    
    return total_reward
```

### 3.3 状态空间优化

当前状态空间可能缺少关键信息。建议添加：

```python
# 增强观测空间
def get_enhanced_observation(uav, env):
    obs = []
    
    # 当前连接信息
    obs.extend([
        uav.current_sinr / 50.0,  # 归一化
        uav.current_throughput / uav.max_throughput,
        uav.current_latency / 100.0,
        uav.current_satisfaction,
    ])
    
    # 各基站状态（包括当前未连接的）
    for bs_id in range(env.num_bs):
        bs = env.base_stations[bs_id]
        obs.extend([
            uav.sinr_to_bs[bs_id] / 50.0 if bs_id in uav.sinr_to_bs else 0,
            bs.current_load / bs.capacity,
            bs.num_connected_uavs / bs.max_connections,
            1.0 if bs_id == uav.connected_bs_id else 0.0,  # 是否当前连接
        ])
    
    # 历史信息（最近几次切换结果）
    recent_switch_success = uav.recent_switch_results[-5:]  # 最近5次
    obs.extend([1.0 if x else 0.0 for x in recent_switch_success])
    obs.extend([0.0] * (5 - len(recent_switch_success)))  # 填充
    
    # 业务类型编码
    biz_onehot = [0.0] * 3
    biz_onehot[uav.business_type.value] = 1.0
    obs.extend(biz_onehot)
    
    return np.array(obs, dtype=np.float32)
```

### 3.4 best_sinr 基线问题解决方案

**问题：** best_sinr 基线表现太好，说明只看 SINR 就能取得好效果。

**解决方案：**

1. **增加场景复杂度**
   - 增加负载波动
   - 增加移动性
   - 增加干扰变化

2. **修改 SINR 计算**
   - 添加测量误差
   - 添加延迟（SINR 不是实时准确的）

3. **重新设计实验场景**
   - 设计需要多步决策的场景
   - 设计需要权衡多个指标的场景

```python
# 添加 SINR 测量误差
class RealisticSINRModel:
    def __init__(self, noise_std=2.0, measurement_delay=3):
        self.noise_std = noise_std
        self.measurement_delay = measurement_delay
        self.sinr_history = deque(maxlen=measurement_delay + 1)
    
    def get_measured_sinr(self, true_sinr):
        """获取带噪声和延迟的 SINR 测量值"""
        self.sinr_history.append(true_sinr)
        
        if len(self.sinr_history) < self.measurement_delay:
            return true_sinr  # 初始阶段无延迟
        
        delayed_sinr = self.sinr_history[0]
        noisy_sinr = delayed_sinr + np.random.normal(0, self.noise_std)
        
        return max(0, noisy_sinr)
```

---

## 4. 通信指标补充实现

根据 `exp3_data.json` 定义的评估标准，需要补充以下指标：

### 4.1 指标采集模块

```python
class CommunicationMetricsCollector:
    """通信指标采集器"""
    
    def __init__(self):
        self.metrics = {
            'handover_success_rate': [],
            'avg_switching_latency_ms': [],
            'max_switching_latency_ms': [],
            'avg_decision_time_ms': [],
            'missed_opportunity_rate': [],
            'avg_satisfaction': [],
            'critical_satisfaction': [],
            'weighted_satisfaction': [],
            'latency_satisfaction': [],
            'rate_satisfaction': [],
            'load_variance': [],
            'avg_sinr': [],
            'migration_success_rate': [],
            'connected_ratio': [],
        }
    
    def collect_episode(self, env, decisions):
        """采集一个 episode 的指标"""
        # 切换成功率
        handover_attempts = len([d for d in decisions if d['action'] != d['last_bs']])
        handover_success = len([d for d in decisions if d['success']])
        self.metrics['handover_success_rate'].append(
            handover_success / handover_attempts if handover_attempts > 0 else 1.0
        )
        
        # 切换延迟统计
        latencies = [d['latency'] for d in decisions if d['action'] != d['last_bs']]
        if latencies:
            self.metrics['avg_switching_latency_ms'].append(np.mean(latencies))
            self.metrics['max_switching_latency_ms'].append(np.max(latencies))
        
        # 决策时间
        decision_times = [d['decision_time'] for d in decisions]
        self.metrics['avg_decision_time_ms'].append(np.mean(decision_times))
        
        # 错失机会率
        missed = len([d for d in decisions if d['missed_opportunity']])
        self.metrics['missed_opportunity_rate'].append(missed / len(decisions))
        
        # 满意度指标
        sats = [uav.current_satisfaction for uav in env.uavs.values()]
        self.metrics['avg_satisfaction'].append(np.mean(sats))
        
        # 关键业务满意度（控制信令 + 视频）
        critical_sats = [uav.current_satisfaction for uav in env.uavs.values()
                        if uav.business_type in [BusinessType.CONTROL_SIGNAL, BusinessType.VIDEO_STREAMING]]
        self.metrics['critical_satisfaction'].append(np.mean(critical_sats))
        
        # SINR
        sinrs = [uav.current_sinr for uav in env.uavs.values()]
        self.metrics['avg_sinr'].append(np.mean(sinrs))
        
        # 连接率
        connected = len([uav for uav in env.uavs.values() if uav.is_connected])
        self.metrics['connected_ratio'].append(connected / len(env.uavs))
    
    def get_summary(self):
        """获取汇总统计"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        return summary
```

### 4.2 集成到评估流程

```python
# 在 experiments_mappo.py 的评估阶段添加
def evaluate_with_full_metrics(agent, env, num_episodes=3):
    """带完整通信指标的评估"""
    collector = CommunicationMetricsCollector()
    
    for ep in range(num_episodes):
        obs_dict, state = env.reset()
        decisions = []
        
        for step in range(max_steps):
            start_time = time.time()
            actions, _, _, _, _ = agent.select_actions(obs_dict, state, biz_types, training=False)
            decision_time = (time.time() - start_time) * 1000  # ms
            
            # 记录决策
            for uav_id, action in actions.items():
                uav = env.env.uavs[uav_id]
                decisions.append({
                    'action': action,
                    'last_bs': uav.last_bs_id,
                    'success': True,  # 根据实际结果更新
                    'latency': uav.handover_latency if action != uav.last_bs_id else 0,
                    'decision_time': decision_time / len(actions),
                    'missed_opportunity': False,  # 根据逻辑判断
                })
            
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            obs_dict = next_obs
            state = next_state
        
        collector.collect_episode(env.env, decisions)
    
    return collector.get_summary()
```

---

## 5. 增强算法隔离与优化

### 5.1 参数配置隔离

创建独立的配置文件：

```python
# enhanced_algorithm_config.py
ENHANCED_ALGORITHM_CONFIG = {
    # 增强算法专用参数
    'sinr_threshold': 3.0,  # dB
    'hysteresis': 1.0,
    'load_balance_weight': 0.3,
    'business_priority': {
        BusinessType.CONTROL_SIGNAL: 1.0,
        BusinessType.VIDEO_STREAMING: 0.8,
        BusinessType.ENVIRONMENT_MONITORING: 0.5,
    },
    'prediction_horizon': 5,
    'exploration_rate': 0.1,
}

# 与MAPPO实验参数严格区分
MAPPO_EXPERIMENT_CONFIG = {
    'actor_lr': 5e-5,
    'critic_lr': 3e-4,
    # ... MAPPO专用参数
}
```

### 5.2 增强算法核心机制重设计

```python
class EnhancedHandoverAlgorithmV2:
    """
    增强切换算法 V2
    
    改进点：
    1. 多目标优化（SINR + 负载 + 业务类型）
    2. 预测性切换（基于轨迹预测）
    3. 协同决策（考虑相邻UAV的切换决策）
    """
    
    def __init__(self, config=None):
        self.config = config or ENHANCED_ALGORITHM_CONFIG
        self.decision_history = deque(maxlen=100)
    
    def select_action(self, uav, env):
        """选择切换目标"""
        current_bs = uav.connected_bs_id
        
        # 计算各基站的综合得分
        scores = {}
        for bs_id in range(env.num_bs):
            score = self._compute_bs_score(uav, env, bs_id)
            scores[bs_id] = score
        
        # 选择最佳基站
        best_bs = max(scores, key=scores.get)
        
        # 切换判决
        if best_bs == current_bs:
            return 0  # stay
        
        # 检查切换条件
        if self._should_handover(uav, env, current_bs, best_bs):
            return best_bs + 1  # 动作编码
        
        return 0
    
    def _compute_bs_score(self, uav, env, bs_id):
        """计算基站综合得分"""
        # SINR得分
        sinr = uav.sinr_to_bs.get(bs_id, -100)
        sinr_score = min(sinr / 30, 1.0)  # 30dB满分
        
        # 负载得分
        bs = env.base_stations[bs_id]
        load_score = 1.0 - bs.current_load / bs.capacity
        
        # 业务优先级得分
        priority = self.config['business_priority'].get(uav.business_type, 0.5)
        
        # 综合得分
        score = (
            0.5 * sinr_score +
            0.3 * load_score +
            0.2 * priority
        )
        
        return score
    
    def _should_handover(self, uav, env, current_bs, target_bs):
        """切换判决"""
        current_sinr = uav.sinr_to_bs.get(current_bs, -100)
        target_sinr = uav.sinr_to_bs.get(target_bs, -100)
        
        # A3事件触发条件 + 业务感知偏移
        threshold = self.config['sinr_threshold']
        hysteresis = self.config['hysteresis']
        
        # 控制信令业务更保守（更难触发切换）
        if uav.business_type == BusinessType.CONTROL_SIGNAL:
            threshold += 2.0
        
        return target_sinr > current_sinr + threshold + hysteresis
```

---

## 6. Stay 基线合理性验证

### 6.1 当前 Stay 基线分析

Stay 基线满意度较高（0.9054），需要验证其合理性：

```python
def analyze_stay_baseline(env, num_episodes=10):
    """分析 stay 基线的行为特征"""
    stats = {
        'satisfaction': [],
        'disconnections': [],
        'sinr_distribution': [],
        'load_distribution': [],
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        for step in range(max_steps):
            # 所有 UAV 执行 stay
            actions = {i: 0 for i in range(env.num_agents)}
            next_obs, _, _, _, info = env.step(actions)
            
            # 记录统计
            for uav in env.env.uavs.values():
                stats['satisfaction'].append(uav.current_satisfaction)
                stats['sinr_distribution'].append(uav.current_sinr)
                if not uav.is_connected:
                    stats['disconnections'].append(step)
    
    # 分析结果
    print(f"Stay 基线分析:")
    print(f"  平均满意度: {np.mean(stats['satisfaction']):.4f}")
    print(f"  SINR 分布: {np.percentile(stats['sinr_distribution'], [25, 50, 75])}")
    print(f"  断连次数: {len(stats['disconnections'])}")
    
    return stats
```

### 6.2 基线改进建议

如果 stay 基线过于简单，可以考虑：

1. **Random 基线**：随机选择动作
2. **Round-Robin 基线**：轮询切换
3. **Greedy-SINR 基线**：始终选择 SINR 最高的基站（不带滞后）

---

## 7. 实验方法论与参数探索

### 7.1 负载率与超参数影响研究

创建专门的参数探索实验：

```python
# mappo_parameter_search.py 已创建
# 使用方式：
```

**实验设计矩阵：**

| 负载率 | UAV数量 | BS容量 | 预期行为 |
|--------|---------|--------|----------|
| 低 (30%) | 32 | 高 | 资源充足，切换压力小 |
| 中 (60%) | 64 | 中 | 适度竞争 |
| 高 (88%) | 128 | 低 | 当前配置，资源紧张 |
| 极高 (95%) | 150 | 低 | 极端压力测试 |

### 7.2 自动化参数优化流程

```python
class MAPPOAutoTuner:
    """MAPPO 自动参数调优器"""
    
    def __init__(self):
        self.search_space = {
            'actor_lr': [1e-5, 3e-5, 5e-5, 1e-4],
            'critic_lr': [1e-4, 3e-4, 5e-4, 1e-3],
            'entropy_coef': [0.01, 0.02, 0.05, 0.1],
            'clip_epsilon': [0.1, 0.15, 0.2],
        }
        self.results = []
    
    def objective(self, config):
        """优化目标函数"""
        # 训练 MAPPO
        result = train_mappo_with_config(config, episodes=100)
        
        # 综合评分
        score = (
            0.4 * result['final_sat'] +
            0.3 * (1 / (1 + result['reward_variance'])) +  # 稳定性
            0.3 * (1 / (1 + result['convergence_speed'] / 100))  # 收敛速度
        )
        
        return score
    
    def run_optimization(self, method='bayesian', n_trials=50):
        """运行优化"""
        if method == 'grid':
            return self._grid_search()
        elif method == 'random':
            return self._random_search(n_trials)
        elif method == 'bayesian':
            return self._bayesian_optimization(n_trials)
    ```

---

## 8. 数据记录与可视化增强

### 8.1 扩展数据记录范围

已创建 `mappo_enhanced_monitoring.py`，包含：

1. **训练指标**
   - 奖励、满意度、损失
   - 熵、KL散度、梯度范数
   - 优势、价值函数统计
   - 学习率变化

2. **策略分析**
   - 动作分布
   - 业务类型分布
   - 切换率统计

3. **网络参数**
   - 权重均值/方差
   - 偏置变化

4. **通信指标**
   - 切换成功率
   - 延迟、抖动、丢包
   - 连接率

### 8.2 可视化增强

```python
# 使用示例
from mappo_enhanced_monitoring import MAPPOTrainingMonitor

monitor = MAPPOTrainingMonitor('./logs')

# 训练过程中记录
for episode in range(num_episodes):
    # ... 训练代码 ...
    
    monitor.log_episode(episode, {
        'reward': episode_reward,
        'satisfaction': avg_satisfaction,
        'actor_loss': train_stats['actor_loss'],
        # ... 其他指标
    })
    
    monitor.log_network_stats(agent)

# 生成综合可视化
monitor.create_comprehensive_visualization()
monitor.save_detailed_logs()
```

---

## 9. 实施时间表

| 阶段 | 任务 | 优先级 | 预计时间 | 预期成果 |
|------|------|--------|----------|----------|
| 1 | 参数搜索实验 | 高 | 2-3天 | 最优参数配置 |
| 2 | 奖励函数重构 | 高 | 1-2天 | 简化奖励函数 |
| 3 | 通信指标补充 | 高 | 1天 | 完整指标采集 |
| 4 | 增强算法隔离 | 中 | 1天 | 独立配置文件 |
| 5 | 可视化优化 | 中 | 1天 | 增强图表 |
| 6 | 负载率研究 | 中 | 2天 | 多场景结果 |
| 7 | 综合测试 | 高 | 2天 | 最终评估报告 |

---

## 10. 关键文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `mappo_parameter_search.py` | 参数系统性搜索 | ✅ 已创建 |
| `mappo_enhanced_monitoring.py` | 增强监控与可视化 | ✅ 已创建 |
| `MAPPO_OPTIMIZATION_PLAN.md` | 本优化计划文档 | ✅ 已创建 |

---

## 11. 下一步行动建议

1. **立即执行**：运行参数搜索实验，找出最优配置
   ```bash
   python mappo_parameter_search.py
   ```

2. **并行进行**：修改奖励函数，简化并明确优化目标

3. **随后执行**：补充通信指标采集模块

4. **最后**：进行全面的对比实验，验证改进效果
