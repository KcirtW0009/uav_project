# UAV业务识别与切换决策系统 - 完整代码结构文档

**版本**: v2.0 (2026-05-11 更新)  
**作者**: UAV Research Team  
**状态**: ✅ 已完成实验1-4，MAPPO训练完成，可视化就绪

---

## 📁 项目总览

```
uav_project/
├── main.py                          # 🚀 主程序入口（唯一启动点）
├── requirements.txt                 # Python依赖列表
│
├── docs/                            # 📚 文档目录
│   ├── README.md                    # 项目说明
│   ├── 快速学习指南.md              # 入门指南（本文档）
│   └── 论文全文.md                  # 毕业论文
│
├── uav_system/                      # 🔧 核心系统模块
│   ├── __init__.py                 # 包初始化
│   │
│   ├── config.py                   # ⚙️ 全局配置中心
│   ├── business.py                 # 📊 业务类型定义（3种5G业务）
│   ├── satisfaction.py             # ⭐ 满意度评估系统（层次化）
│   ├── recognition.py              # 🧠 业务识别模型（ML分类器）
│   ├── reward_functions.py         # 🎯 MAPPO奖励函数设计
│   │
│   ├── entities.py                 # 🏗️ 基础实体定义
│   │   ├── BaseStation            # 基站类
│   │   ├── UAV                    # 无人机类
│   │   └── Channel                # 信道模型
│   │
│   ├── environment.py              # 🌐 网络仿真环境
│   │   ├── NetworkEnvironment     # 基础环境
│   │   └── EnhancedNetworkEnvironment  # 增强环境（含识别）
│   │
│   ├── algorithms.py               # 🔄 切换算法实现
│   │   ├── IntegratedHandoverAlgorithm  # 传统算法
│   │   └── EnhancedHandoverAlgorithm    # 增强算法（5个模块）
│   │
│   ├── mappo_agent_v2.py           # 🤖 MAPPO智能体（V2优化版）
│   ├── mappo_environment.py        # 🎮 MAPPO专用评估环境
│   ├── mappo_optimized_config.py   # 📋 MAPPO超参数配置
│   │
│   ├── visualization.py            # 📈 内置可视化工具
│   ├── enhanced_observation.py     # 👁️ 增强观测空间设计
│   └── communication_metrics.py    # 📡 通信质量指标计算
│   │
│   └── experiments.py              # 🧪 实验管理核心（最重要！）
│       ├── Experiment1            # 实验1：识别准确率影响分析
│       ├── Experiment2            # 实验2：机制有效性验证
│       ├── Experiment3            # 实验3：三算法全面对比
│       └── Experiment4            # 实验4：五场景泛化验证
│
├── plot_exp2_figures.py            # 📊 实验2专业绘图脚本
├── plot_exp3_figures.py            # 📊 实验3专业绘图脚本
├── plot_exp4_figures.py            # 📊 实验4专业绘图脚本（6张图）
│
└── experiment_results/             # 💾 实验数据存储
    ├── exp1_data.json             # 实验1结果
    ├── exp2_data.json             # 实验2结果
    ├── exp3_data.json             # 实验3结果（含三算法）
    ├── exp4_data.json             # 实验4结果（五场景×三算法）
    ├── exp3_mappo_summary.json    # MAPPO详细数据
    ├── exp4_mappo_summary.json    # MAPPO五场景数据
    ├── mappo_models/              # 训练好的模型文件
    │   └── mappo_8bs_300uav.pt   # 最佳模型（300UAV×8BS）
    ├── latest_figures/            # 最新生成的图表
    │   ├── exp4_satisfaction_comparison.png
    │   ├── exp4_critical_satisfaction.png
    │   ├── exp4_connected_ratio.png
    │   ├── exp4_throughput_comparison.png
    │   ├── exp4_sinr_comparison.png
    │   └── exp4_load_variance.png
    └── training_logs/             # MAPPO训练日志
```

---

## 🎯 核心架构（六层体系）

```
┌─────────────────────────────────────────────────────────────┐
│                    L6: 可视化层 (Visualization)              │
│   plot_exp2/3/4_figures.py → 生成论文图表 (6+张)          │
├─────────────────────────────────────────────────────────────┤
│                    L5: 实验管理层 (Experiments)              │
│   experiments.py → Experiment1-4 类                        │
│   ├ 统一调度: 种子管理、缓存模式、自动保存                 │
│   ├ 数据收集: 17项核心指标 + 分业务统计                     │
│   └ 结果输出: 表格打印、统计检验、图表调用                  │
├─────────────────────────────────────────────────────────────┤
│                    L4: 决策算法层 (Algorithms)               │
│   algorithms.py → 传统 vs 增强                              │
│   mappo_agent_v2.py → 多智能体强化学习                     │
│   ├ 传统算法: RSSI阈值 + 负载均衡                           │
│   ├ 增强算法: 5大增强模块                                   │
│   │   ├ 动态阈值调整                                        │
│   │   ├ 业务权重优先级                                      │
│   │   ├ ε-贪婪探索策略                                      │
│   │   ├ 负载均衡优化                                        │
│   │   └ 自适应识别反馈                                      │
│   └ MAPPO: Actor-Critic + 集中式训练分布式执行            │
├─────────────────────────────────────────────────────────────┤
│                    L3: 评价层 (Satisfaction)                 │
│   satisfaction.py → 层次化满意度评估                         │
│   ├ L1: critical_satisfaction (关键业务满足率)             │
│   ├ L2: avg_satisfaction (整体满足率)                       │
│   ├ L3: weighted_satisfaction (加权满足率)                  │
│   └ L4/L5: rate/latency/loss 满足度细分                    │
├─────────────────────────────────────────────────────────────┤
│                    L2: 建模层 (Modeling)                     │
│   business.py → 3种5G业务类型定义                           │
│   recognition.py → ML业务识别模型                          │
│   environment.py → 网络仿真环境                             │
│   entities.py → BS/UAV/Channel实体                          │
├─────────────────────────────────────────────────────────────┤
│                    L1: 配置层 (Configuration)                │
│   config.py → 全局参数中心                                  │
│   ├ GLOBAL_SEED, RESULT_DIR                                │
│   ├ COLORS, CMAP_* 配色方案                                 │
│   └ MAPPOConfig → 24个P1级超参数                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 核心模块详解

### 1️⃣ **config.py** - 全局配置中心 ⭐⭐⭐

**地位**: 整个系统的"单一真相源"(Single Source of Truth)

```python
# 核心常量
GLOBAL_SEED = 30042          # 全局随机种子
RESULT_DIR = "experiment_results"  # 结果保存路径

# 中断检测配置
INTERRUPTION_CONFIG = {
    'threshold': 0.3,        # 满意度中断阈值
    'duration': 5,           # 持续步数
}

# MAPPO超参数（V21版本，解决硬编码问题）
class MAPPOConfig:
    class RewardConfig:      # 24个P1级奖励参数
    class BusinessWeightConfig:  # 业务权重
    class LoadAdaptiveConfig:   # 负载自适应
    class TrainingConfig:      # 训练参数
```

**关键功能**:
- ✅ 集中管理所有可调参数（避免散落在各文件）
- ✅ 提供类型提示和运行时验证
- ✅ 定义学术出版级配色方案
- ✅ 同步numpy/random/torch随机种子

**修改频率**: 低（确定后很少改动）

---

### 2️⃣ **business.py** - 业务类型建模 ⭐⭐

**地位**: 对齐3GPP 5G切片标准

```python
class BusinessType(Enum):
    CONTROL_SIGNAL = 0    # 控制信令 → URLLC
    VIDEO_STREAMING = 1   # 视频回传 → eMBB  
    ENVIRONMENT_MONITORING = 2  # 环境监测 → mMTC

class QoSProfile:
    min_rate: float        # 最低速率要求
    ideal_rate: float      # 理想目标速率
    max_delay: float       # 最大容忍时延
    max_loss_rate: float   # 最大丢包率
    priority: float        # 业务优先级 [0,1]
    criticality: float     # 关键性等级 [0,1]
    latency_sensitivity: float  # 时延敏感度
```

**三种业务对比**:

| 业务 | 带宽 | 时延 | 优先级 | 降级容忍 |
|-----|------|------|-------|---------|
| 控制信令 | 0.15-0.5 Mbps | ≤20ms | 0.99 | 仅5% |
| 视频回传 | 2-10 Mbps | ≤100ms | 0.85 | 允许40% |
| 环境监测 | 0.01-0.1 Mbps | ≤500ms | 0.60 | 允许70% |

---

### 3️⃣ **satisfaction.py** - 满意度评估系统 ⭐⭐⭐

**地位**: 系统的"评价核心"，决定算法好坏的标准

```python
class HierarchicalSatisfactionMetric:
    @staticmethod
    def compute_satisfaction(uav, true_qos) -> dict:
        """单个UAV的多维度满意度评估"""
        return {
            'critical': 0或1,      # 关键指标是否满足（二元）
            'overall': 0或1,       # 整体是否满足最低要求
            'weighted': [0,priority],  # 加权连续评分
            'rate_sat': [0,1],     # 速率满足度
            'latency_sat': [0,1],  # 时延满足度
            'loss_sat': [0,1]      # 丢包满足度
        }
    
    @staticmethod
    def compute_network_metrics(env) -> dict:
        """网络整体统计"""
        return {
            'avg_satisfaction': ...,      # 平均整体满足率
            'critical_satisfaction': ...,  # 关键业务满足率
            'weighted_satisfaction': ...,  # 加权满足率
            'latency_satisfaction': ...,   # 时延满足率
            'rate_satisfaction': ...       # 速率满足率
        }
```

**层次化指标体系**:
```
L1: critical_satisfaction (关键业务)
    ↓ 控制信令必须同时满足速率+时延
L2: avg_satisfaction (整体满足率) 
    ↓ 基于min_rate的硬性判断
L3: weighted_satisfaction (加权满足率)
    ↓ 使用priority作为满分权重
L4/L5: 各维度细分（rate/latency/loss）
```

**使用位置**: 
- `environment.py` 的 `get_state_statistics()` 
- `experiments.py` 的数据收集
- 所有实验的结果评估

---

### 4️⃣ **recognition.py** - 业务识别模型 ⭐⭐

**地位**: ML分类器，实现UAV业务的自动识别

```python
class BusinessRecognitionModel:
    def __init__(self, model_type='random_forest'):
        self.model = RandomForestClassifier()  # 默认RF
        self.scaler = StandardScaler()
    
    def train(self, features, labels):
        """训练识别模型（使用历史数据）"""
        
    def predict(self, features) -> BusinessType:
        """预测单个UAV的业务类型"""
        return predicted_type
    
    def predict_proba(self, features) -> np.ndarray:
        """返回各类别的概率分布"""
```

**技术细节**:
- **输入特征**: 7维特征向量（速率、时延、包大小等）
- **可选模型**: DecisionTree / SVM / MLP / RandomForest / GradientBoosting
- **默认选择**: RandomForest（最佳准确率~100%）
- **警告抑制**: 已添加 `warnings.filterwarnings` 抑制sklearn并行警告

**当前状态**: 
- ✅ 实验三：保留并正常加载
- ❌ 实验四：已剥离（提升速度）

---

### 5️⃣ **environment.py** - 网络仿真环境 ⭐⭐⭐

**地位**: 整个系统的"物理世界模拟器"

```python
class NetworkEnvironmentWithRecognition(EnhancedNetworkEnvironment):
    """带业务识别功能的增强环境"""
    
    def __init__(self, num_bs=8, num_uav=300, 
                 recognition_model=None, scaler=None,
                 seed=42, event_probability=0.05):
        # 初始化基站和UAV
        # 设置信道模型
        # 配置识别模型
        
    def step(self):
        """执行一个仿真步骤"""
        # 1. 更新信道状态
        # 2. 触发移动事件
        # 3. 执行切换决策
        # 4. 分配资源
        # 5. 计算满意度
        
    def get_state_statistics(self) -> dict:
        """获取当前网络统计（17+项指标）"""
        return {
            'total_load': ...,           # 总负载
            'avg_satisfaction': ...,     # 平均满足率
            'connected_count': ...,      # 连接数
            'handover_count': ...,       # 切换次数
            'load_variance': ...,        # 负载方差
            'avg_sinr': ...,             # 平均SINR
            # ... 更多指标
        }
    
    def get_business_type_stats(self) -> dict:
        """按业务类型分组统计（新增！）"""
        return {bt: satisfaction_stats for bt in BusinessType}
```

**关键方法**:

| 方法 | 功能 | 返回值 |
|-----|------|--------|
| `step()` | 推进仿真一步 | None |
| `reset()` | 重置环境状态 | None |
| `get_state_statistics()` | 收集17+项性能指标 | dict |
| `get_business_type_stats()` | 分业务满意度统计 | dict |
| `perform_recognition(uav_id)` | 识别UAV业务类型 | BusinessType |

**环境规模配置**:

| 参数 | 实验3 | 实验4各场景 |
|------|-------|------------|
| 基站数量 | 8 | 8 |
| UAV数量 | 300 | 300-500 |
| 仿真步数 | 350 | 350 |
| 事件概率 | 0.05 | 0.05 |

---

### 6️⃣ **algorithms.py** - 切换算法实现 ⭐⭐⭐

**地位**: 系统的"决策大脑"

#### **传统算法 (IntegratedHandoverAlgorithm)**

```python
class IntegratedHandoverAlgorithm:
    """传统RSSI阈值切换算法"""
    
    def run_step(self):
        for uav in env.uavs.values():
            current_rssi = uav.sinr_db
            if current_rssi < RSSI_THRESHOLD:  # 固定阈值
                target_bs = select_best_bs(uav)
                perform_handover(uav, target_bs)
```

**特点**:
- ❌ 固定阈值（不适应动态变化）
- ❌ 无负载均衡考虑
- ❌ 无业务差异化处理
- ✅ 延迟极低（0.001ms）

---

#### **增强算法 (EnhancedHandoverAlgorithm)** 

```python
class EnhancedHandoverAlgorithm:
    """增强版切换算法（5大模块）"""
    
    def __init__(self, env):
        self.dynamic_threshold = DynamicThresholdModule()
        self.business_weights = BusinessWeightModule()
        self.epsilon_greedy = EpsilonGreedyExplorer()
        self.load_balancer = LoadBalancer()
        self.recognition_feedback = RecognitionFeedbackModule()
    
    def run_step(self, enable_load_balancing=True):
        for uav in env.uavs.values():
            # 1. 动态阈值调整
            threshold = self.dynamic_threshold.adjust(uav)
            
            # 2. 业务权重排序
            priority = self.business_weights.get_priority(uav)
            
            # 3. ε-贪婪探索
            if random() < self.epsilon:
                action = explore()
            else:
                action = exploit(threshold, priority)
            
            # 4. 负载均衡（可选）
            if enable_load_balancing:
                self.load_balancer.rebalance(env)
            
            # 5. 执行切换决策
            execute_decision(uav, action)
```

**五大增强模块**:

| 模块 | 功能 | 性能提升 |
|-----|------|---------|
| **动态阈值** | 自适应调整切换触发条件 | 切换成功率 73%→89% |
| **业务权重** | 高优先级业务优先保障 | 关键业务满足率→100% |
| **ε-贪婪** | 平衡探索与利用 | 整体满足率+4.8% |
| **负载均衡** | 均衡基站间负载 | 负载方差降低95% |
| **识别反馈** | 利用识别结果优化决策 | 综合性能提升 |

---

### 7️⃣ **mappo_agent_v2.py** - MAPPO智能体 ⭐⭐

**地位**: 多智能体强化学习的核心实现

```python
class MAPPOAgentV2:
    """MAPPO智能体 V2优化版"""
    
    def __init__(self, config: MAPPOConfig):
        self.actor = ActorNetwork(obs_dim, act_dim)   # 策略网络
        self.critic = CriticNetwork(obs_dim)          # 价值网络
        
    def select_actions(self, obs_dict, deterministic=False):
        """为所有UAV选择动作"""
        actions = {}
        for uav_id, obs in obs_dict.items():
            action, log_prob = self.actor.sample(obs)
            actions[uav_id] = action
        return actions, log_probs
    
    def update(self, rollouts):
        """PPO更新（集中式训练）"""
        # 1. 计算优势函数 (GAE)
        advantages = compute_gae(rollouts)
        
        # 2. 更新Actor（多轮mini-batch）
        for epoch in range(PPO_EPOCHS):
            actor_loss = compute_policy_loss(advantages)
            self.actor.optimizer.step()
        
        # 3. 更新Critic
        critic_loss = compute_value_loss(returns)
        self.critic.optimizer.step()
```

**网络架构**:
```
Actor Network (策略网络):
  Input: 观测向量 (obs_dim=28维)
  ↓ FC(128) → ReLU
  ↓ FC(128) → ReLU  
  Output: 动作概率分布 (act_dim=9)

Critic Network (价值网络):
  Input: 观测向量 (obs_dim=28维)
  ↓ FC(128) → ReLU
  ↓ FC(128) → ReLU
  Output: 状态价值 V(s) (标量)
```

**训练配置**:
- **算法**: PPO (Proximal Policy Optimization)
- **框架**: MAPPO (Multi-Agent PPO)
- **训练轮数**: 500 episodes
- **每轮步数**: 350 steps × 300 UAVs
- **优化器**: Adam (lr=3e-4)
- **Clip范围**: 0.2
- **GAE lambda**: 0.95

---

### 8️⃣ **mappo_environment.py** - MAPPO专用环境 ⭐⭐

**地位**: 为RL训练设计的接口适配层

```python
class MAPPOEnvironment:
    """MAPPO专用的环境包装器"""
    
    def __init__(self, num_bs=8, num_uav=300, num_steps=350):
        self.env = EnhancedNetworkEnvironment(...)
        self.num_agents = num_uav
        self.obs_dim = 28  # 观测空间维度
        self.act_dim = 9   # 动作空间大小
    
    def reset(self):
        """重置环境，返回初始观测"""
        self.env.reset()
        return self.get_all_obs()
    
    def step(self, actions):
        """执行动作，返回下一状态和奖励"""
        # 1. 解析actions字典
        # 2. 为每个UAV执行切换决策
        # 3. 推进环境一步
        # 4. 计算奖励
        # 5. 返回 (obs, rewards, dones, infos)
        return obs_dict, global_state, rewards, team_reward, done, info
    
    def get_all_obs(self):
        """获取所有UAV的观测"""
        return {uid: self.get_obs(uid) for uid in range(self.num_agents)}
    
    def get_obs(self, uav_id):
        """单个UAV的观测向量（28维）"""
        features = [
            sinr,              # 当前SINR
            load_ratio,        # 目标基站负载
            satisfaction,      # 当前满意度
            business_type,     # 业务类型（one-hot）
            connected_bs_id,   # 连接的基站ID
            neighbor_sinrs,    # 邻居基站SINR（top-3）
            # ... 共28维
        ]
        return np.array(features)
```

**观测空间设计** (28维):

| 特征组 | 维度 | 说明 |
|-------|------|------|
| 当前状态 | 5 | SINR、负载、满意度、连接BS、业务类型 |
| 邻居信息 | 12 | Top-3邻居基站的SINR和负载 |
| 历史信息 | 6 | 过去3步的平均SINR和满意度 |
| 全局信息 | 5 | 全局平均负载、最大负载等 |

**动作空间** (9个离散动作):

| 动作ID | 含义 |
|-------|------|
| 0 | 保持连接（不切换）|
| 1-8 | 切换到基站1-8 |

---

### 9️⃣ **reward_functions.py** - 奖励函数设计 ⭐⭐⭐

**地位**: 引导MAPPO学习的"目标函数"

```python
def compute_mappo_reward(uav, env, config: MAPPOConfig) -> dict:
    """
    计算单个UAV的综合奖励（6个子奖励加权组合）
    
    Returns:
        dict: {
            'total_reward': float,      # 总奖励
            'satisfaction_reward': ..., # 满足率奖励
            'connection_reward': ...,   # 连接保持奖励
            'switch_quality_reward':...,# 切换质量奖励
            'load_balance_reward': ..., # 负载均衡奖励
            'business_priority_reward':.,# 业务优先级奖励
            'penalty': ...              # 惩罚项
        }
    """
```

**奖励组成（V21版本）**:

| 子奖励 | 权重范围 | 作用 |
|-------|---------|------|
| **satisfaction_reward** | 1.0~2.0 | 鼓励提高满足率 |
| **connection_reward** | 0.5~1.5 | 惩罚断连 |
| **switch_quality_reward** | 0.3~1.0 | 鼓励高质量切换 |
| **load_balance_reward** | 0.1~0.5 | 鼓励负载均衡 |
| **business_priority_reward** | 0.2~0.8 | 保护高优先级业务 |
| **penalty** | -0.5~-2.0 | 惩罚频繁无效切换 |

**设计原则**:
- ✅ 稀疏奖励 → 密集奖励（每步都有信号）
- ✅ 单目标 → 多目标（平衡多个指标）
- ✅ 固定权重 → 自适应权重（根据场景动态调整）

---

### 🔟 **experiments.py** - 实验管理核心 ⭐⭐⭐⭐⭐

**地位**: 最重要！整个项目的"指挥中心"，协调所有模块

```python
# ============================================================
# 【实验一】识别准确率影响分析
# ============================================================
class Experiment1:
    @staticmethod
    def run(recognition_model, scaler):
        """
        测试不同识别准确率下的系统性能
        
        准确率级别: 100%, 90%, 80%, 60%, 33%(随机)
        重复次数: 每个级别10次
        输出: 识别准确率→性能损失的映射关系
        """

# ============================================================
# 【实验二】增强算法机制有效性验证
# ============================================================
class Experiment2:
    @staticmethod
    def run():
        """
        逐步添加增强模块，验证每个模块的贡献
        
        测试组合:
        1. traditional (基准)
        2. + dynamic_threshold
        3. + business_weights
        4. + epsilon_greedy
        5. + load_balance
        6. + adaptive_recognition (完整增强算法)
        
        输出: 各模块的性能提升百分比
        """

# ============================================================
# 【实验三】三算法全面对比（单场景）
# ============================================================
class Experiment3:
    METRICS = {
        'avg_satisfaction': '整体满足率',
        'critical_satisfaction': '关键业务满足率',
        'connected_ratio': '连接保持率',
        'total_throughput': '吞吐量',
        'load_variance': '负载方差',
        'avg_switching_latency_ms': '切换延迟',
        # ... 共17项指标
    }
    
    @staticmethod
    def run(recognition_model, scaler, num_steps=350, repeats=10,
             include_mappo=False, use_cache=False):
        """
        三算法对比: Enhanced vs Traditional vs MAPPO
        
        场景: 单一场景（8BS × 300UAV）
        重复: 10次（确保统计显著性）
        
        数据保存:
        - AUTO-SAVE: 每轮MAPPO完成后立即保存
        - FINAL-SAVE: 绘图前保存完整summary
        - save_experiment_data(): 标准化导出
        """

# ============================================================
# 【实验四】五场景泛化能力验证
# ============================================================
class Experiment4:
    SCENARIOS = {
        'agriculture': {
            'name': '农业植保',
            'num_uav': 350,
            'seeds': [30051, 30045, 37436, 35834, 39774],
        },
        'smart_city': {...},      # 智慧城市监控 (400架)
        'industrial_inspection': {...},  # 工业巡检 (300架)
        'emergency_rescue': {...},       # 应急救援 (300架)
        'logistics_delivery': {...},     # 物流配送 (500架)
    }
    
    @staticmethod
    def run(num_steps=350, repeats=5, include_mappo=True,
             use_cache=False, no_cache=False):
        """
        五场景泛化测试 + 三算法对比
        
        特色功能:
        - 场景特定种子（每个场景独立5个种子）
        - 缓存模式（跳过已完成的算法）
        - no-cache模式（强制重新运行）
        - FINAL-SAVE-2: 保存完整数据到exp4_data.json
        
        输出: 5场景 × 3算法 × 17指标的完整数据矩阵
        """
```

**实验四数据流**:

```
┌─────────────────────────────────────────────────────────┐
│  Experiment4.run()                                       │
│    ↓                                                     │
│  [循环] for scenario in SCENARIOS: (5个场景)            │
│    ↓                                                     │
│  [循环] for rep in range(repeats): (5次重复)            │
│    ↓                                                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. 运行增强算法 (EnhancedHandoverAlgorithm)      │    │
│  │    → enh_stats (17项指标 + business_stats)      │    │
│  │                                                 │    │
│  │ 2. 运行传统算法 (IntegratedHandoverAlgorithm)    │    │
│  │    → trad_stats (17项指标 + business_stats)     │    │
│  │                                                 │    │
│  │ 3. 运行MAPPO (mappo_agent_v2 + mappo_env)       │    │
│  │    → mappo_stats (17项指标)                     │    │
│  └─────────────────────────────────────────────────┘    │
│    ↓                                                     │
│  results[scenario][algo].append(stats)                   │
│    ↓                                                     │
│  [汇总] _summarize() → summary dict                     │
│    ↓                                                     │
│  [保存] FINAL-SAVE-2 → exp4_data.json                   │
│    ↓                                                     │
│  [绘图] plot_exp4_figures.py → 6张图表                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 数据收集指标体系（17项核心指标）

### **满意度类 (4项)**

| 指标名 | 键名 | 说明 | 取值范围 |
|-------|------|------|---------|
| 整体满足率 | `avg_satisfaction` | 所有UAV平均满足率 | [0,1] |
| 关键业务满足率 | `critical_satisfaction` | 高优先级业务满足率 | [0,1] |
| 加权满足率 | `weighted_satisfaction` | 优先级加权的满足率 | [0,priority] |
| 连接保持率 | `connected_ratio` | 有连接的UAV占比 | [0,1] 或 % |

### **切换性能类 (5项)**

| 指标名 | 键名 | 说明 | 单位 |
|-------|------|------|------|
| 切换成功率 | `handover_success_rate` | 成功切换/尝试切换 | % |
| 平均切换延迟 | `avg_switching_latency_ms` | 切换决策耗时 | ms |
| 最大切换延迟 | `max_switching_latency_ms` | 最慢的一次切换 | ms |
| 平均决策时间 | `avg_decision_time_ms` | 算法决策耗时 | ms |
| 错失机会率 | `missed_opportunity_rate` | 未抓住的切换时机 | % |

### **网络性能类 (4项)**

| 指标名 | 键名 | 说明 | 单位 |
|-------|------|------|------|
| 系统吞吐量 | `total_throughput` | 所有UAV速率之和 | Mbps |
| 负载方差 | `load_variance` | 基站间负载差异 | 越小越好 |
| 平均SINR | `avg_sinr` | 信号干扰噪声比 | dB |
| 总负载 | `total_load` | 基站总负载量 | Mbps |

### **其他指标 (4项)**

| 指标名 | 键名 | 说明 |
|-------|------|------|
| 迁移成功率 | `migration_success_rate` | 成功迁移比例 |
| 时延满足率 | `latency_satisfaction` | 时延达标的比例 |
| 速率满足率 | `rate_satisfaction` | 速率达标的比例 |
| 分业务满足率 | `business_stats` | 按业务类型分组的统计 |

---

## 🎨 可视化系统

### **实验四图表（最新版本 - 6张）**

| 图表 | 文件名 | 内容 | 尺寸 |
|-----|--------|------|------|
| **1. 整体满足率** | `exp4_satisfaction_comparison.png` | 三算法满足率对比 | 13×7英寸 |
| **2. 关键业务满足率** | `exp4_critical_satisfaction.png` | 关键业务表现 | 13×7英寸 |
| **3. 连接保持率** | `exp4_connected_ratio.png` | 系统稳定性 | 13×7英寸 |
| **4. 吞吐量** | `exp4_throughput_comparison.png` | 系统容量 | **15×8英寸** (大尺寸) |
| **5. 平均SINR** | `exp4_sinr_comparison.png` | 信号质量 | 13×7英寸 |
| **6. 负载方差** | `exp4_load_variance.png` | 负载均衡度 | 13×7英寸 |

**绘图脚本调用流程**:
```python
# plot_exp4_figures.py
def plot_combined_exp4_figures(data):
    data = load_exp4_data()  # 从exp4_data.json加载
    
    fig_paths = []
    fig_paths.append(plot_1_satisfaction(data))      # 图1
    fig_paths.append(plot_2_critical_satisfaction(data))  # 图2
    fig_paths.append(plot_3_connected_ratio(data))    # 图3
    fig_paths.append(plot_4_throughput(data))         # 图4 (大尺寸)
    fig_paths.append(plot_5_sinr(data))               # 图5
    fig_paths.append(plot_7_load_variance(data))      # 图6
    
    return fig_paths  # 返回所有图表路径
```

---

## 🔄 典型工作流程

### **场景1: 从零开始运行全部实验**

```bash
# Step 1: 训练MAPPO模型（如果还没有）
.\venv\Scripts\python.exe main.py --exp mappo
# 预计时间: 8-12小时

# Step 2: 运行实验3（三算法对比，单场景）
.\venv\Scripts\python.exe main.py --exp 3 --include-mappo --no-cache
# 预计时间: 14小时（传统/增强）+ 4小时（MAPPO）

# Step 3: 运行实验4（五场景泛化）
.\venv\Scripts\python.exe main.py --exp 4 --include-mappo --no-cache
# 预计时间: 37小时（传统/增强）+ 20小时（MAPPO）

# 总计: ~75小时（约3天）
```

### **场景2: 只重新跑MAPPO（推荐！使用缓存）**

```bash
# 已经有传统/增强数据，只需更新MAPPO
.\venv\Scripts\python.exe main.py --exp 3 4 --include-mappo --use-cache
# 预计时间: ~24小时（节省51小时！）
```

### **场景3: 快速调试**

```bash
# 小规模快速验证
.\venv\Scripts\python.exe main.py --exp mappo --small
# UAV: 300→128, BS: 8→3, Steps: 350→50, Episodes: 500→100
# 预计时间: 30分钟
```

---

## 📝 代码规范与约定

### **命名规范**

| 类型 | 规范 | 示例 |
|-----|------|------|
| 文件名 | 小写+下划线 | `reward_functions.py` |
| 类名 | 大驼峰 | `EnhancedHandoverAlgorithm` |
| 函数/方法 | 小写下划线 | `compute_satisfaction()` |
| 变量 | 小写下划线 | `total_throughput` |
| 常量 | 大写蛇形 | `GLOBAL_SEED` |
| 配置键 | 小写下划线 | `handover_success_rate` |

### **注释规范**

```python
# 单行注释: 简洁说明作用
enh_stats['connected_ratio'] = connected_count / num_uav

# [TAG] 重要标记注释
# [V27] 版本号标记
# [FIX] Bug修复标记
# [NEW] 新增功能标记
# [TODO] 待办事项

# 模块/类/函数文档字符串（必须包含）
def function_name(param1, param2):
    """
    函数简述（一行）
    
    详细说明（多行，解释设计思路）
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
        
    Returns:
        返回值说明
        
    Example:
        >>> result = function_name(arg1, arg2)
    """
```

### **错误处理模式**

```python
# 安全计算辅助函数（防止空列表崩溃）
def _safe_mean(data, default=0.0):
    return np.mean(data) if len(data) > 0 else default

# 自动保存机制（防止长时间运行崩溃丢失数据）
try:
    with open(save_path, 'w') as f:
        json.dump(data, f)
    print(f"[OK] 数据已保存")
except Exception as e:
    print(f"[WARN] 保存失败: {e}")
    # 不中断主流程，继续执行

# 绘图异常捕获（绘图失败不影响实验结果）
try:
    from plot_exp4_figures import plot_combined_exp4_figures
    fig_paths = plot_combined_exp4_figures(data)
except Exception as vis_err:
    print(f"[WARN] 图表生成失败: {vis_err}")
    # 回退方案或跳过
```

---

## 🔍 关键代码位置速查

| 需要查找的内容 | 文件 | 行号范围 | 关键词 |
|--------------|------|---------|--------|
| **修改全局种子** | `config.py` | L30-50 | `GLOBAL_SEED = 30042` |
| **修改实验四种子** | `experiments.py` | L2400-2450 | `SCENARIOS = {...}` |
| **修改MAPPO超参数** | `config.py` | L200-400 | `class MAPPOConfig` |
| **修改奖励函数权重** | `reward_functions.py` | L50-150 | `compute_mappo_reward` |
| **添加新的评价指标** | `experiments.py` | L1476-1520 | `_summarize()` |
| **修改Y轴范围** | `plot_exp4_figures.py` | L430-500 | `ylim=(...)` |
| **修改图表颜色** | `plot_exp4_figures.py` | L30-80 | `ALGORITHM_COLORS` |
| **查看数据保存逻辑** | `experiments.py` | L2840-2880 | `FINAL-SAVE-2` |
| **查看缓存模式判断** | `main.py` | L200-220 | `--use-cache` |

---

## ⚠️ 常见问题排查

### **Q1: sklearn警告不断弹出？**
✅ **已修复**: `recognition.py:L174` 添加了警告抑制
```python
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
```

### **Q2: 实验运行一半崩溃，数据丢失？**
✅ **已防护**: 三层自动保存机制
- AUTO-SAVE: 每轮完成立即保存
- FINAL-SAVE: 绘图前完整保存
- 标准导出: `save_experiment_data()`

### **Q3: 图表显示不全或柱子太高/太低？**
✅ **已优化**: Y轴范围可调
- 关键业务满足率: `ylim=(0.80, 1.01)`
- 吞吐量: `ylim=(0, 13000)` (大尺寸15×8)
- SINR: `ylim=(0, 30)`

### **Q4: MAPPO总是选择action=0（不切换）？**
⚠️ **可能原因**:
1. 模型未充分训练（<100 episodes）
2. 奖励函数设计不合理（惩罚太重）
3. 探索不足（epsilon太小）

**解决方案**: 检查训练日志中的 `attempts` 和 `success` 字段

### **Q5: 内存不足(OOM)？**
💡 **优化建议**:
1. 减少 `num_uav`: 300→200
2. 减少 `repeats`: 10→5
3. 使用 `--small` 模式调试

---

## 📚 扩展阅读

### **必读文件（按重要性排序）**

1. **📖 [main.py](main.py)** - 了解如何运行系统
2. **📖 [experiments.py](uav_system/experiments.py)** - 理解实验设计
3. **📖 [config.py](uav_system/config.py)** - 掌握配置系统
4. **📖 [satisfaction.py](uav_system/satisfaction.py)** - 理解评价标准
5. **📖 [algorithms.py](uav_system/algorithms.py)** - 学习算法实现
6. **📖 [mappo_agent_v2.py](uav_system/mappo_agent_v2.py)** - 深入MAPPO原理

### **相关文档**

- [快速学习指南](docs/快速学习指南.md) - 入门教程
- [论文全文](docs/论文全文.md) - 学术背景
- [CURRICULUM_GUIDE.md](CURRICULUM_GUIDE.md) - 课程学习路径
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - 实现细节

---

## 🎓 学习路径建议

### **初学者（了解系统）**
```
1. 阅读 main.py → 知道怎么运行
2. 阅读 config.py → 理解配置项
3. 阅读 business.py → 了解业务模型
4. 运行 `--small` 模式 → 亲身体验
```

### **进阶者（理解原理）**
```
1. 阅读 satisfaction.py → 理解评价体系
2. 阅读 algorithms.py → 学习算法设计
3. 阅读 environment.py → 掌握仿真环境
4. 查看 experiment_results/ → 分析真实数据
```

### **高级者（改进系统）**
```
1. 阅读 reward_functions.py → 优化奖励设计
2. 阅读 mappo_agent_v2.py → 改进网络结构
3. 阅读 experiments.py → 设计新实验
4. 修改 config.py → 调参实验
```

---

## ✅ 版本更新日志

### **v2.0 (2026-05-11) - 当前版本**
- ✅ 完成全部实验（实验1-4）
- ✅ MAPPO模型训练完成（500 episodes）
- ✅ 实验四扩展至5个场景
- ✅ 可视化系统升级（6张专业图表）
- ✅ 添加分业务满足率统计
- ✅ 修复sklearn警告抑制
- ✅ 优化数据保存机制（三层保护）

### **v1.5 (2026-05-09)**
- ✅ 集成MAPPO三算法对比
- ✅ 添加17项完整评估指标
- ✅ 实现统计显著性检验

### **v1.0 (2026-04-20)**
- ✅ 初始版本
- ✅ 实现基础框架（实验1-4）

---

**🎉 恭喜！你现在已经完全掌握了整个系统的架构！**

下一步：
- 🚀 运行实验：`python main.py --exp 3 --include-mappo --use-cache`
- 📊 查看结果：`experiment_results/latest_figures/`
- 🔬 深入代码：从感兴趣的模块开始阅读
