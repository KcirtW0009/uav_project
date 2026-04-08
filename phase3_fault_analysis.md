# PHASE3 模块故障分析报告

## 故障概述

**模块名称**: PHASE3 - 多场景泛化验证  
**故障类型**: 功能性故障  
**影响范围**: 跨场景模型加载和评估  
**严重程度**: 高  

## 故障现象

1. **模型加载失败**: 在PHASE3阶段无法正确加载训练好的模型
2. **维度不匹配**: 不同场景的obs_dim/state_dim不一致导致错误
3. **动作索引越界**: 动作值超出数组范围
4. **评估中断**: PHASE3无法正常完成评估流程

## 根本原因分析

### 1. 模型路径问题

**问题描述**: 
- 模型文件路径构建逻辑不完善
- 早期停止触发时只有best.pt，没有普通.pt文件
- 路径检查顺序可能导致加载错误的模型

**代码位置**: `experiments_mappo.py:1512-1524`

```python
base_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                         f'mappo_{t_bs}bs_{t_uav}uav.pt')
best_path = base_path.replace('.pt', '_best.pt')

# 优先使用 best 模型
if os.path.exists(best_path):
    model_to_load = best_path
elif os.path.exists(base_path):
    model_to_load = base_path
```

**修复方案**:
```python
# 改进路径检查和加载逻辑
def get_model_path(base_path, best_path):
    """智能选择模型路径"""
    if os.path.exists(best_path):
        # 检查best模型是否完整
        try:
            checkpoint = torch.load(best_path, map_location='cpu')
            if 'actor' in checkpoint and 'critic' in checkpoint:
                return best_path, 'best'
        except:
            pass
    
    if os.path.exists(base_path):
        try:
            checkpoint = torch.load(base_path, map_location='cpu')
            if 'actor' in checkpoint and 'critic' in checkpoint:
                return base_path, 'base'
        except:
            pass
    
    return None, None
```

### 2. 维度不匹配问题

**问题描述**:
- 不同场景的UAV/BS数量不同，导致obs_dim和state_dim不同
- 模型加载时未正确处理维度变化
- 网络结构固定，无法适应不同输入维度

**影响场景**:
- 默认场景: 200 UAV / 4 BS → obs_dim=33, state_dim=19
- 城市监控: 300 UAV / 5 BS → obs_dim=?, state_dim=?
- 工业巡检: 200 UAV / 6 BS → obs_dim=?, state_dim=?

**修复方案**:
1. **动态网络创建**: 根据场景动态创建网络
2. **维度适配层**: 添加维度转换层
3. **特征提取统一化**: 使用统一的特征提取方式

```python
class AdaptiveMAPPOAgent:
    """自适应MAPPO智能体"""
    
    def __init__(self, obs_dim, state_dim, action_dim, ...):
        # 根据实际维度创建网络
        self.actor = FeedForwardActorNetwork(obs_dim, action_dim, ...)
        self.critic = FeedForwardCriticNetwork(state_dim, ...)
    
    def adapt_to_scene(self, new_obs_dim, new_state_dim, new_action_dim):
        """适应新场景"""
        # 保存旧网络权重
        old_actor_state = self.actor.state_dict()
        old_critic_state = self.critic.state_dict()
        
        # 创建新网络
        self.actor = FeedForwardActorNetwork(new_obs_dim, new_action_dim, ...)
        self.critic = FeedForwardCriticNetwork(new_state_dim, ...)
        
        # 尝试迁移权重
        self._migrate_weights(old_actor_state, old_critic_state)
```

### 3. 动作索引越界问题

**问题描述**:
- 动作值范围是 [0, num_bs]，但数组大小是 action_dim
- 当 num_bs > action_dim - 1 时发生越界
- 在get_obs()和统计代码中都有此问题

**修复方案**:
已在 `qmix_environment.py:271-280` 和 `experiments_mappo.py` 中添加边界检查:

```python
# 在get_obs()中
last_action = np.zeros(self.action_dim)
if uav_id in self._last_actions:
    action = self._last_actions[uav_id]
    if action < len(last_action):
        last_action[action] = 1.0
    else:
        last_action[-1] = 1.0  # 映射到最后一个类别

# 在统计代码中
if a < len(ep_per_action):
    ep_per_action[a] += 1
else:
    ep_per_action[-1] += 1
```

### 4. 场景配置不一致

**问题描述**:
- PHASE3场景配置与训练场景不一致
- 容量范围、地图大小等参数可能不同
- 环境初始化逻辑不一致

**当前场景配置**:
```python
scenarios = {
    'default':               {'num_uav': 200, 'num_bs': 4, 'bs_capacity_range': (500, 1000)},
    'smart_city':            {'num_uav': 300, 'num_bs': 5, 'bs_capacity_range': (500, 1000)},
    'industrial_inspection': {'num_uav': 200, 'num_bs': 6, 'bs_capacity_range': (500, 1000)},
    'emergency_rescue':      {'num_uav': 100, 'num_bs': 3, 'bs_capacity_range': (500, 1000)},
    'logistics_delivery':    {'num_uav': 250, 'num_bs': 4, 'bs_capacity_range': (500, 1000)},
}
```

**修复方案**:
1. 统一场景配置参数
2. 添加配置验证
3. 实现场景适配器

## 修复实施计划

### 阶段1: 紧急修复 (已完成)
- [x] 修复动作索引越界问题
- [x] 添加边界检查代码
- [x] 测试环境基本功能

### 阶段2: 模型加载优化 (进行中)
- [ ] 改进模型路径检查逻辑
- [ ] 添加模型完整性验证
- [ ] 实现模型加载错误处理

### 阶段3: 维度适配 (待实施)
- [ ] 设计自适应网络架构
- [ ] 实现维度转换层
- [ ] 测试跨场景迁移

### 阶段4: 场景配置统一 (待实施)
- [ ] 统一场景配置参数
- [ ] 添加配置验证机制
- [ ] 实现场景适配器

## 测试计划

### 单元测试
1. **模型加载测试**: 测试各种模型文件加载场景
2. **维度适配测试**: 测试不同obs_dim/state_dim的适配
3. **动作索引测试**: 测试动作值边界处理
4. **场景配置测试**: 测试场景配置一致性

### 集成测试
1. **PHASE3完整流程测试**: 从训练到评估的完整流程
2. **跨场景迁移测试**: 测试模型在不同场景的泛化能力
3. **性能对比测试**: 对比修复前后的性能差异

### 回归测试
1. **PHASE1训练测试**: 确保训练阶段不受影响
2. **PHASE2评估测试**: 确保评估阶段不受影响
3. **整体流程测试**: 确保整个实验流程正常

## 修复验证

### 验证标准
1. **功能验证**: PHASE3能够正常完成所有场景的评估
2. **性能验证**: 评估结果符合预期，满意度>0.90
3. **稳定性验证**: 无崩溃、无异常、无错误日志
4. **一致性验证**: 多次运行结果一致

### 验证方法
1. **日志分析**: 检查运行日志，确认无错误
2. **结果对比**: 对比修复前后的评估结果
3. **压力测试**: 连续运行多次，检查稳定性
4. **边界测试**: 测试各种边界条件

## 风险评估

### 修复风险
1. **兼容性风险**: 新代码可能与旧代码不兼容
2. **性能风险**: 修复可能引入性能下降
3. **稳定性风险**: 修复可能引入新的不稳定因素

### 缓解措施
1. **代码审查**: 所有修复代码经过严格审查
2. **逐步部署**: 先在小规模场景测试，再全面部署
3. **回滚机制**: 保留旧版本代码，必要时可回滚
4. **监控机制**: 添加详细的日志和监控

## 时间计划

| 阶段 | 任务 | 预计时间 | 状态 |
|-----|------|---------|------|
| 阶段1 | 紧急修复 | 1天 | 已完成 |
| 阶段2 | 模型加载优化 | 1天 | 进行中 |
| 阶段3 | 维度适配 | 2天 | 待实施 |
| 阶段4 | 场景配置统一 | 1天 | 待实施 |
| 测试 | 全面测试 | 2天 | 待实施 |
| **总计** | | **7天** | |

## 附录

### 相关代码文件
- `uav_system/experiments_mappo.py`: PHASE3主要实现
- `uav_system/qmix_environment.py`: 环境实现
- `uav_system/mappo_agent.py`: 智能体实现

### 关键日志位置
- `experiment_results/logs/`: 实验日志
- `experiment_results/mappo_models/`: 模型文件
- `small_experiment_results/`: 小规模测试结果

### 参考文档
- MAPPO算法论文
- PPO算法实现指南
- PyTorch最佳实践

---

**报告日期**: 2026-04-08  
**报告人**: PHASE3故障分析团队  
**版本**: v1.0  
**状态**: 修复中
