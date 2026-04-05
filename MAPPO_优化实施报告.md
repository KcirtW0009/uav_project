# BA-MAPPO 算法优化实施报告

## 实施概要

基于 `MAPPO_深度诊断与优化方案.md` 中提出的第一阶段（基础稳定性恢复）优化建议，对BA-MAPPO算法代码进行了系统性修改。所有修改已完成并通过语法验证。

## 修改文件清单

| 文件 | 修改项数 | 状态 |
|------|---------|------|
| `uav_system/mappo_agent.py` | 1 | ✅ 完成 |
| `uav_system/qmix_environment.py` | 5 | ✅ 完成 |
| `uav_system/experiments_mappo.py` | 7 | ✅ 完成 |

---

## 详细修改记录

### 1. mappo_agent.py - 移除epsilon-greedy探索机制

**位置**: 第1134-1202行 (select_actions方法)  
**问题根因**: epsilon-greedy与PPO的Categorical采样机制冲突，导致策略梯度估计偏差  
**修改内容**:

```python
# 修改前: epsilon-greedy + 随机动作
epsilon = 0.8 * (1 - progress) + 0.1
if np.random.random() < epsilon:
    action = np.random.randint(0, self.action_dim)

# 修改后: 纯PPO Categorical分布采样
action_tensor = dist.sample()
log_probs_dict[uid] = dist.log_prob(action_tensor).item()
action = action_tensor.item()
```

**预期效果**:
- 消除策略梯度估计偏差，使PPO更新方向正确
- 通过entropy_coef=0.05控制探索强度，而非硬编码epsilon
- 分层策略中高层和底层都使用纯概率采样

### 2. qmix_environment.py - 奖励函数优化

#### 2a. 降低r_value权重 (第598行)
```
修改前: r_value = 2.0 * (predicted_sat - 0.5)
修改后: r_value = 0.5 * (predicted_sat - 0.5)
理由: 减少噪声奖励影响，价值预测在训练初期不准确
```

#### 2b. 业务奖励连续化 (第603-608行)
```python
# 修改前: 阶跃函数
r_biz = 0.5 if new_sat > 0.8 else -0.2

# 修改后: 连续函数
r_biz = 2.0 * (new_sat - threshold)
# 延迟敏感型阈值=0.8, 吞吐量型=0.7, 可靠性型=0.75
```
**理由**: 梯度信号更平滑，避免奖励突变导致训练震荡

#### 2c. 扩大奖励裁剪范围 (约第647行)
```
修改前: np.clip(r_individual, -5.0, 5.0)
修改后: np.clip(r_individual, -10.0, 10.0)
理由: 允许更大的奖励信号传递，特别是在切换场景
```

### 3. experiments_mappo.py - 超参数调整

| 参数 | 修改前 | 修改后 | 调整理由 |
|------|--------|--------|---------|
| gamma | 0.99 | **0.95** | 平衡长短项奖励，避免长期预测偏差 |
| entropy_coef | 0.02 | **0.05** | 增强PPO内在探索能力 |
| num_epochs | 3 | **5** | 提高数据利用效率 |
| batch_size | 32 | **64** | 稳定梯度估计 |
| num_demos | 500 | **1000** | 更充分的预训练数据 |
| pretrain_epochs | 50 | **80** | 更充分的预训练 |

### 4. Phase 3模型加载修复

**位置**: 第921-931行 (_phase3_scenarios方法)  
**问题**: 训练时使用分层网络(hierarchical)，加载时默认标准网络 → state_dict键不匹配  
**修复**: 
```python
mappo_agent = MAPPOAgent(
    ...
    use_hierarchical=True,  # 新增：与训练时保持一致
)
```

---

## 语法验证结果

```
[PASS] mappo_agent.py:     语法正确
[PASS] qmix_environment.py: 语法正确
[PASS] experiments_mappo.py: 语法正确
```

---

## 未实施项说明

以下优化项因时间/复杂度考虑，暂未在本轮实施：

| 项目 | 优先级 | 说明 |
|------|--------|------|
| CTDE违规特征移除 | P1 | "同类型UAV平均满意度"特征需确认是否实际存在 |
| SINR差分特征添加 | P1 | 需要重新定义obs_dim，影响面广 |
| 训练监控代码 | P2 | 梯度范数、价值误差等统计，可在验证阶段添加 |
| 双流业务感知网络 | P2 | 架构级改动，需单独测试 |

---

## 运行指令

完成所有修改后，运行实验：

```bash
python main.py --exp mappo
```

## 验收标准（第一阶段目标）

| 指标 | 当前基线 | 目标值 | 验证方法 |
|------|---------|--------|---------|
| 训练稳定性 | reward跌至-169 | reward始终>0 | 观察训练日志 |
| 最终满足率 | 0.985(简单环境) | >0.92(标准环境) | Phase 2评估 |
| a_loss/c_loss/H | 全为0 | 有非零值 | 每30轮打印信息 |
| 切换成功率 | ~30% | >20% | sw(ok)/sw(total) |

---

## 后续步骤

1. **立即执行**: 运行 `python main.py --exp mappo` 验证修改效果
2. **观察重点**:
   - 标准环境训练时reward是否保持正值
   - a_loss, c_loss, H是否有非零输出
   - 是否仍出现早停现象
3. **如遇问题**: 对照本报告检查具体修改项

---

**编制日期**: 2026年4月5日  
**版本**: v1.0 (第一阶段完整版)  
**状态**: 已完成，待用户运行验证
