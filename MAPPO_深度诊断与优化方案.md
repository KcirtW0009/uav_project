# BA-MAPPO 算法深度诊断与优化方案

> **编制日期**: 2026年4月5日
> **依据**: `mappo_agent.py` (1653行), `qmix_environment.py` (769行), `experiments_mappo.py` (1154行), MAPPO_系统性分析报告
> **说明**: 实验结果txt文件为空(0 bytes)，本诊断完全基于代码实现审查与分析报告交叉验证

---

## 一、Phase 1 训练问题深度分析

### 1.1 算法超参数敏感性分析

#### 1.1.1 当前超参数配置审查

基于 `experiments_mappo.py:256-281` 的实际配置：

| 参数 | 当前值 | 代码位置 | 问题评级 |
|------|--------|---------|---------|
| actor_lr | 1e-4 | `experiments_mappo.py:128` | ✅ 合理 |
| critic_lr | 3e-4 | `experiments_mappo.py:128` | ✅ 合理（3x actor） |
| gamma | 0.99 | `experiments_mappo.py:266` | ⚠️ 偏高 |
| gae_lambda | 0.95 | `experiments_mappo.py:267` | ✅ 合理 |
| clip_epsilon | 0.2 | `experiments_mappo.py:268` | ✅ 标准值 |
| entropy_coef | 0.02 | `experiments_mappo.py:269` | ⚠️ 偏低 |
| value_coef | 0.5 | `experiments_mappo.py:270` | ✅ 标准值 |
| num_epochs | 3 | `experiments_mappo.py:272` | ⚠️ 偏低 |
| batch_size | 32 | `experiments_mappo.py:273` | ⚠️ 偏小 |
| rollout_length | 150 | `experiments_mappo.py:271` | ⚠️ 与num_steps耦合 |
| max_grad_norm | 2.0 | `mappo_agent.py:911` | ✅ 合理 |
| warmup_steps | 50 | `mappo_agent.py:998` | ⚠️ 占比仅5% |

#### 1.1.2 关键超参数问题诊断

**问题1: γ=0.99 导致长期依赖过强**

`mappo_agent.py:931` 中 `self.gamma = 0.99`，在100步episode中：
- 第50步的奖励权重 = 0.99^50 ≈ 0.605（仅衰减40%）
- 第90步的奖励权重 = 0.99^90 ≈ 0.405（仍保留40%）
- **后果**: Critic需准确估计长期价值，但当前奖励信号噪声大，导致价值函数偏差放大

**建议**: γ=0.95，在100步episode中：
- 第50步奖励权重 = 0.95^50 ≈ 0.077（适度衰减）
- 更关注中期(10-30步)的满意度变化趋势

**问题2: entropy_coef=0.02 抑制了探索**

`mappo_agent.py:1353` 中 `entropy_loss = -entropy.mean()`，乘以0.02后：
- 熵奖励对总loss贡献极小（<5%）
- 结合 `mappo_agent.py:1143` 的epsilon-greedy探索（0.8→0.1线性衰减），PPO自身的熵正则几乎不起作用
- **关键矛盾**: epsilon-greedy与PPO的随机策略探索是两种冲突的探索机制

**建议**: 移除epsilon-greedy，将entropy_coef提高到0.05-0.08

**问题3: num_epochs=3 数据利用不足**

`experiments_mappo.py:272` 中每个episode仅用3个epoch更新PPO，且 `mappo_agent.py:1346` 的KL early-stop阈值(0.015)可能进一步缩短有效epoch数。

**建议**: num_epochs=5-8，配合更大的KL阈值(0.02-0.03)

#### 1.1.3 超参数敏感性测试矩阵

| 配置编号 | actor_lr | critic_lr | γ | entropy_coef | num_epochs | batch_size | 预期效果 |
|---------|----------|-----------|---|-------------|-----------|-----------|---------|
| C0(当前) | 1e-4 | 3e-4 | 0.99 | 0.02 | 3 | 32 | 基线 |
| C1 | 1e-4 | 3e-4 | 0.95 | 0.05 | 5 | 64 | ★推荐 |
| C2 | 5e-5 | 1.5e-4 | 0.95 | 0.08 | 5 | 64 | 高探索 |
| C3 | 2e-4 | 6e-4 | 0.95 | 0.05 | 8 | 32 | 快速收敛 |
| C4 | 1e-4 | 3e-4 | 0.90 | 0.05 | 5 | 64 | 短期响应 |

### 1.2 状态空间特征重要性评估与维度优化

#### 1.2.1 当前状态空间构成分析

基于 `qmix_environment.py:173-190` 的 `_calc_obs_dim()`:

```
obs_dim = 4 * num_bs + 9 + action_dim + 2
       = 4 * 8 + 9 + 6 + 2 = 49维 (num_bs=8时)
```

**逐项审查**:

| 特征 | 代码位置 | 维度 | 重要性 | 问题 |
|------|---------|------|--------|------|
| SINR向量 | `qmix_environment.py:218-219` | 8 | ★★★★★ | ✅ 核心信号，但归一化方式 `(sinr+10)/40` 固定，未自适应 |
| 基站负载率 | `qmix_environment.py:222` | 8 | ★★★★☆ | ✅ 但缺少"负载趋势"信息 |
| 连接BS one-hot | `qmix_environment.py:225-227` | 8 | ★★★★☆ | ✅ 冗余但有效 |
| 容量需求比 | `qmix_environment.py:230-237` | 8 | ★★★☆☆ | ⚠️ 归一化上限2.0，大容量差异被压缩 |
| 业务类型 one-hot | `qmix_environment.py:240-241` | 3 | ★★★★★ | ✅ 核心差异化特征 |
| 当前满意度 | `qmix_environment.py:244` | 1 | ★★★★★ | ✅ 即时反馈 |
| 连接状态 | `qmix_environment.py:247` | 1 | ★★★★☆ | ✅ |
| 移动速度 | `qmix_environment.py:250` | 1 | ★★☆☆☆ | ⚠️ 速度归一化上限30m/s可能过高 |
| 上次动作 one-hot | `qmix_environment.py:253-255` | 6 | ★☆☆☆☆ | ❌ 冗余：GRU已编码历史 |
| 满意度趋势 | `qmix_environment.py:258-266` | 1 | ★★★☆☆ | ✅ |
| 同类型UAV平均满意度 | `qmix_environment.py:269-273` | 1 | ★★☆☆☆ | ⚠️ 全局信息泄漏，CTDE不合规 |
| 历史满意度 | `qmix_environment.py:276-282` | 3 | ★★★☆☆ | ⚠️ 与GRU历史编码重复 |

#### 1.2.2 状态空间设计缺陷

**缺陷1: 状态平滑机制削弱信号区分度**

`qmix_environment.py:289-299` 使用5步移动平均平滑观测值：
```python
smoothed_obs = np.mean(self._state_history[uav_id], axis=0)
```
- **问题**: SINR的瞬时跳变（切换触发条件）被平滑掉
- **影响**: 策略网络接收到的信号是"模糊"的，无法区分"应该立即切换"和"正在恶化但暂不需要切换"
- **严重度**: ★★★★★（直接削弱切换决策能力）

**缺陷2: 同类型UAV平均满意度违反CTDE原则**

`qmix_environment.py:269-273`:
```python
for other_uid, other_uav in self.env.uavs.items():
    if other_uid != uav_id and other_uav.true_business_type == uav.true_business_type:
        peer_sats.append(other_uav.current_satisfaction)
peer_avg = np.array([np.mean(peer_sats) if peer_sats else 0.0])
```
- **问题**: 执行时需要其他UAV的满意度信息，CTDE架构下不可获取
- **影响**: 训练-执行不一致（train-deploy mismatch）
- **严重度**: ★★★★☆

**缺陷3: 容量需求比归一化信息丢失**

`qmix_environment.py:232-233`:
```python
ratio = min(bs.available_capacity / required, 2.0) / 2.0
```
- **问题**: 当available_capacity远大于required时（如10倍），被截断为1.0
- **影响**: 无法区分"容量充足"和"容量极度充足"

#### 1.2.3 状态空间优化方案

**移除的特征** (减5维):
- 上次动作one-hot (6维) → 依赖GRU历史编码
- 同类型UAV平均满意度 (1维) → 违反CTDE

**修改的特征**:
- 容量需求比: 改用 `log(1 + ratio)` 归一化，保留大容量差异
- SINR归一化: 添加当前SINR与最大SINR的差分特征 (新增1维)

**新增的特征** (加3维):
- 当前基站剩余容量绝对值 (归一化)
- SINR差分: 当前SINR - 历史SINR (趋势强度)
- 切换计数: 当前episode切换次数/最大允许次数

**优化后维度**: 49 - 6 - 1 + 1 + 3 = 46维

### 1.3 奖励函数构造逻辑审查

#### 1.3.1 奖励函数完整代码审查

基于 `qmix_environment.py:564-657` (V8奖励机制):

```python
r_delta = 5.0 * delta_sat                          # 权重5.0
r_value = 2.0 * (predicted_sat - 0.5)              # 权重2.0
r_biz = {0: 0.5/-0.2, 1: 0.3/-0.1, 2: 0.4/-0.15} # 权重0.1-0.5
r_action = {good:3.0, bad:-0.3, neutral:0.8, ...}  # 权重-0.15~3.0
r_connect = 1.0/-2.0/-3.0                          # 权重1.0~-3.0
```

#### 1.3.2 奖励信号问题诊断

**问题1: 满意度预测值(predicted_sat)质量差**

`qmix_environment.py:733-763` 的 `predict_future_satisfaction()`:
```python
trend = (recent_sats[-1] - recent_sats[0]) / len(recent_sats)
predicted_sat = current_sat + trend * steps
```
- **问题**: 仅使用线性外推5步历史数据，预测能力极弱
- **影响**: r_value信号近乎随机噪声，权重2.0×噪声 = 放大噪声
- **证据**: 当SINR突变(如UAV进入基站盲区)时，线性预测完全失效
- **建议**: 降低r_value权重至0.5，或改用更鲁棒的预测方法

**问题2: 业务类型奖励与满意度阈值硬编码**

`qmix_environment.py:603-608`:
```python
if biz_type == 0: r_biz = 0.5 if new_sat > 0.8 else -0.2
elif biz_type == 1: r_biz = 0.3 if new_sat > 0.7 else -0.1
elif biz_type == 2: r_biz = 0.4 if new_sat > 0.75 else -0.15
```
- **问题**: 阈值(0.8/0.7/0.75)是硬编码的，与实际满意度分布不匹配
- **影响**: 如果大部分时间满意度在0.6-0.7区间，r_biz始终为负值
- **建议**: 使用连续函数替代阶跃函数：`r_biz = k * (sat - threshold)`，k为缩放系数

**问题3: 连接状态奖励过度惩罚**

`qmix_environment.py:637-641`:
```python
r_connect = 1.0 if is_connected else -2.0
if not is_connected and was_connected: r_connect -= 1.0  # 额外-1.0
```
- **问题**: 断连惩罚-3.0，但成功切换奖励仅3.0
- **分析**: 在 `qmix_environment.py:526-550` 的切换逻辑中，切换失败有两个路径：
  1. 目标BS容量不足 → 回滚旧BS（不触发断连）
  2. 旧BS也已满 → 断连（触发-3.0惩罚）
- **但**: 在高负载场景下路径2的概率显著上升，导致模型学会"永远不切换"

**问题4: 奖励归一化时机不当**

`qmix_environment.py:660`:
```python
rewards = self._reward_normalizer.normalize(rewards_raw)
```
- **问题**: `RunningNormalizer` 使用EMA更新均值和方差（`qmix_environment.py:42-51`）
- **影响**: 训练初期EMA统计不稳定，归一化后的奖励信号失真
- **更关键**: 在 `mappo_agent.py:176` 的GAE计算中使用的是归一化后的奖励，但 `mappo_agent.py:1131-1132` 的Critic估值使用的是归一化后的观测，两者的尺度不一致

**问题5: 奖励裁剪范围[-5.0, 5.0]可能过窄**

`qmix_environment.py:647`:
```python
r_individual = np.clip(r_individual, -5.0, 5.0)
```
- **问题**: 好的切换(Δsat=+0.2, r_delta=1.0 + r_action=3.0 + r_connect=1.0 + r_biz=0.5 = 5.5)被裁剪
- **建议**: 扩大至[-8.0, 8.0]或使用tanh软裁剪

#### 1.3.3 奖励信号稀疏性与延迟性分析

| 信号 | 频率 | 延迟 | 稀疏性评估 |
|------|------|------|-----------|
| r_delta (满意度变化) | 每步 | 0步 | ✅ 密集且即时 |
| r_value (预测值) | 每步 | 5步 | ⚠️ 伪密集（但噪声大） |
| r_biz (业务奖励) | 每步 | 0步 | ⚠️ 阶跃函数导致梯度稀疏 |
| r_action (动作奖励) | 切换时 | 1步 | ❌ 仅在切换时非零 |
| r_connect (连接状态) | 每步 | 0步 | ✅ 密集但偏向常数 |

**核心问题**: r_action仅在切换时非零，而不切换时r_action=-0.05~-0.15（几乎为零）。
这导致：**如果不切换，奖励主要由r_delta和r_connect决定；如果切换，奖励增加3.0但风险-2.0~-3.0**
→ 模型收敛到"少切换"的保守策略。

### 1.4 训练过程动态监控数据趋势分析

#### 1.4.1 当前监控指标审查

基于 `mappo_agent.py:1421-1432` 和 `experiments_mappo.py:416-419`:

| 监控项 | 代码位置 | 是否记录 | 频率 |
|--------|---------|---------|------|
| actor_loss | `mappo_agent.py:1429` | ✅ | 每episode |
| critic_loss | `mappo_agent.py:1430` | ✅ | 每episode |
| entropy | `mappo_agent.py:1431` | ✅ | 每episode |
| approx_kl | `mappo_agent.py:1326` | ✅ | 每PPO update |
| 策略梯度范数 | ❌ | 未监控 | - |
| 价值函数估计误差 | ❌ | 未监控 | - |
| 优势函数分布 | ❌ | 未监控 | - |
| 探索利用率 | ❌ | 未监控 | - |
| 协调一致性 | ❌ | 未监控 | - |

#### 1.4.2 监控盲区的影响分析

**盲区1: 无法检测"假收敛"**

当前系统报告中提到的 `a_loss=0, c_loss=0, H=0` 现象：
- actor_loss=0 说明策略梯度为零
- entropy=0 说明策略已完全确定性
- **根因**: `mappo_agent.py:1143` 的epsilon-greedy机制在训练后期(epsilon=0.1)仍引入随机性，但这些随机动作的log_prob被记录但未反映在loss中
- **诊断**: PPO更新实际未改变策略参数，模型停留在预训练或初始化状态

**盲区2: 无法判断Critic是否准确**

缺少V(s)与实际return的对比监控。如果Critic严重高估，会导致：
- 优势函数A(s,a) = Q(s,a) - V(s)偏负
- 策略倾向于"不改变当前行为"（因为当前行为的V已被高估）

#### 1.4.3 必须添加的监控代码

```python
# 在 mappo_agent.py 的 train() 方法中添加:
# 1. 策略梯度范数
grad_norms = [p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None]
avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

# 2. 价值函数估计误差 (explained variance)
with torch.no_grad():
    value_pred = self.critic(states_batch, obs_all_batch)
    ev = 1 - torch.var(ret_batch - value_pred) / torch.var(ret_batch)
    explained_variance = ev.item()

# 3. 优势函数统计
adv_mean = adv_batch.mean().item()
adv_std = adv_batch.std().item()

# 4. 动作分布熵的详细分解
entropy_per_action = dist.entropy()
entropy_std = entropy_per_action.std().item()
```

### 1.5 环境交互样本效率与探索-利用平衡评估

#### 1.5.1 探索机制双重冲突分析

代码中存在**两种并行的探索机制**：

**机制1: Epsilon-Greedy** (`mappo_agent.py:1136-1145`)
```python
progress = min(1.0, self._current_train_step / self._total_train_steps)
epsilon = 0.8 * (1 - progress) + 0.1  # 0.8 → 0.1
```
- 探索方式: 完全随机动作（与策略网络无关）
- 问题: 随机动作收集的经验无法有效训练策略网络（因为这些动作不是策略产生的）

**机制2: PPO熵正则** (`mappo_agent.py:1353-1354`)
```python
entropy_loss = -entropy.mean()
ppo_loss = actor_loss + self.entropy_coef * entropy_loss  # coef=0.02
```
- 探索方式: 鼓励策略保持随机性
- 问题: 与epsilon-greedy冲突——当epsilon高时，PPO的熵正则几乎无效果

**机制3: 数据增强噪声** (`mappo_agent.py:1105-1107`)
```python
noise = np.random.normal(0, self.augmentation_noise, obs_batch_norm.shape)
obs_batch_norm = obs_batch_norm + noise
```
- 探索方式: 观测空间扰动
- 问题: augmentation_noise=0.01对49维观测的扰动极小，且每次前向传播都加噪声导致训练/评估不一致

**结论**: 三种探索机制互相干扰，应统一为一种（推荐纯PPO熵正则 + 可选的noisy net）

#### 1.5.2 样本效率评估

| 指标 | 当前值 | 评估 |
|------|--------|------|
| 每episode步数 | 100 | ✅ 合理 |
| rollout后PPO更新epochs | 3 | ⚠️ 偏低 |
| 预训练样本数 | 500 episodes | ⚠️ 偏少 |
| 预训练epochs | 50 | ⚠️ 偏少 |
| Domain Randomization | 容量范围±20% | ✅ 合理 |
| burn-in步数 | 5 | ✅ 合理 |

**关键发现**: `experiments_mappo.py:290` 中预训练仅收集500个episode的示范，每个episode 100步×30 UAV = 15000个样本，但 `mappo_agent.py:1606-1610` 的预训练batch_size=32，50个epoch = 50 × (15000/32) ≈ 23437次梯度更新——可能过拟合示范数据。

---

## 二、Phase 2 性能量化评估

### 2.1 关键性能指标对比体系

基于分析报告中的数据和代码实现，建立完整的5+维评估体系：

| 指标编号 | 指标名称 | 定义 | 采集方式 |
|---------|---------|------|---------|
| KPI-1 | 平均满足率 | mean(all UAV satisfaction) | `experiments_mappo.py:768-769` |
| KPI-2 | 关键业务满足率 | mean(控制信令+视频回传 sat) | `experiments_mappo.py:80-82` |
| KPI-3 | 收敛速度 | 达到95%最终性能的episode数 | `experiments_mappo.py:300-682` |
| KPI-4 | 资源利用率 | avg(各BS实际负载/总容量) | 需新增采集 |
| KPI-5 | 稳定性 | std(多次重复评估的avg_sat) | `experiments_mappo.py:748-773` |
| KPI-6 | 切换效率 | 成功切换次数/尝试切换次数 | `qmix_environment.py:686-689` |
| KPI-7 | 业务公平性 | min(per_biz_sat) - max(per_biz_sat) | `experiments_mappo.py:822` |

### 2.2 已有实验数据量化分析

基于 `MAPPO_系统性分析报告.md` 第189-196行的Phase 2数据：

#### 2.2.1 BA-MAPPO vs 增强算法（核心对比）

| 维度 | BA-MAPPO | 增强算法 | 差异 | 显著性 |
|------|----------|---------|------|--------|
| 平均满足率(UAV=30) | 0.9534 | 0.9349 | +1.98% | p=0.003 |
| 标准差 | 0.0051 | 0.0747 | -93.2% | — |
| 平均奖励 | 182.4 | 156.2 | +16.8% | — |

**定量分析**:
- BA-MAPPO在稳定性上显著优于增强算法（std降低93.2%）
- 但满足率提升仅1.98%，未达到预期（目标>3%）
- **瓶颈定位**: 满足率提升有限的原因是当前场景下增强算法已经表现良好（0.9349），提升空间有限

#### 2.2.2 BA-MAPPO vs best_sinr基线（异常现象）

| 维度 | BA-MAPPO | best_sinr | 差异 |
|------|----------|-----------|------|
| 平均满足率 | 0.9534 | 0.9922 | -3.91% |
| 平均奖励 | 182.4 | 245.8 | -25.8% |

**根因分析**:

1. **best_sinr本质上是每步最优贪心**: 在8基站30UAV场景下（UAV/BS=3.75），资源竞争不激烈，每步选最大SINR的基站几乎是最优策略
2. **BA-MAPPO的多步规划优势未体现**: 当单步贪心已经很好时，多步规划反而引入次优（因为Critic估计不准）
3. **best_sinr无切换代价**: `qmix_environment.py:428-436` 的action=1直接选择最优BS，没有考虑切换中断风险

### 2.3 统计假设检验验证

**当前问题**: 分析报告声称进行了配对t检验，但 `experiments_mappo.py:753-773` 中的评估仅10次重复。

**统计效力分析**:
- n=10, α=0.05, 检测d=0.5(中等效应量)的统计效力仅约0.32
- 要达到0.80效力，需要n≈26

**建议**: 
- 评估重复次数从10增加到至少20
- 报告效应量Cohen's d和置信区间
- 使用Wilcoxon符号秩检验（非参数）替代t检验（应对非正态分布）

### 2.4 消融实验设计

基于代码中的模块开关，设计完整的消融矩阵：

| 实验ID | 配置修改 | 对应代码开关 | 预期影响 | 验证目标 |
|--------|---------|------------|---------|---------|
| ABL-1 | use_biz_heads=False | `mappo_agent.py:914` | -5~10% | BA机制有效性 |
| ABL-2 | use_attention_critic=False | `mappo_agent.py:915` | -3~8% | 注意力Critic有效性 |
| ABL-3 | use_hierarchical=False | `mappo_agent.py:950` | -5~15% | 分层策略有效性 |
| ABL-4 | use_data_augmentation=False | `mappo_agent.py:954` | -2~5% | 数据增强有效性 |
| ABL-5 | use_pretrain=False | `mappo_agent.py:949` | -10~20% | 预训练贡献度 |
| ABL-6 | use_distillation=False | `mappo_agent.py:947` | -1~3% | 策略蒸馏有效性 |
| ABL-7 | gamma=0.90 | `experiments_mappo.py:266` | ±5% | 折扣因子敏感性 |
| ABL-8 | entropy_coef=0.05 | `experiments_mappo.py:269` | ±3% | 熵系数敏感性 |
| ABL-9 | 移除epsilon-greedy | `mappo_agent.py:1136-1145` | ±5~10% | 探索机制对比 |
| ABL-10 | 移除状态平滑 | `qmix_environment.py:289-299` | ±5~8% | 状态平滑影响 |

**实施建议**: ABL-5是最高优先级——如果移除预训练后性能大幅下降，说明当前MAPPO的"学习"能力很弱，主要依赖预训练的模仿。

### 2.5 不同任务复杂度下的性能衰减

基于已有数据(UAV=30/50)和分析报告的预测：

```
复杂度梯度 (UAV/BS比值):
UAV/BS = 3.75 (30/8):  BA-MAPPO=0.9534
UAV/BS = 6.25 (50/8):  BA-MAPPO=0.9500
UAV/BS = 10.0 (80/8):  预测~0.920 (指数衰减拐点)
UAV/BS = 15.0 (120/8): 预测~0.880
UAV/BS = 25.0 (200/8): 预测~0.800
```

**关键发现**: 性能衰减拐点在UAV/BS≈10，这与基站容量设计直接相关。`main.py:144` 中 `bs_capacity_range=(60, 120)`，8个基站总容量≈720，80个UAV平均需求≈80×30=2400（假设每UAV需求30），容量严重不足。

---

## 三、参考基线优化建议

### 3.1 best_sinr基线表现异常的量化分析

#### 3.1.1 性能量化对比

| 场景 | best_sinr | 增强算法 | 传统算法 | BA-MAPPO | best_sinr排名 |
|------|-----------|---------|---------|----------|-------------|
| UAV=30(默认) | **0.9922** | 0.9349 | 0.9339 | 0.9534 | 第1 |
| UAV=50 | **0.98xx** | ~0.93xx | ~0.92xx | 0.9500 | 第1 |

**核心矛盾**: best_sinr作为"每步贪心"策略，在所有场景下都是最优的。

#### 3.1.2 best_sinr表现好的理论原因

**原因1: 资源竞争不充分**

当前容量设计 `bs_capacity_range=(60, 120)`，8基站总容量≈720。
- 30个UAV，假设每UAV需求~20单位 → 总需求~600，负载率≈83%
- 但best_sinr分散了UAV到不同基站，实际负载更均衡
- **关键**: `qmix_environment.py:428-436` 的best_sinr实现没有切换代价，是理想化的贪心

**原因2: SINR与满意度高度线性相关**

在此仿真环境中，SINR是满意度的主导因素（权重最大），因此best_sinr≈"每步最大化满意度"。

**原因3: 缺少切换中断时间建模**

真实网络中，切换有50-200ms中断时间，期间的吞吐量为零。`qmix_environment.py:525-550` 的切换逻辑中，虽然建模了"目标BS容量不足"的失败，但未建模"切换中断期间满意度下降"。

#### 3.1.3 best_sinr基线设计缺陷

| 缺陷 | 代码位置 | 量化影响 |
|------|---------|---------|
| 无切换代价 | `qmix_environment.py:428-436` | 每步都切换也不会被惩罚 |
| 无乒乓切换惩罚 | 无相关代码 | 可能在两个BS间来回切换 |
| 无信令开销 | 无相关代码 | 忽略了切换对网络的负担 |
| 全知信息 | 需要所有BS的SINR | 实际中难以实时获取 |

### 3.2 基线剔除与实验设计调整

#### 3.2.1 新基线体系

| 基线 | 描述 | 类型 | 保留建议 |
|------|------|------|---------|
| 传统算法(3GPP A3) | 工业界标准 | 核心对比 | ✅ 必须保留 |
| 增强算法(本文) | 前序工作 | 直接改进对象 | ✅ 必须保留 |
| stay | 永不切换 | 下界参考 | ✅ 保留（验证切换必要性） |
| best_sinr | 理想贪心 | 上界参考 | ⚠️ 保留但标注"无切换代价" |
| random_bs | 随机选择 | 随机下界 | ✅ 保留 |
| BL-Capacity | 仅考虑容量 | 资源导向对比 | ❌ 新增 |

#### 3.2.2 对照实验变量控制

**当前问题**: `experiments_mappo.py:716-725` 中评估时的seed与训练时不同：
- 训练seed: `GLOBAL_SEED + num_uav * 100`
- 评估seed: `GLOBAL_SEED + num_uav * 200`

**建议**: 评估时使用完全独立的种子集（不与训练种子相关），避免数据泄露。

### 3.3 基线纳入/排除准则

| 准则 | 条件 | 操作 |
|------|------|------|
| 纳入 | 该基线代表一类已知方法 | 保留 |
| 纳入 | 该基线提供性能上/下界 | 保留 |
| 排除 | 该基线使用了MAPPO无法获取的信息 | 标注后保留 |
| 排除 | 该基线在所有场景中都是最优/最差 | 分析原因 |
| 修改 | best_sinr无切换代价 | 添加切换惩罚后重新评估 |

---

## 四、Phase 3 实验设计重构

### 4.1 当前Phase 3覆盖不足分析

基于 `experiments_mappo.py:880-886` 的当前场景设计：

```python
scenarios = {
    'default':               {'num_uav': 50, 'bs_capacity_range': (500, 1000)},
    'smart_city':            {'num_uav': 400, 'bs_capacity_range': (1500, 2400)},
    'industrial_inspection': {'num_uav': 300, 'bs_capacity_range': (1400, 2300)},
    'emergency_rescue':      {'num_uav': 300, 'bs_capacity_range': (900, 1200)},
    'logistics_delivery':    {'num_uav': 500, 'bs_capacity_range': (1200, 2100)},
}
```

**关键缺陷**:
1. **仅5个场景**，缺少异构业务、动态拓扑、资源稀缺等关键场景
2. **MAPPO仅在UAV=50时被评估**（`experiments_mappo.py:909`），其他场景无MAPPO数据
3. **业务分布固定为均匀**，未测试业务偏斜场景
4. **缺少难度梯度设计**，无法分析性能衰减曲线
5. **缺少场景间迁移能力评估**

### 4.2 新八场景测试矩阵

| 场景ID | 名称 | UAV | BS容量 | 业务分布 | 难度 | 核心测试维度 | UAV/BS |
|--------|------|-----|--------|---------|------|------------|--------|
| S1 | 默认均衡 | 50 | 60-120 | 均匀33% | ★★ | 基准性能 | 6.25 |
| S2 | 密集城市 | 80 | 80-160 | 延迟60% | ★★★★ | 高密度+业务偏斜 | 10.0 |
| S3 | 工业巡检 | 60 | 50-100 | 可靠50% | ★★★★ | 可靠性保障 | 7.5 |
| S4 | 应急救援 | 100 | 40-80 | 混合突发 | ★★★★★ | 极端资源竞争 | 12.5 |
| S5 | 物流配送 | 120 | 100-200 | 吞吐70% | ★★★★ | 大规模协调 | 15.0 |
| S6 | 异构业务 | 50 | 60-120 | 控制信令80% | ★★★ | 业务严重偏斜 | 6.25 |
| S7 | 基站故障 | 50 | 60-120 | 均匀 | ★★★★ | 鲁棒性(1-2个BS故障) | 6.25 |
| S8 | 资源稀缺 | 80 | 30-60 | 均匀 | ★★★★★ | 极度竞争 | 10.0 |

### 4.3 场景间迁移策略

**核心问题**: 当前训练使用固定的UAV数量(30/50)，评估使用不同的UAV数量。

**策略1: Zero-Shot迁移**（当前方式）
- 直接用训练好的模型在新场景评估
- 优点: 简单
- 缺点: action_dim可能不同（如果num_bs变化）

**策略2: Fine-Tuning迁移**
- 在新场景上继续训练少量episode（如100 episode）
- 优点: 适应新场景
- 缺点: 需要额外训练时间

**策略3: Multi-Scenario联合训练**（推荐）
- 训练时交替采样不同场景
- 优点: 模型天然具备泛化能力
- 缺点: 训练时间增加

**注意**: 当前代码中action_dim固定为6（`qmix_environment.py:152`），且基站数量固定为8，因此Zero-Shot迁移在UAV数量变化时是可行的（因为每个UAV独立决策，obs_dim中num_bs相关特征维度不变）。

### 4.4 泛化性能评估指标

| 指标 | 公式 | 目标 | 权重 |
|------|------|------|------|
| 场景平均满足率 | mean(sat_i across 8 scenarios) | >0.90 | 0.30 |
| 最差场景满足率 | min(sat_i) | >0.80 | 0.25 |
| 场景间标准差 | std(sat_i) | <0.03 | 0.15 |
| 迁移衰减率 | (sat_train - sat_test) / sat_train | <0.10 | 0.20 |
| 业务公平性 | min_biz_sat - avg_biz_sat | >-0.05 | 0.10 |

---

## 五、MAPPO 算法核心优化方案

### 5.1 网络结构调整

#### 改进1: 移除epsilon-greedy，纯PPO探索

**修改位置**: `mappo_agent.py:1136-1145`

**当前代码问题**:
```python
# mappo_agent.py:1136-1145
if training:
    progress = min(1.0, self._current_train_step / self._total_train_steps)
    epsilon = 0.8 * (1 - progress) + 0.1
```

**修改方案**: 删除epsilon-greedy块，完全依赖PPO的Categorical分布采样。将entropy_coef提高到0.05。

#### 改进2: 禁用状态平滑

**修改位置**: `qmix_environment.py:289-299`

**当前问题**: 5步移动平均平滑了SINR突变信号。

**修改方案**: 将 `use_state_smoothing` 默认设为 `False`。或改为仅对"慢变化特征"平滑（如基站负载），不对SINR平滑。

#### 改进3: 修复分层策略的log_prob计算效率

**修改位置**: `mappo_agent.py:609-628`

**当前问题**: `evaluate_actions()` 中使用for循环逐样本计算log_prob，效率极低。

```python
# mappo_agent.py:609
for i in range(obs.shape[0]):
    action = actions[i].item()
    if action == 0:  # stay
        ...
```

**修改方案**: 向量化计算（batch处理所有样本）。

#### 改进4: 移除CTDE违规特征

**修改位置**: `qmix_environment.py:269-273`

**修改方案**: 移除"同类型UAV平均满意度"，或改为"同类型UAV数量占比"（不泄露具体值）。

### 5.2 奖励函数优化

#### 优化1: 降低r_value权重

**修改位置**: `qmix_environment.py:598`

```python
# 当前: r_value = 2.0 * (predicted_sat - 0.5)
# 修改: r_value = 0.5 * (predicted_sat - 0.5)
```

#### 优化2: 业务奖励连续化

**修改位置**: `qmix_environment.py:603-608`

```python
# 当前: 阶跃函数
if biz_type == 0: r_biz = 0.5 if new_sat > 0.8 else -0.2
# 修改: 连续函数
thresholds = {0: 0.8, 1: 0.7, 2: 0.75}
r_biz = 2.0 * (new_sat - thresholds[biz_type])  # 连续奖励
```

#### 优化3: 扩大奖励裁剪范围

**修改位置**: `qmix_environment.py:647`

```python
# 当前: r_individual = np.clip(r_individual, -5.0, 5.0)
# 修改: r_individual = np.clip(r_individual, -10.0, 10.0)
```

### 5.3 多智能体协调机制优化

**当前状态**: CTDE架构中，智能体间完全无通信。Critic使用Attention聚合全局信息（`mappo_agent.py:785-813`），但Actor仅使用局部观测。

**建议**: 暂不添加显式通信（增加复杂度），而是通过改进Critic的全局信息聚合来间接提升协调：
- 在Critic中添加"区域拥挤度"特征（当前基站附近UAV数量）
- 在全局状态中添加"切换热度"特征（最近N步的切换次数）

### 5.4 分阶段实验参数调整方案

#### 第一阶段: 基础稳定性恢复 (预计1-2天)

| 参数 | 当前值 | 目标值 | 修改位置 |
|------|--------|--------|---------|
| gamma | 0.99 | 0.95 | `experiments_mappo.py:266` |
| entropy_coef | 0.02 | 0.05 | `experiments_mappo.py:269` |
| num_epochs | 3 | 5 | `experiments_mappo.py:272` |
| batch_size | 32 | 64 | `experiments_mappo.py:273` |
| epsilon-greedy | 0.8→0.1 | 移除 | `mappo_agent.py:1136-1145` |
| state_smoothing | True | False | `QMixHandoverEnv.__init__` |
| r_value权重 | 2.0 | 0.5 | `qmix_environment.py:598` |
| 奖励裁剪 | [-5, 5] | [-10, 10] | `qmix_environment.py:647` |
| pretrain_demos | 500 | 1000 | `experiments_mappo.py:290` |
| pretrain_epochs | 50 | 80 | `experiments_mappo.py:292` |

**验收标准**:
- ✅ reward全程>0
- ✅ sat>0.92
- ✅ actor_loss, critic_loss, entropy均为非零
- ✅ 切换成功率>20%

#### 第二阶段: 性能提升 (预计2-3天)

| 参数 | 第一阶段值 | 目标值 | 修改位置 |
|------|----------|--------|---------|
| augmentation_noise | 0.01 | 0.03 | `mappo_agent.py:925` |
| train_episodes | 1000 | 1500 | `main.py:142` |
| early_stop_patience | 200 | 300 | 保持 |
| 业务奖励 | 阶跃 | 连续 | `qmix_environment.py:603-608` |
| 移除peer_avg | 存在 | 移除 | `qmix_environment.py:269-273` |
| 添加SINR差分 | 无 | +1维 | `qmix_environment.py:get_obs()` |
| KL early-stop | 0.015 | 0.025 | `mappo_agent.py:1346` |

**验收标准**:
- ✅ sat>0.97 (UAV=30)
- ✅ sat>0.95 (UAV=50)
- ✅ 超越增强算法>3%

#### 第三阶段: 泛化验证 (预计2天)

使用4.2节的8场景矩阵进行评估。

**验收标准**:
- ✅ 场景平均sat>0.90
- ✅ 最差场景sat>0.80
- ✅ 场景间std<0.03

### 5.5 消融实验执行流程

```
优先级排序（按预期影响）:
1. ABL-5 (移除预训练) — 验证RL学习能力的核心
2. ABL-3 (移除分层策略) — 验证分层决策的贡献
3. ABL-9 (移除epsilon-greedy) — 验证探索机制改进
4. ABL-1 (移除BA heads) — 验证业务感知贡献
5. ABL-10 (移除状态平滑) — 验证状态表示改进
6. ABL-2 (移除Attention Critic) — 验证Critic改进

每组实验: 3种UAV配置 × 10次重复 = 30次评估
总计: 6组 × 30次 = 180次评估
预计时间: 3-4小时 (GPU) / 6-8小时 (CPU)
```

### 5.6 性能提升量化目标

| 指标 | 当前值 | 阶段1目标 | 阶段2目标 | 阶段3目标 | 优先级 |
|------|--------|----------|----------|----------|--------|
| avg_sat(UAV=30) | 0.9534 | >0.93 | >0.97 | >0.97 | P0 |
| avg_sat(UAV=50) | 0.9500 | >0.92 | >0.95 | >0.95 | P0 |
| 训练稳定性(std) | 0.0051 | <0.01 | <0.005 | <0.003 | P1 |
| 收敛速度(ep) | ~534 | <800 | <500 | <400 | P2 |
| 切换成功率 | ~30% | >20% | >40% | >50% | P0 |
| 最差场景sat | ~0.85 | >0.82 | >0.88 | >0.80(8场景) | P0 |

### 5.7 风险控制措施

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 移除epsilon后探索不足 | 高 | 高 | 提高entropy_coef至0.08 |
| gamma降低后短期主义 | 中 | 中 | 监控V(s)趋势 |
| 奖励修改后训练不稳定 | 中 | 高 | 逐步修改，每步验证 |
| 消融实验时间不足 | 中 | 低 | 优先做ABL-5和ABL-3 |

**回退方案**: 如果第一阶段未通过验收，回退到仅修改gamma和entropy_coef，保持其他不变。

### 5.8 后续实验实施计划

```
Day 1-2:  第一阶段实施 + 验证
  - 修改超参数 (gamma, entropy_coef, num_epochs, batch_size)
  - 移除epsilon-greedy
  - 禁用状态平滑
  - 运行训练 + 检查CP-1

Day 3-4:  第一阶段调优
  - 根据训练曲线调整参数
  - 添加监控代码
  - 运行Phase 2评估

Day 5-6:  第二阶段实施
  - 修改奖励函数
  - 优化状态空间
  - 运行消融实验(ABL-5, ABL-3, ABL-9)

Day 7-8:  第二阶段验证
  - 运行Phase 2完整评估
  - 统计检验
  - 检查CP-2

Day 9-10: 第三阶段实施
  - 运行8场景泛化实验
  - 分析场景间迁移能力
  - 检查CP-3

Day 11-12: 报告撰写 + 可视化更新
```

---

## 六、总结

### 核心发现

1. **训练不稳定的根因**: 三重探索机制冲突(epsilon-greedy + PPO熵 + 数据增强噪声) + 状态平滑削弱信号 + 奖励信号噪声大
2. **性能瓶颈**: 当前场景下增强算法已表现良好(~0.935)，提升空间有限；预训练是性能的主要来源而非RL学习
3. **基线设计问题**: best_sinr无切换代价，不应作为核心对比对象
4. **泛化覆盖不足**: 仅5个场景，MAPPO仅在其中1个(UAV=50)被评估

### 最高优先级行动项

1. **移除epsilon-greedy** → 统一为纯PPO探索 (`mappo_agent.py:1136-1145`)
2. **禁用状态平滑** → 恢复SINR突变信号 (`qmix_environment.py:289-299`)
3. **降低gamma至0.95** → 平衡长短项 (`experiments_mappo.py:266`)
4. **运行ABL-5** → 量化预训练的实际贡献

### 预期改善幅度

- 训练稳定性: 显著改善（消除策略梯度为零的问题）
- 满足率: +1~3%（通过修复探索和奖励函数）
- 收敛速度: +20~30%（通过正确的超参数）
- 泛化能力: 需要第三阶段验证

---

**报告结束**
