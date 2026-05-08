# PMSF v3.0 课程学习系统 - 使用指南

## 🎯 系统概述

PMSF v3.0 是基于课程学习（Curriculum Learning）+ 对比学习（Contrastive Learning）的混合策略微调系统，旨在将MAPPO模型的全场景平均满意度从 **79% 提升到 85-88%**。

### 核心创新

1. **课程学习框架**: 由易到难的4阶段训练
   - Phase 0: 强场景巩固 (8 eps)
   - Phase 1: 中等突破 (25 eps)
   - Phase 2: 大规模攻坚 (35 eps)
   - Phase 3: 联合精调 (20 eps)

2. **场景特定奖励塑造**: 针对不同场景优化目标
   - 工业巡检: 连接稳定性优先 (+35%)
   - 智慧城市: 负载均衡优先 (+35%)
   - 物流配送: 切换成功率优先 (+45%)

3. **对比学习辅助监督**: 学习通用切换决策表征
   - InfoNCE损失 (λ=0.1)
   - 正/负样本缓冲区

4. **渐进式UAV规模扩展**: 400→450→500 UAV

---

## 🚀 快速开始

### 前置条件

```bash
# 确保基础模型存在
ls results/mappo_models/mappo_8bs_300uav_best.pt

# 如果不存在，需要先运行主实验获取预训练模型
```

### 运行命令

#### 方式1: 快速测试模式 (推荐首次使用)

```bash
# Quick模式: 压缩版，约2-3小时完成
.\venv\Scripts\python.exe curriculum_learning.py --mode quick
```

**Quick模式配置**:
- Phase 0: 3 episodes (原8)
- Phase 1: 10 episodes (原25)
- Phase 2: 15 episodes (原35)
- Phase 3: 8 episodes (原20)
- 最大迭代次数: 2次/阶段 (原3次)

#### 方式2: 完整训练模式

```bash
# Full模式: 完整版，预计6-10小时完成
.\venv\Scripts\python.exe curriculum_learning.py --mode full
```

#### 方式3: 从指定阶段继续 (断点续训)

```bash
# 从Phase 1开始 (跳过Phase 0)
.\venv\Scripts\python.exe curriculum_learning.py --from-phase 1
```

#### 方式4: 指定自定义模型

```bash
# 使用其他预训练模型
.\venv\Scripts\python.exe curriculum_learning.py --model path/to/your_model.pt
```

---

## 📊 训练流程详解

### 阶段0: 强场景巩固 (Phase 0)

**目标**: 确保农业(95.97%)和应急(90.49%)不下降超过2%

```
配置:
├─ Episodes: 8 (Quick: 3)
├─ 场景: ['agriculture', 'emergency_rescue']
├─ LR因子: 0.8 (保守)
├─ Entropy因子: 0.8 (减少探索)
├─ 目标改进: 0% (维持即可)
└─ 早停耐心: 5 episodes
```

**预期结果**:
- 农业: 95.97% → 94-96%
- 应急: 90.49% → 89-91%

---

### 阶段1: 中等突破 (Phase 1)

**目标**: 工业巡检从67.55%提升到78-82%

```
配置:
├─ Episodes: 25 (Quick: 10)
├─ 场景: ['industrial_inspection'] (单点突破!)
├─ LR因子: 1.0 (标准)
├─ Entropy因子: 1.2 (增加探索)
├─ 目标改进: +8%
├─ 早停耐心: 10 episodes
└─ 奖励权重:
    ├─ connection_stability: 35%
    ├─ handover_success: 40%
    ├─ load_balance: 15%
    └─ satisfaction: 10%
```

**关键策略**:
- 只聚焦一个最弱场景 (工业巡检)
- 高探索率以避免局部最优
- 连接稳定性优先的奖励塑造

**预期结果**:
- 工业巡检: 67.55% → 75-82% (+7-14pp)

---

### 阶段2: 大规模攻坚 (Phase 2)

**目标**: 智慧城市(70.47%)和物流(71.04%)提升到80%+

```
配置:
├─ Episodes: 35 (Quick: 15)
├─ 场景: ['smart_city', 'logistics_delivery']
├─ LR因子: 0.9 (略保守)
├─ Entropy因子: 1.1 (适度探索)
├─ 目标改进: +10%
├─ 早停耐心: 12 episodes
├─ 渐进式UAV: [400, 450, 500] ← 关键!
└─ 奖励权重:
    ├─ 智慧城市: load_balance=35% (负载均衡优先)
    └─ 物流配送: handover_success=45% (切换成功优先)
```

**关键策略**:
- 渐进式UAV规模 (先400再500)
- 不同场景不同优化重点
- 对比学习辅助 (λ=0.1)

**预期结果**:
- 智慧城市: 70.47% → 78-83% (+8-13pp)
- 物流配送: 71.04% → 77-82% (+6-11pp)

---

### 阶段3: 联合精调 (Phase 3)

**目标**: 所有场景联合优化，达到全局最优

```
配置:
├─ Episodes: 20 (Quick: 8)
├─ 场景: 全部5个场景
├─ LR因子: 0.6 (精细调整)
├─ Entropy因子: 0.7 (减少探索)
├─ 目标改进: +3%
├─ 早停耐心: 8 episodes
└─ 策略: 加权采样 (弱场景优先)
```

**关键策略**:
- 低学习率精细调优
- 弱场景有更高采样概率
- 保持强场景性能不下降

**预期结果**:
- 全局平均: 85-88% (从79%提升6-9pp)

---

## 📁 输出文件结构

训练完成后，在 `experiment_results/curriculum_v3_TIMESTAMP/` 目录下会生成:

```
curriculum_v3_20260509_XXXXXX/
├── training_log.txt              # 详细训练日志
├── final_result.json             # 最终结果 (JSON格式)
├── phase_0_consolidation_best.pt # Phase 0最佳模型
├── phase_1_medium_breakthrough_best.pt
├── phase_2_large_scale_best.pt
├── phase_3_joint_finetune_best.pt
└── curriculum_final.pt           # 最终模型 (推荐使用)
```

---

## 🔍 监控训练进度

### 实时日志输出示例

```
======================================================================
  PMSF v3.0 - 课程学习 + 对比学习 混合策略
======================================================================

  [*] 训练配置:
     基础模型: mappo_8bs_300uav_best.pt
     总阶段数: 4
     对比学习: 启用 (λ=0.1)
     奖励塑造: 启用

───────────────────────────────────────────────────────
 ▶ Phase: 强场景巩固
    Episodes: 8 | Scenarios: 2
    Target: 0.0% improvement
───────────────────────────────────────────────────────

  [CUR] Ep   1/ 8 ( 12.5%) | AGR | Rwd:  12.34 | A-L:-0.0023 | C-L:0.2341 | ...
  [CUR] Ep   2/ 8 ( 25.0%) | EMG | Rwd:  15.67 | A-L:+0.0012 | C-L:0.1987 | ...
  ...

  [EVAL] 农业植保      : 0.9543 ± 0.0021
  [EVAL] 应急救援      : 0.9012 ± 0.0034

[PHASE SUMMARY] 强场景巩固
   Episodes完成: 8
   耗时: 23.5分钟
   全局平均: 0.9278

   各场景得分:
      农业植保      : 0.9543 (基线0.9597, -0.54%)
      应急救援      : 0.9012 (基线0.9049, -0.41%)

[ADVANCE] 进入下一阶段...
```

### 关键指标解读

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| `Rwd` | 缩放后奖励 | >5.0 为良好 |
| `A-L` | Actor Loss | 接近0为收敛 |
| `C-L` | Critic Loss | <0.5为健康 |
| `Ent` | 熵系数 | 0.5-1.5为正常 |
| `Sat` | 平均满意度 | >0.7为目标 |

---

## ⚠️ 故障排除

### 问题1: 内存不足 (OOM)

**症状**: `CUDA out of memory` 或程序崩溃

**解决方案**:
```bash
# 使用Quick模式减少显存占用
.\venv\Scripts\python.exe curriculum_learning.py --mode quick

# 或减少batch_size (需修改源码)
# 在CurriculumConfig中设置: batch_size = 32
```

### 问题2: 训练停滞不前

**症状**: Episode分数长时间不变

**可能原因**:
1. 学习率过高 → 降低lr_factor (如0.5)
2. 探索不足 → 增加entropy_factor (如1.5)
3. 场景太难 → 回退到上一阶段重新训练

**解决方案**:
```bash
# 从当前阶段重新开始
.\venv\Scripts\python.exe curriculum_learning.py --from-phase 1
```

### 问题3: 强场景性能下降过多

**症状**: 农业或应急下降>5%

**原因**: Phase 2/3的训练影响了已学到的策略

**解决方案**:
1. 减少Phase 2/3的episodes数
2. 增大早停patience (如15→20)
3. 手动回退到Phase 0/1的最佳模型继续训练

### 问题4: 对比学习损失为0

**症状**: 日志中`contrastive_loss=0.0000`

**原因**: 正负样本缓冲区未收集足够数据

**解决方案**:
- 这是正常的！前几个episode缓冲区为空
- 通常在第10个episode后会开始生效
- 如果持续为0，检查contrastive_enabled是否为True

---

## 📈 预期效果 vs 实际效果

### 理论预期 (基于设计)

| 场景 | 基线 | 目标 | 提升 |
|------|------|------|------|
| 工业巡检 | 67.55% | 78-82% | +10-14pp |
| 智慧城市 | 70.47% | 80-84% | +10-14pp |
| 物流配送 | 71.04% | 79-83% | +8-12pp |
| 应急救援 | 90.49% | 88-90% | -2~0pp |
| 农业植保 | 95.97% | 94-96% | -2~0pp |
| **全局平均** | **79.10%** | **85-87%** | **+6-8pp** |

### 判断标准

| 结果范围 | 行动建议 |
|----------|---------|
| **全局 >88%** | 🎉 **完美!** 可直接用于论文 |
| **全局 85-88%** | ✅ **优秀!** 达到预期目标 |
| **全局 82-85%** | ⚠️ **良好** 可接受，考虑再跑一轮Full模式 |
| **全局 79-82%** | 😐 **一般** 分析瓶颈，尝试调整超参数 |
| **全局 <79%** | ❌ **失败** 检查代码或回到v2.0 |

---

## 💡 高级用法

### 自定义场景配置

编辑 `curriculum_learning.py` 中的 `CurriculumConfig` 类:

```python
@dataclass
class CurriculumConfig:
    # ... 其他配置 ...
    
    phase_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'phase_0_consolidation': {
            'name': '强场景巩固',
            'episodes': 10,  # ← 修改这里增加episodes
            'scenarios': ['agriculture', 'emergency_rescue'],
            # ... 
        },
        # ... 其他阶段 ...
    })
```

### 自定义奖励权重

```python
scenario_reward_weights: Dict[str, Dict] = field(default_factory=lambda: {
    'industrial_inspection': {
        'connection_stability': 0.40,  # ← 增加连接稳定性权重
        'handover_success': 0.35,
        'load_balance': 0.15,
        'satisfaction': 0.10,
    },
    # ... 其他场景 ...
})
```

### 调整对比学习强度

```python
@dataclass
class CurriculumConfig:
    # 对比学习参数
    contrastive_lambda: float = 0.15  # ← 从0.1增加到0.15 (更强约束)
    contrastive_temperature: float = 0.07  # ← 从0.1降低到0.07 (更尖锐分布)
```

---

## 🔄 与v2.0的对比

| 维度 | v2.0 (finetune_multi_scenario) | v3.0 (curriculum_learning) |
|------|-------------------------------|---------------------------|
| **训练策略** | 随机多场景同时训练 | 由易到难分阶段训练 |
| **奖励函数** | 统一奖励 | 场景特定塑造 |
| **辅助监督** | 无 | 对比学习 (InfoNCE) |
| **UAV规模处理** | 固定 | 渐进式扩展 |
| **预期提升** | +1.5% (实测) | **+6-9%** (理论) |
| **训练时间** | 2-3小时 | 6-10小时 (Full) |
| **复杂度** | 低 | 中等 |
| **可控性** | 一般 | **优秀** |

---

## 📝 论文写作建议

如果v3.0达到预期效果，论文可以这样描述:

### 方法论部分

> 本文提出了一种基于课程学习的渐进式多场景微调框架(PMSF v3.0)。该框架包含三个核心组件:
> 
> 1. **课程调度器(CurriculumScheduler)**: 将训练过程划分为四个由易到难的学习阶段。初始阶段专注于巩固高基线场景的性能，随后逐步引入复杂场景进行专项突破，最终实现全场景联合优化。
> 
> 2. **场景感知奖励塑造(Scenario-Aware Reward Shaping)**: 针对不同应用场景的业务特性和QoS需求，动态调整奖励函数的权重分配。例如，工业巡检场景强调连接稳定性(35%)，而物流配送场景则侧重切换成功率(45%)。
> 
> 3. **对比表征学习(Contrastive Representation Learning)**: 引入InfoNCE风格的对比损失作为辅助监督信号，促使网络学习场景无关的通用切换决策表征，提升跨场景泛化能力。

### 实验结果部分

> 实验结果表明，PMSF v3.0将MAPPO模型的全场景平均满意度从79.10%提升至**XX.XX%**(相对提升X.X%)。其中，弱场景表现显著改善: 工业巡检提升**XX.XX个百分点**，智慧城市监控提升**XX.XX个百分点**，物流配送提升**XX.XX个百分点**。与此同时，强场景性能保持在可接受范围内(下降<2%)。

---

## 🆘 技术支持

遇到问题时:

1. **查看日志**: `experiment_results/curriculum_v3_XXXX/training_log.txt`
2. **运行验证**: `python test_curriculum_learning.py`
3. **检查模型**: 确认`mappo_8bs_300uav_best.pt`存在且有效

---

## 📌 下一步行动

```bash
# 1. 运行快速测试 (2-3小时)
.\venv\Scripts\python.exe curriculum_learning.py --mode quick

# 2. 检查结果
cat experiment_results/curriculum_v3_*/final_result.json

# 3. 如果效果好，运行完整版 (6-10小时)
.\venv\Scripts\python.exe curriculum_learning.py --mode full

# 4. 使用最终模型进行评估
# (模型路径: experiment_results/curriculum_v3_*/curriculum_final.pt)
```

**祝训练顺利! 🚀**
