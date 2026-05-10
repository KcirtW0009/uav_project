# UAV业务识别与切换决策系统 - 代码架构文档

## 📋 目录
- [1. 系统概述](#1-系统概述)
- [2. 核心模块架构](#2-核心模块架构)
- [3. 实验3详解](#3-实验3详解)
- [4. 实验4详解](#4-实验4详解)
- [5. 数据流向图](#5-数据流向图)
- [6. 关键配置参数](#6-关键配置参数)
- [7. 运行指南](#7-运行指南)

---

## 1. 系统概述

### 1.1 项目目标
本项目实现了一个**无人机(UAV)业务感知切换决策系统**，核心创新点：
- **业务类型识别**：自动识别UAV当前运行的业务类型（控制信令/视频回传/环境监测）
- **智能切换决策**：基于QoS需求和网络状态做出最优基站切换决策
- **多算法对比**：传统算法 vs 增强算法 vs MAPPO强化学习算法

### 1.2 三种算法对比

| 算法 | 特点 | 切换策略 |
|------|------|---------|
| **传统算法** | 基于SINR阈值 | 仅当信号强度低于阈值时切换 |
| **增强算法** | +动态阈值+业务权重+ε-greedy+负载均衡 | 综合考虑信号、负载、业务需求 |
| **MAPPO** | 多智能体强化学习 | 通过训练学习最优策略 |

---

## 2. 核心模块架构

```
uav_system/
├── main.py                    # 主程序入口（命令行解析）
├── experiments.py             # 实验管理（实验1-4）⭐ 核心
├── config.py                  # 全局配置（种子、路径、颜色）
│
├── environment/
│   └── environment.py         # 网络仿真环境（BS/UAV/信道模型）
│
├── algorithms/
│   ├── algorithms.py          # 传统/增强切换算法实现
│   └── mappo_agent_v2.py      # MAPPO智能体（Actor-Critic）
│
├── recognition/
│   └── recognition.py         # 业务识别模型（决策树/随机森林）
│
├── satisfaction/
│   └── satisfaction.py        # 层次化满意度评估（速率/时延/丢包）
│
├── business/
│   └── business.py            # 业务类型定义与QoS配置
│
├── visualization/
│   └── visualization.py       # 可视化工具（图表生成）
│
└── mappo_environment.py       # MAPPO专用环境（纯净版评估）
```

### 2.1 模块职责说明

#### **experiments.py** - 实验管理中心
```
职责：
  ✅ 协调所有模块的调用顺序
  ✅ 管理实验流程（数据收集→统计→检验→可视化）
  ✅ 提供缓存模式（跳过已完成的算法）
  ✅ 自动保存机制（防止数据丢失）

核心类：
  Experiment1: 识别准确性验证（5种准确率等级）
  Experiment2: 机制有效性验证（逐步添加机制）
  Experiment3: 增强vs传统全面对比（8BS×300UAV）
  Experiment4: 多场景泛化测试（5个典型场景）
  
核心函数：
  evaluate_mappo_in_experiment()  ← MAPPO评估入口
  compare_algorithms_with_tests() ← 统计显著性检验
  save_experiment_data()          ← 数据持久化
```

---

## 3. 实验3详解

### 3.1 实验目标
**在标准场景下（8个基站×300架UAV），全面对比三种算法的性能差异**

### 3.2 实验配置

```python
# 环境配置
num_bs = 8                    # 基站数量
num_uav = 300                 # UAV数量 (~77%负载率)
num_steps = 350               # 仿真步数
repeats = 10                  # 重复次数（不同随机种子）
global_seed = 30042           # 基础种子 (30042~30051)

# MAPPO种子重排（避免初始种子过于简单）
mappo_seed_order = [5,7,3,8,1,9,2,6,0,4]  # 先跑有挑战性的种子
```

### 3.3 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment3.run()                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 加载业务识别模型                                     │
│   train_or_load_recognition_model(force_compare=False)       │
│   → 输出: recognition_model (dt/rf), scaler (StandardScaler) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: [缓存模式] 加载传统/增强算法数据                      │
│   IF use_cache=True:                                        │
│     从 exp3_data.json 读取 {enhanced, traditional}           │
│     跳过 ~14小时的传统/增强算法运行                           │
│   ELSE:                                                     │
│     FOR rep in range(10):                                   │
│       ├─ 创建环境 (EnhancedNetworkEnvironment)              │
│       ├─ 运行增强算法 (EnhancedHandoverAlgorithm)           │
│       ├─ 运行传统算法 (IntegratedHandoverAlgorithm)         │
│       └─ 收集统计数据到 results[]                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: [可选] MAPPO评估（10轮，使用种子重排）                │
│   FOR idx, rep in enumerate([5,7,3,8,1,9,2,6,0,4]):        │
│     ├─ 设置种子: set_global_seed(30042 + rep)               │
│     ├─ 调用 evaluate_mappo_in_experiment()                  │
│     │   ├─ 创建 MultiAgentHandoverEnv                       │
│     │   ├─ 加载 MAPPO 模型 (mappo_8bs_300uav_best.pt)      │
│     │   └─ 运行350步仿真                                    │
│     ├─ [AUTO-SAVE] 每轮保存到 exp3_mappo_raw_results.json   │
│     └─ 显示17个指标的详细数据                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 统计汇总与显著性检验                                  │
│   summary = _summarize(enhanced, traditional, mappo)        │
│   → 计算 mean ± std for all 17 metrics                     │
│                                                              │
│   [两算法检验]                                               │
│   compare_algorithms_with_tests(enhanced, traditional)      │
│   → t-test / Mann-Whitney U (自动选择)                     │
│   → 输出 p值, 效应量, 显著性标记                             │
│                                                              │
│   [三算法检验] (如果include_mappo=True)                     │
│   compare_three_algorithms_with_tests(mappo, enh, trad)    │
│   → MAPPO vs 增强, MAPPO vs 传统, 增强 vs 传统             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 结果展示与保存                                       │
│   _print_results_table(summary)  → 打印对比表格             │
│   [FINAL-SAVE] 保存到 exp3_mappo_summary.json               │
│   _plot(summary)                 → 生成16张可视化图表        │
│   save_experiment_data('exp3')   → 保存到 exp3_data.json   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 评估指标体系（17个指标）

| 类别 | 指标名 | 变量名 | 说明 |
|------|--------|--------|------|
| **切换性能** | 切换成功率 | `handover_success_rate` | 成功切换/总尝试 |
| | 平均切换时延 | `avg_switching_latency_ms` | 切换决策→完成时间 |
| | 最大切换时延 | `max_switching_latency_ms` | 最坏情况延迟 |
| | 平均决策时间 | `avg_decision_time_ms` | 算法计算耗时 |
| **连接质量** | 连接保持率 | `connected_ratio` | 保持连接的UAV比例 |
| | 错失机会率 | `missed_opportunity_rate` | 未抓住的切换机会 |
| | 迁移成功率 | `migration_success_rate` | 成功迁移到更好BS |
| **用户满意度** | 整体满足率 | `avg_satisfaction` | 综合满意度均值 |
| | 关键业务满足率 | `critical_satisfaction` | 重要业务的满足程度 |
| | 加权满足率 | `weighted_satisfaction` | 按业务权重加权 |
| | 时延满足率 | `latency_satisfaction` | 延迟需求的满足度 |
| | 速率满足率 | `rate_satisfaction` | 速率需求的满足度 |
| **系统效率** | 系统吞吐量 | `total_throughput` | 总传输速率(Mbps) |
| | 负载方差 | `load_variance` | BS间负载均衡程度 |
| | 平均SINR | `avg_sinr_db` | 平均信干噪比(dB) |
| **辅助指标** | 识别准确率 | `recognition_accuracy` | 业务类型分类准确率 |

### 3.5 自动保存机制（三层保护）

```python
# 第1层：每轮完成后立即保存
exp3_mappo_raw_results.json
{
  "timestamp": "2026-05-10T15:30:00",
  "total_completed": 3,        # 已完成轮数
  "seed_order": [5,7,3,...],  # 种子重排顺序
  "results": [...]            # 原始数据列表
}

# 第2层：全部完成后的完整summary
exp3_mappo_summary.json
{
  "total_mappo_runs": 10,
  "seed_order": [...],
  "summary": {                # 统计汇总
    "mappo": {
      "handover_success_rate": (0.75, 0.15),  # (mean, std)
      ...
    }
  },
  "raw_results": [...]        # 所有原始数据
}

# 第3层：绘图异常捕获
try:
    Experiment3._plot(summary)
except Exception as e:
    print(f"[WARN] 绘图出错 (数据已保存): {e}")
```

---

## 4. 实验4详解

### 4.1 实验目标
**测试算法在不同5G应用场景下的泛化能力（零样本泛化）**

### 4.2 五个测试场景

| 场景ID | 场景名称 | UAV数量 | 5G特性 | 典型应用 |
|--------|---------|---------|-------|---------|
| scenario1 | 密集城区 | 500 | mMTC大规模接入 | 智慧城市传感网 |
| scenario2 | 高速移动 | 400 | uRLLC超可靠低时延 | 自驾驶车队 |
| scenario3 | 广域覆盖 | 300 | eMBB增强宽带 | 农业监测 |
| scenario4 | 热点区域 | 450 | 网络切片 | 体育赛事直播 |
| scenario5 | 混合场景 | 350 | 边缘计算MEC | 工业互联网 |

### 4.3 与实验3的区别

```
┌─ 实验3 ──────────────────────────────────┐
│ 单一场景: 8BS × 300UAV                   │
│ 目标: 全面对比三种算法性能                │
│ 重点: 统计显著性检验 (t-test/Wilcoxon)    │
│ 时间: ~4.7小时 (含MAPPO)                 │
└───────────────────────────────────────────┘

┌─ 实验4 ──────────────────────────────────┐
│ 5个场景: 不同UAV数量 (300-500)            │
│ 目标: 验证MAPPO零样本泛化能力             │
│ 重点: 跨场景性能稳定性分析                │
│ 时间: ~20小时 (含MAPPO)                  │
└───────────────────────────────────────────┘
```

### 4.4 执行流程

```
FOR scenario in ['scenario1', 'scenario2', ..., 'scenario5']:
    num_uav = SCENARIOS[scenario]['num_uav']  # 300/350/400/450/500
    
    # [缓存模式] 只运行MAPPO（传统/增强从exp4_data.json加载）
    FOR rep in range(10):
        mappo_stats = evaluate_mappo_in_experiment(
            num_bs=8,
            num_uav=num_uav,           # ★ 不同场景不同UAV数
            model_path=mappo_model_path
        )
        
        # [AUTO-SAVE] 每轮保存（按场景分组）
        results[scenario]['mappo'].append(mappo_stats)

# 最终汇总 + 可视化
Experiment4._print_results_table(summary)
Experiment4._plot(summary)  # 生成热力图、雷达图等
```

---

## 5. 数据流向图

### 5.1 完整数据流

```
┌──────────────┐
│ 业务识别模型  │ recognition.py
│ (dt/rf)      │
└──────┬───────┘
       │  输出: predicted_business_type
       ▼
┌──────────────┐     ┌──────────────────┐
│  网络环境     │────▶│  切换算法        │
│ environment  │     │ algorithms.py    │
│              │     │                  │
│ • BS状态     │     │ • 传统算法        │
│ • UAV状态    │     │ • 增强算法        │
│ • 信道质量   │     │ • QoS适配决策     │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       │  env.step()          │  algo.run_step()
       │                      │
       ▼                      ▼
┌──────────────────────────────────────────┐
│         统计数据收集                        │
│  experiments.py                           │
│                                           │
│  env.get_state_statistics()               │
│  algo.get_detailed_stats()               │
│                                           │
│  → avg_satisfaction                      │
│  → handover_success_rate                 │
│  → connected_ratio                       │
│  → ... (共17个指标)                       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         数据持久化                          │
│                                           │
│  exp3_data.json / exp4_data.json          │
│  exp3_mappo_raw_results.json              │
│  exp3_mappo_summary.json                  │
│  experiment_results/*.png                 │
└──────────────────────────────────────────┘
```

### 5.2 MAPPO评估专用流程

```
┌────────────────┐
│ 训练好的模型    │ mappo_8bs_300uav_best.pt
└───────┬────────┘
        │ torch.load()
        ▼
┌────────────────┐     ┌──────────────────────┐
│  MAPPO Agent   │────▶│  MultiAgentHandoverEnv│
│ (Actor-Critic) │     │  mappo_environment.py │
│                │     │                      │
│ • 策略网络     │     │ • 纯净版（无预检查）  │
│ • 价值网络     │     │ • 无降级分配          │
│                │     │ • 有限回滚机制        │
└───────┬────────┘     └──────────┬───────────┘
        │                         │
        │  agent.select_action()  │  env.step(action)
        │◀────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Reward计算 (V12增强版)                    │
│                                           │
│  R = w1*满意度提升 + w2*切换成功          │
│    + w3*负载均衡 + w4*惩罚断连             │
│    + w5*关键业务优先 + w6*同类排名         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
        返回 stats_dict (17个指标)
```

---

## 6. 关键配置参数

### 6.1 全局配置 (config.py)

```python
GLOBAL_SEED = 30042          # 基础随机种子
RESULT_DIR = "experiment_results"  # 结果保存目录

# 可视化配色方案
COLORS = {
    'primary': '#2E86AB',    # 主色调（蓝色）
    'success': '#28A745',    # 成功（绿色）
    'warning': '#FFC107',    # 警告（黄色）
    'danger': '#DC3545',     # 危险（红色）
    'neutral': '#6C757D',    # 中性（灰色）
}
```

### 6.2 环境参数

```python
# 基站配置
bs_capacity_range = (500, 1000)  # BS容量范围 (Mbps)
pos_range = 1000                 # 地图大小 (米×米)

# UAV配置
event_probability = 0.05         # 移动事件概率 (每步5%UAV移动)
required_rates = [10, 50, 20]   # 各业务所需速率 (Mbps)

# 仿真参数
max_steps = 350                 # 每轮仿真步数
repeats = 10                    # 重复实验次数
```

### 6.3 MAPPO模型参数

```python
# 网络结构
hidden_dim = 64                 # Actor隐藏层维度
critic_hidden_dim = 128         # Critic隐藏层维度

# 训练参数 (参考)
episodes = 500                  # 训练轮数
batch_size = 256                # 批次大小
lr_actor = 3e-4                 # Actor学习率
lr_critic = 1e-3                # Critic学习率
gamma = 0.99                    # 折扣因子
clip_ratio = 0.2                # PPO裁剪比例
```

---

## 7. 运行指南

### 7.1 快速开始

```bash
# 进入项目目录
cd "f:\桌面\本科毕业论文\结题\uav_project"

# 激活虚拟环境
.\venv\Scripts\activate
```

### 7.2 常用命令

```bash
# 实验3 + MAPPO (缓存模式，推荐)
python main.py --exp 3 --include-mappo --use-cache \
  --mappo-model "experiment_results/mappo_models/mappo_8bs_300uav_best.pt"

# 实验3 + 实验4 + MAPPO (完整三算法对比)
python main.py --exp 3 4 --include-mappo --use-cache

# 仅运行传统/增强算法（不含MAPPO）
python main.py --exp 3

# MAPPO训练
python main.py --exp mappo

# 小规模调试 (128UAV/3BS)
python main.py --exp mappo --small
```

### 7.3 数据恢复工具

```bash
# 检查是否有已保存的数据
python recover_mappo_data.py --check

# 从已有数据生成报告
python recover_mappo_data.py --report

# 验证exp3_data.json完整性
python verify_exp3_data.py
```

### 7.4 预期运行时间

| 实验 | 场景 | 不含MAPPO | 含MAPPO | 缓存模式 |
|------|------|----------|---------|---------|
| **实验3** | 1场景×10轮 | ~14小时 | ~18小时 | **~4小时** |
| **实验4** | 5场景×10轮 | ~37小时 | ~57小时 | **~20小时** |
| **总计** | - | ~51小时 | ~75小时 | **~24小时** |

> 💡 **提示**: 使用 `--use-cache` 参数可节省约51小时！

---

## 附录A: 文件结构说明

```
experiment_results/
├── backup_20260510/              # 重要数据备份
│   ├── exp3_data.json
│   ├── exp4_data.json
│   └── exp3_data.json.bak_before_hosr_fix
│
├── exp3_data.json                # 实验3传统/增强算法数据 (17KB)
├── exp4_data.json                # 实验4传统/增强算法数据 (12KB)
├── exp3_mappo_raw_results.json   # MAPPO原始数据 (自动保存)
├── exp3_mappo_summary.json       # MAPPO汇总数据 (最终保存)
├── exp4_mappo_raw_results.json   # 实验4 MAPPO数据
├── exp4_mappo_summary.json       # 实验4 MAPPO汇总
│
├── separated_figs/               # 分离式图表 (32张PNG)
│   ├── exp3_*.png                #   实验3可视化
│   └── exp4_*.png                #   实验4可视化
│
├── mappo_models/                 # MAPPO模型文件
│   └── mappo_8bs_300uav_best.pt  # 最佳模型 (推荐使用)
│
└── training_logs/                # 训练日志
    └── mappo_*/                  # 各次训练记录
        ├── metrics.json          # 训练曲线数据
        └── training_curves.png   # Loss/Reward曲线
```

## 附录B: 常见问题排查

### Q1: 切换成功率显示100%是否正常？
**答**: 在纯净版评估模式下，42%-99%都是正常范围。如果是100%，可能是种子过于简单。

### Q2: 如何判断是否存在过拟合？
**答**: 对比训练环境和评估环境的性能差距。如果训练时99%但评估时<80%，可能存在过拟合。

### Q3: 种子重排的作用？
**答**: 避免前几个种子表现过于理想（如100%成功率），导致无法观察真实性能差异。

### Q4: 缓存数据被覆盖怎么办？
**答**: 从 `backup_20260510/` 文件夹恢复原始数据。

---

**文档版本**: v2.0
**最后更新**: 2026-05-10
**适用代码**: experiments.py (实验3/4完整版)
