# 🚀 UAV业务识别与切换决策系统 - 快速学习指南

**版本**: v2.0 (2026-05-11 更新)  
**目标读者**: 新加入项目的开发者、论文评审者  
**预计阅读时间**: 30分钟（初学者）/ 10分钟（有经验者）

---

## 📌 本指南能帮你做什么？

✅ **5分钟** - 知道怎么运行系统  
✅ **15分钟** - 理解核心模块的作用  
✅ **30分钟** - 能够修改参数和运行实验  
✅ **1小时** - 能够添加新功能或设计新实验  

---

## 🎯 一句话总结

> **这是一个基于多智能体强化学习(MAPPO)的无人机切换决策系统，通过业务识别+增强算法实现比传统方法更好的服务质量保障。**

---

## ⚡ 快速开始（3步上手）

### **Step 1: 环境准备**

```bash
# 进入项目目录
cd "f:\桌面\本科毕业论文\结题\uav_project"

# 激活虚拟环境（必须！）
.\venv\Scripts\activate

# 验证Python版本
python --version  # 应该显示 Python 3.8+
```

### **Step 2: 运行第一个实验**

```bash
# 最简单的命令：只运行实验3（传统 vs 增强）
.\venv\Scripts\python.exe main.py --exp 3
```

**预期输出**：
```
=======================================================
实验3：增强算法 vs 传统算法（全面对比）
=======================================================

--- 重复 1/10 ---
 增强算法 - 满足率: 0.981, 切换成功率: 100.0%, 吞吐量: 4706.5 Mbps
 传统算法 - 满足率: 0.763, 切换成功率: 56.4%, 吞吐量: 2863.5 Mbps

[Visualization] 生成实验三专业图表...
  ✅ 已生成 3 张图表

[FINAL-SAVE] 完整结果已保存 → experiment_results/exp3_data.json
```

### **Step 3: 查看结果**

生成的图表位置：
```
experiment_results/latest_figures/
├── exp3_satisfaction_metrics_comparison.png    # 满足率对比图
├── exp3_stability_metrics_comparison.png       # 稳定性指标
└── exp3_performance_metrics_comparison.png     # 性能效率指标
```

双击打开任意 `.png` 文件即可查看！

---

## 🏗️ 系统架构（一图胜千言）

```
┌─────────────────────────────────────────────────────┐
│                    用户界面层                        │
│              main.py (命令行启动)                    │
└───────────────────────┬─────────────────────────────┘
                        │ 调用
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌──────────┐    ┌──────────┐
   │ 实验1   │    │ 实验2    │    │ 实验3-4  │
   │ 识别影响│    │ 机制验证 │    │ 三算法对比│
   └────┬────┘    └────┬─────┘    └────┬─────┘
        │              │               │
        ▼              ▼               ▼
   ┌──────────────────────────────────────────┐
   │            核心模块层                      │
   │  ┌─────────┬─────────┬────────────────┐  │
   │  │环境仿真 │ 切换算法│ 业务识别模型   │  │
   │  │env.py   │algo.py  │recognition.py  │  │
   │  └─────────┴─────────┴────────────────┘  │
   │  ┌─────────┬─────────┬────────────────┐  │
   │  │满意度评估│ MAPPO  │ 配置中心      │  │
   │  │satisf.. │agent_v2│ config.py     │  │
   │  └─────────┴─────────┴────────────────┘  │
   └──────────────────────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────┐
   │            数据与可视化                   │
   │  experiment_results/ → JSON数据 + 图表    │
   └──────────────────────────────────────────┘
```

---

## 🔑 核心概念（必读！）

### **1️⃣ 什么是"业务识别"？**

**类比**: 就像快递员根据包裹大小选择不同运输方式一样，系统根据UAV的业务类型（控制信令/视频回传/环境监测）采用不同的切换策略。

**三种业务类型**:

| 业务 | 类比 | 需求特点 |
|-----|------|---------|
| **控制信令** | 🆘 急救电话 | 必须立即接通，不能断线 |
| **视频回传** | 📹 直播 | 要清晰流畅，偶尔卡顿可接受 |
| **环境监测** | 🌡️ 温度计 | 数据量小，慢一点没关系 |

**技术实现**:
- 使用机器学习模型（随机森林）自动识别UAV的业务类型
- 准确率接近100%
- 为后续差异化处理提供依据

---

### **2️⃣ 什么是"切换"？**

**定义**: 当UAV移动到当前基站信号变差时，需要"切换"到信号更好的基站。

**传统方法的痛点**:
```
❌ 固定阈值: 所有UAV用同一个标准判断是否切换
❌ 不考虑负载: 可能切换到一个已经很拥挤的基站
❌ 无业务差异: 视频和控制信令同等对待
```

**我们的改进**:
```
✅ 动态阈值: 根据实际情况调整触发条件
✅ 负载均衡: 避免所有UAV挤到同一个基站
✅ 业务优先级: 关键业务优先保障
```

---

### **3️⃣ 什么是MAPPO？**

**全称**: Multi-Agent Proximal Policy Optimization  
**中文名**: 多智能体近端策略优化  
**本质**: 一种深度强化学习算法，让多个UAV学会"聪明地切换"

**工作原理**:
```
1. 观察(Observation): 每个UAV看到周围的环境状态
2. 决策(Action): MAPPO告诉每个UAV该怎么做（保持/切换到哪个基站）
3. 执行(Execution): 系统执行这些决策
4. 反馈(Reward): 根据结果给奖励或惩罚
5. 学习(Learning): MAPPO根据反馈调整策略（重复数千次）
```

**为什么MAPPO效果好？**
- ✅ 全局视角：能看到所有UAV的状态（不是各自为战）
- ✅ 经验共享：多个UAV的学习经验可以互相借鉴
- ✅ 自适应：能适应不同场景（农业/城市/工业等）

---

## 📦 模块详解（按重要性排序）

### **🥇 第一重要：experiments.py（实验指挥官）**

**角色**: 整个系统的"大脑"，协调所有模块完成实验

**核心职责**:
```python
class Experiment3:
    """三算法对比实验"""
    
    def run():
        # 1. 创建环境（包含300个UAV和8个基站）
        env_enh = EnhancedNetworkEnvironment(num_bs=8, num_uav=300)
        
        # 2. 运行三种算法
        for step in range(350):  # 模拟350个时间步
            enhanced_algo.run_step()   # 增强算法
            traditional_algo.run_step() # 传统算法
            mappo_agent.step()         # MAPPO
        
        # 3. 收集17项性能指标
        stats = env.get_state_statistics()
        
        # 4. 保存数据并生成图表
        save_experiment_data('exp3', summary)
        plot_combined_exp3_figures(data)
```

**关键数据流**:
```
原始数据（每步）→ 统计汇总（10次重复）→ 平均值±标准差 → 表格/图表
```

**你需要知道的**:
- 实验3：单场景（8BS×300UAV），重复10次
- 实验4：五场景（农业/城市/工业/应急/物流），每个场景重复5次
- 自动保存机制：防止崩溃丢失数据

---

### **🥈 第二重要：config.py（配置管家）**

**角色**: 管理所有可调参数的"中央数据库"

**常用配置项**:
```python
GLOBAL_SEED = 30042          # 随机种子（确保实验可复现）
RESULT_DIR = "experiment_results"  # 结果保存位置

# 中断检测阈值
INTERRUPTION_CONFIG = {
    'threshold': 0.3,        # 满意度低于0.3视为中断
    'duration': 5,           # 连续5步才算中断事件
}

# MAPPO训练超参数
class TrainingConfig:
    NUM_EPISODES = 500        # 训练轮数
    PPO_EPOCHS = 10           # PPO更新轮数
    LEARNING_RATE = 3e-4      # 学习率
```

**什么时候需要修改它？**
- 改实验规模（UAV数量、基站数量）
- 调整训练参数（学习率、训练轮数）
- 修改评估标准（中断阈值）

---

### **🥉 第三重要：satisfaction.py（评分员）**

**角色**: 给每个UAV的服务质量打分

**评分维度**:
```python
def compute_satisfaction(uav):
    scores = {
        'rate_sat': 当前速率 / 目标速率,           # 速率满足度 [0,1]
        'latency_sat': 1 - (当前时延 / 最大时延),   # 时延满足度 [0,1]  
        'loss_sat': 1 - 当前丢包率,                 # 丢包满足度 [0,1]
        
        'overall': 是否达到最低要求,                # 二元 [0或1]
        'critical': 关键指标是否满足,               # 二元 [0或1]
        'weighted': 加权综合得分,                   # 连续 [0,priority]
    }
    return scores
```

**关键规则**:
- 控制信令（安全关键）：必须同时满足速率+时延，否则整体为0
- 其他业务：只需满足最低速率要求即可
- 加权分数考虑了业务优先级（控制信令权重最高）

---

### **第四梯队：其他重要模块**

#### **algorithms.py（两种算法）**

| 特性 | 传统算法 | 增强算法 |
|------|---------|---------|
| **切换触发** | 固定RSSI阈值 | 动态自适应阈值 |
| **负载考虑** | ❌ 无 | ✅ 负载均衡 |
| **业务差异** | ❌ 无 | ✅ 权重优先级 |
| **探索能力** | ❌ 无 | ✅ ε-贪婪探索 |
| **决策延迟** | 极低(~0.001ms) | 较低(~0.01ms) |

#### **environment.py（仿真世界）**

模拟真实网络环境：
- 300架UAV在空间中移动
- 8个基站提供覆盖
- 信道质量随时间变化
- UAV产生不同类型的业务流量

#### **mappo_agent_v2.py（AI大脑）**

神经网络结构：
- **Actor网络**: 输入观测→输出动作概率
- **Critic网络**: 输入观测→评估状态好坏
- 训练方式：PPO（近端策略优化）

#### **reward_functions.py（奖励设计）**

告诉MAPPO什么是"好的行为"：
```python
total_reward = (
    1.5 × satisfaction_reward     # 满足率高→正奖励
  + 1.0 × connection_reward       # 保持连接→正奖励
  + 0.5 × switch_quality_reward   # 成功切换→正奖励
  + 0.3 × load_balance_reward     # 负载均衡→正奖励
  - 1.0 × penalty                # 频繁无效切换→惩罚
)
```

---

## 🛠️ 常用操作手册

### **场景1: 我想重新跑实验三**

```bash
# 完整模式（含MAPPO，不使用缓存）
.\venv\Scripts\python.exe main.py --exp 3 --include-mappo --no-cache
# 时间: ~18小时
```

### **场景2: 我只想更新MAPPO的数据**

```bash
# 缓存模式（跳过传统/增强算法）
.\venv\Scripts\python.exe main.py --exp 3 --include-mappo --use-cache
# 时间: ~4小时（节省14小时！）
```

### **场景3: 我要调试代码，快速验证**

```bash
# 小规模模式
.\venv\Scripts\python.exe main.py --exp mappo --small
# UAV: 300→128, BS: 8→3, 训练时间: 12小时→30分钟
```

### **场景4: 运行全部实验（从零开始）**

```bash
# 第一步：训练MAPPO模型
.\venv\Scripts\python.exe main.py --exp mappo

# 第二步：运行实验3+4（含MAPPO）
.\venv\Scripts\python.exe main.py --exp 3 4 --include-mappo --no-cache

# 总时间: ~75小时（建议周末运行）
```

### **场景5: 只看已有的图表**

直接打开文件夹：
```
experiment_results/latest_figures/
```
双击 `.png` 文件查看！

---

## 📊 结果解读指南

### **实验三图表解读（3张）**

#### **图1: Satisfaction Metrics Comparison**
```
X轴: 4个指标
  ├ avg_satisfaction (整体满足率)
  ├ critical_satisfaction (关键业务满足率)  
  ├ weighted_satisfaction (加权满足率)
  └ connected_ratio (连接保持率)

Y轴: 归一化值 [0-1]

柱子颜色:
  ■ Enhanced (天蓝色) - 我们的算法
  ■ Traditional (灰色) - 传统方法
  ■ MAPPO (金黄色) - 强化学习方法
```

**如何判断好坏？**
- ✅ 柱子越高越好（除了反向指标）
- ✅ Enhanced应该明显高于Traditional
- ✅ MAPPO应该接近或超过Enhanced

#### **图2: Stability Metrics Comparison**
```
X轴: 3个稳定性指标
  ├ handover_success_rate (切换成功率)
  ├ load_variance (负载方差) ← 越低越好!
  └ migration_success_rate (迁移成功率)
```

**注意**: `load_variance` 是**反向指标**（越低越好），图中会标注 ↓ 符号

#### **图3: Performance Efficiency Comparison**
```
X轴: 4个效率指标
  ├ avg_sinr (平均SINR)
  ├ avg_switching_latency_ms (切换延迟) ← 越低越好!
  ├ avg_decision_time_ms (决策时间) ← 越低越好!
  └ total_throughput (吞吐量) ← 越高越好!
```

---

### **实验四图表解读（6张，更详细！）**

实验四增加了**五场景对比**，每张图展示5种应用场景：

| 场景 | UAV数量 | 典型应用 |
|-----|--------|---------|
| agriculture | 350 | 农业植保喷洒 |
| smart_city | 400 | 智慧城市监控 |
| industrial_inspection | 300 | 工业设备巡检 |
| emergency_rescue | 300 | 应急救援搜索 |
| logistics_delivery | 500 | 物流配送运输 |

**图表特色**:
- ✅ 分组柱状图（每组3根柱子：Enhanced/Traditional/MAPPO）
- ✅ 包含误差棒（显示标准差）
- ✅ Y轴范围经过精心调优（避免视觉误导）
- ✅ 学术出版级配色方案

---

## 🔍 故障排查（FAQ）

### **Q1: 运行时提示 "ModuleNotFoundError: No module named 'torch'"**

**原因**: 没有使用虚拟环境中的Python

**解决**:
```bash
# 错误做法
python main.py  # 使用系统Python

# 正确做法
.\venv\Scripts\python.exe main.py  # 使用虚拟环境Python
```

---

### **Q2: sklearn警告不断弹出，很烦人**

**状态**: ✅ 已修复！

**修复位置**: [recognition.py:174](uav_system/recognition.py#L174)

如果仍然出现，检查是否使用了最新版本的代码。

---

### **Q3: 运行到一半崩溃了，数据丢失了吗？**

**状态**: ✅ 有三层保护！

1. **AUTO-SAVE**: 每完成1轮就自动保存
2. **FINAL-SAVE**: 绘图前保存完整数据
3. **标准导出**: 最终保存到JSON文件

**恢复方法**:
```bash
# 检查是否有部分数据
ls experiment_results/exp3_mappo_raw_results.json
ls experiment_results/exp3_mappo_summary.json

# 如果有，可以使用缓存模式继续
.\venv\Scripts\python.exe main.py --exp 3 --include-mappo --use-cache
```

---

### **Q4: 图表中的柱子太高或太低，看不清差异**

**状态**: ✅ 已优化Y轴范围！

**当前设置**（实验四）:
- 关键业务满足率: `(0.80, 1.01)` - 突出细微差异
- 连接保持率: `(0.60, 1.02)` - 展示全部范围
- 吞吐量: `(0, 13000)` - 大尺寸15×8英寸
- SINR: `(0, 30)` - 清晰展示dB值

**自定义调整**: 编辑 [plot_exp4_figures.py](plot_exp4_figures.py) 的 `ylim` 参数

---

### **Q5: MAPPO总是选择action=0（不切换任何基站）**

**可能原因**:
1. **模型未充分训练** (<100 episodes)
2. **奖励函数惩罚太重** (频繁切换被严厉惩罚)
3. **探索不足** (epsilon太小，没尝试过其他动作)

**诊断步骤**:
```bash
# 查看训练日志
cat experiment_results/training_logs/mappo_training.log

# 关注这两个字段
attempts=209   # 尝试切换次数
success=76     # 成功次数
# 如果 attempts 很少，说明探索不足
```

---

## 💡 进阶技巧

### **技巧1: 快速验证代码改动**

不要每次都跑完整的18小时实验！使用小规模模式：

```bash
# 30分钟内完成一次完整流程
.\venv\Scripts\python.exe main.py --exp mappo --small
```

然后检查输出是否符合预期。

---

### **技巧2: 只重新绘图（不重跑实验）**

如果你只是想调整图表样式：

```python
# 在 plot_exp4_figures.py 末尾添加
if __name__ == '__main__':
    from plot_exp4_figures import load_exp4_data, plot_combined_exp4_figures
    data = load_exp4_data()
    fig_paths = plot_combined_exp4_figures(data)
    print(f'Generated {len(fig_paths)} figures')
```

然后运行：
```bash
.\venv\Scripts\python.exe plot_exp4_figures.py
```

---

### **技巧3: 分析特定场景的数据**

```python
import json

with open('experiment_results/exp4_data.json', 'r') as f:
    data = json.load(f)

# 查看智慧城市场景的增强算法数据
city_enhanced = data['smart_city']['enhanced']
print(f"平均满足率: {city_enhanced['avg_satisfaction'][0]:.3f}")
print(f"标准差: {city_enhanced['avg_satisfaction'][1]:.3f}")
print(f"连接保持率: {city_enhanced['connected_ratio'][0]*100:.1f}%")
```

---

### **技巧4: 添加新的评价指标**

如果你想追踪一个新指标：

**Step 1**: 在 `environment.py` 的 `get_state_statistics()` 中计算
```python
stats['my_new_metric'] = calculate_something(env)
```

**Step 2**: 在 `experiments.py` 的 `_summarize()` 中收集
```python
summary['enhanced']['my_new_metric'] = np.mean([r['my_new_metric'] for r in results])
```

**Step 3**: 在 `plot_exp4_figures.py` 中可视化
```python
{
    'key': 'my_new_metric',
    'name': 'My New Metric',
    ...
}
```

---

## 📚 学习路径推荐

### **路径A: 快速上手（1小时）**

```
⏱️ 0-5分钟: 阅读本指南的"快速开始"章节
⏱️ 5-15分钟: 运行 `--small` 模式体验一下
⏱️ 15-30分钟: 查看 experiment_results/ 目录下的数据和图表
⏱️ 30-60分钟: 阅读 config.py 和 experiments.py 的前100行
```

**目标**: 能够独立运行实验并理解基本流程

---

### **路径B: 理解原理（半天）**

```
⏰ 上午:
  1. 阅读 business.py → 了解三种业务类型
  2. 阅读 satisfaction.py → 理解评价体系
  3. 阅读 algorithms.py → 对比两种算法的差异

⏰ 下午:
  4. 阅读 environment.py → 掌握仿真环境
  5. 阅读 reward_functions.py → 理解奖励设计
  6. 阅读 mappo_agent_v2.py → 学习MAPPO原理
```

**目标**: 能够解释为什么增强算法比传统算法好

---

### **路径C: 改进系统（1-2周）**

```
📅 Day 1-2: 
  - 通读 CODE_STRUCTURE.md（完整架构文档）
  - 运行完整实验获取基线数据
  
📅 Day 3-5:
  - 选择一个改进方向（如优化奖励函数）
  - 使用小规模模式快速迭代
  
📅 Day 6-7:
  - 在完整规模上验证改进效果
  - 收集数据并绘制对比图表
  
📅 Day 8-10:
  - 撰写改进报告或论文补充材料
```

**目标**: 能够提出并验证自己的改进方案

---

## 🎯 常用命令速查表

| 操作 | 命令 | 时间 |
|-----|------|------|
| **最小化测试** | `main.py --exp mappo --small` | 30min |
| **仅实验3** | `main.py --exp 3` | 14h |
| **实验3+MAPPO** | `main.py --exp 3 --include-mappo` | 18h |
| **实验3+4+MAPPO** | `main.py --exp 3 4 --include-mappo` | 55h |
| **使用缓存加速** | `... --use-cache` | 节省51h! |
| **强制重新运行** | `... --no-cache` | 完整时间 |
| **查看已有数据** | 打开 `experiment_results/` | 即刻 |
| **重新绘图** | `python plot_exp4_figures.py` | 1min |

---

## 📖 相关文档索引

| 文档名 | 用途 | 难度 |
|-------|------|------|
| **本指南** | 快速入门 | ⭐ |
| [CODE_STRUCTURE.md](CODE_STRUCTURE.md) | 完整架构文档 | ⭐⭐ |
| [CURRICULUM_GUIDE.md](CURRICULUM_GUIDE.md) | 课程式学习路径 | ⭐⭐⭐ |
| [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) | 实现细节深度解析 | ⭐⭐⭐⭐ |
| [docs/论文全文.md](docs/论文全文.md) | 学术背景和理论基础 | ⭐⭐⭐⭐⭐ |

---

## ✅ 学习自检清单

学完本指南后，你应该能够：

- [ ] **运行** 实验并查看结果图表
- [ ] **解释** 三种业务类型的区别
- [ ] **描述** 传统算法和增强算法的核心差异
- [ ] **理解** MAPPO的工作原理（观察→决策→执行→反馈→学习）
- [ ] **读取** 实验结果的JSON数据文件
- [ ] **修改** config.py中的参数并观察影响
- [ ] **定位** 问题代码的位置（使用本文档的"关键位置速查"）

**如果能打勾✅以上所有项，恭喜你已经掌握了这个系统的基础！**

---

## 🆘 遇到问题怎么办？

### **第一步：查阅本文档**
- 使用 `Ctrl+F` 搜索关键词
- 查看"故障排查"章节

### **第二步：查看完整架构文档**
- 打开 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)
- 搜索相关模块的详细说明

### **第三步：检查日志文件**
```bash
# 查看最近的错误信息
type experiment_results\training_logs\*.log | findstr /i "error warning"

# 查看实验运行的详细过程
type experiment_results\exp3_mappo_summary.json
```

### **第四步：小规模复现**
```bash
# 用--small模式快速定位问题
.\venv\Scripts\python.exe main.py --exp mappo --small
```

---

## 🎉 下一步行动建议

### **如果你是开发者**:
1. 运行一次完整的实验流程
2. 尝试修改一个参数（如 `GLOBAL_SEED`）
3. 添加一个新的评价指标
4. 设计一个新的实验场景

### **如果你是论文作者**:
1. 整理实验四的五场景数据表格
2. 截取关键的对比图表
3. 撰写"实验设置"和"结果分析"章节
4. 准备答辩PPT的可视化素材

### **如果你是评审者**:
1. 查看 `experiment_results/` 下的原始数据
2. 验证统计显著性检验的结果
3. 检查图表的Y轴范围是否合理
4. 对比不同场景下的性能表现

---

## 📞 最后的话

**记住三个要点**:

1. **config.py是配置中心** - 改参数先找这里
2. **experiments.py是指挥官** - 看流程看这里
3. **satisfaction.py是评分员** - 理解好坏的标准在这里

**遇到问题时**:
- 不要慌！先查日志，再查文档
- 小规模复现是调试的好帮手
- 三层数据保护让你不会丢失成果

**祝你使用愉快！** 🚀

---

**版本历史**:
- v2.0 (2026-05-11): 全面更新，增加故障排查和学习路径
- v1.0 (2026-04-20): 初始版本

**维护者**: UAV Research Team  
**最后更新**: 2026-05-11
