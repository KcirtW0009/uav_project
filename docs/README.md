# 5G无人机异构网络智能切换决策系统

## 项目简介

本系统针对5G无人机(UAV)网络中的切换决策问题，设计了**业务感知增强算法**和**BA-MAPPO多智能体强化学习算法**，并通过四组递进式实验验证了其性能优势。

### 核心创新
1. **业务感知增强切换算法**：基于QoS优先级的多维度效用函数 + 动态阈值 + 降级搜索 + 回滚保护 + 全局负载均衡
2. **BA-MAPPO（业务感知多智能体近端策略优化）**：融合模仿学习预训练、策略蒸馏、注意力机制的MAPPO架构

---

## 项目结构

```
uav_project/
├── main.py                    # ★ 主入口：统一运行所有实验
├── requirements.txt           # Python依赖
├── .gitignore                 # Git忽略规则
│
├── docs/                      # ★ 项目文档
│   ├── 论文全文.md             #   毕业论文全文
│   └── README.md              #   本文档
│
├── uav_system/                # ★ 核心代码包(24个模块)
│   ├── config.py              #   全局配置(种子/颜色/matplotlib)
│   ├── business.py            #   业务类型定义与QoS配置
│   ├── entities.py            #   UAV/基站实体定义
│   ├── environment.py         #   网络仿真环境(EnhancedNetworkEnvironment)
│   ├── algorithms.py          #   切换算法(传统A3 / 增强算法)
│   ├── recognition.py         #   业务识别模型(训练/推理)
│   ├── satisfaction.py        #   分层满意度度量
│   ├── visualization.py       #   可视化工具(图表生成)
│   ├── reward_functions.py    #   MAPPO奖励函数设计
│   ├── mappo_agent_v2.py      #   MAPPO Agent (Actor-Critic + Attention)
│   ├── mappo_environment.py   #   MAPPO专用环境(MultiAgentHandoverEnv)
│   ├── mappo_optimized_config.py # MAPPO超参数配置
│   ├── enhanced_observation.py    # 增强观测空间设计
│   ├── communication_metrics.py   # 通信指标计算
│   ├── experiments.py         # ★ 实验1-4实现(含可视化)
│   └── experiments_mappo.py   # ★ MAPPO训练与评估
│
├── experiment_results/        # 实验输出结果(图表+数据)
├── experiment_logs/           # 训练日志与模型检查点
├── figures/                   # 论文配图
├── archive/                   # 归档文件(旧脚本/数据/结果)
└── venv/                      # Python虚拟环境(含PyTorch等依赖)
```

---

## 快速开始

### 环境要求
- Python 3.9+
- PyTorch >= 1.12
- numpy, scipy, matplotlib, scikit-learn, pandas

```bash
# 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 运行实验

```bash
# 默认: 运行实验3 (主场景三算法对比: 传统 vs 增强 vs MAPPO)
.\venv\Scripts\python.exe main.py

# 运行全部实验 (实验1/2/2b/2c/3/4，不含MAPPO)
.\venv\Scripts\python.exe main.py --all

# 实验3 + MAPPO三算法对比
.\venv\Scripts\python.exe main.py --exp 3 --include-mappo

# 实验4 + MAPPO泛化评估 (五场景)
.\venv\Scripts\python.exe main.py --exp 4 --include-mappo

# 训练MAPPO模型 (8基站×300UAV)
.\venv\Scripts\python.exe main.py --exp mappo

# 小规模调试 (128UAV/3BS, 快速验证)
.\venv\Scripts\python.exe main.py --exp mappo --small

# 加载已有MAPPO模型跳过训练，直接评估
.\venv\Scripts\python.exe main.py --exp mappo --rl-load
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp` | 实验ID: 1, 2, 2b, 2c, 3, 4, mappo | 3 |
| `--all` | 运行所有非MAPPO实验 | False |
| `--include-mappo` | 在实验3/4中集成MAPPO评估 | False |
| `--retrain` | 强制重新训练业务识别模型 | False |
| `--rl-load` | MAPPO加载已有模型 | False |
| `--small` | 小规模快速调试模式 | False |

---

## 实验体系

| 实验 | 名称 | 内容 | 对应论文章节 |
|------|------|------|-------------|
| **实验1** | 业务识别模块验证 | 多分类器对比、特征工程效果 | 4.2.1 |
| **实验2** | 切换算法消融分析 | 6版本逐步添加机制(基线→动态阈值→权重→ε-greedy→负载→识别) | 4.3.1-4.3.4 |
| **实验2b** | 动态阈值专项分析 | 阈值参数敏感性分析 | 4.3.x |
| **实验2c** | 自适应识别更新机制 | 识别置信度衰减与重训练触发 | 4.3.x |
| **实验3** | 三算法主对比 | 传统A3 vs 增强算法 vs BA-MAPPO (8BS×300UAV×10次重复) | 4.4.1-4.4.3 |
| **实验4** | 五场景泛化测试 | 智慧城市/工业巡检/农业植保/应急救援/物流配送 (300-500 UAV) | 4.5.1-4.5.4 |
| **MAPPO训练** | BA-MAPPO训练 | 500 episodes × 350步/episode | 3.4-3.5 |

---

## 核心算法参数说明

### EnhancedHandoverAlgorithm 的 weight_config 选项

| 配置名 | 使用场景 | 权重来源 | 控制信令权重(sinr/load/rate) |
|--------|---------|----------|---------------------------|
| `'new_env'` | **默认** - 所有主实验(1/2/3/4) | 贝叶斯优化(2026-04-20) | 0.695/0.050/0.255 |
| `'optimized'` | MAPPO对比实验中增强算法侧 | 手动调优 | 0.65/0.10/0.25 |
| `'paper'` | 论文原始参数(旧环境) | 经验设定 | 0.50/0.20/0.30 |

> **当前默认值为 `new_env`**，论文中的所有表4-2数据均基于此配置产生。

### 关键全局参数 (`uav_system/config.py`)
```
GLOBAL_SEED = 30042     # 随机种子
RESULT_DIR = "experiment_results"  # 结果输出目录
INTERRUPTION_CONFIG = {threshold: 0.3, duration: 5}  # 中断检测
```

---

## 输出结果

运行后结果保存在以下位置：
- **实验数据**: `experiment_results/exp{N}_data.{pkl,json}`
- **实验图表**: `experiment_results/exp{N}_results.png`
- **模型可视化**: `experiment_results/model_visualization.png`
- **MAPPO模型**: `experiment_results/mappo_models/mappo_8bs_300uav.pt`

---

## 归档目录 (`archive/`)

包含所有开发过程中的临时脚本、旧版结果和调试文件：
- `archive/scripts/` - 已归档的临时脚本 (~100个)
- `archive/data/` - 已归档的数据文件 (.pkl/.json/.txt)
- `archive/results/` - 已归档的旧结果目录

> 这些文件不影响核心功能，仅作历史参考保留。

---

## 注意事项

1. **必须使用 venv 中的 Python** 运行，不要用系统Python
2. MAPPO训练需要GPU支持(CUDA)，CPU模式下会非常慢
3. 完整运行全部实验(`--all --include-mappo`)预计需要数小时
4. 随机种子已固定为30042，确保可复现性
