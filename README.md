# UAV业务识别与切换决策联动系统

## 📋 项目概述

本项目是一个基于**多智能体强化学习(MAPPO)**的无人机(UAV)网络切换决策系统，用于解决5G网络中UAV的业务识别与基站切换优化问题。

系统实现了从传统算法到增强算法再到深度强化学习的完整技术路线，包含**四大核心实验**和**全面的性能评估体系**。

## ✨ 核心功能

### 🎯 业务识别模块
- 基于机器学习的业务类型自动识别（决策树/随机森林等）
- 支持三种5G典型业务类型：
  - **控制信令** (URLLC) - 遥控指令、状态上报
  - **视频回传** (eMBB) - 实时视频流传输
  - **环境监测** (mMTC) - 传感器数据采集

### 🔄 切换决策算法
- **传统算法**: 基于RSS的贪婪切换策略
- **增强算法**: 融合动态阈值、业务权重、ε-greedy探索、负载均衡等6大优化机制
- **MAPPO算法**: 基于多智能体近端策略优化的深度强化学习方法

### 📊 实验验证体系
- **实验1**: 业务识别准确性影响分析（5级准确率测试）
- **实验2**: 增强机制有效性验证（逐步添加机制A/B测试）
- **实验3**: 三算法全面对比（8BS×300UAV场景，17个评估指标）⭐核心实验
- **实验4**: 多场景泛化能力测试（5个典型5G应用场景）
- **MAPPO训练**: BA-MAPPO多智能体强化学习训练与评估

## 🛠️ 技术栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 编程语言 | Python | 3.8+ (推荐3.9+) |
| 数值计算 | NumPy | ≥1.21.0 |
| 数据处理 | Pandas | ≥1.3.0 |
| 可视化 | Matplotlib | ≥3.4.0 |
| 统计图表 | Seaborn | ≥0.11.0 |
| 机器学习 | Scikit-learn | ≥1.0.0 |
| 科学计算 | SciPy | ≥1.7.0 |
| 深度学习 | PyTorch | 用于MAPPO训练 |

## 📦 安装步骤

### 1. 克隆仓库
```bash
git clone https://github.com/KcirtW0009/uav_project.git
cd uav_project
```

### 2. 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

> ⚠️ **注意**: 如需运行MAPPO相关功能，还需安装PyTorch:
> ```bash
> pip install torch torchvision
> ```

## 🚀 使用方法

### 快速开始
```bash
# 运行默认实验（实验3）
python main.py

# 运行全部基础实验（不含MAPPO）
python main.py --all

# 仅运行指定实验
python main.py --exp 3        # 仅实验3
python main.py --exp 4        # 仅实验4
python main.py --exp 3 4      # 同时运行实验3和4
```

### 三算法对比模式（含MAPPO）
```bash
# 实验3 + 传统 vs 增强 vs MAPPO 对比
python main.py --exp 3 --include-mappo

# 使用缓存加速（节省51小时！强烈推荐）
python main.py --exp 3 4 --include-mappo --use-cache
```

### MAPPO训练与评估
```bash
# 训练MAPPO模型
python main.py --exp mappo

# 加载已有模型进行评估
python main.py --exp mappo --rl-load

# 小规模快速调试（128 UAV）
python main.py --exp mappo --small
```

### 其他常用选项
```bash
--retrain          # 强制重新训练识别模型
--force-compare    # 强制重新对比识别模型并选取最优
--mappo-model PATH # 指定MAPPO模型文件路径
```

## 📁 项目结构

```
uav_project/
├── main.py                      # 主程序入口（参数解析与调度）
├── requirements.txt             # Python依赖列表
├── README.md                    # 项目说明文档
├── LICENSE                      # MIT许可证
│
├── plot_exp2_figures.py         # 实验2可视化脚本
├── plot_exp3_figures.py         # 实验3可视化脚本
├── plot_exp4_figures.py         # 实验4可视化脚本
├── redraw_figures_large_font.py # 图表重绘工具
│
└── uav_system/                  # 核心系统模块
    ├── __init__.py              # 包初始化
    ├── config.py                # 全局配置中心（超参数/颜色方案/MAPPO配置）
    ├── environment.py           # 网络仿真环境（UAV/基站建模）
    ├── entities.py              # 实体定义（UAV/基站/业务类）
    ├── business.py              # 业务类型定义（QoS需求/特征生成）
    ├── recognition.py           # 业务识别模块（ML模型训练/预测）
    ├── algorithms.py            # 切换决策算法（传统/增强算法实现）
    ├── reward_functions.py      # MAPPO奖励函数设计
    ├── satisfaction.py          # 满意度评估模块
    ├── communication_metrics.py # 通信指标计算
    ├── enhanced_observation.py  # 增强观测空间设计
    ├── visualization.py         # 可视化工具函数
    │
    ├── mappo_environment.py     # MAPPO专用环境封装
    ├── mappo_agent_v2.py        # MAPPO智能体V2（PPO+中心化价值函数）
    ├── mappo_optimized_config.py# MAPPO优化配置
    ├── experiments.py           # 实验管理中心（Experiment1-4调度）
    ├── experiments_mappo.py     # MAPPO实验执行器
    │
    └── deprecated/              # 已废弃的历史版本代码
        ├── rl_agent.py
        ├── qmix_agent.py
        └── ...
```

## 🔬 实验结果预期

### 输出文件结构
运行完成后在 `experiment_results/` 目录下生成：
- `exp3_data.json` / `exp4_data.json` - 实验统计数据
- `exp3_results.png` / `exp4_results.png` - 可视化图表
- `separated_figs/` - 分离式高清图表（32张PNG）
- `mappo_models/` - 训练好的MAPPO模型文件（.pt）
- `training_logs/` - 训练过程日志

## 📊 评估指标（17项）

系统从以下维度进行全面评估：
- **连接质量**: 切换成功率、平均切换次数、中断率
- **用户体验**: 用户满意度（加权QoS满足度）、关键业务满意度
- **资源效率**: 负载标准差、带宽利用率、 Jain's公平性指数
- **通信性能**: 平均吞吐量、平均时延、丢包率
- **算法特性**: 探索利用率、负载均衡改善度、决策时间


## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 👨‍💻 作者

**KcirtW0009**  
邮箱: KcirtW0009@outlook.com

## 🙏 致谢

- 感谢指导老师的悉心指导
- 感谢开源社区提供的优秀工具（NumPy, PyTorch, Scikit-learn等）
- 本项目为本科毕业设计作品

---

⭐ 如果这个项目对你有帮助，请给一个Star支持一下！
