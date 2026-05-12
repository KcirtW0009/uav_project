"""
=============================================================================
  UAV业务识别与切换决策联动系统 - 主程序入口 (main.py)
=============================================================================

【文件定位】
本文件是整个系统的**唯一入口点**，负责：
1. 解析命令行参数
2. 初始化业务识别模型
3. 调度实验执行（实验1-4 + MAPPO训练）
4. 汇总结果并生成可视化

【运行环境要求】
✅ Python版本: 3.8+ (推荐3.9+)
✅ 必须使用虚拟环境: .\venv\Scripts\python.exe
✅ 不要使用系统Python（可能缺少依赖包）

【快速开始】
```bash
# 0. 进入项目目录
cd "f:/桌面/本科毕业论文/结题/uav_project"

# 1. 激活虚拟环境 (可选，如果已激活可跳过)
.\venv\Scripts\activate

# 2. 运行实验 (选择以下任一命令)
```

【常用命令速查表】

┌─────────────────────────────────────────────────────────────────────┐
│ 命令                                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 【基础实验】                                                         │
│   python main.py                        默认运行实验3               │
│   python main.py --all                  运行全部实验(无MAPPO)        │
│   python main.py --exp 3                仅运行实验3                   │
│   python main.py --exp 4                仅运行实验4                   │
│   python main.py --exp 3 4              同时运行实验3和4             │
│                                                                     │
│ 【三算法对比模式】(含MAPPO)                                           │
│   python main.py --exp 3 --include-mappo                            │
│         实验3 + 传统 vs 增强 vs MAPPO                                │
│                                                                     │
│   python main.py --exp 3 4 --include-mappo                           │
│         实验3+4 + 三算法对比                                         │
│                                                                     │
│ 【缓存加速模式】(强烈推荐!)                                            │
│   python main.py --exp 3 4 --include-mappo --use-cache              │
│         实验3+4 + 三算法对比 + 跳过传统/增强(节省51小时)            │
│                                                                     │
│ 【MAPPO训练/评估】                                                    │
│   python main.py --exp mappo              训练MAPPO模型              │
│   python main.py --exp mappo --rl-load     加载模型仅评估           │
│   python main.py --exp mappo --small        小规模调试(128UAV)       │
│                                                                     │
│ 【其他选项】                                                         │
│   --retrain          强制重新训练识别模型                             │
│   --force-compare    强制重新对比识别模型(dt/rf等)，选取最优          │
│   --mappo-model PATH 指定MAPPO模型路径                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【参数详细说明】

--include-mappo [可选]
  作用: 在实验3/4中集成MAPPO评估，实现三种算法的全面对比
  影响: 
    - 增加约4-20小时运行时间（取决于场景数量）
    - 输出额外的统计显著性检验结果(t-test/Wilcoxon)
    - 生成包含MAPPO的三方对比图表
  
--use-cache [可选, 强烈推荐]
  作用: 从已有的JSON文件读取传统/增强算法数据，跳过重新运行
  适用场景:
    - 已经运行过传统/增强算法，只需重新跑MAPPO
    - 调试MAPPO参数时反复测试
  时间节省:
    - 实验3: 节省 ~14小时
    - 实验4: 节省 ~37小时
    - 总计: 节省 ~51小时!

--mappo-model PATH [可选]
  作用: 指定要使用的MAPPO模型文件路径
  默认值: experiment_results/mappo_models/mappo_8bs_300uav_best.pt
  常用选项:
    - mappo_8bs_300uav_best.pt  (最佳模型，推荐)
    - mappo_8bs_300uav.pt      (最终模型)
    - 自定义路径

--small [仅MAPPO训练时有效]
  作用: 缩减训练规模用于快速调试
  配置变化:
    - UAV数量: 300 → 128
    - BS数量: 8 → 3  
    - 训练步数: 350 → 50
    - 训练轮数: 500 → 100
  适用: 快速验证代码正确性，不适合正式实验

--force-compare [可选]
  作用: 强制重新对比所有识别模型类型(dt/rf/svm/knn等)
  默认行为: 如果已有保存模型，直接加载不重新对比
  使用时机: 怀疑当前模型不是最优时

【实验架构说明】

整个系统包含5大模块:

┌──────────────────────────────────────────────────────────────────┐
│                        系统架构图                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐                                                  │
│  │  main.py   │ ← 你在这里! 主入口，负责参数解析和调度            │
│  └─────┬──────┘                                                  │
│        │                                                          │
│        ▼                                                          │
│  ┌────────────┐     ┌──────────────────────────────────────┐     │
│  │ recognition │     │         experiments.py               │     │
│  │    .py      │────▶│  实验管理中心                         │     │
│  │ 业务识别模型 │     │  • Experiment1-4                    │     │
│  │ (dt/rf)     │     │  • 数据收集/统计/检验/可视化          │     │
│  └────────────┘     └──────────┬───────────────────────────┘     │
│                                │                                 │
│        ┌───────────────────────┼──────────────────────┐         │
│        ▼                       ▼                      ▼         │
│  ┌──────────┐          ┌────────────┐        ┌────────────┐    │
│  │ environ- │          │ algorithms │        │ mappo_*    │    │
│  │ ment.py  │          │   .py      │        │ .py        │    │
│  │ 网络仿真 │          │ 切换算法   │        │ MAPPO组件  │    │
│  └──────────┘          └────────────┘        └────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

【实验内容详解】

实验1: 业务识别准确性验证
  目的: 验证不同识别准确率对系统性能的影响
  方法: 设置5种准确率等级(100%/90%/80%/60%/33%)进行对比
  输出: 准确率-性能关系曲线

实验2: 增强机制有效性验证
  目的: 验证每个增强机制的独立贡献
  方法: 逐步添加机制(动态阈值→业务权重→ε-greedy→负载均衡...)
  输出: 各机制性能提升柱状图

实验3: 增强算法 vs 传统算法 全面对比 ⭐核心实验
  场景: 8个基站 × 300架UAV (~77%负载率)
  方法: 10次重复实验(不同随机种子)，17个指标全面对比
  可选: +MAPPO 实现三算法统计显著性检验
  输出: 对比表格、统计检验报告、16张可视化图表

实验4: 多场景泛化能力测试
  场景: 5个典型5G应用场景(密集城区/高速移动/广域覆盖/热点/混合)
  方法: 测试算法在不同UAV数量(300-500)下的泛化能力
  可选: +MAPPO 零样本泛化评估(用实验3训练的模型直接测试)
  输出: 场景间性能热力图、雷达图

MAPPO训练: BA-MAPPO多智能体强化学习
  环境: 与实验3对齐(8BS×300UAV, 350步)
  算法: PPO + 中心化价值函数
  Reward: V12增强版(满意度+切换成功+负载均衡+关键业务优先)
  输出: 训练曲线、最优模型(.pt文件)

【预期运行时间】

┌──────────┬──────────┬──────────┬──────────────┐
│ 实验     │ 不含MAPPO│ 含MAPPO  │ 缓存模式      │
├──────────┼──────────┼──────────┼──────────────┤
│ 实验3    │ ~14小时  │ ~18小时  │ **~4小时**   │
│ 实验4    │ ~37小时  │ ~57小时  │ **~20小时**  │
│ MAPPO训练│ -        │ ~12小时  │ -            │
├──────────┼──────────┼──────────┼──────────────┤
│ 总计     │ ~51小时  │ ~87小时  │ **~24小时**  │
└──────────┴──────────┴──────────┴──────────────┘

> 💡 使用 --use-cache 参数可将总时间从87小时缩减到24小时!

【输出文件结构】

运行完成后，在 experiment_results/ 目录下生成:

experiment_results/
├── exp3_data.json                 # 实验3统计数据(JSON格式)
├── exp3_results.png               # 实验3可视化图表
├── exp3_mappo_summary.json        # MAPPO汇总数据(如启用)
├── exp4_data.json                 # 实验4统计数据
├── exp4_results.png               # 实验4可视化图表
├── separated_figs/                # 分离式图表(32张PNG)
│   ├── exp3_*.png
│   └── exp4_*.png
├── mappo_models/                 # MAPPO模型文件
│   └── mappo_8bs_300uav_best.pt
└── training_logs/                # 训练日志
    └── mappo_*/metrics.json

【常见问题排查】

Q: 提示 "ModuleNotFoundError"?
A: 确保使用 venv 中的Python: .\venv\Scripts\python.exe

Q: 运行很慢怎么办?
A: 添加 --use-cache 参数，可节省51小时

Q: 如何只运行单个实验?
A: 使用 --exp 参数指定: --exp 3 或 --exp 4

Q: MAPPO模型在哪里?
A: 默认路径: experiment_results/mappo_models/mappo_8bs_300uav_best.pt

Q: 如何中断并恢复?
A: 直接Ctrl+C中断，数据会自动保存。下次运行自动从断点继续

【依赖检查】

运行前请确保已安装:
✅ numpy >= 1.19.0
✅ scipy >= 1.5.0 (统计检验需要)
✅ matplotlib >= 3.3.0 (可视化需要)
✅ scikit-learn >= 0.24.0 (识别模型需要)
✅ torch >= 1.7.0 (MAPPO需要)
✅ joblib >= 0.17.0 (模型序列化需要)

安装命令:
  pip install numpy scipy matplotlib scikit-learn torch joblib

【作者】: UAV Research Team
【版本】: v2.0 (2026-05-10 更新)
【联系方式】: 见项目README.md
=============================================================================
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')  # [V27] 抑制sklearn并行配置警告
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.visualization import RecognitionModelVisualizer
from uav_system.experiments import Experiment1, Experiment2, Experiment2b, Experiment2c, Experiment3, Experiment4


def main(force_retrain=False, run_experiments=None,
         rl_load=False, rl_phase='both', small_scale=False,
         include_mappo=False, mappo_model_path=None,
         use_cache=False, mappo_repeats=None):  # [NEW] MAPPO差异化重复次数
    """
    主函数：初始化模型并运行实验

    Args:
        force_retrain: 是否强制重新训练识别模型
        run_experiments: 要运行的实验列表，如 [1, 2, 3, 4] 或 ['2b', 'mappo']，默认 [3]
        rl_load: RL 实验是否加载已有模型（跳过训练）
        rl_phase: RL 实验运行阶段 'both' / 'phase1' / 'phase2'
        small_scale: 是否缩减训练规模（快速调试用）
        include_mappo: 在实验3/4中包含MAPPO评估（三算法对比模式）
        mappo_model_path: MAPPO模型文件路径（None=自动检测默认路径）
        use_cache: 是否读取已有的传统/增强算法数据（跳过重新运行，大幅节省时间）
        mappo_repeats: MAPPO评估的重复次数 (None=与传统/增强算法相同)
                       设置为较小值(如3-5)可大幅缩短时间，同时保持统计公平性
    """
    print("\n" + "=" * 80)
    print("无人机业务识别与切换决策联动系统")
    if include_mappo:
        print("  [模式] 三算法对比: 传统算法 vs 增强算法 vs MAPPO")
    print("=" * 80)

    if run_experiments is None:
        run_experiments = [3]
    run_experiments_str = [str(exp) for exp in run_experiments]
    only_mappo = (len(run_experiments_str) == 1 and run_experiments_str[0] == 'mappo')

    recognition_model = None
    scaler = None
    all_model_results = None

    # [V28] 判断是否需要加载业务识别模型（实验3/4已切断识别模块）
    needs_recognition = any(exp in run_experiments_str for exp in ['1', '2', '2b', '2c', '3', 'mappo'])  # [回档版] 实验三使用识别模型

    if needs_recognition:
        # V17: 始终加载业务识别模型（评估阶段需要接入预测噪声）
        if not only_mappo:
            print("\n步骤1: 初始化业务识别模型...")
        else:
            print("\n步骤1: 初始化业务识别模型（MAPPO评估阶段使用）...")

        recognition_model, all_model_results = train_or_load_recognition_model(
            force_retrain=force_retrain, compare_models=True, verbose=True,
            force_compare=args.force_compare
        )
        scaler = recognition_model.scaler

        if all_model_results is None and not force_retrain:
            import pickle
            all_results_file = "all_model_results.pkl"
            if os.path.exists(all_results_file):
                with open(all_results_file, 'rb') as f:
                    all_model_results = pickle.load(f)

        recognition_model.print_model_info()
    else:
        # [V28] 实验3/4不需要识别模块，跳过加载以加速启动
        print("\n[INFO] 跳过业务识别模型加载 (实验3/4使用真实业务类型)")
        print("  → 使用 ground truth 业务类型，零识别误差")
        print("  → 启动速度更快，无sklearn警告")

    results = {}
    exp_map = {
        '1': lambda: Experiment1.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2': lambda: Experiment2.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2b': lambda: Experiment2b.run(recognition_model, scaler, num_steps=150, repeats=8),
        '2c': lambda: Experiment2c.run(recognition_model, scaler, num_steps=200, repeats=6),
        '3': lambda: Experiment3.run(recognition_model, scaler,  # [回档版] 使用识别模型
                                       include_mappo=include_mappo,
                                       mappo_model_path=mappo_model_path,
                                       use_cache=use_cache,
                                       mappo_repeats=mappo_repeats),  # [NEW] MAPPO差异化重复
        '4': lambda: Experiment4.run(None, None, num_steps=350, repeats=5,  # [V30] 优化参数：350步×5次
                                      include_mappo=include_mappo,
                                      mappo_model_path=mappo_model_path,
                                      use_cache=use_cache),  # [V27] 统一repeats=5
        'mappo': lambda: _run_exp_mappo(load_models=rl_load,
                                          phase=rl_phase,
                                          small_scale=small_scale,
                                          recognition_model=recognition_model,
                                          scaler=scaler),
    }

    for exp_id_str in run_experiments_str:
        print(f"\n{'=' * 80}")
        print(f"运行实验 {exp_id_str}")
        print('=' * 80)
        if exp_id_str in exp_map:
            results[f'exp{exp_id_str}'] = exp_map[exp_id_str]()
        else:
            print(f"警告: 未知的实验ID '{exp_id_str}', 跳过")

    # 可视化
    if not only_mappo and recognition_model is not None:
        print("\n" + "=" * 80)
        print("所有实验运行完成！")
        print(f"结果已保存至: {os.path.abspath(RESULT_DIR)}")
        print("=" * 80)
        print("\n生成模型可视化...")
        RecognitionModelVisualizer.visualize_model(recognition_model, all_model_results, show=False)
    else:
        print("\n" + "=" * 80)
        print("MAPPO 实验运行完成！")
        print(f"结果已保存至: {os.path.abspath(RESULT_DIR)}")
        print("=" * 80)

    return results


def _run_exp_mappo(load_models=False, phase='both', small_scale=False,
                   recognition_model=None, scaler=None):
    """运行 BA-MAPPO 多智能体强化学习实验
    
    训练配置 (Step3 已对齐实验3):
      环境: 8 BS × 300 UAV ≈ 77%负载率
      步数: 350 步/episode (与实验3一致)
      训练: 500 episodes (增强确保低负载下充分收敛)
      评估: 10 episodes × 350步
      Reward: V12 (V11 + 负载自适应 + 目标差距 + 同类排名)
      
    小规模调试 (--small):
      128 UAV / 3 BS, 50步, 100 episodes (快速验证代码正确性)
    
    Args:
        load_models: 加载已有模型，跳过训练
        phase: 运行阶段 'both' / 'phase1' / 'phase2'
        small_scale: 小规模快速调试模式
    """
    from uav_system.experiments_mappo import ExperimentBAMAPPO
    _cap = (500, 1000)
    
    if small_scale:
        # 快速调试: 小规模环境
        # V17: 缩小容量范围使负载率达到 ~78%（与正式模式对齐）
        #   128 UAV / 3 BS, 容量(50, 90) → 总容=210~270, 负载率=47%~61%
        #   容量(40, 75) → 总容=120~225, 负载率=57%~107%
        _cap_small = (45, 80)
        return ExperimentBAMAPPO.run(
            num_uav_list=(128,),
            num_bs_list=(3,),
            num_steps=60,
            train_episodes=60,          # V17: 60轮足够看趋势
            eval_episodes=5,
            bs_capacity_range=_cap_small,
            pos_range=800,
            load_models=load_models,
            phase=phase,
            verbose=True,
            train_sample_agents=20,
            attention_sample_agents=20,
            num_parallel_envs=1,
        )
    
    # 标准训练: 对齐实验3环境 (8BS×300UAV, ~77%负载率)
    # V17: 评估阶段传入业务识别模型（带预测噪声，更接近真实部署场景）
    return ExperimentBAMAPPO.run(
        num_uav_list=(300,),       # 与实验3一致的UAV数量
        num_bs_list=(8,),          # 与实验3一致的BS数量
        num_steps=350,             # 与实验3一致的步数
        train_episodes=500,        # 增强轮次确保低负载下收敛
        eval_episodes=10,
        bs_capacity_range=_cap,
        pos_range=1000,
        load_models=load_models,
        phase=phase,
        verbose=True,
        train_sample_agents=50,
        attention_sample_agents=50,
        num_parallel_envs=2,       # 并行加速数据收集
        recognition_model=recognition_model,
        scaler=scaler,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='无人机业务识别与切换决策联动系统')
    parser.add_argument('--exp', nargs='+', default=[3],
                        help='要运行的实验，如: --exp 1 2 2b 3 4 mappo（默认: 3）')
    parser.add_argument('--all', action='store_true',
                        help='运行所有实验 (1, 2, 2b, 3, 4)')
    parser.add_argument('--retrain', action='store_true',
                        help='强制重新训练识别模型')
    parser.add_argument('--rl-load', action='store_true',
                        help='RL实验: 加载已有模型跳过训练')
    parser.add_argument('--rl-phase', choices=['both', 'phase1', 'phase2'], default='both',
                        help='RL实验: 运行阶段 (both/phase1/phase2)')
    parser.add_argument('--small', action='store_true',
                        help='小规模测试: 缩减训练规模，用于快速调试')
    parser.add_argument('--include-mappo', action='store_true',
                        help='在实验3/4中集成MAPPO评估，实现三算法对比')
    parser.add_argument('--mappo-model', type=str, default=None,
                        help='指定MAPPO模型路径（默认自动检测 mappo_models/mappo_8bs_300uav.pt）')
    parser.add_argument('--force-compare', action='store_true',
                        help='强制重新对比所有识别模型并选取最优（忽略已有模型）')
    parser.add_argument('--use-cache', action='store_true',
                        help='实验3/4: 读取已有的传统/增强算法数据（跳过重新运行，大幅节省时间）')
    parser.add_argument('--no-cache', action='store_true',
                        help='实验3/4: 强制不使用缓存，完整重新运行所有算法（默认行为，可省略）')
    parser.add_argument('--mappo-repeats', type=int, default=None,
                        help='MAPPO评估的重复次数 (默认: 与传统/增强算法相同，即repeats参数值)。'
                             '设置为较小值(如3-5)可大幅缩短实验4时间，同时保持统计公平性。'
                             '示例: --mappo-repeats 5 表示MAPPO只跑5次，传统/增强从缓存中采样5次')
    args = parser.parse_args()

    if args.all:
        run_experiments = [1, 2, '2b', '2c', 3, 4]
    else:
        run_experiments = [int(e) if isinstance(e, str) and e.isdigit() else e for e in args.exp]

    # 处理缓存参数：--no-cache 优先级高于 --use-cache
    use_cache = args.use_cache and not args.no_cache

    set_global_seed(GLOBAL_SEED)
    main(force_retrain=args.retrain, run_experiments=run_experiments,
         rl_load=args.rl_load, rl_phase=args.rl_phase, small_scale=args.small,
         include_mappo=args.include_mappo, mappo_model_path=args.mappo_model,
         use_cache=use_cache, mappo_repeats=args.mappo_repeats)  # [NEW] 传入MAPPO重复次数参数


# ==================== 快速参考 ====================
# .\venv\Scripts\python.exe main.py                              默认: 实验3
# .\venv\Scripts\python.exe main.py --all                          全部实验(无MAPPO)
# .\venv\Scripts\python.exe main.py --exp 3 --include-mappo         实验3 + 三算法对比
# .\venv\Scripts\python.exe main.py --exp 4 --include-mappo         实验4 + MAPPO泛化
# .\venv\Scripts\python.exe main.py --exp mappo                    训练MAPPO(8BSx300UAV)
#     --mappo-model results/mappo_models/custom_model.pt           指定自定义模型
#
# 缓存模式 (快速!):
# .\venv\Scripts\python.exe main.py --exp 3 4 --include-mappo --use-cache  完整三算法对比 (~几小时)
#     从已有数据加载传统/增强算法，仅运行MAPPO评估（纯净版）
