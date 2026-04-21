"""
无人机业务识别与切换决策联动系统 — 主入口

【必须使用 venv 中的 Python 运行（不要用系统Python）】

用法:
    .\venv\Scripts\python.exe main.py                        默认: 实验3
    .\venv\Scripts\python.exe main.py --all                    全部实验(无MAPPO)
    .\venv\Scripts\python.exe main.py --exp 3 --include-mappo   实验3 + 三算法对比
    .\venv\Scripts\python.exe main.py --exp 4 --include-mappo   实验4 + MAPPO泛化评估

BA-MAPPO 训练:
    .\venv\Scripts\python.exe main.py --exp mappo              训练MAPPO (8BSx300UAV, 对齐实验3)
    .\venv\Scripts\python.exe main.py --exp mappo --rl-load     加载模型，仅评估
    .\venv\Scripts\python.exe main.py --exp mappo --rl-phase phase  仅训练阶段
    .\venv\Scripts\python.exe main.py --exp mappo --small        小规模调试 (128UAV/3BS)

参数说明:
    --include-mappo:  实验3/4中集成MAPPO，实现三算法对比（传统 vs 增强 vs MAPPO）
    --mappo-model:    指定MAPPO模型路径（默认自动检测 8BSx300UAV 模型）
    --small:           缩减训练规模（快速调试用）
    --retrain:         强制重新训练业务识别模型

架构说明:
    - 实验1/2/2b/2c: 业务识别 + 切换算法设计验证
    - 实验3: 增强算法 vs 传统算法 全面对比 (8BS×300UAV, ~77%负载率)
             可选 +MAPPO 实现三算法统计检验(t-test/Wilcoxon/p值)
    - 实验4: 多场景泛化测试 (5个典型5G应用场景, 300-500 UAV)
             可选 +MAPPO 零样本泛化评估
    - MAPPO训练: BA-MAPPO 多智能体强化学习 (V12 Reward增强)
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.visualization import RecognitionModelVisualizer
from uav_system.experiments import Experiment1, Experiment2, Experiment2b, Experiment2c, Experiment3, Experiment4


def main(force_retrain=False, run_experiments=None,
         rl_load=False, rl_phase='both', small_scale=False,
         include_mappo=False, mappo_model_path=None):
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

    # V17: 始终加载业务识别模型（评估阶段需要接入预测噪声）
    if not only_mappo:
        print("\n步骤1: 初始化业务识别模型...")
    else:
        print("\n步骤1: 初始化业务识别模型（MAPPO评估阶段使用）...")

    recognition_model, all_model_results = train_or_load_recognition_model(
        force_retrain=force_retrain, compare_models=True, verbose=True
    )
    scaler = recognition_model.scaler

    if all_model_results is None and not force_retrain:
        import pickle
        all_results_file = "all_model_results.pkl"
        if os.path.exists(all_results_file):
            with open(all_results_file, 'rb') as f:
                all_model_results = pickle.load(f)

    recognition_model.print_model_info()

    results = {}
    exp_map = {
        '1': lambda: Experiment1.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2': lambda: Experiment2.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2b': lambda: Experiment2b.run(recognition_model, scaler, num_steps=150, repeats=8),
        '2c': lambda: Experiment2c.run(recognition_model, scaler, num_steps=200, repeats=6),
        '3': lambda: Experiment3.run(recognition_model, scaler, include_mappo=include_mappo,
                                       mappo_model_path=mappo_model_path),
        '4': lambda: Experiment4.run(recognition_model, scaler, num_steps=150, repeats=10,
                                       include_mappo=include_mappo,
                                       mappo_model_path=mappo_model_path),
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
    args = parser.parse_args()

    if args.all:
        run_experiments = [1, 2, '2b', '2c', 3, 4]
    else:
        run_experiments = [int(e) if isinstance(e, str) and e.isdigit() else e for e in args.exp]

    set_global_seed(GLOBAL_SEED)
    main(force_retrain=args.retrain, run_experiments=run_experiments,
         rl_load=args.rl_load, rl_phase=args.rl_phase, small_scale=args.small,
         include_mappo=args.include_mappo, mappo_model_path=args.mappo_model)


# ==================== 快速参考 ====================
# .\venv\Scripts\python.exe main.py                              默认: 实验3
# .\venv\Scripts\python.exe main.py --all                          全部实验(无MAPPO)
# .\venv\Scripts\python.exe main.py --exp 3 --include-mappo         实验3 + 三算法对比
# .\venv\Scripts\python.exe main.py --exp 4 --include-mappo         实验4 + MAPPO泛化
# .\venv\Scripts\python.exe main.py --exp mappo                    训练MAPPO(8BSx300UAV)
#     --mappo-model results/mappo_models/custom_model.pt           指定自定义模型
