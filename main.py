"""
无人机业务识别与切换决策联动系统 — 主入口

【重要】必须使用 venv 中的 Python 运行（不要用系统Python）:
    .\venv\Scripts\python.exe main.py                  # 默认运行实验3
    .\venv\Scripts\python.exe main.py --all            # 运行所有实验(1, 2, 2b, 3, 4, mappo)
    .\venv\Scripts\python.exe main.py --exp mappo      # 运行 BA-MAPPO 多智能体强化学习实验

    BA-MAPPO 实验用法:
    .\venv\Scripts\python.exe main.py --exp mappo --rl-load       # 加载已有模型，跳过训练
    .\venv\Scripts\python.exe main.py --exp mappo --rl-phase phase1  # 仅运行训练阶段
    .\venv\Scripts\python.exe main.py --exp mappo --rl-phase phase2  # 仅运行评估阶段
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
         rl_load=False, rl_phase='both', small_scale=False):
    """
    主函数：初始化模型并运行实验

    Args:
        force_retrain: 是否强制重新训练识别模型
        run_experiments: 要运行的实验列表，如 [1, 2, 3, 4] 或 ['2b', 'mappo']，默认 [3]
        rl_load: RL 实验是否加载已有模型（跳过训练）
        rl_phase: RL 实验运行阶段 'both' / 'phase1' / 'phase2'
    """
    print("\n" + "=" * 80)
    print("无人机业务识别与切换决策联动系统")
    print("=" * 80)

    # 判断是否仅运行 MAPPO 实验（不需要业务识别模型）
    if run_experiments is None:
        run_experiments = [3]
    run_experiments_str = [str(exp) for exp in run_experiments]
    only_mappo = (len(run_experiments_str) == 1 and run_experiments_str[0] == 'mappo')

    recognition_model = None
    scaler = None
    all_model_results = None

    if not only_mappo:
        # 步骤1: 初始化业务识别模型（仅非 MAPPO 实验需要）
        print("\n步骤1: 初始化业务识别模型...")
        recognition_model, all_model_results = train_or_load_recognition_model(
            force_retrain=force_retrain, compare_models=True, verbose=True
        )
        scaler = recognition_model.scaler

        # 如果是加载已有模型，尝试加载保存的模型对比结果
        if all_model_results is None and not force_retrain:
            import pickle
            all_results_file = "all_model_results.pkl"
            if os.path.exists(all_results_file):
                with open(all_results_file, 'rb') as f:
                    all_model_results = pickle.load(f)

        recognition_model.print_model_info()
    else:
        print("\n[跳过] MAPPO 实验不依赖业务识别模型，跳过模型加载")

    # 步骤2: 运行实验
    results = {}

    exp_map = {
        '1': lambda: Experiment1.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2': lambda: Experiment2.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2b': lambda: Experiment2b.run(recognition_model, scaler, num_steps=150, repeats=8),
        '2c': lambda: Experiment2c.run(recognition_model, scaler, num_steps=200, repeats=6),
        '3': lambda: Experiment3.run(recognition_model, scaler),
        '4': lambda: Experiment4.run(recognition_model, scaler, num_steps=150, repeats=10),
        # ---- [已废弃] DQN 单智能体实验 (原 exp 5/5b/5c) ----
        # '5': lambda: _run_exp5_unified(load_models=rl_load,
        #                                  phase=rl_phase,
        #                                  ablation_train_episodes=rl_abl_ep),
        # '5b': lambda: _run_exp5_unified(load_models=rl_load,
        #                                   phase=rl_phase,
        #                                   ablation_train_episodes=rl_abl_ep),
        # '5c': lambda: _run_exp5_unified(load_models=rl_load,
        #                                   phase=rl_phase,
        #                                   ablation_train_episodes=rl_abl_ep),
        # ---- [已废弃] QMIX 多智能体实验 ----
        # 'qmix': lambda: _run_exp_qmix(load_models=rl_load,
        #                                 phase=rl_phase),
        'mappo': lambda: _run_exp_mappo(load_models=rl_load,
                                          phase=rl_phase,
                                          small_scale=small_scale),
    }

    for exp_id_str in run_experiments_str:
        print(f"\n{'=' * 80}")
        print(f"运行实验 {exp_id_str}")
        print('=' * 80)
        if exp_id_str in exp_map:
            results[f'exp{exp_id_str}'] = exp_map[exp_id_str]()
        else:
            print(f"警告: 未知的实验ID '{exp_id_str}', 跳过")

    # 步骤3: 生成可视化（仅非 MAPPO 实验）
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


# ---- [已废弃] DQN 单智能体实验 ----
# def _run_exp5_unified(load_models=False, phase='both', ablation_train_episodes=None):
#     from experiments_rl import Experiment5Unified
#     return Experiment5Unified.run(...)
#
# def _run_exp_qmix(load_models=False, phase='both'):
#     from uav_system.experiments_qmix import ExperimentQMIX
#     return ExperimentQMIX.run(...)


def _run_exp_mappo(load_models=False, phase='both', small_scale=False):
    """运行 BA-MAPPO 多智能体强化学习实验

    环境配置: BS 容量 (500, 1000) Mbps, pos_range=1000m
    通过 UAV/BS 数量比控制负载率:
      小规模: 150 UAV / 3 BS → 负载率 ~103%
      大规模: 200 UAV / 4 BS → ~103%, 280 UAV / 5 BS → ~116%
    """
    from uav_system.experiments_mappo import ExperimentBAMAPPO
    # 统一容量范围 (500, 1000) Mbps — 与实验2/3/4保持一致
    # 负载率: 小规模 150UAV/3BS→~103%, 标准 200UAV/4BS→~103%, 280UAV/5BS→~116%
    _cap = (500, 1000)
    if small_scale:
        return ExperimentBAMAPPO.run(
            num_uav_list=(150,),
            num_bs_list=(3,),       # 4→3，提高单基站负载
            num_steps=50,
            train_episodes=200,
            eval_episodes=3,
            bs_capacity_range=_cap,
            pos_range=1000,
            load_models=load_models,
            phase=phase,
            verbose=True,
            train_sample_agents=50,
            attention_sample_agents=50,
            num_parallel_envs=1,
        )
    return ExperimentBAMAPPO.run(
        num_uav_list=(200, 280),
        num_bs_list=(4, 5),
        num_steps=100,
        train_episodes=1000,
        eval_episodes=10,
        bs_capacity_range=_cap,
        pos_range=1000,
        load_models=load_models,
        phase=phase,
        verbose=True,
        train_sample_agents=50,
        attention_sample_agents=50,
        num_parallel_envs=4,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='无人机业务识别与切换决策联动系统')
    parser.add_argument('--exp', nargs='+', default=[3],
                        help='要运行的实验，如: --exp 1 2 2b 3 4 mappo（默认: 3）')
    parser.add_argument('--all', action='store_true',
                        help='运行所有实验 (1, 2, 2b, 3, 4, mappo)')
    parser.add_argument('--retrain', action='store_true',
                        help='强制重新训练识别模型')
    parser.add_argument('--rl-load', action='store_true',
                        help='RL 实验: 加载已有模型跳过训练（仅评估）')
    parser.add_argument('--rl-phase', choices=['both', 'phase1', 'phase2'], default='both',
                        help='RL 实验: 运行阶段 (both=全部, phase1=训练, phase2=评估)')
    parser.add_argument('--small', action='store_true',
                        help='小规模测试: 大幅缩减训练/评估规模，用于快速调试')
    args = parser.parse_args()

    if args.all:
        run_experiments = [1, 2, '2b', '2c', 3, 4, 'mappo']
    else:
        run_experiments = [int(e) if isinstance(e, str) and e.isdigit() else e for e in args.exp]

    set_global_seed(GLOBAL_SEED)
    main(force_retrain=args.retrain, run_experiments=run_experiments,
         rl_load=args.rl_load, rl_phase=args.rl_phase, small_scale=args.small)


# 【必须使用venv】.\venv\Scripts\python.exe main.py                    只运行实验3
# 【必须使用venv】.\venv\Scripts\python.exe main.py --all              运行全部实验
# 【必须使用venv】.\venv\Scripts\python.exe main.py --exp mappo        运行 BA-MAPPO 多智能体实验
# 【必须使用venv】.\venv\Scripts\python.exe main.py --exp mappo --rl-load  加载已有模型，仅评估
# 【必须使用venv】.\venv\Scripts\python.exe main.py --exp mappo --rl-phase phase1  仅运行训练阶段
# 【必须使用venv】.\venv\Scripts\python.exe main.py --exp mappo --rl-phase phase2  仅运行评估阶段
# venv\Scripts\python.exe main.py --exp mappo --rl-phase both --small
