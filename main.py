"""
无人机业务识别与切换决策联动系统 — 主入口

用法:
    python main.py                  # 默认运行实验3
    python main.py --all            # 运行所有实验(1, 2, 2b, 3, 4, 5)
    python main.py --exp 5          # 运行统一 RL 实验（原 5+5b+5c）
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.visualization import RecognitionModelVisualizer
from uav_system.experiments import Experiment1, Experiment2, Experiment2b, Experiment3, Experiment4


def main(force_retrain=False, run_experiments=None):
    """
    主函数：初始化模型并运行实验

    Args:
        force_retrain: 是否强制重新训练识别模型
        run_experiments: 要运行的实验列表，如 [1, 2, 3, 4] 或 ['2b', 5]，默认 [3]
    """
    print("\n" + "=" * 80)
    print("无人机业务识别与切换决策联动系统")
    print("=" * 80)

    # 步骤1: 初始化业务识别模型
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

    # 步骤2: 运行实验
    results = {}
    if run_experiments is None:
        run_experiments = [3]

    run_experiments_str = [str(exp) for exp in run_experiments]

    exp_map = {
        '1': lambda: Experiment1.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2': lambda: Experiment2.run(recognition_model, scaler, num_steps=150, repeats=10),
        '2b': lambda: Experiment2b.run(recognition_model, scaler, num_steps=150, repeats=8),
        '3': lambda: Experiment3.run(recognition_model, scaler),
        '4': lambda: Experiment4.run(recognition_model, scaler, num_steps=150, repeats=10),
        '5': lambda: _run_exp5_unified(),
        '5b': lambda: _run_exp5_unified(),   # 向后兼容，重定向到统一实验
        '5c': lambda: _run_exp5_unified(),   # 向后兼容，重定向到统一实验
    }

    for exp_id_str in run_experiments_str:
        print(f"\n{'=' * 80}")
        print(f"运行实验 {exp_id_str}")
        print('=' * 80)
        if exp_id_str in exp_map:
            results[f'exp{exp_id_str}'] = exp_map[exp_id_str]()
        else:
            print(f"警告: 未知的实验ID '{exp_id_str}', 跳过")

    # 步骤3: 生成可视化
    print("\n" + "=" * 80)
    print("所有实验运行完成！")
    print(f"结果已保存至: {os.path.abspath(RESULT_DIR)}")
    print("=" * 80)

    print("\n生成模型可视化...")
    RecognitionModelVisualizer.visualize_model(recognition_model, all_model_results, show=False)

    return results


def _run_exp5_unified():
    """运行统一 RL 实验（合并原 5 + 5b + 5c）"""
    from experiments_rl import Experiment5Unified
    return Experiment5Unified.run(
        uav_counts=(10, 20, 30, 40),
        num_steps=150,
        repeats=10,
        num_bs=8,
        target_uav_id=0,
        dqn_train_episodes=1000,
        bs_capacity_range=(250, 450),
        verbose=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='无人机业务识别与切换决策联动系统')
    parser.add_argument('--exp', nargs='+', default=[1, 2, '2b', 3, 4],
                        help='要运行的实验，如: --exp 1 2 2b 3 4 5（默认: 1 2 2b 3 4）')
    parser.add_argument('--all', action='store_true',
                        help='运行所有实验 (1, 2, 2b, 3, 4, 5)')
    parser.add_argument('--retrain', action='store_true',
                        help='强制重新训练识别模型')
    args = parser.parse_args()

    if args.all:
        run_experiments = [1, 2, '2b', 3, 4, 5]
    else:
        run_experiments = [int(e) if e.isdigit() else e for e in args.exp]

    set_global_seed(GLOBAL_SEED)
    main(force_retrain=args.retrain, run_experiments=run_experiments)

# python main.py 	只运行实验3（和之前一样）
# python main.py --all 运行全部实验
# python main.py --exp 5 运行统一 RL 实验（原 5+5b+5c）
