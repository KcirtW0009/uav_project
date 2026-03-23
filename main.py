# main.py
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

# 将当前目录添加到 Python 路径（确保 uav_system 能被找到）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.visualization import RecognitionModelVisualizer
from uav_system.experiments import Experiment1, Experiment2, Experiment3, Experiment4, Experiment5

def main(force_retrain=False, run_experiments=None):
    print("\n" + "="*80)
    print("无人机业务识别与切换决策联动系统")
    print("="*80)

    print("\n步骤1: 初始化业务识别模型...")
    recognition_model = train_or_load_recognition_model(
        force_retrain=force_retrain, compare_models=True, verbose=True)
    scaler = recognition_model.scaler

    recognition_model.print_model_info()
    RecognitionModelVisualizer.visualize_model(recognition_model)

    results = {}
    if run_experiments is None:
        run_experiments = [1, 2, 3, 4, 5]

    for exp_id in run_experiments:
        print(f"\n{'='*80}")
        print(f"运行实验 {exp_id}")
        print('='*80)
        if exp_id == 1:
            results['exp1'] = Experiment1.run(recognition_model, scaler, num_steps=150, repeats=5)
        elif exp_id == 2:
            results['exp2'] = Experiment2.run(recognition_model, scaler, num_steps=200, repeats=10)
        elif exp_id == 3:
            results['exp3'] = Experiment3.run(recognition_model, scaler, num_steps=200, repeats=5)
        elif exp_id == 4:
            results['exp4'] = Experiment4.run(recognition_model, scaler, num_steps=150, repeats=3)
        elif exp_id == 5:
            results['exp5'] = Experiment5.run(recognition_model, scaler, num_steps=150, repeats=3)

    print("\n" + "="*80)
    print("所有实验运行完成！")
    print(f"结果已保存至: {os.path.abspath(RESULT_DIR)}")
    print("="*80)
    return results

if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    # main(force_retrain=False, run_experiments=[1, 2, 3, 4, 5])
    main(force_retrain=False, run_experiments=[1])
    # main(force_retrain=False, run_experiments=[1, 2, 3, 4, 5])