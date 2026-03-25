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
from uav_system.experiments import Experiment1, Experiment2, Experiment2b, Experiment3, Experiment4

def main(force_retrain=False, run_experiments=None):
    print("\n" + "="*80)
    print("无人机业务识别与切换决策联动系统")
    print("="*80)

    print("\n步骤1: 初始化业务识别模型...")
    recognition_model, all_model_results = train_or_load_recognition_model(
        force_retrain=force_retrain, compare_models=True, verbose=True)
    scaler = recognition_model.scaler

    # 如果是加载已有模型，尝试加载保存的模型对比结果
    if all_model_results is None and not force_retrain:
        import pickle
        all_results_file = "all_model_results.pkl"
        if os.path.exists(all_results_file):
            with open(all_results_file, 'rb') as f:
                all_model_results = pickle.load(f)

    recognition_model.print_model_info()

    results = {}
    if run_experiments is None:
        run_experiments = [1, 2, 3, 4]

    # 将数字转换为字符串,统一处理
    run_experiments_str = [str(exp) for exp in run_experiments]

    for exp_id_str in run_experiments_str:
        print(f"\n{'='*80}")
        print(f"运行实验 {exp_id_str}")
        print('='*80)
        if exp_id_str == '1':
            results['exp1'] = Experiment1.run(recognition_model, scaler, num_steps=150, repeats=10)  # 实验1：识别准确性的价值验证
        elif exp_id_str == '2':
            results['exp2'] = Experiment2.run(recognition_model, scaler, num_steps=150, repeats=10)  # 实验2：机制有效性验证
        elif exp_id_str == '2b':
            results['exp2b'] = Experiment2b.run(recognition_model, scaler, num_steps=150, repeats=8)  # 实验2b：机制组合验证
        elif exp_id_str == '3':
            results['exp3'] = Experiment3.run(recognition_model, scaler)  # 实验3：增强算法 vs 传统算法（使用方案C默认参数）
        elif exp_id_str == '4':
            results['exp4'] = Experiment4.run(recognition_model, scaler, num_steps=150, repeats=10)  # 实验4：多场景对比
        else:
            print(f"警告: 未知的实验ID '{exp_id_str}', 跳过")

    print("\n" + "="*80)
    print("所有实验运行完成！")
    print(f"结果已保存至: {os.path.abspath(RESULT_DIR)}")
    print("="*80)

    # 在所有实验完成后生成可视化
    print("\n生成模型可视化...")
    RecognitionModelVisualizer.visualize_model(recognition_model, all_model_results, show=False)

    return results

if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)
    # 运行实验2b（机制组合验证）
    # 可以传入数字或字符串: [1, 2, '2b', 3, 4]
    main(force_retrain=False, run_experiments=[3])
    # 运行所有实验
    # main(force_retrain=False, run_experiments=[ 1,2,'2b', 4])