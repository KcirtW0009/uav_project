"""
MAPPO标准环境训练启动脚本

整合所有优化配置，直接在标准环境中运行MAPPO训练。
支持CPU/GPU自动切换，包含训练监控和早停机制。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入CUDA修复模块
from cuda_fix import setup_environment

# 设置环境
DEVICE = setup_environment()

# 尝试导入PyTorch（处理可能的导入失败）
try:
    import torch
    TORCH_OK = True
    print(f"[OK] PyTorch {torch.__version__} 已加载")
    print(f"[OK] 设备: {DEVICE.upper()}")
except (OSError, ImportError) as e:
    TORCH_OK = False
    print(f"[ERROR] PyTorch加载失败: {e}")
    print("\n" + "=" * 60)
    print("PyTorch DLL加载失败解决方案:")
    print("=" * 60)
    print("1. 安装Visual C++ Redistributable:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("")
    print("2. 重新安装PyTorch (CPU版本，避免CUDA问题):")
    print("   pip uninstall torch -y")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("")
    print("3. 或者使用纯numpy环境进行算法验证")
    print("=" * 60)
    print("\n推荐执行:")
    print("  1. 下载并安装 VC_redist.x64.exe")
    print("  2. 在venv中执行:")
    print("     .\\venv\\Scripts\\activate")
    print("     pip uninstall torch -y")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("")
    print("如果仍有问题，运行 python verify_env.py 进行详细诊断")
    print("=" * 60)
    sys.exit(1)

# 导入配置和实验模块
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.experiments_mappo import ExperimentBAMAPPO
from mappo_standard_config import get_config, print_config


def run_mappo_training(config_type="standard", phase="both"):
    """
    运行MAPPO训练
    
    Args:
        config_type: "standard" | "small_test"
        phase: "both" | "phase1" | "phase2"
    """
    # 获取配置
    config = get_config(config_type)
    
    print("\n" + "=" * 80)
    print(f"BA-MAPPO 标准环境训练")
    print("=" * 80)
    print(f"  设备: {DEVICE.upper()}")
    print(f"  配置类型: {config_type}")
    print(f"  运行阶段: {phase}")
    
    # 打印配置
    print_config(config)
    
    # 设置随机种子
    set_global_seed(GLOBAL_SEED)
    
    # 运行实验
    print("\n开始训练...")
    print("=" * 80 + "\n")
    
    try:
        results = ExperimentBAMAPPO.run(
            # 环境配置
            num_uav_list=config["num_uav_list"],
            num_bs_list=config["num_bs_list"],
            num_steps=config["num_steps"],
            bs_capacity_range=config["bs_capacity_range"],
            pos_range=config["pos_range"],
            
            # 训练配置
            train_episodes=config["train_episodes"],
            eval_episodes=config["eval_episodes"],
            rollout_length=config["rollout_length"],
            
            # 网络配置
            hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["critic_hidden_dim"],
            use_biz_heads=config["use_biz_heads"],
            use_attention_critic=config["use_attention_critic"],
            
            # 学习率
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            
            # PPO参数
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_epsilon=config["clip_epsilon"],
            entropy_coef=config["entropy_coef"],
            value_loss_coef=config["value_loss_coef"],
            
            # 批次参数
            batch_size=config["batch_size"],
            num_parallel_envs=config["num_parallel_envs"],
            
            # 其他配置
            load_models=False,
            phase=phase,
            verbose=True,
        )
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        
        # 打印关键结果
        if 'training' in results:
            print("\n训练结果摘要:")
            for num_uav, tr_data in results['training'].items():
                if isinstance(tr_data, dict) and 'final_avg_sat' in tr_data:
                    print(f"  UAV={num_uav}: 最终满意度={tr_data['final_avg_sat']:.4f}")
        
        if 'evaluation' in results:
            print("\n评估结果摘要:")
            for num_uav, ev_data in results['evaluation'].items():
                if isinstance(ev_data, dict) and 'mappo' in ev_data:
                    mappo_avg = ev_data['mappo']['avg']
                    print(f"  UAV={num_uav}: 满意度={mappo_avg[0]:.4f}+/-{mappo_avg[1]:.4f}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return None
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BA-MAPPO 标准环境训练")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='standard',
        choices=['standard', 'small_test'],
        help='配置类型: standard(标准) 或 small_test(小规模测试)'
    )
    parser.add_argument(
        '--phase', '-p',
        type=str,
        default='both',
        choices=['both', 'phase1', 'phase2'],
        help='运行阶段: both(训练+评估) 或 phase1(仅训练) 或 phase2(仅评估)'
    )
    parser.add_argument(
        '--uav', '-u',
        type=int,
        nargs='+',
        default=None,
        help='指定UAV数量列表 (例如: 30 80 150)'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=None,
        help='训练episode数量'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='设备选择: auto(自动) 或 cpu 或 cuda'
    )
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config(args.config)
    
    # 命令行参数覆盖配置
    if args.uav:
        config["num_uav_list"] = tuple(args.uav)
        # 自动调整BS数量
        config["num_bs_list"] = tuple(min(max(u // 10, 4), 10) for u in args.uav)
    
    if args.episodes:
        config["train_episodes"] = args.episodes
    
    # 设备选择
    if args.device != 'auto':
        global DEVICE
        DEVICE = args.device
    
    # 运行训练
    run_mappo_training(config_type=args.config, phase=args.phase)


if __name__ == "__main__":
    main()
