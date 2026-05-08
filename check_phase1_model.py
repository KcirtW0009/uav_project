"""
快速验证脚本: 检查 Phase 1 模型的真实配置并进行正确评估

用途:
1. 检查 phase1_r1.pt 的实际 hidden_dim 配置
2. 使用正确的维度进行评估
3. 对比基线分数，显示真实的提升效果

运行方式:
    python check_phase1_model.py
    
预计耗时: 5~8分钟 (仅评估，无训练)
"""

import sys
import os
import torch
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgent
from uav_system.business import BusinessType


def inspect_model(model_path: str) -> dict:
    """检查模型的实际配置"""
    print(f"\n{'='*70}")
    print(f"[INSPECT] 检查模型文件: {os.path.basename(model_path)}")
    print(f"{'='*70}")
    
    info = {
        'file_size_kb': os.path.getsize(model_path) / 1024,
        'path': model_path,
    }
    
    try:
        # [FIX] PyTorch 2.6+ 兼容性
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"\n  [OK] 模型加载成功")
        print(f"     文件大小: {info['file_size_kb']:.1f} KB")
        print(f"     包含的键: {list(checkpoint.keys())}")
        
        # 检查是否有config信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n  [CONFIG] 模型包含配置信息:")
            for key, value in config.items():
                print(f"     {key}: {value}")
            
            info['has_config'] = True
            info['config'] = config
            info['hidden_dim'] = config.get('hidden_dim', None)
            info['critic_hidden_dim'] = config.get('critic_hidden_dim', None)
        else:
            print(f"\n  [WARN] 模型不包含config信息，需要从权重推断...")
            info['has_config'] = False
        
        # 从权重大小推断hidden_dim
        if 'actor' in checkpoint:
            actor_state = checkpoint['actor']
            print(f"\n  [ACTOR] Actor网络层数: {len(actor_state)}")
            
            for key, tensor in actor_state.items():
                if 'fc1.weight' in key or 'layers.0.weight' in key:
                    inferred_dim = tensor.shape[0]
                    print(f"     [{key}] shape={tensor.shape} → 推断 hidden_dim={inferred_dim}")
                    
                    if info.get('hidden_dim') is None:
                        info['hidden_dim'] = inferred_dim
                    elif info['hidden_dim'] != inferred_dim:
                        print(f"     [WARN] 与之前推断的值({info['hidden_dim']})不一致!")
                    break
            
            # 计算总参数量
            total_params = sum(p.numel() for p in actor_state.values())
            print(f"     总参数量: {total_params:,}")
            info['actor_params'] = total_params
        
        if 'critic' in checkpoint:
            critic_state = checkpoint['critic']
            print(f"\n  [CRITIC] Critic网络层数: {len(critic_state)}")
            
            for key, tensor in critic_state.items():
                if 'fc1.weight' in key or 'layers.0.weight' in key:
                    inferred_critic_dim = tensor.shape[0]
                    print(f"     [{key}] shape={tensor.shape} → 推断 critic_hidden_dim={inferred_critic_dim}")
                    
                    if info.get('critic_hidden_dim') is None:
                        info['critic_hidden_dim'] = inferred_critic_dim
                    elif info['critic_hidden_dim'] != inferred_critic_dim:
                        print(f"     [WARN] 与之前推断的值({info['critic_hidden_dim']})不一致!")
                    break
            
            total_params = sum(p.numel() for p in critic_state.values())
            print(f"     总参数量: {total_params:,}")
            info['critic_params'] = total_params
        
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return info
        
    except Exception as e:
        print(f"\n  [FAIL] 检查失败: {e}")
        info['error'] = str(e)
        return info


def evaluate_with_correct_config(
    model_path: str,
    hidden_dim: int,
    critic_hidden_dim: int,
    num_uav: int,
    scenario_id: str,
    num_eval_episodes: int = 5
) -> float:
    """使用正确的配置评估单个场景"""
    
    satisfactions = []
    scenario_hash = hash(scenario_id) % 10000
    GLOBAL_SEED = 30042  # 使用与主脚本相同的基线种子
    
    for rep in range(num_eval_episodes):
        seed = GLOBAL_SEED + scenario_hash + num_uav * 10 + rep * 7
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = MultiAgentHandoverEnv(
            num_bs=8, num_uav=num_uav,
            max_steps=150,
            seed=seed,
            bs_capacity_range=(500, 1000),
            pos_range=1000,
            scenario=scenario_id,
        )
        
        obs_dict, global_state = env.reset()
        
        # [KEY] 使用正确的维度!
        agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
        )
        
        # 加载模型并捕获可能的错误
        load_success = False
        try:
            agent.load(model_path)
            load_success = True
            
            # [DEBUG] 第一次加载时打印诊断信息
            if rep == 0:
                first_layer = None
                for name, param in agent.actor.named_parameters():
                    if 'weight' in name and len(param.shape) == 2:
                        first_layer = param
                        break
                
                if first_layer is not None:
                    weight_norm = torch.norm(first_layer).item()
                    print(f"         [OK] 权重norm={weight_norm:.2f} (>{100}说明已加载预训练权重)")
                
        except Exception as e:
            if rep == 0:
                print(f"         [FAIL] 加载失败: {e}")
            continue
        
        if not load_success:
            continue
        
        # 评估循环
        for step in range(150):
            biz_types = {
                uid: env.env.uavs[uid].true_business_type.value
                for uid in range(env.num_agents)
            }
            
            actions, _, _, _, _ = agent.select_actions(
                obs_dict, global_state,
                biz_types=biz_types, training=False
            )
            
            obs_dict, global_state, _, _, done, info = env.step(actions)
            
            if done:
                break
        
        # 计算满意度
        final_sat = np.mean([
            env.env.uavs[uid].current_satisfaction
            for uid in range(env.num_agents)
        ])
        
        satisfactions.append(final_sat)
        
        del agent, env
    
    mean_sat = np.mean(satisfactions)
    std_sat = np.std(satisfactions)
    
    return mean_sat, std_sat


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" " * 20 + "Phase 1 模型验证工具")
    print(" " * 15 + "(检查配置 + 正确评估)")
    print("="*70)
    
    # 模型路径
    model_dir = r"experiment_results\mappo_models\finetune_multi_v2"
    model_path = os.path.join(model_dir, "phase1_r1.pt")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        print("请确保已完成 Phase 1 训练")
        return
    
    # Step 1: 检查模型配置
    model_info = inspect_model(model_path)
    
    if 'error' in model_info:
        print("\n[ABORT] 无法继续，模型检查失败")
        return
    
    # 提取正确的维度
    hidden_dim = model_info.get('hidden_dim', 64)
    critic_hidden_dim = model_info.get('critic_hidden_dim', 128)
    
    print(f"\n{'='*70}")
    print(f"[CONFIG] 确定的正确配置:")
    print(f"{'='*70}")
    print(f"     hidden_dim:        {hidden_dim}")
    print(f"     critic_hidden_dim: {critic_hidden_dim}")
    print(f"\n  [COMPARE] 评估函数中硬编码的错误值: hidden_dim=64, critic_hidden_dim=128")
    
    if hidden_dim != 64 or critic_hidden_dim != 128:
        print(f"  [ALERT] ⚠️ 维度不匹配! 这就是'零提升'的根本原因!")
    else:
        print(f"  [INFO] 维度相同(64/128)，可能不是维度问题，需要进一步调查")
    
    # Step 2: 定义场景
    SCENARIOS = {
        'industrial_inspection': {
            'name': '工业巡检',
            'num_uav': 300,
            'expected_sat': 0.96,
        },
        'agriculture': {
            'name': '农业植保',
            'num_uav': 350,
            'expected_sat': 0.93,
        },
        'smart_city': {
            'name': '智慧城市监控',
            'num_uav': 400,
            'expected_sat': 0.90,
        },
        'emergency_rescue': {
            'name': '应急救援',
            'num_uav': 300,
            'expected_sat': 0.95,
        },
        'logistics_delivery': {
            'name': '物流配送',
            'num_uav': 500,
            'expected_sat': 0.88,
        },
    }
    
    # 基线分数（从之前的日志获取）
    BASELINE_SCORES = {
        'industrial_inspection': 0.6735,
        'agriculture': 0.9456,
        'smart_city': 0.6879,
        'emergency_rescue': 0.9068,
        'logistics_delivery': 0.6978,
    }
    
    # Step 3: 使用正确配置进行评估
    print(f"\n{'='*70}")
    print(f"[EVAL] 开始全场景评估 (使用正确的维度配置)...")
    print(f"{'='*70}")
    
    results = {}
    
    for sid, scenario in SCENARIOS.items():
        print(f"\n  评估: {scenario['name']} ({scenario['num_uav']} UAVs)...")
        print(f"       Agent配置: hidden_dim={hidden_dim}, critic_hidden_dim={critic_hidden_dim}")
        
        mean_sat, std_sat = evaluate_with_correct_config(
            model_path=model_path,
            hidden_dim=hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            num_uav=scenario['num_uav'],
            scenario_id=sid,
            num_eval_episodes=5
        )
        
        baseline = BASELINE_SCORES[sid]
        improvement = (mean_sat - baseline) / max(baseline, 1e-6) * 100
        
        results[sid] = {
            'name': scenario['name'],
            'mean': mean_sat,
            'std': std_sat,
            'baseline': baseline,
            'improvement_pct': improvement,
        }
        
        status = "✅" if mean_sat >= scenario['expected_sat'] else ("~" if mean_sat >= baseline * 1.05 else "❌")
        
        print(f"       结果: {mean_sat:.4f} ± {std_sat:.4f} "
              f"(基线: {baseline:.4f}, 变化: {improvement:+.2f}%) {status}")
    
    # Step 4: 汇总报告
    print(f"\n\n{'='*70}")
    print(f"[REPORT] Phase 1 真实效果评估报告")
    print(f"{'='*70}")
    
    print(f"\n  {'场景':12s} │ {'基线':>7s} │ {'Phase1后':>9s} │ {'变化':>8s} │ {'达标?':>6s}")
    print(f"  {'-'*12}-┼-{ '-'*7 }-┼-{ '-'*9 }-┼-{ '-'*8 }-┼-{ '-'*6 }-")
    
    total_baseline = 0
    total_phase1 = 0
    
    for sid, result in results.items():
        baseline = result['baseline']
        phase1 = result['mean']
        change = result['improvement_pct']
        expected = SCENARIOS[sid]['expected_sat']
        
        status = "✅ 达标" if phase1 >= expected else ("~ 接近" if phase1 >= baseline * 1.05 else "❌ 未达")
        
        print(f"  {result['name']:10s} │ {baseline:7.4f} │ {phase1:9.4f} │ {change:+7.2f}% │ {status:>6s}")
        
        total_baseline += baseline
        total_phase1 += phase1
    
    avg_baseline = total_baseline / len(results)
    avg_phase1 = total_phase1 / len(results)
    avg_change = (avg_phase1 - avg_baseline) / max(avg_baseline, 1e-6) * 100
    
    print(f"  {'-'*12}-┼-{ '-'*7 }-┼-{ '-'*9 }-┼-{ '-'*8 }-┼-{ '-'*6 }-")
    print(f"  {'全局平均':8s} │ {avg_baseline:7.4f} │ {avg_phase1:9.4f} │ {avg_change:+7.2f}% │")
    
    print(f"\n  [CONCLUSION]")
    if avg_change > 5:
        print(f"  🎉🎉🎉 Phase 1 训练效果显著! 平均提升 {avg_change:.2f}%")
        print(f"      这证明之前的'零提升'确实是评估bug导致的假象!")
    elif avg_change > 2:
        print(f"  ✅ Phase 1 有一定效果, 提升 {avg_change:.2f}%")
        print(f"      建议继续 Phase 2/3 以获得更好效果")
    elif avg_change > 0:
        print(f"  ⚠️ Phase 1 效果微弱, 仅提升 {avg_change:.2f}%")
        print(f"      可能需要调整超参数或增加训练轮数")
    else:
        print(f"  ❌ Phase 1 无提升甚至退步 ({avg_change:+.2f}%)")
        print(f"      需要深入排查训练过程是否有其他问题")
    
    print(f"\n  [NEXT STEPS]")
    print(f"  1. 如果效果显著(>5%), 可以考虑提前停止或继续优化")
    print(f"  2. 如果效果一般(2-5%), 建议进入 Phase 2 微调")
    print(f"  3. 如果效果不佳(<2%), 需要回顾训练参数设置")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
