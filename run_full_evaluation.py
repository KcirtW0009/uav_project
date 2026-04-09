# -*- coding: utf-8 -*-
"""
完整评估方案

使用已保存的优化模型，在原有评估框架中进行完整评估
对比：传统算法、增强算法、BA-MAPPO（优化后）
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.experiments_mappo import ExperimentBAMAPPO


def load_and_evaluate_model(model_path, num_uav=128, num_bs=3, num_episodes=3, seed=42):
    """
    加载已保存的模型并进行完整评估
    
    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估模型: {model_path}")
    print(f"{'='*60}")
    
    set_global_seed(seed)
    
    # 创建环境
    env = QMixHandoverEnv(
        num_uav=num_uav,
        num_bs=num_bs,
        pos_range=1000,
        seed=seed
    )
    
    # 获取维度
    obs_dict, global_state = env.reset()
    obs_dim = len(obs_dict[0])
    state_dim = len(global_state)
    action_dim = env.action_dim
    
    print(f"观测维度: {obs_dim}, 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建agent（使用优化后的参数）
    agent = MAPPOAgentV2(
        num_agents=num_uav,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        critic_hidden_dim=256,
        actor_lr=3e-5,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.99,
        clip_epsilon=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        rollout_length=150,
        num_epochs=5,
        batch_size=256,
        use_biz_heads=True,
        use_attention_critic=True,
        train_sample_agents=50,
        attention_sample_agents=50,
    )
    
    # 加载模型
    agent.load(model_path)
    print(f"模型已加载")
    
    # 进行评估
    results = {
        'satisfaction': [],
        'handover_success': [],
        'handover_latency': [],
        'ping_jitter': [],
        'packet_loss': [],
        'qos_violation': [],
        'rewards': [],
    }
    
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        
        episode_reward = 0
        episode_sats = []
        
        for step in range(150):
            # 获取业务类型
            biz_types = {}
            for uid in range(env.num_agents):
                uav = env.env.uavs[uid]
                biz_types[uid] = uav.true_business_type.value
            
            # 选择动作
            actions, _, _, _, _ = agent.select_actions(
                obs_dict, global_state, biz_types, training=False, env=env
            )
            
            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            
            obs_dict = next_obs
            global_state = next_state
            
            if done:
                break
        
        results['rewards'].append(episode_reward)
        results['satisfaction'].append(np.mean(episode_sats))
        
        print(f"Episode {ep+1}/{num_episodes}: reward={episode_reward:.2f}, sat={np.mean(episode_sats):.4f}")
    
    # 汇总结果
    summary = {
        'avg_satisfaction': np.mean(results['satisfaction']),
        'std_satisfaction': np.std(results['satisfaction']),
        'avg_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
    }
    
    print(f"\n评估结果汇总:")
    print(f"  平均满意度: {summary['avg_satisfaction']:.4f} ± {summary['std_satisfaction']:.4f}")
    print(f"  平均奖励: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    
    return summary


def find_latest_model():
    """查找最新的模型文件"""
    log_dir = './experiment_logs'
    if not os.path.exists(log_dir):
        return None
    
    dirs = [d for d in os.listdir(log_dir) if d.startswith('mappo_high_')]
    if not dirs:
        return None
    
    latest_dir = sorted(dirs)[-1]
    
    # 优先使用best_model
    model_path = os.path.join(log_dir, latest_dir, 'best_model.pt')
    if os.path.exists(model_path):
        return model_path
    
    model_path = os.path.join(log_dir, latest_dir, 'final_model.pt')
    if os.path.exists(model_path):
        return model_path
    
    return None


def main():
    """主函数"""
    print("="*60)
    print("MAPPO 优化模型完整评估")
    print("="*60)
    
    # 查找最新模型
    model_path = find_latest_model()
    if model_path is None:
        print("错误: 未找到模型文件")
        return
    
    print(f"使用模型: {model_path}")
    
    # 进行评估
    results = load_and_evaluate_model(
        model_path=model_path,
        num_uav=128,
        num_bs=3,
        num_episodes=3,
        seed=42
    )
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    
    # 对比基准
    print("\n与基准对比:")
    print("  原MAPPO实验满意度: ~0.96")
    print(f"  优化后满意度: {results['avg_satisfaction']:.4f}")
    improvement = (results['avg_satisfaction'] - 0.96) / 0.96 * 100
    print(f"  改进: {improvement:+.2f}%")


if __name__ == '__main__':
    main()
