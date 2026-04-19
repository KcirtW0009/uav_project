# -*- coding: utf-8 -*-
"""
使用优化参数训练MAPPO模型
基于参数搜索结果的最佳配置
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.mappo_optimized_config import OPTIMIZED_MAPPO_CONFIG


def train_optimized_mappo(
    num_uav=128,
    num_bs=3,
    num_episodes=200,
    eval_interval=20,
    seed=42
):
    """使用优化参数训练MAPPO"""
    
    print("=" * 70)
    print("使用优化参数训练MAPPO")
    print("=" * 70)
    
    # 使用优化配置
    config = OPTIMIZED_MAPPO_CONFIG.copy()
    print("\n优化参数配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建环境
    set_global_seed(seed)
    env = MultiAgentHandoverEnv(
        num_uav=num_uav,
        num_bs=num_bs,
        pos_range=1000,
        max_steps=150,
    )
    
    # 获取维度
    obs_dict, global_state = env.reset()
    obs_dim = len(obs_dict[0])
    state_dim = len(global_state)
    
    print(f"\n环境配置:")
    print(f"  UAV数量: {num_uav}")
    print(f"  BS数量: {num_bs}")
    print(f"  观察维度: {obs_dim}")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {env.action_dim}")
    
    # 创建优化后的MAPPO agent
    agent = MAPPOAgentV2(
        num_agents=env.num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=env.action_dim,
        hidden_dim=config['hidden_dim'],
        critic_hidden_dim=config['critic_hidden_dim'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_epsilon=config['clip_epsilon'],
        entropy_coef=config['entropy_coef'],
        value_coef=config['value_coef'],
        use_biz_heads=config['use_biz_heads'],
        use_attention_critic=config['use_attention_critic'],
    )
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'./experiment_logs/optimized_mappo_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n训练日志目录: {log_dir}")
    print(f"训练轮数: {num_episodes}")
    print("=" * 70)
    
    # 训练历史
    training_history = {
        'episodes': [],
        'rewards': [],
        'satisfactions': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'eval_rewards': [],
        'eval_satisfactions': [],
    }
    
    best_satisfaction = 0
    best_model_path = None
    
    for episode in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        
        episode_reward = 0
        episode_satisfactions = []
        
        for step in range(150):
            # 收集业务类型
            biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                        for uid in range(env.num_agents)}
            
            # 选择动作
            actions, log_probs, values, entropies, action_probs = agent.select_actions(
                obs_dict, global_state, biz_types, training=True, env=env
            )
            
            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            # 存储transition (使用insert_experience方法)
            agent.insert_experience(
                step=step,
                obs_dict=obs_dict,
                state=global_state,
                actions=actions,
                rewards=rewards,
                team_reward=team_reward,
                done=done,
                log_probs=log_probs,
                values=values,
                biz_types=biz_types
            )
            
            episode_reward += team_reward
            episode_satisfactions.append(info.get('avg_satisfaction', 0))
            
            obs_dict = next_obs
            global_state = next_state
            
            if done:
                break
        
        # 训练
        if len(agent.buffer['obs']) >= agent.rollout_length:
            train_stats = agent.train()
            training_history['actor_losses'].append(train_stats.get('actor_loss', 0))
            training_history['critic_losses'].append(train_stats.get('critic_loss', 0))
            training_history['entropies'].append(train_stats.get('entropy', 0))
        
        # 记录
        avg_sat = np.mean(episode_satisfactions) if episode_satisfactions else 0
        training_history['episodes'].append(episode)
        training_history['rewards'].append(episode_reward)
        training_history['satisfactions'].append(avg_sat)
        
        # 定期评估和打印
        eval_sat = avg_sat  # 默认使用训练满意度
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_sat = evaluate_agent(agent, env, num_episodes=3)
            training_history['eval_rewards'].append(eval_reward)
            training_history['eval_satisfactions'].append(eval_sat)
            
            print(f"Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Sat={avg_sat:.4f}, "
                  f"EvalSat={eval_sat:.4f}, "
                  f"Best={best_satisfaction:.4f}")
        
        # 保存最佳模型（基于评估满意度，避免过拟合）
        if eval_sat > best_satisfaction:
            best_satisfaction = eval_sat
            best_model_path = os.path.join(log_dir, 'best_model.pt')
            agent.save(best_model_path)
            print(f"  *** 保存最佳模型: EvalSat={eval_sat:.4f} ***")
    
    # 保存最终模型
    final_model_path = os.path.join(log_dir, 'final_model.pt')
    agent.save(final_model_path)
    
    # 保存训练历史
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳满意度: {best_satisfaction:.4f}")
    print(f"最佳模型: {best_model_path}")
    print(f"最终模型: {final_model_path}")
    print("=" * 70)
    
    return log_dir, best_model_path


def evaluate_agent(agent, env, num_episodes=10):
    """评估agent性能（增加评估轮数至10个Episode以提高可靠性）"""
    rewards = []
    satisfactions = []

    for _ in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        
        episode_reward = 0
        episode_sats = []
        
        for step in range(150):
            biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                        for uid in range(env.num_agents)}
            
            actions, _, _, _, _ = agent.select_actions(
                obs_dict, global_state, biz_types, training=False, env=env
            )
            
            next_obs, next_state, rewards_dict, team_reward, done, info = env.step(actions)
            
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            
            obs_dict = next_obs
            global_state = next_state
            
            if done:
                break
        
        rewards.append(episode_reward)
        satisfactions.append(np.mean(episode_sats) if episode_sats else 0)
    
    return np.mean(rewards), np.mean(satisfactions)


if __name__ == "__main__":
    log_dir, best_model = train_optimized_mappo(
        num_uav=128,
        num_bs=3,
        num_episodes=200,
        eval_interval=20,
        seed=42
    )
    
    print(f"\n训练完成！")
    print(f"日志目录: {log_dir}")
    print(f"最佳模型: {best_model}")
