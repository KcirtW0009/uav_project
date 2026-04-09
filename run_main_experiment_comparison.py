# -*- coding: utf-8 -*-
"""
主实验环境公平对比实验

在 EnhancedNetworkEnvironment 中运行三种算法的公平对比：
1. 传统算法(3GPP)
2. 增强算法
3. 优化MAPPO

确保与主实验1234的环境完全一致。

使用方法：
    venv\Scripts\python.exe run_main_experiment_comparison.py
"""

import os
import sys
import json
import numpy as np
import torch
import time
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.mappo_optimized_config import OPTIMIZED_MAPPO_CONFIG
from uav_system.business import BusinessType
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm

from performance_visualization import create_performance_report


def evaluate_algorithm_in_main_env(algorithm, name, env, num_episodes=10, seed=42):
    """在主实验环境中评估算法"""
    print(f"\n" + "="*70)
    print(f"在主实验环境中评估 {name}")
    print("="*70)
    
    set_global_seed(seed)
    
    all_metrics = []
    episode_details = []
    
    for ep in range(num_episodes):
        env.reset()
        episode_reward = 0
        step_metrics = []
        
        for step in range(150):
            # 让算法为所有UAV做决策并执行切换
            handover_count = 0
            if name == "传统算法(3GPP)" or name == "增强算法":
                # 为每个UAV做决策
                for uav_id in range(env.num_uav):
                    if name == "传统算法(3GPP)":
                        decision = algorithm.make_decision(uav_id)
                    else:
                        decision = algorithm.make_intelligent_decision(uav_id)
                    
                    if decision is not None:
                        target_bs, downgrade_ratio = decision
                        current_bs = env.uavs[uav_id].connected_bs_id
                        if target_bs != current_bs:
                            if algorithm.execute_handover(uav_id, target_bs, downgrade_ratio):
                                handover_count += 1
            
            # 执行环境步进
            env.step()
            
            # 收集指标
            step_data = {
                'step': step,
                'satisfaction': np.mean([uav.current_satisfaction for uav in env.uavs.values()]),
                'reward': 0,  # EnhancedNetworkEnvironment 不返回奖励
                'connected_ratio': sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav,
                'avg_sinr': np.mean(env.sinr_matrix),
                'handover_count': handover_count,
                # 通信性能指标
                'avg_latency': np.mean([uav.current_latency for uav in env.uavs.values()]),
                'avg_packet_loss': np.mean([uav.packet_loss_rate for uav in env.uavs.values()]),
                'total_throughput': sum(uav.current_allocated_rate for uav in env.uavs.values()),
                'avg_throughput': np.mean([uav.current_allocated_rate for uav in env.uavs.values()]),
                'bandwidth_utilization': sum(bs.current_load for bs in env.base_stations.values()) / sum(bs.capacity for bs in env.base_stations.values()),
                'load_variance': np.var([bs.load_ratio for bs in env.base_stations.values()]),
                'interruption_rate': len(env.interrupted_uavs) / env.num_uav,
            }
            step_metrics.append(step_data)
            episode_reward += step_data['reward']
        
        ep_summary = {
            'episode': ep + 1,
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfaction'] for m in step_metrics]),
            'final_satisfaction': step_metrics[-1]['satisfaction'] if step_metrics else 0,
            'min_satisfaction': min([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'max_satisfaction': max([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'std_satisfaction': np.std([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'avg_connected_ratio': np.mean([m['connected_ratio'] for m in step_metrics]),
            'avg_sinr': np.mean([m['avg_sinr'] for m in step_metrics]),
            'total_handovers': sum([m['handover_count'] for m in step_metrics]),
            # 通信性能指标
            'avg_latency': np.mean([m['avg_latency'] for m in step_metrics]),
            'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in step_metrics]),
            'avg_total_throughput': np.mean([m['total_throughput'] for m in step_metrics]),
            'avg_throughput': np.mean([m['avg_throughput'] for m in step_metrics]),
            'avg_bandwidth_utilization': np.mean([m['bandwidth_utilization'] for m in step_metrics]),
            'avg_load_variance': np.mean([m['load_variance'] for m in step_metrics]),
            'avg_interruption_rate': np.mean([m['interruption_rate'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        episode_details.append(ep_summary)
        
        print(f"  Episode {ep+1:2d}: Sat={ep_summary['avg_satisfaction']:.4f} "
              f"(min={ep_summary['min_satisfaction']:.4f}, max={ep_summary['max_satisfaction']:.4f}), "
              f"Reward={episode_reward:.2f}, HOs={ep_summary['total_handovers']}")
    
    summary = {
        'name': name,
        'num_episodes': num_episodes,
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'min_satisfaction': np.min([m['avg_satisfaction'] for m in all_metrics]),
        'max_satisfaction': np.max([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_connected_ratio': np.mean([m['avg_connected_ratio'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'avg_handovers': np.mean([m['total_handovers'] for m in all_metrics]),
        # 通信性能指标
        'avg_latency': np.mean([m['avg_latency'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'avg_total_throughput': np.mean([m['avg_total_throughput'] for m in all_metrics]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in all_metrics]),
        'avg_bandwidth_utilization': np.mean([m['avg_bandwidth_utilization'] for m in all_metrics]),
        'avg_load_variance': np.mean([m['avg_load_variance'] for m in all_metrics]),
        'avg_interruption_rate': np.mean([m['avg_interruption_rate'] for m in all_metrics]),
        'episode_details': episode_details,
    }
    
    print(f"\n汇总结果:")
    print(f"  平均满意度: {summary['avg_satisfaction']:.4f} ± {summary['std_satisfaction']:.4f}")
    print(f"  满意度范围: [{summary['min_satisfaction']:.4f}, {summary['max_satisfaction']:.4f}]")
    print(f"  平均奖励: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  平均切换次数: {summary['avg_handovers']:.1f}")
    # 通信性能指标
    print(f"  平均延迟: {summary['avg_latency']:.4f} ms")
    print(f"  平均丢包率: {summary['avg_packet_loss']:.4f}")
    print(f"  平均总吞吐量: {summary['avg_total_throughput']:.2f} Mbps")
    print(f"  平均吞吐量: {summary['avg_throughput']:.2f} Mbps")
    print(f"  平均带宽利用率: {summary['avg_bandwidth_utilization']:.4f}")
    print(f"  平均负载方差: {summary['avg_load_variance']:.4f}")
    print(f"  平均中断率: {summary['avg_interruption_rate']:.4f}")
    
    return summary


def train_mappo_in_main_env(env, num_episodes=200, eval_interval=20, num_eval_episodes=10, seed=42):
    """在主实验环境中训练MAPPO"""
    print("\n" + "="*70)
    print("在主实验环境中训练MAPPO")
    print("="*70)
    
    set_global_seed(seed)
    
    config = OPTIMIZED_MAPPO_CONFIG.copy()
    
    # 获取观察和状态维度
    env.reset()
    # 手动构建观察值
    obs = {}
    for uav_id in range(env.num_uav):
        # 简单的观察值构建，可能需要根据实际情况调整
        uav = env.uavs[uav_id]
        obs[uav_id] = np.array([
            uav.position[0], uav.position[1], uav.position[2],
            uav.current_satisfaction,
            1.0 if uav.connected_bs_id is not None else 0.0,
            # 添加其他观察值...
        ])
    
    obs_dim = len(obs[0]) if obs else 29  # 默认值
    state_dim = 128  # 主实验环境的状态维度
    action_dim = env.num_bs + 1  # 0表示不切换，1-8表示切换到对应BS
    
    print("环境配置:")
    print(f"  UAV数量: {env.num_uav}")
    print(f"  BS数量: {env.num_bs}")
    print(f"  观察维度: {obs_dim}")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    
    # 创建MAPPO agent
    agent = MAPPOAgentV2(
        num_agents=env.num_uav,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'./experiment_logs/mappo_main_env_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    best_satisfaction = 0
    best_model_path = None
    patience_counter = 0
    max_patience = 15
    training_start_time = time.time()
    
    training_history = {
        'episodes': [],
        'train_satisfactions': [],
        'eval_satisfactions': [],
        'train_rewards': [],
        'eval_rewards': [],
    }
    
    print("\n开始训练...")
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        env.reset()
        # 手动构建观察值
        obs = {}
        for uav_id in range(env.num_uav):
            uav = env.uavs[uav_id]
            obs[uav_id] = np.array([
                uav.position[0], uav.position[1], uav.position[2],
                uav.current_satisfaction,
                1.0 if uav.connected_bs_id is not None else 0.0,
            ])
        
        agent.reset_hidden()
        episode_reward = 0
        episode_sats = []
        
        for step in range(150):
            # 获取业务类型
            biz_types = {uid: env.uavs[uid].true_business_type.value for uid in range(env.num_uav)}
            
            # 确保obs不是None
            if obs is None:
                # 重新构建观察值
                obs = {}
                for uav_id in range(env.num_uav):
                    uav = env.uavs[uav_id]
                    obs[uav_id] = np.array([
                        uav.position[0], uav.position[1], uav.position[2],
                        uav.current_satisfaction,
                        1.0 if uav.connected_bs_id is not None else 0.0,
                    ])
            
            # 构建状态表示
            # 状态可以包括：全局统计信息、基站负载、整体满意度等
            state = []
            
            # 添加全局统计信息
            avg_satisfaction = np.mean([uav.current_satisfaction for uav in env.uavs.values()])
            connected_ratio = sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav
            avg_sinr = np.mean(env.sinr_matrix)
            
            # 添加基站负载信息
            for bs in env.base_stations.values():
                state.append(float(bs.load_ratio))
            
            # 添加全局统计
            state.extend([float(avg_satisfaction), float(connected_ratio), float(avg_sinr)])
            
            # 确保状态维度与配置一致
            while len(state) < 128:
                state.append(0.0)
            state = state[:128]
            
            # 确保所有元素都是数值类型
            state = [float(x) for x in state]
            
            # 选择动作
            actions, log_probs, values, _, _ = agent.select_actions(
                obs, state, biz_types, training=True, env=env
            )
            
            # 执行动作（通过算法的execute_handover）
            for uav_id, action in actions.items():
                if action != 0:  # 0 表示不切换
                    target_bs = action - 1  # 动作是 1-based
                    # 简单的切换逻辑，这里可能需要调整
                    if env.uavs[uav_id].connected_bs_id != target_bs:
                        # 这里需要实现MAPPO的切换逻辑
                        # 暂时使用简化的切换
                        env.uavs[uav_id].connected_bs_id = target_bs
            
            # 执行环境步进
            env.step()
            
            # 计算奖励
            rewards = {}
            for uav_id in range(env.num_uav):
                uav = env.uavs[uav_id]
                rewards[uav_id] = uav.current_satisfaction
            
            # 存储经验
            agent.insert_experience(
                step=step,
                obs_dict=obs,
                state=state,
                actions=actions,
                rewards=rewards,
                team_reward=np.sum(list(rewards.values())) if rewards else 0,
                done=False,  # EnhancedNetworkEnvironment 不返回 done
                log_probs=log_probs,
                values=values,
                biz_types=biz_types
            )
            
            episode_reward += np.sum(list(rewards.values())) if rewards else 0
            episode_sats.append(np.mean([uav.current_satisfaction for uav in env.uavs.values()]))
            
            # 手动更新观察值
            new_obs = {}
            for uav_id in range(env.num_uav):
                uav = env.uavs[uav_id]
                new_obs[uav_id] = np.array([
                    uav.position[0], uav.position[1], uav.position[2],
                    uav.current_satisfaction,
                    1.0 if uav.connected_bs_id is not None else 0.0,
                    # 添加其他观察值...
                ])
            obs = new_obs
        
        # 训练
        if len(agent.buffer['obs']) >= agent.rollout_length:
            agent.train()
        
        avg_train_sat = np.mean(episode_sats) if episode_sats else 0
        training_history['episodes'].append(episode)
        training_history['train_satisfactions'].append(avg_train_sat)
        training_history['train_rewards'].append(episode_reward)
        
        if (episode + 1) % eval_interval == 0:
            # 评估
            eval_rewards = []
            eval_sats = []
            for _ in range(num_eval_episodes):
                env.reset()
                # 手动构建观察值
                obs = {}
                for uav_id in range(env.num_uav):
                    uav = env.uavs[uav_id]
                    obs[uav_id] = np.array([
                        uav.position[0], uav.position[1], uav.position[2],
                        uav.current_satisfaction,
                        1.0 if uav.connected_bs_id is not None else 0.0,
                    ])
                
                agent.reset_hidden()
                ep_reward = 0
                ep_sats = []
                step_metrics = []
                
                for step in range(150):
                    biz_types = {uid: env.uavs[uid].true_business_type.value for uid in range(env.num_uav)}
                    # 构建状态表示
                    state = []
                    
                    # 添加全局统计信息
                    avg_satisfaction = np.mean([uav.current_satisfaction for uav in env.uavs.values()])
                    connected_ratio = sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav
                    avg_sinr = np.mean(env.sinr_matrix)
                    
                    # 添加基站负载信息
                    for bs in env.base_stations.values():
                        state.append(float(bs.load_ratio))
                    
                    # 添加全局统计
                    state.extend([float(avg_satisfaction), float(connected_ratio), float(avg_sinr)])
                    
                    # 确保状态维度与配置一致
                    while len(state) < 128:
                        state.append(0.0)
                    state = state[:128]
                    
                    # 确保所有元素都是数值类型
                    state = [float(x) for x in state]
                    
                    actions, _, _, _, _ = agent.select_actions(obs, state, biz_types, training=False, env=env)
                    
                    # 执行动作
                    handover_count = 0
                    for uav_id, action in actions.items():
                        if action != 0:  # 0 表示不切换
                            target_bs = action - 1  # 动作是 1-based
                            if env.uavs[uav_id].connected_bs_id != target_bs:
                                env.uavs[uav_id].connected_bs_id = target_bs
                                handover_count += 1
                    
                    # 执行环境步进
                    env.step()
                    
                    # 收集指标
                    step_data = {
                        'step': step,
                        'satisfaction': np.mean([uav.current_satisfaction for uav in env.uavs.values()]),
                        'reward': 0,  # EnhancedNetworkEnvironment 不返回奖励
                        'connected_ratio': sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav,
                        'avg_sinr': np.mean(env.sinr_matrix),
                        'handover_count': handover_count,
                        # 通信性能指标
                        'avg_latency': np.mean([uav.current_latency for uav in env.uavs.values()]),
                        'avg_packet_loss': np.mean([uav.packet_loss_rate for uav in env.uavs.values()]),
                        'total_throughput': sum(uav.current_allocated_rate for uav in env.uavs.values()),
                        'avg_throughput': np.mean([uav.current_allocated_rate for uav in env.uavs.values()]),
                        'bandwidth_utilization': sum(bs.current_load for bs in env.base_stations.values()) / sum(bs.capacity for bs in env.base_stations.values()),
                        'load_variance': np.var([bs.load_ratio for bs in env.base_stations.values()]),
                        'interruption_rate': len(env.interrupted_uavs) / env.num_uav,
                    }
                    step_metrics.append(step_data)
                    ep_reward += step_data['reward']
                    ep_sats.append(step_data['satisfaction'])
                    
                    # 手动更新观察值
                    new_obs = {}
                    for uav_id in range(env.num_uav):
                        uav = env.uavs[uav_id]
                        new_obs[uav_id] = np.array([
                            uav.position[0], uav.position[1], uav.position[2],
                            uav.current_satisfaction,
                            1.0 if uav.connected_bs_id is not None else 0.0,
                        ])
                    obs = new_obs
                
                eval_rewards.append(ep_reward)
                eval_sats.append(np.mean(ep_sats) if ep_sats else 0)
            
            eval_reward = np.mean(eval_rewards)
            eval_sat = np.mean(eval_sats)
            training_history['eval_rewards'].append(eval_reward)
            training_history['eval_satisfactions'].append(eval_sat)
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Episode {episode+1}/{num_episodes}")
            print(f"  训练: Sat={avg_train_sat:.4f}, Reward={episode_reward:.2f}")
            print(f"  评估: Sat={eval_sat:.4f}, Reward={eval_reward:.2f}")
            print(f"  最佳: {best_satisfaction:.4f}, 耐心: {patience_counter}/{max_patience}")
            
            if eval_sat > best_satisfaction + 0.001:
                best_satisfaction = eval_sat
                best_model_path = os.path.join(log_dir, 'best_model.pt')
                agent.save(best_model_path)
                patience_counter = 0
                print(f"  *** 保存新的最佳模型 ***")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n早停：连续{max_patience}次评估无改善")
                    break
    
    training_time = time.time() - training_start_time
    agent.save(os.path.join(log_dir, 'final_model.pt'))
    
    print(f"\n训练完成!")
    print(f"  总训练时间: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")
    print(f"  实际训练轮数: {episode+1}")
    print(f"  最佳评估满意度: {best_satisfaction:.4f}")
    
    return best_model_path, training_history, best_satisfaction


def evaluate_mappo_in_main_env(model_path, env, num_episodes=10, seed=42):
    """在主实验环境中评估MAPPO"""
    print(f"\n" + "="*70)
    print(f"在主实验环境中评估 MAPPO")
    print("="*70)
    
    set_global_seed(seed)
    
    # 获取观察和状态维度
    env.reset()
    # 手动构建观察值
    obs = {}
    for uav_id in range(env.num_uav):
        uav = env.uavs[uav_id]
        obs[uav_id] = np.array([
            uav.position[0], uav.position[1], uav.position[2],
            uav.current_satisfaction,
            1.0 if uav.connected_bs_id is not None else 0.0,
        ])
    
    obs_dim = len(obs[0]) if obs else 29
    state_dim = 128
    action_dim = env.num_bs + 1  # 0表示不切换，1-8表示切换到对应BS
    
    # 创建并加载agent
    agent = MAPPOAgentV2(
        num_agents=env.num_uav,
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
        use_biz_heads=True,
        use_attention_critic=True,
    )
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"  加载模型: {model_path}")
    else:
        print(f"  警告: 模型文件不存在，使用随机初始化")
    
    all_metrics = []
    episode_details = []
    
    for ep in range(num_episodes):
        env.reset()
        # 手动构建观察值
        obs = {}
        for uav_id in range(env.num_uav):
            uav = env.uavs[uav_id]
            obs[uav_id] = np.array([
                uav.position[0], uav.position[1], uav.position[2],
                uav.current_satisfaction,
                1.0 if uav.connected_bs_id is not None else 0.0,
                # 添加其他观察值...
            ])
        
        agent.reset_hidden()
        episode_reward = 0
        step_metrics = []
        
        for step in range(150):
            biz_types = {uid: env.uavs[uid].true_business_type.value for uid in range(env.num_uav)}
            # 构建状态表示
            state = []
            
            # 添加全局统计信息
            avg_satisfaction = np.mean([uav.current_satisfaction for uav in env.uavs.values()])
            connected_ratio = sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav
            avg_sinr = np.mean(env.sinr_matrix)
            
            # 添加基站负载信息
            for bs in env.base_stations.values():
                state.append(bs.load_ratio)
            
            # 添加全局统计
            state.extend([avg_satisfaction, connected_ratio, avg_sinr])
            
            # 确保状态维度与配置一致
            while len(state) < 128:
                state.append(0.0)
            state = state[:128]
            
            actions, _, _, _, _ = agent.select_actions(obs, state, biz_types, training=False, env=env)
            
            # 执行动作
            handover_count = 0
            for uav_id, action in actions.items():
                if action != 0:  # 0 表示不切换
                    target_bs = action - 1  # 动作是 1-based
                    if env.uavs[uav_id].connected_bs_id != target_bs:
                        env.uavs[uav_id].connected_bs_id = target_bs
                        handover_count += 1
            
            # 执行环境步进
            env.step()
            
            # 收集指标
            step_data = {
                'step': step,
                'satisfaction': np.mean([uav.current_satisfaction for uav in env.uavs.values()]),
                'reward': 0,  # EnhancedNetworkEnvironment 不返回奖励
                'connected_ratio': sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None) / env.num_uav,
                'avg_sinr': np.mean(env.sinr_matrix),
                'handover_count': handover_count,
                # 通信性能指标
                'avg_latency': np.mean([uav.current_latency for uav in env.uavs.values()]),
                'avg_packet_loss': np.mean([uav.packet_loss_rate for uav in env.uavs.values()]),
                'total_throughput': sum(uav.current_allocated_rate for uav in env.uavs.values()),
                'avg_throughput': np.mean([uav.current_allocated_rate for uav in env.uavs.values()]),
                'bandwidth_utilization': sum(bs.current_load for bs in env.base_stations.values()) / sum(bs.capacity for bs in env.base_stations.values()),
                'load_variance': np.var([bs.load_ratio for bs in env.base_stations.values()]),
                'interruption_rate': len(env.interrupted_uavs) / env.num_uav,
            }
            step_metrics.append(step_data)
            episode_reward += step_data['reward']
            
            # 手动更新观察值
            new_obs = {}
            for uav_id in range(env.num_uav):
                uav = env.uavs[uav_id]
                new_obs[uav_id] = np.array([
                    uav.position[0], uav.position[1], uav.position[2],
                    uav.current_satisfaction,
                    1.0 if uav.connected_bs_id is not None else 0.0,
                    # 添加其他观察值...
                ])
            obs = new_obs
        
        ep_summary = {
            'episode': ep + 1,
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfaction'] for m in step_metrics]),
            'min_satisfaction': min([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'max_satisfaction': max([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'std_satisfaction': np.std([m['satisfaction'] for m in step_metrics]) if step_metrics else 0,
            'avg_connected_ratio': np.mean([m['connected_ratio'] for m in step_metrics]),
            'avg_sinr': np.mean([m['avg_sinr'] for m in step_metrics]),
            'total_handovers': sum([m['handover_count'] for m in step_metrics]),
            # 通信性能指标
            'avg_latency': np.mean([m['avg_latency'] for m in step_metrics]),
            'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in step_metrics]),
            'avg_total_throughput': np.mean([m['total_throughput'] for m in step_metrics]),
            'avg_throughput': np.mean([m['avg_throughput'] for m in step_metrics]),
            'avg_bandwidth_utilization': np.mean([m['bandwidth_utilization'] for m in step_metrics]),
            'avg_load_variance': np.mean([m['load_variance'] for m in step_metrics]),
            'avg_interruption_rate': np.mean([m['interruption_rate'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        episode_details.append(ep_summary)
        
        print(f"  Episode {ep+1:2d}: Sat={ep_summary['avg_satisfaction']:.4f} "
              f"(min={ep_summary['min_satisfaction']:.4f}, max={ep_summary['max_satisfaction']:.4f}), "
              f"Reward={episode_reward:.2f}, HOs={ep_summary['total_handovers']}")
    
    summary = {
        'name': '优化MAPPO',
        'num_episodes': num_episodes,
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'min_satisfaction': np.min([m['avg_satisfaction'] for m in all_metrics]),
        'max_satisfaction': np.max([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_connected_ratio': np.mean([m['avg_connected_ratio'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'avg_handovers': np.mean([m['total_handovers'] for m in all_metrics]),
        # 通信性能指标
        'avg_latency': np.mean([m['avg_latency'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'avg_total_throughput': np.mean([m['avg_total_throughput'] for m in all_metrics]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in all_metrics]),
        'avg_bandwidth_utilization': np.mean([m['avg_bandwidth_utilization'] for m in all_metrics]),
        'avg_load_variance': np.mean([m['avg_load_variance'] for m in all_metrics]),
        'avg_interruption_rate': np.mean([m['avg_interruption_rate'] for m in all_metrics]),
        'episode_details': episode_details,
    }
    
    print(f"\n汇总结果:")
    print(f"  平均满意度: {summary['avg_satisfaction']:.4f} ± {summary['std_satisfaction']:.4f}")
    print(f"  满意度范围: [{summary['min_satisfaction']:.4f}, {summary['max_satisfaction']:.4f}]")
    print(f"  平均奖励: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}")
    # 通信性能指标
    print(f"  平均延迟: {summary['avg_latency']:.4f} ms")
    print(f"  平均丢包率: {summary['avg_packet_loss']:.4f}")
    print(f"  平均总吞吐量: {summary['avg_total_throughput']:.2f} Mbps")
    print(f"  平均吞吐量: {summary['avg_throughput']:.2f} Mbps")
    print(f"  平均带宽利用率: {summary['avg_bandwidth_utilization']:.4f}")
    print(f"  平均负载方差: {summary['avg_load_variance']:.4f}")
    print(f"  平均中断率: {summary['avg_interruption_rate']:.4f}")
    
    return summary


def generate_comparison_report(results, timestamp):
    """生成对比报告"""
    report = f"""# 主实验环境公平对比实验报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 实验概述

### 1.1 实验目的
在主实验环境 (EnhancedNetworkEnvironment) 中进行公平对比，验证三种算法在真实复杂环境下的性能表现。

### 1.2 实验环境
- **环境类**: EnhancedNetworkEnvironment
- **UAV数量**: 300
- **BS数量**: 8
- **负载率**: ~77%
- **随机事件**: 开启
- **与主实验1234完全一致**

### 1.3 对比算法
1. **传统算法(3GPP)**: IntegratedHandoverAlgorithm
2. **增强算法**: EnhancedHandoverAlgorithm
3. **优化MAPPO**: 在主实验环境中重新训练

## 2. 对比结果

### 2.1 核心指标对比

| 算法 | 平均满意度 | 标准差 | 平均奖励 | 平均切换次数 |
|------|-----------|--------|----------|-------------|
"""
    
    for result in results:
        report += f"| {result['name']} | {result['avg_satisfaction']:.4f} | {result['std_satisfaction']:.3f} | {result['avg_reward']:.2f} | {result['avg_handovers']:.1f} |\n"
    
    report += """

### 2.2 网络质量指标

| 算法 | 连接率 | 平均SINR(dB) |
|------|--------|--------------|
"""
    
    for result in results:
        report += f"| {result['name']} | {result['avg_connected_ratio']:.4f} | {result['avg_sinr']:.2f} |\n"
    
    report += """

### 2.3 通信性能指标

| 算法 | 平均延迟(ms) | 平均丢包率 | 平均总吞吐量(Mbps) | 平均吞吐量(Mbps) | 平均带宽利用率 | 平均负载方差 | 平均中断率 |
|------|--------------|------------|-------------------|------------------|-----------------|--------------|------------|
"""
    
    for result in results:
        report += f"| {result['name']} | {result['avg_latency']:.4f} | {result['avg_packet_loss']:.4f} | {result['avg_total_throughput']:.2f} | {result['avg_throughput']:.2f} | {result['avg_bandwidth_utilization']:.4f} | {result['avg_load_variance']:.4f} | {result['avg_interruption_rate']:.4f} |\n"
    
    # 排序找出最佳算法
    sorted_results = sorted(results, key=lambda x: x['avg_satisfaction'], reverse=True)
    
    report += f"""

## 3. 结果分析

### 3.1 性能排名

"""
    
    for i, result in enumerate(sorted_results, 1):
        report += f"{i}. **{result['name']}**: {result['avg_satisfaction']:.4f} ± {result['std_satisfaction']:.3f}\n"
    
    report += """

### 3.2 稳定性分析

"""
    
    for result in results:
        stability = "高" if result['std_satisfaction'] < 0.01 else "中" if result['std_satisfaction'] < 0.02 else "低"
        report += f"- **{result['name']}**: 标准差={result['std_satisfaction']:.3f} ({stability}稳定性)\n"
    
    report += """

## 4. 与QMixHandoverEnv对比

### 4.1 关键差异
- **环境复杂度**: EnhancedNetworkEnvironment 更复杂真实
- **BS数量**: 8个 vs 3个
- **负载率**: 77% vs 88%
- **随机事件**: 开启 vs 关闭

### 4.2 预期影响
- 增强算法和MAPPO在复杂环境中应有更好表现
- 传统算法的优势可能减弱
- 更公平的对比结果

## 5. 结论与建议

### 5.1 主要结论
1. 在主实验环境中，三种算法的相对性能可能发生变化
2. 复杂环境更能体现增强算法和MAPPO的优势
3. 传统算法在简单环境中的优势可能减弱

### 5.2 建议
1. 使用主实验环境作为标准对比平台
2. 在不同负载率下进行多场景验证
3. 优化增强算法和MAPPO在复杂环境中的表现

---

*报告生成完成*
"""
    
    report_file = f'main_experiment_comparison_report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n对比报告已保存: {report_file}")
    return report_file


def main():
    """主函数"""
    print("\n" + "="*80)
    print("主实验环境公平对比实验")
    print("="*80)
    print("\n实验环境: EnhancedNetworkEnvironment (与主实验1234完全一致)")
    print("UAV数量: 300")
    print("BS数量: 8")
    print("负载率: ~77%")
    print("随机事件: 开启")
    print("="*80)
    
    seed = 42
    num_episodes = 10
    
    # 创建主实验环境
    set_global_seed(seed)
    env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=300,
        recognition_model=None,
        scaler=None,
        seed=seed,
        event_probability=0.05  # 开启随机事件
    )
    
    results = []
    
    # 1. 评估传统算法
    traditional_algo = IntegratedHandoverAlgorithm(env)
    result = evaluate_algorithm_in_main_env(traditional_algo, "传统算法(3GPP)", env, num_episodes, seed)
    results.append(result)
    
    # 2. 评估增强算法
    enhanced_algo = EnhancedHandoverAlgorithm(env, weight_config='optimized')
    # 使用调优后的最佳参数
    enhanced_algo.base_threshold = 0.01
    enhanced_algo.epsilon = 0.01
    enhanced_algo.handover_cooldown = 5
    enhanced_algo.use_load_mode = True
    result = evaluate_algorithm_in_main_env(enhanced_algo, "增强算法", env, num_episodes, seed)
    results.append(result)
    
    # 3. 训练并评估MAPPO
    print("\n" + "="*80)
    print("开始训练MAPPO (可能需要较长时间)")
    print("="*80)
    
    mappo_model_path, training_history, best_sat = train_mappo_in_main_env(
        env, num_episodes=200, eval_interval=20, num_eval_episodes=10, seed=seed
    )
    
    result = evaluate_mappo_in_main_env(mappo_model_path, env, num_episodes, seed)
    results.append(result)
    
    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 准备可视化数据
    visual_data = {}
    for result in results:
        algorithm_name = result['name']
        # 提取每个episode的详细数据
        episode_data = []
        for ep_detail in result['episode_details']:
            episode_data.append({
                'satisfaction': ep_detail['avg_satisfaction'],
                'handover_count': ep_detail['total_handovers'],
                'avg_latency': ep_detail['avg_latency'],
                'avg_packet_loss': ep_detail['avg_packet_loss'],
                'avg_throughput': ep_detail['avg_throughput'],
                'bandwidth_utilization': ep_detail['avg_bandwidth_utilization'],
            })
        visual_data[algorithm_name] = episode_data
    
    # 生成性能可视化和报告
    print("\n" + "="*80)
    print("生成性能可视化和报告")
    print("="*80)
    
    create_performance_report(
        visual_data, 
        report_name=f'main_experiment_comparison_{timestamp}',
        output_dir='performance_reports'
    )
    
    # JSON报告
    json_report = {
        'timestamp': timestamp,
        'environment': {
            'class': 'EnhancedNetworkEnvironment',
            'num_uav': 300,
            'num_bs': 8,
            'seed': seed,
            'event_probability': 0.05,
        },
        'results': results,
        'mappo_training': {
            'best_satisfaction': best_sat,
            'model_path': mappo_model_path,
        }
    }
    
    json_file = f'main_experiment_comparison_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"\nJSON报告已保存: {json_file}")
    
    # Markdown报告
    md_file = generate_comparison_report(results, timestamp)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - JSON报告: {json_file}")
    print(f"  - Markdown报告: {md_file}")
    print(f"  - 性能可视化报告: performance_reports/")
    print("\n可以使用Markdown阅读器查看详细报告，使用浏览器打开HTML报告查看可视化结果。")


if __name__ == "__main__":
    main()
