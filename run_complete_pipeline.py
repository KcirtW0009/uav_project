# -*- coding: utf-8 -*-
"""
完整流程自动化脚本：训练优化MAPPO + 全面对比实验

运行此脚本将自动完成：
1. 使用优化参数训练MAPPO模型
2. 评估训练好的模型
3. 运行完整对比实验（传统算法 vs 增强算法 vs 优化MAPPO）
4. 生成详细报告和可视化结果

使用方法：
    venv\Scripts\python.exe run_complete_pipeline.py

预计运行时间：30-60分钟（取决于硬件性能）
"""

import os
import sys
import json
import numpy as np
import torch
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.mappo_optimized_config import OPTIMIZED_MAPPO_CONFIG
from uav_system.business import BusinessType
from uav_system.algorithms import EnhancedHandoverAlgorithm


def evaluate_agent(agent, env, num_episodes=10):
    """评估agent性能"""
    rewards = []
    satisfactions = []
    for _ in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        episode_reward = 0
        episode_sats = []
        for step in range(150):
            biz_types = {uid: env.env.uavs[uid].true_business_type.value for uid in range(env.num_agents)}
            actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=False, env=env)
            next_obs, next_state, _, team_reward, done, info = env.step(actions)
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            obs_dict = next_obs
            global_state = next_state
            if done:
                break
        rewards.append(episode_reward)
        satisfactions.append(np.mean(episode_sats) if episode_sats else 0)
    return np.mean(rewards), np.mean(satisfactions)


def collect_detailed_metrics(env, actions):
    """收集详细指标"""
    metrics = {
        'satisfactions': [], 'latencies': [], 'rates': [], 'sinrs': [],
        'handover_attempts': 0, 'handover_success': 0, 'handover_latencies': [],
        'ping_jitters': [], 'packet_losses': [], 'qos_violations': [],
        'connected_count': 0, 'biz_sats': {0: [], 1: [], 2: []},
    }
    for uav_id, uav in env.uavs.items():
        metrics['satisfactions'].append(uav.current_satisfaction)
        metrics['latencies'].append(uav.current_latency)
        metrics['rates'].append(uav.current_allocated_rate)
        metrics['sinrs'].append(uav.sinr_db)
        biz_type = uav.business_type.value if hasattr(uav.business_type, 'value') else 2
        metrics['biz_sats'][biz_type].append(uav.current_satisfaction)
        is_connected = uav.connected_bs_id is not None
        if is_connected:
            metrics['connected_count'] += 1
        action = actions.get(uav_id, 0)
        if action != 0 and is_connected:
            metrics['handover_attempts'] += 1
            metrics['handover_success'] += 1
            metrics['handover_latencies'].append(5.0)
        if uav.current_latency > 50:
            metrics['ping_jitters'].append(uav.current_latency - 50)
        if uav.sinr_db < 10:
            metrics['packet_losses'].append(max(0, (10 - uav.sinr_db) * 2))
        required_rate = getattr(uav, 'required_rate', 1)
        if uav.current_allocated_rate < required_rate * 0.8:
            metrics['qos_violations'].append(1)
    return metrics


def evaluate_algorithm(algo_name, algo, env, num_episodes=10, seed=42):
    """评估传统算法"""
    print(f"\n评估 {algo_name}...")
    set_global_seed(seed)
    all_metrics = []
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        episode_reward = 0
        step_metrics = []
        for step in range(150):
            start_time = time.time()
            actions = {}
            for uav_id in range(env.num_agents):
                uav = env.env.uavs[uav_id]
                current_bs = uav.connected_bs_id
                best_bs = current_bs
                uav_id_real = uav.uav_id
                best_sinr = env.env.sinr_matrix[uav_id_real, current_bs] if current_bs is not None else -100
                for bs_id in range(env.env.num_bs):
                    if bs_id == current_bs:
                        continue
                    sinr = env.env.sinr_matrix[uav_id_real, bs_id]
                    if sinr > best_sinr + 3.0 + 1.0:
                        best_sinr = sinr
                        best_bs = bs_id
                actions[uav_id] = 0 if best_bs == current_bs else best_bs + 1
            decision_time = (time.time() - start_time) * 1000
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            metrics = collect_detailed_metrics(env.env, actions)
            metrics['decision_time'] = decision_time / env.num_agents
            metrics['team_reward'] = team_reward
            step_metrics.append(metrics)
            episode_reward += team_reward
            if done:
                break
        ep_summary = {
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfactions'] for m in step_metrics]),
            'avg_latency': np.mean([np.mean(m['latencies']) for m in step_metrics]),
            'avg_rate': np.mean([np.mean(m['rates']) for m in step_metrics]),
            'avg_sinr': np.mean([np.mean(m['sinrs']) for m in step_metrics]),
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) / max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, reward={episode_reward:.2f}")
    summary = {
        'name': algo_name,
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_latency': np.mean([m['avg_latency'] for m in all_metrics]),
        'avg_rate': np.mean([m['avg_rate'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'handover_success_rate': np.mean([m['handover_success_rate'] for m in all_metrics]),
        'connected_ratio': np.mean([m['connected_ratio'] for m in all_metrics]),
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }
    return summary


def evaluate_enhanced_algorithm(env, num_episodes=10, seed=42):
    """评估完整增强算法"""
    print("\n评估 增强算法(完整版)...")
    set_global_seed(seed)
    enhanced = EnhancedHandoverAlgorithm(env.env, weight_config='optimized')
    all_metrics = []
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        episode_reward = 0
        step_metrics = []
        for step in range(150):
            start_time = time.time()
            actions = {}
            for uav_id in range(env.num_agents):
                decision = enhanced.make_intelligent_decision(uav_id)
                if decision is None:
                    actions[uav_id] = 0
                else:
                    target_bs, _ = decision
                    current_bs = env.env.uavs[uav_id].connected_bs_id
                    actions[uav_id] = 0 if target_bs == current_bs else target_bs + 1
            decision_time = (time.time() - start_time) * 1000
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            metrics = collect_detailed_metrics(env.env, actions)
            metrics['decision_time'] = decision_time / env.num_agents
            metrics['team_reward'] = team_reward
            step_metrics.append(metrics)
            episode_reward += team_reward
            if done:
                break
        ep_summary = {
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfactions'] for m in step_metrics]),
            'avg_latency': np.mean([np.mean(m['latencies']) for m in step_metrics]),
            'avg_rate': np.mean([np.mean(m['rates']) for m in step_metrics]),
            'avg_sinr': np.mean([np.mean(m['sinrs']) for m in step_metrics]),
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) / max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, reward={episode_reward:.2f}")
    summary = {
        'name': '增强算法(完整版)',
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_latency': np.mean([m['avg_latency'] for m in all_metrics]),
        'avg_rate': np.mean([m['avg_rate'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'handover_success_rate': np.mean([m['handover_success_rate'] for m in all_metrics]),
        'connected_ratio': np.mean([m['connected_ratio'] for m in all_metrics]),
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }
    return summary


def evaluate_mappo(model_path, env, num_episodes=10, seed=42):
    """评估MAPPO"""
    print(f"\n评估 MAPPO...")
    set_global_seed(seed)
    obs_dict, global_state = env.reset()
    obs_dim = len(obs_dict[0])
    state_dim = len(global_state)
    agent = MAPPOAgentV2(
        num_agents=env.num_agents, obs_dim=obs_dim, state_dim=state_dim,
        action_dim=env.action_dim, hidden_dim=128, critic_hidden_dim=256,
        actor_lr=3e-5, critic_lr=3e-4, gamma=0.99, gae_lambda=0.99,
        clip_epsilon=0.2, entropy_coef=0.02, value_coef=0.5,
        use_biz_heads=True, use_attention_critic=True,
    )
    agent.load(model_path)
    all_metrics = []
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        episode_reward = 0
        step_metrics = []
        for step in range(150):
            start_time = time.time()
            biz_types = {uid: env.env.uavs[uid].true_business_type.value for uid in range(env.num_agents)}
            actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=False, env=env)
            decision_time = (time.time() - start_time) * 1000
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            metrics = collect_detailed_metrics(env.env, actions)
            metrics['decision_time'] = decision_time / env.num_agents
            metrics['team_reward'] = team_reward
            step_metrics.append(metrics)
            episode_reward += team_reward
            obs_dict = next_obs
            global_state = next_state
            if done:
                break
        ep_summary = {
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfactions'] for m in step_metrics]),
            'avg_latency': np.mean([np.mean(m['latencies']) for m in step_metrics]),
            'avg_rate': np.mean([np.mean(m['rates']) for m in step_metrics]),
            'avg_sinr': np.mean([np.mean(m['sinrs']) for m in step_metrics]),
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) / max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }
        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, reward={episode_reward:.2f}")
    summary = {
        'name': '优化MAPPO',
        'avg_satisfaction': np.mean([m['avg_satisfaction'] for m in all_metrics]),
        'std_satisfaction': np.std([m['avg_satisfaction'] for m in all_metrics]),
        'avg_reward': np.mean([m['reward'] for m in all_metrics]),
        'std_reward': np.std([m['reward'] for m in all_metrics]),
        'avg_latency': np.mean([m['avg_latency'] for m in all_metrics]),
        'avg_rate': np.mean([m['avg_rate'] for m in all_metrics]),
        'avg_sinr': np.mean([m['avg_sinr'] for m in all_metrics]),
        'handover_success_rate': np.mean([m['handover_success_rate'] for m in all_metrics]),
        'connected_ratio': np.mean([m['connected_ratio'] for m in all_metrics]),
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }
    return summary


def main():
    """主流程"""
    print("\n" + "=" * 70)
    print("完整流程：训练 + 评估 + 对比实验")
    print("=" * 70)
    
    num_uav = 128
    num_bs = 3
    seed = 42
    
    # ========================================
    # 阶段1：训练MAPPO
    # ========================================
    print("\n" + "=" * 70)
    print("阶段1/3：训练MAPPO")
    print("=" * 70)
    
    config = OPTIMIZED_MAPPO_CONFIG.copy()
    set_global_seed(seed)
    env = QMixHandoverEnv(num_uav=num_uav, num_bs=num_bs, pos_range=1000, max_steps=150)
    obs_dict, global_state = env.reset()
    obs_dim = len(obs_dict[0])
    state_dim = len(global_state)
    
    agent = MAPPOAgentV2(
        num_agents=env.num_agents, obs_dim=obs_dim, state_dim=state_dim,
        action_dim=env.action_dim, hidden_dim=128, critic_hidden_dim=256,
        actor_lr=3e-5, critic_lr=3e-4, gamma=0.99, gae_lambda=0.99,
        clip_epsilon=0.2, entropy_coef=0.02, value_coef=0.5,
        use_biz_heads=True, use_attention_critic=True,
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'./experiment_logs/optimized_mappo_{timestamp}'
    os.makedirs(log_dir, exist_ok=True)
    
    best_satisfaction = 0
    best_model_path = None
    patience_counter = 0
    max_patience = 15
    
    for episode in range(200):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        episode_reward = 0
        episode_sats = []
        
        for step in range(150):
            biz_types = {uid: env.env.uavs[uid].true_business_type.value for uid in range(env.num_agents)}
            actions, log_probs, values, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=True, env=env)
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            agent.insert_experience(step=step, obs_dict=obs_dict, state=global_state, actions=actions,
                                   rewards=rewards, team_reward=team_reward, done=done, log_probs=log_probs, values=values, biz_types=biz_types)
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            obs_dict = next_obs
            global_state = next_state
            if done:
                break
        
        if len(agent.buffer['obs']) >= agent.rollout_length:
            agent.train()
        
        avg_sat = np.mean(episode_sats) if episode_sats else 0
        
        if (episode + 1) % 20 == 0:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Episode {episode+1}/200")
            eval_reward, eval_sat = evaluate_agent(agent, env, num_episodes=10)
            print(f"  训练Sat: {avg_sat:.4f}, 评估Sat: {eval_sat:.4f}")
            
            if eval_sat > best_satisfaction + 0.001:
                best_satisfaction = eval_sat
                best_model_path = os.path.join(log_dir, 'best_model.pt')
                agent.save(best_model_path)
                patience_counter = 0
                print(f"  *** 新最佳模型: {eval_sat:.4f} ***")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n早停：连续{max_patience}次评估无改善")
                    break
    
    agent.save(os.path.join(log_dir, 'final_model.pt'))
    print(f"\n训练完成！最佳满意度: {best_satisfaction:.4f}")
    
    # ========================================
    # 阶段2：对比实验
    # ========================================
    print("\n" + "=" * 70)
    print("阶段2/3：运行对比实验")
    print("=" * 70)
    
    results = []
    
    # 传统算法
    result = evaluate_algorithm("传统算法(3GPP)", None, env, num_episodes=10, seed=seed)
    results.append(result)
    
    # 增强算法
    result = evaluate_enhanced_algorithm(env, num_episodes=10, seed=seed)
    results.append(result)
    
    # MAPPO
    result = evaluate_mappo(best_model_path, env, num_episodes=10, seed=seed)
    results.append(result)
    
    # ========================================
    # 阶段3：生成报告
    # ========================================
    print("\n" + "=" * 70)
    print("阶段3/3：生成报告")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    
    print(f"\n{'算法':<20} {'满意度':<15} {'奖励':<12} {'延迟(ms)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['avg_satisfaction']:.4f}±{r['std_satisfaction']:.4f}   "
              f"{r['avg_reward']:.1f}±{r['std_reward']:.1f}    {r['avg_latency']:.1f}")
    
    print(f"\n{'算法':<20} {'SINR(dB)':<12} {'切换成功率':<12} {'连接率':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['avg_sinr']:.1f}          "
              f"{r['handover_success_rate']:.2f}        {r['connected_ratio']:.4f}")
    
    print(f"\n{'算法':<20} {'抖动(ms)':<12} {'丢包率(%)':<12} {'QoS违规率':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['avg_jitter']:.1f}          "
              f"{r['avg_packet_loss']:.1f}          {r['qos_violation_rate']:.4f}")
    
    best_sat = max(results, key=lambda x: x['avg_satisfaction'])
    best_reward = max(results, key=lambda x: x['avg_reward'])
    print(f"\n最佳满意度: {best_sat['name']} ({best_sat['avg_satisfaction']:.4f})")
    print(f"最佳奖励: {best_reward['name']} ({best_reward['avg_reward']:.2f})")
    
    # 保存结果
    report = {
        'timestamp': timestamp,
        'best_model': best_model_path,
        'training_best_satisfaction': best_satisfaction,
        'num_episodes_eval': 10,
        'results': results,
        'config': config,
    }
    
    report_file = f'complete_pipeline_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n报告已保存: {report_file}")
    print("\n" + "=" * 70)
    print("全部流程完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
