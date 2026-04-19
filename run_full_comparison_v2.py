# -*- coding: utf-8 -*-
"""
完整对比实验 V2：传统算法 vs 增强算法 vs 优化MAPPO

采集所有可获取的指标（除recognition_accuracy外）
"""

import os
import sys
import json
import numpy as np
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.business import BusinessType
from uav_system.algorithms import EnhancedHandoverAlgorithm


class TraditionalAlgorithm:
    """传统3GPP切换算法"""
    
    def __init__(self, sinr_threshold=3.0, hysteresis=1.0):
        self.sinr_threshold = sinr_threshold
        self.hysteresis = hysteresis
    
    def select_action(self, uav, env):
        """选择切换目标"""
        current_bs = uav.connected_bs_id
        best_bs = current_bs
        uav_id = uav.uav_id
        
        # 从环境的sinr_matrix获取SINR值
        best_sinr = env.sinr_matrix[uav_id, current_bs] if current_bs is not None else -100
        
        for bs_id in range(env.num_bs):
            if bs_id == current_bs:
                continue
            sinr = env.sinr_matrix[uav_id, bs_id]
            if sinr > best_sinr + self.sinr_threshold + self.hysteresis:
                best_sinr = sinr
                best_bs = bs_id
        
        return 0 if best_bs == current_bs else best_bs + 1


class EnhancedAlgorithm:
    """增强切换算法（业务感知）"""
    
    def __init__(self):
        self.sinr_threshold = 3.0
        self.hysteresis = 1.0
        self.biz_priority = {0: 1.0, 1: 0.8, 2: 0.5}
    
    def select_action(self, uav, env):
        """选择切换目标"""
        current_bs = uav.connected_bs_id
        biz_type = uav.business_type.value if hasattr(uav.business_type, 'value') else 2
        uav_id = uav.uav_id
        
        best_bs = current_bs
        best_score = -1000
        
        for bs_id in range(env.num_bs):
            # 从环境的sinr_matrix获取SINR值
            sinr = env.sinr_matrix[uav_id, bs_id]
            sinr_score = min(sinr / 30, 1.0)
            
            if hasattr(env, 'base_stations') and bs_id in env.base_stations:
                bs = env.base_stations[bs_id]
                load_score = 1.0 - bs.current_load / bs.capacity
            else:
                load_score = 0.5
            
            priority = self.biz_priority.get(biz_type, 0.5)
            score = 0.5 * sinr_score + 0.3 * load_score + 0.2 * priority
            
            if bs_id != current_bs:
                current_sinr = env.sinr_matrix[uav_id, current_bs] if current_bs is not None else -100
                if sinr > current_sinr + self.sinr_threshold + self.hysteresis:
                    if score > best_score:
                        best_score = score
                        best_bs = bs_id
        
        return 0 if best_bs == current_bs else best_bs + 1


def collect_detailed_metrics(env, actions):
    """收集详细的评估指标
    
    Args:
        env: NetworkEnvironmentWithRecognition对象（底层环境）
        actions: 动作字典
    """
    metrics = {
        'satisfactions': [],
        'latencies': [],
        'rates': [],
        'sinrs': [],
        'handover_attempts': 0,
        'handover_success': 0,
        'handover_latencies': [],
        'ping_jitters': [],
        'packet_losses': [],
        'qos_violations': [],
        'connected_count': 0,
        'biz_sats': {0: [], 1: [], 2: []},
    }
    
    for uav_id, uav in env.uavs.items():
        # 基础指标
        metrics['satisfactions'].append(uav.current_satisfaction)
        metrics['latencies'].append(uav.current_latency)
        # UAV使用current_allocated_rate而不是current_throughput
        metrics['rates'].append(uav.current_allocated_rate)
        # UAV使用sinr_db而不是current_sinr
        metrics['sinrs'].append(uav.sinr_db)
        
        # 业务类型满意度
        biz_type = uav.business_type.value if hasattr(uav.business_type, 'value') else 2
        metrics['biz_sats'][biz_type].append(uav.current_satisfaction)
        
        # 连接状态（connected_bs_id不为None表示已连接）
        is_connected = uav.connected_bs_id is not None
        if is_connected:
            metrics['connected_count'] += 1
        
        # 切换统计
        action = actions.get(uav_id, 0)
        if action != 0:  # 发生切换
            metrics['handover_attempts'] += 1
            # 简化的切换成功判断
            if is_connected:
                metrics['handover_success'] += 1
                metrics['handover_latencies'].append(5.0)  # 假设5ms切换延迟
        
        # 通信质量指标（简化计算）
        if uav.current_latency > 50:  # 延迟超过50ms认为有抖动
            metrics['ping_jitters'].append(uav.current_latency - 50)
        
        # 丢包率估算（基于SINR）
        if uav.sinr_db < 10:
            packet_loss = max(0, (10 - uav.sinr_db) * 2)
            metrics['packet_losses'].append(packet_loss)
        
        # QoS违规（基于业务需求）
        required_rate = getattr(uav, 'required_rate', 1)
        if uav.current_allocated_rate < required_rate * 0.8:
            metrics['qos_violations'].append(1)
    
    return metrics


def evaluate_algorithm(algo_name, algo, env, num_episodes=3, seed=42):
    """评估算法性能（完整指标版）"""
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
                action = algo.select_action(uav, env.env)
                actions[uav_id] = action
            
            decision_time = (time.time() - start_time) * 1000  # ms
            
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            # 收集详细指标（传入底层环境env.env）
            metrics = collect_detailed_metrics(env.env, actions)
            metrics['decision_time'] = decision_time / env.num_agents
            metrics['team_reward'] = team_reward
            step_metrics.append(metrics)
            
            episode_reward += team_reward
            
            if done:
                break
        
        # 汇总episode指标
        ep_summary = {
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfactions'] for m in step_metrics]),
            'avg_latency': np.mean([np.mean(m['latencies']) for m in step_metrics]),
            'avg_rate': np.mean([np.mean(m['rates']) for m in step_metrics]),
            'avg_sinr': np.mean([np.mean(m['sinrs']) for m in step_metrics]),
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) / 
                                    max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            # 新增通信指标
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }
        
        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, "
              f"reward={episode_reward:.2f}, "
              f"ho_rate={ep_summary['handover_success_rate']:.2f}, "
              f"jitter={ep_summary['avg_jitter']:.1f}ms, "
              f"loss={ep_summary['avg_packet_loss']:.1f}%")
    
    # 最终汇总
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
        # 新增通信指标
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }
    
    return summary


def evaluate_enhanced_algorithm(env, num_episodes=3, seed=42):
    """评估完整的增强算法（使用EnhancedHandoverAlgorithm）"""
    print("\n评估 增强算法(完整版)...")

    set_global_seed(seed)

    # 创建完整的增强算法实例
    enhanced = EnhancedHandoverAlgorithm(env.env, weight_config='optimized')

    all_metrics = []

    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()

        episode_reward = 0
        step_metrics = []

        for step in range(150):
            start_time = time.time()

            # 使用增强算法选择动作
            actions = {}
            for uav_id in range(env.num_agents):
                decision = enhanced.make_intelligent_decision(uav_id)
                if decision is None:
                    actions[uav_id] = 0  # 不切换
                else:
                    target_bs, downgrade_ratio = decision
                    # 动作编码：0=stay, 1=bs0, 2=bs1, ...
                    current_bs = env.env.uavs[uav_id].connected_bs_id
                    if target_bs == current_bs:
                        actions[uav_id] = 0
                    else:
                        actions[uav_id] = target_bs + 1

            decision_time = (time.time() - start_time) * 1000  # ms

            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

            # 收集详细指标
            metrics = collect_detailed_metrics(env.env, actions)
            metrics['decision_time'] = decision_time / env.num_agents
            metrics['team_reward'] = team_reward
            step_metrics.append(metrics)

            episode_reward += team_reward

            if done:
                break

        # 汇总episode指标
        ep_summary = {
            'reward': episode_reward,
            'avg_satisfaction': np.mean([m['satisfactions'] for m in step_metrics]),
            'avg_latency': np.mean([np.mean(m['latencies']) for m in step_metrics]),
            'avg_rate': np.mean([np.mean(m['rates']) for m in step_metrics]),
            'avg_sinr': np.mean([np.mean(m['sinrs']) for m in step_metrics]),
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) /
                                    max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            # 新增通信指标
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }

        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, "
              f"reward={episode_reward:.2f}, "
              f"ho_rate={ep_summary['handover_success_rate']:.2f}, "
              f"jitter={ep_summary['avg_jitter']:.1f}ms, "
              f"loss={ep_summary['avg_packet_loss']:.1f}%")

    # 最终汇总
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
        # 新增通信指标
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }

    return summary


def evaluate_optimized_mappo(env, num_episodes=3, seed=42):
    """评估优化后的MAPPO"""
    print("\n评估 优化MAPPO...")
    
    # 查找最新的优化MAPPO模型
    log_dir = './experiment_logs'
    
    # 首先尝试查找optimized_mappo模型
    opt_dirs = [d for d in os.listdir(log_dir) if d.startswith('optimized_mappo_')]
    if opt_dirs:
        latest_dir = sorted(opt_dirs)[-1]
        print(f"  找到优化模型目录: {latest_dir}")
    else:
        # 如果没有优化模型，尝试查找mappo_high模型
        dirs = [d for d in os.listdir(log_dir) if d.startswith('mappo_high_')]
        if not dirs:
            print("  错误: 未找到MAPPO模型")
            return None
        latest_dir = sorted(dirs)[-1]
        print(f"  警告: 未找到优化模型，使用旧模型: {latest_dir}")
    
    model_path = os.path.join(log_dir, latest_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, latest_dir, 'final_model.pt')
    
    print(f"  使用模型: {model_path}")
    
    # 创建agent
    set_global_seed(seed)
    obs_dict, global_state = env.reset()
    obs_dim = len(obs_dict[0])
    state_dim = len(global_state)
    
    agent = MAPPOAgentV2(
        num_agents=env.num_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=env.action_dim,
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
    
    agent.load(model_path)
    
    all_metrics = []
    
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        
        episode_reward = 0
        step_metrics = []
        
        for step in range(150):
            start_time = time.time()
            
            biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                        for uid in range(env.num_agents)}
            
            actions, _, _, _, _ = agent.select_actions(
                obs_dict, global_state, biz_types, training=False, env=env
            )
            
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
            'handover_success_rate': sum([m['handover_success'] for m in step_metrics]) /
                                    max(sum([m['handover_attempts'] for m in step_metrics]), 1),
            'connected_ratio': np.mean([m['connected_count'] / env.num_agents for m in step_metrics]),
            # 新增通信指标
            'avg_jitter': np.mean([np.mean(m['ping_jitters']) if m['ping_jitters'] else 0 for m in step_metrics]),
            'avg_packet_loss': np.mean([np.mean(m['packet_losses']) if m['packet_losses'] else 0 for m in step_metrics]),
            'qos_violation_rate': np.mean([sum(m['qos_violations']) / max(len(m['qos_violations']), 1) for m in step_metrics]),
            'avg_decision_time': np.mean([m['decision_time'] for m in step_metrics]),
        }

        all_metrics.append(ep_summary)
        print(f"  Episode {ep+1}: sat={ep_summary['avg_satisfaction']:.4f}, "
              f"reward={episode_reward:.2f}, "
              f"jitter={ep_summary['avg_jitter']:.1f}ms, "
              f"loss={ep_summary['avg_packet_loss']:.1f}%")

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
        # 新增通信指标
        'avg_jitter': np.mean([m['avg_jitter'] for m in all_metrics]),
        'avg_packet_loss': np.mean([m['avg_packet_loss'] for m in all_metrics]),
        'qos_violation_rate': np.mean([m['qos_violation_rate'] for m in all_metrics]),
        'avg_decision_time': np.mean([m['avg_decision_time'] for m in all_metrics]),
    }
    
    return summary


def main():
    """主函数"""
    print("="*70)
    print("完整对比实验 V2：传统算法 vs 增强算法 vs 优化MAPPO")
    print("="*70)
    
    # 使用高负载场景
    num_uav = 128
    num_bs = 3
    seed = 42
    
    print(f"\n实验配置:")
    print(f"  UAV数量: {num_uav}")
    print(f"  BS数量: {num_bs}")
    print(f"  负载率: ~88%")
    print(f"  评估轮数: 3")
    
    # 创建环境
    env = MultiAgentHandoverEnv(
        num_uav=num_uav,
        num_bs=num_bs,
        pos_range=1000,
        seed=seed
    )
    
    # 评估三种算法
    results = []
    
    # 1. 传统算法
    traditional = TraditionalAlgorithm()
    result = evaluate_algorithm("传统算法(3GPP)", traditional, env, num_episodes=3, seed=seed)
    results.append(result)
    
    # 2. 增强算法（使用完整的EnhancedHandoverAlgorithm）
    result = evaluate_enhanced_algorithm(env, num_episodes=3, seed=seed)
    results.append(result)
    
    # 3. 优化MAPPO
    result = evaluate_optimized_mappo(env, num_episodes=3, seed=seed)
    if result:
        results.append(result)
    
    # 打印对比结果
    print("\n" + "="*70)
    print("对比结果汇总")
    print("="*70)
    
    print(f"\n{'算法':<20} {'满意度':<15} {'奖励':<12} {'延迟(ms)':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['avg_satisfaction']:.4f}±{r['std_satisfaction']:.4f}   "
              f"{r['avg_reward']:.1f}±{r['std_reward']:.1f}    {r['avg_latency']:.1f}")

    print(f"\n{'算法':<20} {'SINR(dB)':<12} {'切换成功率':<12} {'连接率':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['avg_sinr']:.1f}          "
              f"{r['handover_success_rate']:.2f}        {r['connected_ratio']:.4f}")

    # 新增通信指标展示
    print(f"\n{'算法':<20} {'抖动(ms)':<12} {'丢包率(%)':<12} {'QoS违规率':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r.get('avg_jitter', 0):.1f}          "
              f"{r.get('avg_packet_loss', 0):.1f}          "
              f"{r.get('qos_violation_rate', 0):.4f}")

    print(f"\n{'算法':<20} {'决策时间(ms)':<15} {'速率(Mbps)':<12}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<20} {r.get('avg_decision_time', 0):.2f}             "
              f"{r['avg_rate']:.1f}")

    # 找出最佳算法
    best_sat = max(results, key=lambda x: x['avg_satisfaction'])
    best_reward = max(results, key=lambda x: x['avg_reward'])

    print(f"\n最佳满意度: {best_sat['name']} ({best_sat['avg_satisfaction']:.4f})")
    print(f"最佳奖励: {best_reward['name']} ({best_reward['avg_reward']:.2f})")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'comparison_results_v2_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {result_file}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
