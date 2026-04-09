# -*- coding: utf-8 -*-
"""
完整对比实验：传统算法 vs 增强算法 vs 优化MAPPO

使用优化后的MAPPO（参数搜索后的最佳配置）与传统算法、增强算法进行全面对比
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2


class TraditionalAlgorithm:
    """传统3GPP切换算法"""
    
    def __init__(self, sinr_threshold=3.0, hysteresis=1.0):
        self.sinr_threshold = sinr_threshold
        self.hysteresis = hysteresis
    
    def select_action(self, uav, env):
        """选择切换目标"""
        current_bs = uav.connected_bs_id
        
        # 找到SINR最好的基站
        best_bs = current_bs
        best_sinr = uav.sinr_to_bs.get(current_bs, -100)
        
        for bs_id in range(env.num_bs):
            if bs_id == current_bs:
                continue
            sinr = uav.sinr_to_bs.get(bs_id, -100)
            # A3事件：目标基站SINR比当前基站高threshold + hysteresis
            if sinr > best_sinr + self.sinr_threshold + self.hysteresis:
                best_sinr = sinr
                best_bs = bs_id
        
        # 返回动作：0=stay, 1+=switch to BS
        if best_bs == current_bs:
            return 0
        else:
            return best_bs + 1


class EnhancedAlgorithm:
    """增强切换算法（业务感知）"""
    
    def __init__(self):
        self.sinr_threshold = 3.0
        self.hysteresis = 1.0
        self.load_balance_weight = 0.3
        # 业务优先级
        self.biz_priority = {0: 1.0, 1: 0.8, 2: 0.5}  # 控制、视频、环境
    
    def select_action(self, uav, env):
        """选择切换目标"""
        current_bs = uav.connected_bs_id
        biz_type = uav.business_type.value if hasattr(uav.business_type, 'value') else 2
        
        # 计算各基站得分
        best_bs = current_bs
        best_score = -1000
        
        for bs_id in range(env.num_bs):
            # SINR得分
            sinr = uav.sinr_to_bs.get(bs_id, -100)
            sinr_score = min(sinr / 30, 1.0)
            
            # 负载得分
            if hasattr(env, 'base_stations') and bs_id in env.base_stations:
                bs = env.base_stations[bs_id]
                load_score = 1.0 - bs.current_load / bs.capacity
            else:
                load_score = 0.5
            
            # 业务优先级
            priority = self.biz_priority.get(biz_type, 0.5)
            
            # 综合得分
            score = 0.5 * sinr_score + 0.3 * load_score + 0.2 * priority
            
            # 切换判决
            if bs_id == current_bs:
                current_score = score
            elif score > best_score:
                # 检查切换条件
                current_sinr = uav.sinr_to_bs.get(current_bs, -100)
                if sinr > current_sinr + self.sinr_threshold + self.hysteresis:
                    best_score = score
                    best_bs = bs_id
        
        return 0 if best_bs == current_bs else best_bs + 1


def evaluate_algorithm(algo_name, algo, env, num_episodes=3, seed=42):
    """
    评估算法性能
    
    Args:
        algo_name: 算法名称
        algo: 算法实例
        env: 环境
        num_episodes: 评估轮数
        seed: 随机种子
    
    Returns:
        评估结果字典
    """
    print(f"\n评估 {algo_name}...")
    
    set_global_seed(seed)
    
    results = {
        'satisfaction': [],
        'rewards': [],
        'handover_count': [],
        'stay_count': [],
    }
    
    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        
        episode_reward = 0
        episode_sats = []
        handover_count = 0
        stay_count = 0
        
        for step in range(150):
            actions = {}
            for uav_id in range(env.num_agents):
                uav = env.env.uavs[uav_id]
                action = algo.select_action(uav, env.env)
                actions[uav_id] = action
                
                if action == 0:
                    stay_count += 1
                else:
                    handover_count += 1
            
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            
            if done:
                break
        
        results['satisfaction'].append(np.mean(episode_sats))
        results['rewards'].append(episode_reward)
        results['handover_count'].append(handover_count)
        results['stay_count'].append(stay_count)
        
        print(f"  Episode {ep+1}: sat={np.mean(episode_sats):.4f}, reward={episode_reward:.2f}")
    
    # 汇总
    summary = {
        'name': algo_name,
        'avg_satisfaction': np.mean(results['satisfaction']),
        'std_satisfaction': np.std(results['satisfaction']),
        'avg_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'avg_handover': np.mean(results['handover_count']),
        'avg_stay': np.mean(results['stay_count']),
    }
    
    return summary


def evaluate_optimized_mappo(env, num_episodes=3, seed=42):
    """评估优化后的MAPPO"""
    print("\n评估 优化MAPPO...")
    
    # 查找最新的高负载模型
    log_dir = './experiment_logs'
    dirs = [d for d in os.listdir(log_dir) if d.startswith('mappo_high_')]
    if not dirs:
        print("  错误: 未找到MAPPO模型")
        return None
    
    latest_dir = sorted(dirs)[-1]
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
    
    results = {
        'satisfaction': [],
        'rewards': [],
    }
    
    for ep in range(num_episodes):
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
            
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            episode_reward += team_reward
            episode_sats.append(info.get('avg_satisfaction', 0))
            
            if done:
                break
        
        results['satisfaction'].append(np.mean(episode_sats))
        results['rewards'].append(episode_reward)
        
        print(f"  Episode {ep+1}: sat={np.mean(episode_sats):.4f}, reward={episode_reward:.2f}")
    
    summary = {
        'name': '优化MAPPO',
        'avg_satisfaction': np.mean(results['satisfaction']),
        'std_satisfaction': np.std(results['satisfaction']),
        'avg_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
    }
    
    return summary


def main():
    """主函数"""
    print("="*70)
    print("完整对比实验：传统算法 vs 增强算法 vs 优化MAPPO")
    print("="*70)
    
    # 使用高负载场景（与原实验一致）
    num_uav = 128
    num_bs = 3
    seed = 42
    
    print(f"\n实验配置:")
    print(f"  UAV数量: {num_uav}")
    print(f"  BS数量: {num_bs}")
    print(f"  负载率: ~88%")
    print(f"  评估轮数: 3")
    
    # 创建环境
    env = QMixHandoverEnv(
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
    
    # 2. 增强算法
    enhanced = EnhancedAlgorithm()
    result = evaluate_algorithm("增强算法", enhanced, env, num_episodes=3, seed=seed)
    results.append(result)
    
    # 3. 优化MAPPO
    result = evaluate_optimized_mappo(env, num_episodes=3, seed=seed)
    if result:
        results.append(result)
    
    # 打印对比结果
    print("\n" + "="*70)
    print("对比结果汇总")
    print("="*70)
    
    print(f"\n{'算法':<20} {'满意度':<15} {'奖励':<15}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<20} {r['avg_satisfaction']:.4f}±{r['std_satisfaction']:.4f}   {r['avg_reward']:.2f}±{r['std_reward']:.2f}")
    
    # 找出最佳算法
    best_sat = max(results, key=lambda x: x['avg_satisfaction'])
    best_reward = max(results, key=lambda x: x['avg_reward'])
    
    print(f"\n最佳满意度: {best_sat['name']} ({best_sat['avg_satisfaction']:.4f})")
    print(f"最佳奖励: {best_reward['name']} ({best_reward['avg_reward']:.2f})")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'comparison_results_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {result_file}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
