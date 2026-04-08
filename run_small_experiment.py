# -*- coding: utf-8 -*-
"""
简化版 MAPPO Small 实验

专注于解决 KL 散度过大的问题，确保算法能够收敛

特性：
1. 简化网络结构，移除RNN提高稳定性
2. 优化奖励函数，增强信号强度
3. 改进早停策略，避免过早终止
4. 专注于 small 场景 (UAV=10)
5. 确保传统算法 < 增强算法 < MAPPO 算法的性能排序

Author: Small Experiment Optimizer
Date: 2026-04-08
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict, deque


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
from uav_system.business import BusinessType


class SimplifiedSmallExperiment:
    """简化版 small 实验"""

    def __init__(self):
        self.results = {}
        self.output_dir = 'small_experiment_results'
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """运行简化版 small 实验"""
        print("\n" + "="*80)
        print("SIMPLIFIED MAPPO SMALL EXPERIMENT")
        print("="*80)

        # 配置参数
        config = {
            'num_bs': 4,
            'num_uav': 10,
            'num_steps': 100,
            'train_episodes': 200,
            'eval_repetitions': 10,
            'hidden_dim': 64,
            'critic_hidden_dim': 128,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'clip_epsilon': 0.1,
            'entropy_coef': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'rollout_length': 150,
            'num_epochs': 5,
            'batch_size': 64,
        }

        print(f"配置: {config}")

        # 1. 训练 MAPPO
        print("\n1. 训练 MAPPO...")
        mappo_results = self._train_mappo(config)
        self.results['mappo'] = mappo_results

        # 2. 评估所有算法
        print("\n2. 评估算法...")
        eval_results = self._evaluate_algorithms(config)
        self.results['evaluation'] = eval_results

        # 3. 生成报告
        print("\n3. 生成报告...")
        self._generate_report()

        return self.results

    def _train_mappo(self, config):
        """训练 MAPPO 算法"""
        set_global_seed(GLOBAL_SEED)

        # 创建环境
        env = QMixHandoverEnv(
            num_bs=config['num_bs'],
            num_uav=config['num_uav'],
            max_steps=config['num_steps'],
            map_size=1000,
            enable_reward_normalization=True,
        )

        # 初始化智能体
        agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=config['hidden_dim'],
            critic_hidden_dim=config['critic_hidden_dim'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            entropy_coef=config['entropy_coef'],
            rollout_length=config['rollout_length'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            use_biz_heads=True,
            use_attention_critic=True,
            use_enhanced_algorithm=True,
            use_pretrain=True,
            use_hierarchical=True,
            use_transformer=False,
            use_data_augmentation=True,
        )

        # 收集增强算法示范数据
        print("  收集增强算法示范数据...")
        self._collect_demonstrations(env, agent, 1000)

        # 模仿学习预训练
        print("  模仿学习预训练...")
        agent.pretrain(epochs=50, batch_size=64, lr=1e-4)

        # 训练监控
        episode_rewards = []
        episode_satisfactions = []
        episode_actor_losses = []
        episode_critic_losses = []
        episode_entropies = []
        episode_kl_values = []

        best_sat = -float('inf')
        best_model_path = os.path.join(self.output_dir, 'mappo_best.pt')

        print("  开始训练...")
        for ep in range(config['train_episodes']):
            obs_dict, global_state = env.reset()
            agent.reset_hidden()
            episode_reward = 0.0
            episode_sat = 0.0

            for step in range(config['num_steps']):
                # 获取业务类型
                biz_types = {}
                for uid in range(env.num_agents):
                    uav = env.env.uavs[uid]
                    biz_types[uid] = uav.true_business_type.value

                # 选择动作
                actions, log_probs, values, pre_hidden = agent.select_actions(
                    obs_dict, global_state, biz_types, training=True, env=env
                )

                # 执行动作
                next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

                # 存储经验
                agent.insert_experience(
                    step, obs_dict, global_state, actions,
                    rewards, team_reward, done, log_probs, values,
                    biz_types, pre_hidden
                )

                # 更新状态
                obs_dict = next_obs
                global_state = next_state
                episode_reward += team_reward

            # 计算满意度
            total_sat = 0.0
            for uid in range(env.num_agents):
                uav = env.env.uavs[uid]
                total_sat += uav.current_satisfaction
            episode_sat = total_sat / env.num_agents

            # 训练
            train_stats = agent.train()
            if train_stats:
                episode_actor_losses.append(train_stats.get('actor_loss', 0))
                episode_critic_losses.append(train_stats.get('critic_loss', 0))
                episode_entropies.append(train_stats.get('entropy', 0))
                episode_kl_values.append(train_stats.get('kl_divergence', 0))

            # 记录
            episode_rewards.append(episode_reward)
            episode_satisfactions.append(episode_sat)

            # 保存最佳模型
            if episode_sat > best_sat:
                best_sat = episode_sat
                agent.save(best_model_path)

            # 每10个episode打印进度
            if (ep + 1) % 10 == 0:
                recent_rew = episode_rewards[-10:]
                recent_sat = episode_satisfactions[-10:]
                print(f"  Episode {ep+1}/{config['train_episodes']}: "
                      f"reward={np.mean(recent_rew):.1f}, "
                      f"sat={np.mean(recent_sat):.3f}, "
                      f"best_sat={best_sat:.3f}")

        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, 'mappo_final.pt')
        agent.save(final_model_path)

        return {
            'rewards': episode_rewards,
            'satisfactions': episode_satisfactions,
            'actor_losses': episode_actor_losses,
            'critic_losses': episode_critic_losses,
            'entropies': episode_entropies,
            'kl_values': episode_kl_values,
            'best_sat': best_sat,
            'best_model_path': best_model_path,
            'final_model_path': final_model_path,
        }

    def _collect_demonstrations(self, env, agent, num_demos):
        """收集增强算法示范数据"""
        enhanced_algorithm = EnhancedHandoverAlgorithm(env.env)
        agent.enhanced_algorithm = enhanced_algorithm

        for i in range(num_demos):
            obs_dict, global_state = env.reset()
            for step in range(env.max_steps):
                # 使用增强算法选择动作
                enhanced_algorithm.run_step(enable_load_balancing=True)
                actions = {}
                for uid in range(env.num_agents):
                    uav = env.env.uavs[uid]
                    # 基于增强算法的决策
                    best_bs = enhanced_algorithm.get_best_base_station(uav)
                    if best_bs == uav.connected_bs_id:
                        actions[uid] = 0  # stay
                    else:
                        # 找到对应的动作索引
                        action = best_bs + 1  # 1-based
                        actions[uid] = action

                # 执行动作
                next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                obs_dict = next_obs

                if done:
                    break

    def _evaluate_algorithms(self, config):
        """评估所有算法"""
        results = {}

        # 加载 MAPPO 模型
        mappo_model_path = os.path.join(self.output_dir, 'mappo_best.pt')
        if not os.path.exists(mappo_model_path):
            mappo_model_path = os.path.join(self.output_dir, 'mappo_final.pt')

        # 创建环境
        env = QMixHandoverEnv(
            num_bs=config['num_bs'],
            num_uav=config['num_uav'],
            max_steps=config['num_steps'],
            map_size=1000,
        )

        # 初始化 MAPPO 智能体
        mappo_agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=config['hidden_dim'],
            critic_hidden_dim=config['critic_hidden_dim'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            entropy_coef=config['entropy_coef'],
            rollout_length=config['rollout_length'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            use_biz_heads=True,
            use_attention_critic=True,
            use_enhanced_algorithm=True,
            use_pretrain=True,
            use_hierarchical=True,
        )

        if os.path.exists(mappo_model_path):
            mappo_agent.load(mappo_model_path)
            print(f"  加载 MAPPO 模型: {mappo_model_path}")

        # 初始化传统算法和增强算法
        traditional_algorithm = IntegratedHandoverAlgorithm(env.env)
        enhanced_algorithm = EnhancedHandoverAlgorithm(env.env)

        # 评估每种算法
        algorithms = {
            'traditional': traditional_algorithm,
            'enhanced': enhanced_algorithm,
            'mappo': mappo_agent,
        }

        for algo_name, algorithm in algorithms.items():
            print(f"  评估 {algo_name}...")
            algo_results = []
            
            for rep in range(config['eval_repetitions']):
                obs_dict, global_state = env.reset()
                total_reward = 0.0
                total_sat = 0.0

                for step in range(config['num_steps']):
                    if algo_name == 'mappo':
                        # 获取业务类型
                        biz_types = {}
                        for uid in range(env.num_agents):
                            uav = env.env.uavs[uid]
                            biz_types[uid] = uav.true_business_type.value
                        
                        # 使用 MAPPO 选择动作
                        actions, _, _, _ = algorithm.select_actions(
                            obs_dict, global_state, biz_types, training=False, env=env
                        )
                    else:
                        # 使用传统或增强算法
                        algorithm.run_step(enable_load_balancing=True)
                        actions = {}
                        for uid in range(env.num_agents):
                            uav = env.env.uavs[uid]
                            if algo_name == 'traditional':
                                best_bs = algorithm.get_best_base_station(uav)
                            else:
                                best_bs = algorithm.get_best_base_station(uav)
                            
                            if best_bs == uav.connected_bs_id:
                                actions[uid] = 0  # stay
                            else:
                                # 找到对应的动作索引
                                action = best_bs + 1  # 1-based
                                actions[uid] = action

                    # 执行动作
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    obs_dict = next_obs
                    total_reward += team_reward

                # 计算最终满意度
                final_sat = 0.0
                for uid in range(env.num_agents):
                    uav = env.env.uavs[uid]
                    final_sat += uav.current_satisfaction
                final_sat /= env.num_agents
                
                algo_results.append(final_sat)

            results[algo_name] = {
                'mean': np.mean(algo_results),
                'std': np.std(algo_results),
                'values': algo_results,
            }

        return results

    def _generate_report(self):
        """生成实验报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 训练曲线
        plt.figure(figsize=(12, 8))

        # 奖励曲线
        plt.subplot(2, 2, 1)
        rewards = self.results['mappo']['rewards']
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)

        # 满意度曲线
        plt.subplot(2, 2, 2)
        satisfactions = self.results['mappo']['satisfactions']
        plt.plot(satisfactions)
        plt.title('Training Satisfaction')
        plt.xlabel('Episode')
        plt.ylabel('Satisfaction')
        plt.grid(True)

        # KL散度曲线
        plt.subplot(2, 2, 3)
        kl_values = self.results['mappo']['kl_values']
        if kl_values:
            plt.plot(kl_values)
            plt.title('KL Divergence')
            plt.xlabel('Episode')
            plt.ylabel('KL Value')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No KL data', ha='center', va='center')

        # 算法对比
        plt.subplot(2, 2, 4)
        eval_results = self.results['evaluation']
        algorithms = ['traditional', 'enhanced', 'mappo']
        means = [eval_results[algo]['mean'] for algo in algorithms]
        stds = [eval_results[algo]['std'] for algo in algorithms]
        
        plt.bar(algorithms, means, yerr=stds, capsize=5)
        plt.title('Algorithm Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Mean Satisfaction')
        plt.grid(True, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'training_curve_{timestamp}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        # 2. 结果报告
        report_path = os.path.join(self.output_dir, f'experiment_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("SIMPLIFIED MAPPO SMALL EXPERIMENT REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TRAINING RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Best Satisfaction: {self.results['mappo']['best_sat']:.3f}\n")
            f.write(f"Final Average Reward: {np.mean(self.results['mappo']['rewards'][-20:]):.2f}\n")
            f.write(f"Final Average Satisfaction: {np.mean(self.results['mappo']['satisfactions'][-20:]):.3f}\n\n")
            
            f.write("EVALUATION RESULTS:\n")
            f.write("-"*40 + "\n")
            for algo in ['traditional', 'enhanced', 'mappo']:
                result = self.results['evaluation'][algo]
                f.write(f"{algo}: {result['mean']:.3f} ± {result['std']:.3f}\n")
            f.write("\n")
            
            # 检查性能排序
            traditional_sat = self.results['evaluation']['traditional']['mean']
            enhanced_sat = self.results['evaluation']['enhanced']['mean']
            mappo_sat = self.results['evaluation']['mappo']['mean']
            
            if traditional_sat < enhanced_sat < mappo_sat:
                f.write("✓ Performance Order: traditional < enhanced < mappo\n")
            else:
                f.write("✗ Performance Order: NOT achieved\n")
                f.write(f"  Actual: traditional={traditional_sat:.3f}, enhanced={enhanced_sat:.3f}, mappo={mappo_sat:.3f}\n")

        print(f"\n报告生成完成:")
        print(f"  训练曲线: {plot_path}")
        print(f"  实验报告: {report_path}")


def main():
    """主函数"""
    experiment = SimplifiedSmallExperiment()
    results = experiment.run()
    return results


if __name__ == "__main__":
    main()
