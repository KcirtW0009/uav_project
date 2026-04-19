# -*- coding: utf-8 -*-
"""
优化后的 MAPPO 实验运行脚本

整合所有优化：
1. 改进的奖励函数
2. 增强的状态空间
3. 完整的通信指标采集
4. 优化的超参数配置
"""

import numpy as np
import torch
import os
import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, RESULT_DIR
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2
from uav_system.mappo_optimized_config import get_optimized_config, print_config
from uav_system.reward_functions import get_reward_function, RewardNormalizer
from uav_system.enhanced_observation import EnhancedObservationSpace
from uav_system.communication_metrics import CommunicationMetricsCollector, compute_critical_satisfaction, compute_weighted_satisfaction
from mappo_enhanced_monitoring import MAPPOTrainingMonitor


class OptimizedMAPPOExperiment:
    """优化后的MAPPO实验类"""
    
    def __init__(self, scenario='high', config_overrides=None):
        """
        初始化实验
        
        Args:
            scenario: 负载场景 ('low', 'medium', 'high', 'extreme')
            config_overrides: 配置覆盖
        """
        # 获取配置
        self.config = get_optimized_config(scenario, config_overrides)
        self.scenario = scenario
        
        # 创建日志目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(
            self.config['data_logging']['log_dir'],
            f'mappo_{scenario}_{timestamp}'
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 初始化组件
        self.reward_func = None
        self.reward_normalizer = RewardNormalizer()
        self.obs_space = None
        self.metrics_collector = CommunicationMetricsCollector()
        self.monitor = MAPPOTrainingMonitor(self.log_dir)
        
        # 训练状态
        self.agent = None
        self.env = None
        
    def setup(self, seed=42):
        """设置实验环境"""
        set_global_seed(seed)
        
        # 创建环境
        self.env = MultiAgentHandoverEnv(
            num_uav=self.config['num_uav'],
            num_bs=self.config['num_bs'],
            pos_range=self.config['map_size'],
            seed=seed
        )
        
        # 获取实际的观测维度
        obs_sample = self.env.reset()[0][0]
        obs_dim = len(obs_sample)
        
        # 初始化观测空间
        self.obs_space = EnhancedObservationSpace(
            num_bs=self.config['num_bs'],
            history_length=self.config['observation']['history_length'],
            use_relative_position=self.config['observation']['use_relative_position']
        )
        
        # 创建奖励函数
        reward_config = self.config.get('reward_function', {}).copy()
        version = reward_config.pop('version', 'v2')  # 提取version，避免重复
        self.reward_func = get_reward_function(version, **reward_config)
        
        # 创建Agent
        self.agent = MAPPOAgentV2(
            num_agents=self.config['num_uav'],
            obs_dim=obs_dim,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim=self.config['hidden_dim'],
            critic_hidden_dim=self.config['critic_hidden_dim'],
            actor_lr=self.config['actor_lr'],
            critic_lr=self.config['critic_lr'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_epsilon=self.config['clip_epsilon'],
            entropy_coef=self.config['entropy_coef'],
            value_coef=self.config['value_coef'],
            rollout_length=self.config['rollout_length'],
            num_epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            use_biz_heads=self.config['use_biz_heads'],
            use_attention_critic=self.config['use_attention_critic'],
            train_sample_agents=self.config['train_sample_agents'],
            attention_sample_agents=self.config['attention_sample_agents'],
            use_early_stopping=self.config['use_early_stopping'],
            early_stop_patience=self.config['early_stop_patience'],
        )
        
        print(f"实验设置完成:")
        print(f"  场景: {self.scenario}")
        print(f"  UAV数量: {self.config['num_uav']}")
        print(f"  观测维度: {obs_dim}")
        print(f"  日志目录: {self.log_dir}")
    
    def train(self, num_episodes=None):
        """
        训练MAPPO
        
        Args:
            num_episodes: 训练轮数，None使用配置值
        """
        if num_episodes is None:
            num_episodes = self.config['train_episodes']
        
        print(f"\n开始训练 {num_episodes} 轮...")
        print("=" * 60)
        
        best_sat = 0.0
        no_improve_count = 0
        
        for episode in range(num_episodes):
            # 开始新episode
            self.metrics_collector.start_episode()
            
            obs_dict, global_state = self.env.reset()
            self.agent.reset_hidden()
            
            episode_reward = 0
            episode_sats = []
            episode_decisions = []
            
            for step in range(self.config['rollout_length']):
                # 获取业务类型
                biz_types = {}
                for uid in range(self.env.num_agents):
                    uav = self.env.env.uavs[uid]
                    biz_types[uid] = uav.true_business_type.value
                
                # 选择动作
                import time
                start_time = time.time()
                actions, log_probs, values, _, _ = self.agent.select_actions(
                    obs_dict, global_state, biz_types, training=True, env=self.env
                )
                decision_time = (time.time() - start_time) * 1000  # ms
                
                # 记录决策
                for uav_id, action in actions.items():
                    uav = self.env.env.uavs[uav_id]
                    self.metrics_collector.record_decision(
                        uav_id=uav_id,
                        action=action,
                        last_bs=getattr(uav, 'last_bs_id', 0),
                        target_bs=action - 1 if action > 0 else getattr(uav, 'connected_bs_id', 0),
                        decision_time_ms=decision_time / len(actions),
                        uav=uav,
                        env=self.env.env
                    )
                
                # 执行动作
                next_obs, next_state, rewards, team_reward, done, info = self.env.step(actions)
                
                # 存储经验
                self.agent.insert_experience(
                    step, obs_dict, global_state, actions,
                    rewards, team_reward, done, log_probs, values,
                    biz_types, None
                )
                
                episode_reward += team_reward
                episode_sats.append(info.get('avg_satisfaction', 0))
                
                # 记录step统计
                self.metrics_collector.record_step_stats(
                    env=self.env.env,
                    info=info,
                    recognition_accuracy=info.get('recognition_accuracy', 0)
                )
                
                obs_dict = next_obs
                global_state = next_state
                
                if done:
                    break
            
            # 训练
            train_stats = self.agent.train()
            
            # 结束episode
            self.metrics_collector.end_episode()
            
            # 计算指标
            avg_sat = np.mean(episode_sats) if episode_sats else 0
            
            # 记录到监控器
            monitor_metrics = {
                'reward': episode_reward,
                'satisfaction': avg_sat,
            }
            if train_stats:
                monitor_metrics.update({
                    'actor_loss': train_stats.get('actor_loss', 0),
                    'critic_loss': train_stats.get('critic_loss', 0),
                    'entropy': train_stats.get('entropy', 0),
                    'kl_divergence': train_stats.get('kl_divergence', 0),
                    'value_mse': train_stats.get('value_mse', 0),
                    'actor_grad_norm': train_stats.get('actor_grad_norm', 0),
                    'critic_grad_norm': train_stats.get('critic_grad_norm', 0),
                })
            
            self.monitor.log_episode(episode, monitor_metrics)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"reward={episode_reward:.2f}, sat={avg_sat:.4f}")
                
                # 早停检查
                if avg_sat > best_sat + self.config['min_delta']:
                    best_sat = avg_sat
                    no_improve_count = 0
                    # 保存最佳模型
                    self.agent.save(os.path.join(self.log_dir, 'best_model.pt'))
                else:
                    no_improve_count += 1
                
                if no_improve_count >= self.config['early_stop_patience']:
                    print(f"早停触发！{no_improve_count}轮无改善")
                    break
        
        # 保存最终模型
        self.agent.save(os.path.join(self.log_dir, 'final_model.pt'))
        
        # 生成可视化
        self.monitor.create_comprehensive_visualization()
        self.monitor.save_detailed_logs()
        
        print("=" * 60)
        print("训练完成！")
        print(f"最佳满意度: {best_sat:.4f}")
        print(f"日志保存至: {self.log_dir}")
    
    def evaluate(self, num_episodes=None):
        """
        评估训练好的模型
        
        Args:
            num_episodes: 评估轮数
        """
        if num_episodes is None:
            num_episodes = self.config['eval_episodes']
        
        print(f"\n开始评估 {num_episodes} 轮...")
        
        # 重置采集器
        self.metrics_collector.reset()
        
        for ep in range(num_episodes):
            self.metrics_collector.start_episode()
            
            obs_dict, global_state = self.env.reset()
            self.agent.reset_hidden()
            
            for step in range(self.config['rollout_length']):
                # 获取业务类型
                biz_types = {}
                for uid in range(self.env.num_agents):
                    uav = self.env.env.uavs[uid]
                    biz_types[uid] = uav.true_business_type.value
                
                # 选择动作（不探索）
                actions, _, _, _, _ = self.agent.select_actions(
                    obs_dict, global_state, biz_types, training=False, env=self.env
                )
                
                # 记录决策
                for uav_id, action in actions.items():
                    uav = self.env.env.uavs[uav_id]
                    self.metrics_collector.record_decision(
                        uav_id=uav_id,
                        action=action,
                        last_bs=getattr(uav, 'last_bs_id', 0),
                        target_bs=action - 1 if action > 0 else getattr(uav, 'connected_bs_id', 0),
                        decision_time_ms=0,
                        uav=uav,
                        env=self.env.env
                    )
                
                # 执行动作
                next_obs, next_state, rewards, team_reward, done, info = self.env.step(actions)
                
                # 记录统计
                self.metrics_collector.record_step_stats(
                    env=self.env.env,
                    info=info
                )
                
                obs_dict = next_obs
                global_state = next_state
                
                if done:
                    break
            
            self.metrics_collector.end_episode()
        
        # 获取汇总
        summary = self.metrics_collector.get_summary()
        
        # 保存结果
        results_file = os.path.join(self.log_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印结果
        print("\n评估结果:")
        print("=" * 60)
        for metric_name, stats in summary.items():
            print(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("=" * 60)
        
        return summary


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='运行优化后的MAPPO实验')
    parser.add_argument('--scenario', type=str, default='high',
                       choices=['low', 'medium', 'high', 'extreme'],
                       help='负载场景')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--eval', action='store_true', help='评估模式')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 打印配置
    config = get_optimized_config(args.scenario)
    print_config(config)
    
    # 创建实验
    experiment = OptimizedMAPPOExperiment(scenario=args.scenario)
    experiment.setup(seed=args.seed)
    
    # 训练
    if args.train or (not args.train and not args.eval):
        experiment.train(num_episodes=args.episodes)
    
    # 评估
    if args.eval or (not args.train and not args.eval):
        experiment.evaluate()


if __name__ == '__main__':
    main()
