# -*- coding: utf-8 -*-
"""
MAPPO 参数系统性探索实验框架

针对收敛抖动、学习率调度、经验回放等关键参数进行系统性测试
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import itertools
import copy

from uav_system.config import set_global_seed, RESULT_DIR
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2


class MAPPOParameterSearch:
    """MAPPO参数搜索与优化类"""
    
    # 参数搜索空间定义
    PARAM_SPACE = {
        # 学习率相关
        'actor_lr': [1e-5, 3e-5, 5e-5, 1e-4],
        'critic_lr': [1e-4, 3e-4, 5e-4, 1e-3],
        'lr_schedule': ['cosine', 'step', 'exponential', 'none'],
        
        # PPO核心参数
        'clip_epsilon': [0.1, 0.15, 0.2, 0.3],
        'entropy_coef': [0.01, 0.02, 0.05, 0.1],
        'value_coef': [0.3, 0.5, 0.7, 1.0],
        
        # GAE参数
        'gae_lambda': [0.9, 0.95, 0.99],
        'gamma': [0.95, 0.99, 0.995],
        
        # 训练参数
        'num_epochs': [3, 5, 8, 10],
        'batch_size': [64, 128, 256, 512],
        'rollout_length': [100, 150, 200],
        
        # 网络结构
        'hidden_dim': [32, 64, 128],
        'critic_hidden_dim': [64, 128, 256],
        
        # 早停与收敛
        'early_stop_patience': [50, 100, 150],
        'min_delta': [0.0001, 0.001, 0.01],
        
        # 奖励归一化
        'reward_norm': ['none', 'running_mean', 'batch_norm'],
        'advantage_norm': [True, False],
        
        # 探索策略
        'use_linear_decay': [True, False],
        'initial_entropy_coef': [0.1, 0.2, 0.5],
    }
    
    def __init__(self, base_config: Dict, num_uav: int = 128, num_bs: int = 3, 
                 train_episodes: int = 100, eval_episodes: int = 3):
        """
        初始化参数搜索实验
        
        Args:
            base_config: 基础配置字典
            num_uav: UAV数量
            num_bs: 基站数量
            train_episodes: 每个参数组合的训练轮数
            eval_episodes: 评估轮数
        """
        self.base_config = base_config
        self.num_uav = num_uav
        self.num_bs = num_bs
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        
        self.results_dir = os.path.join(RESULT_DIR, 'param_search')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.results = []
        
    def create_agent(self, config: Dict) -> MAPPOAgentV2:
        """根据配置创建MAPPO Agent"""
        return MAPPOAgentV2(
            num_agents=self.num_uav,
            obs_dim=config.get('obs_dim', 20),
            state_dim=config.get('state_dim', 30),
            action_dim=config.get('action_dim', self.num_bs + 1),
            hidden_dim=config.get('hidden_dim', 64),
            critic_hidden_dim=config.get('critic_hidden_dim', 128),
            actor_lr=config.get('actor_lr', 5e-5),
            critic_lr=config.get('critic_lr', 3e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_epsilon=config.get('clip_epsilon', 0.1),
            entropy_coef=config.get('entropy_coef', 0.02),
            value_coef=config.get('value_coef', 0.5),
            rollout_length=config.get('rollout_length', 150),
            num_epochs=config.get('num_epochs', 5),
            batch_size=config.get('batch_size', 128),
            use_biz_heads=config.get('use_biz_heads', True),
            use_attention_critic=config.get('use_attention_critic', True),
            train_sample_agents=config.get('train_sample_agents', 50),
            attention_sample_agents=config.get('attention_sample_agents', 50),
            use_early_stopping=config.get('use_early_stopping', True),
            early_stop_patience=config.get('early_stop_patience', 100),
        )
    
    def train_single_config(self, config: Dict, config_id: int) -> Dict:
        """
        训练单个参数配置
        
        Returns:
            包含训练结果的字典
        """
        print(f"\n{'='*60}")
        print(f"Config {config_id}: {config}")
        print(f"{'='*60}")
        
        # 设置随机种子
        seed = 42 + config_id
        set_global_seed(seed)
        
        # 创建环境
        env = MultiAgentHandoverEnv(
            num_uav=self.num_uav,
            num_bs=self.num_bs,
            pos_range=1000,
            seed=seed
        )
        
        # 获取实际的观测维度和状态维度
        obs_dict, global_state = env.reset()
        actual_obs_dim = len(obs_dict[0])
        actual_state_dim = len(global_state)
        config['obs_dim'] = actual_obs_dim
        config['state_dim'] = actual_state_dim
        
        print(f"  Actual obs_dim: {actual_obs_dim}, state_dim: {actual_state_dim}")
        
        # 创建agent
        agent = self.create_agent(config)
        
        # 训练记录
        episode_rewards = []
        episode_satisfactions = []
        episode_actor_losses = []
        episode_critic_losses = []
        episode_entropies = []
        episode_kls = []
        
        # 训练循环
        for ep in range(self.train_episodes):
            obs_dict, global_state = env.reset()
            agent.reset_hidden()
            
            episode_reward = 0
            episode_sat = []
            
            for step in range(config.get('rollout_length', 150)):
                # 获取业务类型
                biz_types = {}
                for uid in range(env.num_agents):
                    uav = env.env.uavs[uid]
                    biz_types[uid] = uav.true_business_type.value
                
                # 选择动作
                actions, log_probs, values, _, _ = agent.select_actions(
                    obs_dict, global_state, biz_types, training=True, env=env
                )
                
                # 执行动作
                next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                
                # 存储经验
                agent.insert_experience(
                    step, obs_dict, global_state, actions,
                    rewards, team_reward, done, log_probs, values,
                    biz_types, None
                )
                
                episode_reward += team_reward
                episode_sat.append(info.get('avg_satisfaction', 0))
                
                obs_dict = next_obs
                global_state = next_state
                
                if done:
                    break
            
            # 训练
            train_stats = agent.train()
            
            episode_rewards.append(episode_reward)
            episode_satisfactions.append(np.mean(episode_sat))
            
            if train_stats:
                episode_actor_losses.append(train_stats.get('actor_loss', 0))
                episode_critic_losses.append(train_stats.get('critic_loss', 0))
                episode_entropies.append(train_stats.get('entropy', 0))
                episode_kls.append(train_stats.get('kl_divergence', 0))
            
            if (ep + 1) % 20 == 0:
                print(f"  Ep {ep+1}/{self.train_episodes}: "
                      f"reward={episode_reward:.2f}, sat={np.mean(episode_sat):.3f}")
        
        # 计算收敛指标
        result = {
            'config_id': config_id,
            'config': config,
            'final_reward': np.mean(episode_rewards[-10:]),
            'final_sat': np.mean(episode_satisfactions[-10:]),
            'reward_variance': np.var(episode_rewards[-20:]),
            'sat_variance': np.var(episode_satisfactions[-20:]),
            'convergence_speed': self._compute_convergence_speed(episode_satisfactions),
            'reward_curve': episode_rewards,
            'sat_curve': episode_satisfactions,
            'actor_losses': episode_actor_losses,
            'critic_losses': episode_critic_losses,
            'entropies': episode_entropies,
            'kls': episode_kls,
        }
        
        return result
    
    def _compute_convergence_speed(self, values: List[float]) -> int:
        """计算收敛速度（达到最终90%性能所需的轮数）"""
        if len(values) < 20:
            return len(values)
        
        final_value = np.mean(values[-10:])
        threshold = final_value * 0.9 if final_value > 0 else final_value * 1.1
        
        for i, v in enumerate(values):
            if np.mean(values[i:i+5]) >= threshold:
                return i
        
        return len(values)
    
    def grid_search(self, param_subset: Dict = None, max_combinations: int = 20):
        """
        网格搜索参数组合
        
        Args:
            param_subset: 要搜索的参数子集，None则使用默认子集
            max_combinations: 最大组合数
        """
        if param_subset is None:
            # 默认搜索关键参数
            param_subset = {
                'actor_lr': [3e-5, 5e-5, 1e-4],
                'critic_lr': [3e-4, 5e-4, 1e-3],
                'clip_epsilon': [0.1, 0.2],
                'entropy_coef': [0.01, 0.02, 0.05],
                'gae_lambda': [0.95, 0.99],
            }
        
        # 生成所有参数组合
        keys = list(param_subset.keys())
        values = [param_subset[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        print(f"总共 {len(combinations)} 个参数组合，将测试前 {min(max_combinations, len(combinations))} 个")
        
        for i, combo in enumerate(combinations[:max_combinations]):
            config = self.base_config.copy()
            config.update(dict(zip(keys, combo)))
            
            result = self.train_single_config(config, i)
            self.results.append(result)
        
        # 保存结果
        self._save_results()
        self._analyze_results()
    
    def random_search(self, num_samples: int = 20):
        """
        随机搜索参数空间
        
        Args:
            num_samples: 随机采样数量
        """
        print(f"随机搜索 {num_samples} 个参数配置")
        
        for i in range(num_samples):
            config = self.base_config.copy()
            
            # 随机采样参数
            config['actor_lr'] = np.random.choice(self.PARAM_SPACE['actor_lr'])
            config['critic_lr'] = np.random.choice(self.PARAM_SPACE['critic_lr'])
            config['clip_epsilon'] = np.random.choice(self.PARAM_SPACE['clip_epsilon'])
            config['entropy_coef'] = np.random.choice(self.PARAM_SPACE['entropy_coef'])
            config['gae_lambda'] = np.random.choice(self.PARAM_SPACE['gae_lambda'])
            config['num_epochs'] = np.random.choice(self.PARAM_SPACE['num_epochs'])
            config['batch_size'] = np.random.choice(self.PARAM_SPACE['batch_size'])
            
            result = self.train_single_config(config, i)
            self.results.append(result)
        
        self._save_results()
        self._analyze_results()
    
    def _convert_to_serializable(self, obj):
        """递归转换numpy类型为Python类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int_)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_results(self):
        """保存搜索结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, f'param_search_{timestamp}.json')
        
        # 转换numpy类型为Python类型以便JSON序列化
        serializable_results = []
        for r in self.results:
            sr = self._convert_to_serializable(r)
            serializable_results.append(sr)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n结果已保存: {filename}")
    
    def _analyze_results(self):
        """分析搜索结果并输出最佳配置"""
        print(f"\n{'='*60}")
        print("参数搜索分析结果")
        print(f"{'='*60}")
        
        # 按最终满意度排序
        sorted_results = sorted(self.results, 
                               key=lambda x: x['final_sat'], 
                               reverse=True)
        
        print("\nTop 5 最佳配置 (按满意度):")
        for i, r in enumerate(sorted_results[:5]):
            print(f"\n  Rank {i+1}:")
            print(f"    Config ID: {r['config_id']}")
            print(f"    Final Satisfaction: {r['final_sat']:.4f}")
            print(f"    Final Reward: {r['final_reward']:.2f}")
            print(f"    Reward Variance: {r['reward_variance']:.4f}")
            print(f"    Convergence Speed: {r['convergence_speed']} episodes")
            print(f"    Key Params: lr={r['config'].get('actor_lr')}, "
                  f"clip={r['config'].get('clip_epsilon')}, "
                  f"ent={r['config'].get('entropy_coef')}")
        
        # 按收敛稳定性排序（低方差）
        stable_results = sorted(self.results,
                               key=lambda x: x['reward_variance'])
        
        print("\nTop 5 最稳定配置 (按奖励方差):")
        for i, r in enumerate(stable_results[:5]):
            print(f"\n  Rank {i+1}:")
            print(f"    Config ID: {r['config_id']}")
            print(f"    Reward Variance: {r['reward_variance']:.4f}")
            print(f"    Final Satisfaction: {r['final_sat']:.4f}")
        
        # 参数敏感性分析
        self._parameter_sensitivity_analysis()
    
    def _parameter_sensitivity_analysis(self):
        """参数敏感性分析"""
        print(f"\n{'='*60}")
        print("参数敏感性分析")
        print(f"{'='*60}")
        
        # 分析每个参数对满意度的影响
        for param in ['actor_lr', 'critic_lr', 'clip_epsilon', 'entropy_coef', 'gae_lambda']:
            param_values = {}
            for r in self.results:
                val = r['config'].get(param)
                if val not in param_values:
                    param_values[val] = []
                param_values[val].append(r['final_sat'])
            
            print(f"\n  {param}:")
            for val, sats in sorted(param_values.items()):
                mean_sat = np.mean(sats)
                std_sat = np.std(sats)
                print(f"    {val}: sat={mean_sat:.4f}±{std_sat:.4f} (n={len(sats)})")


def run_parameter_search():
    """运行参数搜索实验"""
    # 基础配置
    base_config = {
        'obs_dim': 20,
        'state_dim': 30,
        'action_dim': 4,  # 3 BS + stay
        'hidden_dim': 64,
        'critic_hidden_dim': 128,
        'use_biz_heads': True,
        'use_attention_critic': True,
        'train_sample_agents': 50,
        'attention_sample_agents': 50,
        'use_early_stopping': False,  # 禁用早停以获得完整曲线
    }
    
    # 创建参数搜索器
    searcher = MAPPOParameterSearch(
        base_config=base_config,
        num_uav=32,  # 使用小规模加速测试
        num_bs=3,
        train_episodes=100,  # 每配置100轮
        eval_episodes=3
    )
    
    # 运行网格搜索
    print("开始网格搜索...")
    searcher.grid_search(max_combinations=16)
    
    # 运行随机搜索
    print("\n开始随机搜索...")
    searcher.random_search(num_samples=10)


if __name__ == '__main__':
    run_parameter_search()
