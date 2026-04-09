# -*- coding: utf-8 -*-
"""
MAPPO 增强数据记录与可视化系统

扩展数据记录范围，包括但不限于：
- 中间训练状态
- 梯度变化
- 策略分布
- 价值函数变化
- 探索率变化
- 业务类型分布
- 切换决策分析
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque


class MAPPOTrainingMonitor:
    """MAPPO训练过程增强监控器"""
    
    def __init__(self, log_dir: str, max_history: int = 1000):
        """
        初始化监控器
        
        Args:
            log_dir: 日志保存目录
            max_history: 最大历史记录长度
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 训练指标历史
        self.history = {
            'episode': [],
            'reward': [],
            'satisfaction': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'value_mse': [],
            'actor_grad_norm': [],
            'critic_grad_norm': [],
            'advantage_mean': [],
            'advantage_std': [],
            'value_mean': [],
            'value_std': [],
            'exploration_rate': [],
            'learning_rate_actor': [],
            'learning_rate_critic': [],
        }
        
        # 策略分析
        self.policy_stats = {
            'action_distribution': defaultdict(lambda: defaultdict(int)),
            'business_type_distribution': defaultdict(lambda: defaultdict(int)),
            'switch_rate_by_biz': defaultdict(list),
            'stay_rate_by_biz': defaultdict(list),
        }
        
        # 网络参数统计
        self.network_stats = {
            'actor_weight_mean': [],
            'actor_weight_std': [],
            'critic_weight_mean': [],
            'critic_weight_std': [],
            'actor_bias_mean': [],
            'critic_bias_mean': [],
        }
        
        # 通信指标
        self.communication_stats = {
            'handover_success_rate': [],
            'handover_latency': [],
            'ping_jitter': [],
            'packet_loss': [],
            'qos_violation': [],
            'migration_success': [],
            'connected_ratio': [],
        }
        
        self.max_history = max_history
        
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """记录一个episode的指标"""
        self.history['episode'].append(episode)
        
        for key in self.history.keys():
            if key != 'episode' and key in metrics:
                self.history[key].append(metrics[key])
        
        # 限制历史长度
        if len(self.history['episode']) > self.max_history:
            for key in self.history:
                self.history[key] = self.history[key][-self.max_history:]
    
    def log_policy_stats(self, episode: int, actions: List[int], 
                         biz_types: List[int], info: Dict):
        """记录策略统计信息"""
        # 动作分布
        for action in actions:
            self.policy_stats['action_distribution'][episode][action] += 1
        
        # 业务类型分布
        for biz_type in biz_types:
            self.policy_stats['business_type_distribution'][episode][biz_type] += 1
        
        # 切换率统计
        if 'switch_stats' in info:
            for biz_type, stats in info['switch_stats'].items():
                switch_rate = stats.get('switch_rate', 0)
                stay_rate = stats.get('stay_rate', 0)
                self.policy_stats['switch_rate_by_biz'][biz_type].append(switch_rate)
                self.policy_stats['stay_rate_by_biz'][biz_type].append(stay_rate)
    
    def log_network_stats(self, agent):
        """记录网络参数统计"""
        # Actor网络参数统计
        actor_weights = []
        actor_biases = []
        for p in agent.actor.parameters():
            if p.dim() > 1:  # 权重
                actor_weights.extend(p.data.cpu().numpy().flatten())
            else:  # 偏置
                actor_biases.extend(p.data.cpu().numpy().flatten())
        
        if actor_weights:
            self.network_stats['actor_weight_mean'].append(np.mean(actor_weights))
            self.network_stats['actor_weight_std'].append(np.std(actor_weights))
        if actor_biases:
            self.network_stats['actor_bias_mean'].append(np.mean(actor_biases))
        
        # Critic网络参数统计
        critic_weights = []
        critic_biases = []
        for p in agent.critic.parameters():
            if p.dim() > 1:
                critic_weights.extend(p.data.cpu().numpy().flatten())
            else:
                critic_biases.extend(p.data.cpu().numpy().flatten())
        
        if critic_weights:
            self.network_stats['critic_weight_mean'].append(np.mean(critic_weights))
            self.network_stats['critic_weight_std'].append(np.std(critic_weights))
        if critic_biases:
            self.network_stats['critic_bias_mean'].append(np.mean(critic_biases))
    
    def log_communication_stats(self, episode: int, stats: Dict[str, float]):
        """记录通信指标"""
        for key in self.communication_stats.keys():
            if key in stats:
                self.communication_stats[key].append(stats[key])
    
    def create_comprehensive_visualization(self, save_name: str = None):
        """创建综合可视化图表"""
        if save_name is None:
            save_name = f'training_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        save_path = os.path.join(self.log_dir, save_name)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        episodes = self.history['episode']
        
        # 1. 奖励曲线 (带平滑)
        ax1 = fig.add_subplot(gs[0, 0])
        if self.history['reward']:
            rewards = np.array(self.history['reward'])
            ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
            # 平滑曲线
            if len(rewards) >= 10:
                smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
                ax1.plot(episodes[9:], smoothed, color='red', linewidth=2, label='Smoothed')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 满意度曲线 (带平滑)
        ax2 = fig.add_subplot(gs[0, 1])
        if self.history['satisfaction']:
            sats = np.array(self.history['satisfaction'])
            ax2.plot(episodes, sats, alpha=0.3, color='green')
            if len(sats) >= 10:
                smoothed = np.convolve(sats, np.ones(10)/10, mode='valid')
                ax2.plot(episodes[9:], smoothed, color='darkgreen', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Satisfaction')
            ax2.set_title('Average Satisfaction')
            ax2.grid(True, alpha=0.3)
            # 设置y轴范围以突出变化
            if len(sats) > 0:
                ymin, ymax = np.min(sats), np.max(sats)
                margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                ax2.set_ylim(ymin - margin, ymax + margin)
        
        # 3. 损失曲线
        ax3 = fig.add_subplot(gs[0, 2])
        if self.history['actor_loss'] and self.history['critic_loss']:
            ax3.plot(episodes, self.history['actor_loss'], label='Actor Loss', alpha=0.7)
            ax3.plot(episodes, self.history['critic_loss'], label='Critic Loss', alpha=0.7)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Losses')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # 4. 熵和KL散度
        ax4 = fig.add_subplot(gs[1, 0])
        if self.history['entropy']:
            ax4.plot(episodes, self.history['entropy'], color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Entropy')
            ax4.set_title('Policy Entropy')
            ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        if self.history['kl_divergence']:
            ax5.plot(episodes, self.history['kl_divergence'], color='orange')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('KL Divergence')
            ax5.set_title('KL Divergence')
            ax5.grid(True, alpha=0.3)
        
        # 5. 梯度范数
        ax6 = fig.add_subplot(gs[1, 2])
        if self.history['actor_grad_norm'] and self.history['critic_grad_norm']:
            ax6.plot(episodes, self.history['actor_grad_norm'], label='Actor', alpha=0.7)
            ax6.plot(episodes, self.history['critic_grad_norm'], label='Critic', alpha=0.7)
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Gradient Norm')
            ax6.set_title('Gradient Norms')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 6. 优势和价值统计
        ax7 = fig.add_subplot(gs[2, 0])
        if self.history['advantage_mean']:
            ax7.plot(episodes, self.history['advantage_mean'], label='Mean')
            if self.history['advantage_std']:
                advantage_std = np.array(self.history['advantage_std'])
                ax7.fill_between(episodes, 
                                np.array(self.history['advantage_mean']) - advantage_std,
                                np.array(self.history['advantage_mean']) + advantage_std,
                                alpha=0.3)
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Advantage')
            ax7.set_title('Advantage Statistics')
            ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 1])
        if self.history['value_mean']:
            ax8.plot(episodes, self.history['value_mean'], label='Mean', color='green')
            if self.history['value_std']:
                value_std = np.array(self.history['value_std'])
                ax8.fill_between(episodes,
                                np.array(self.history['value_mean']) - value_std,
                                np.array(self.history['value_mean']) + value_std,
                                alpha=0.3, color='green')
            ax8.set_xlabel('Episode')
            ax8.set_ylabel('Value')
            ax8.set_title('Value Function Statistics')
            ax8.grid(True, alpha=0.3)
        
        # 7. 学习率变化
        ax9 = fig.add_subplot(gs[2, 2])
        if self.history['learning_rate_actor']:
            ax9.plot(episodes, self.history['learning_rate_actor'], label='Actor LR')
            ax9.plot(episodes, self.history['learning_rate_critic'], label='Critic LR')
            ax9.set_xlabel('Episode')
            ax9.set_ylabel('Learning Rate')
            ax9.set_title('Learning Rate Schedule')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            ax9.set_yscale('log')
        
        # 8. 网络参数统计
        ax10 = fig.add_subplot(gs[3, 0])
        if self.network_stats['actor_weight_mean']:
            ax10.plot(self.network_stats['actor_weight_mean'], label='Actor Weight Mean')
            ax10.plot(self.network_stats['critic_weight_mean'], label='Critic Weight Mean')
            ax10.set_xlabel('Update Step')
            ax10.set_ylabel('Mean Weight')
            ax10.set_title('Network Weight Statistics')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        ax11 = fig.add_subplot(gs[3, 1])
        if self.network_stats['actor_weight_std']:
            ax11.plot(self.network_stats['actor_weight_std'], label='Actor Weight Std')
            ax11.plot(self.network_stats['critic_weight_std'], label='Critic Weight Std')
            ax11.set_xlabel('Update Step')
            ax11.set_ylabel('Weight Std')
            ax11.set_title('Network Weight Std Dev')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        
        # 9. 通信指标
        ax12 = fig.add_subplot(gs[3, 2])
        comm_metrics = ['handover_success_rate', 'connected_ratio']
        for metric in comm_metrics:
            if self.communication_stats[metric]:
                ax12.plot(self.communication_stats[metric], label=metric.replace('_', ' ').title())
        ax12.set_xlabel('Episode')
        ax12.set_ylabel('Rate')
        ax12.set_title('Communication Metrics')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        
        # 10. 动作分布热力图
        ax13 = fig.add_subplot(gs[4, :])
        if self.policy_stats['action_distribution']:
            # 获取最近的动作分布
            recent_episodes = list(self.policy_stats['action_distribution'].keys())[-50:]
            action_dist_matrix = []
            for ep in recent_episodes:
                dist = self.policy_stats['action_distribution'][ep]
                total = sum(dist.values())
                if total > 0:
                    action_dist_matrix.append([dist.get(i, 0) / total for i in range(4)])
            
            if action_dist_matrix:
                im = ax13.imshow(np.array(action_dist_matrix).T, aspect='auto', cmap='YlOrRd')
                ax13.set_xlabel('Episode (last 50)')
                ax13.set_ylabel('Action')
                ax13.set_title('Action Distribution Over Time')
                ax13.set_yticks(range(4))
                ax13.set_yticklabels(['Stay', 'BS0', 'BS1', 'BS2'])
                plt.colorbar(im, ax=ax13)
        
        # 11. 业务类型切换率
        ax14 = fig.add_subplot(gs[5, 0])
        biz_names = {0: 'Control', 1: 'Video', 2: 'Env'}
        for biz_type in [0, 1, 2]:
            if self.policy_stats['switch_rate_by_biz'][biz_type]:
                rates = self.policy_stats['switch_rate_by_biz'][biz_type]
                ax14.plot(rates, label=biz_names.get(biz_type, f'Biz{biz_type}'), alpha=0.7)
        ax14.set_xlabel('Episode')
        ax14.set_ylabel('Switch Rate')
        ax14.set_title('Switch Rate by Business Type')
        ax14.legend()
        ax14.grid(True, alpha=0.3)
        
        # 12. 收敛分析
        ax15 = fig.add_subplot(gs[5, 1:])
        if len(self.history['satisfaction']) >= 20:
            sats = np.array(self.history['satisfaction'])
            window = 20
            rolling_mean = np.convolve(sats, np.ones(window)/window, mode='valid')
            rolling_std = np.array([np.std(sats[i:i+window]) for i in range(len(sats)-window+1)])
            
            ax15.plot(episodes[window-1:], rolling_mean, label='Rolling Mean', color='blue')
            ax15.fill_between(episodes[window-1:], 
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.3, color='blue')
            ax15.set_xlabel('Episode')
            ax15.set_ylabel('Satisfaction')
            ax15.set_title(f'Convergence Analysis (Window={window})')
            ax15.legend()
            ax15.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"综合可视化已保存: {save_path}")
        return save_path
    
    def save_detailed_logs(self, filename: str = None):
        """保存详细日志到JSON"""
        if filename is None:
            filename = f'training_detailed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        save_path = os.path.join(self.log_dir, filename)
        
        # 准备可序列化的数据
        data = {
            'history': self.history,
            'network_stats': self.network_stats,
            'communication_stats': self.communication_stats,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 转换policy_stats
        data['policy_stats'] = {
            'switch_rate_by_biz': {k: v for k, v in self.policy_stats['switch_rate_by_biz'].items()},
            'stay_rate_by_biz': {k: v for k, v in self.policy_stats['stay_rate_by_biz'].items()},
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"详细日志已保存: {save_path}")
        return save_path


class EnhancedLogger:
    """增强版日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_logs = []
        self.step_logs = []
        self.current_episode_log = {}
        
    def start_episode(self, episode: int):
        """开始记录一个episode"""
        self.current_episode_log = {
            'episode': episode,
            'start_time': datetime.now().isoformat(),
            'steps': [],
        }
    
    def log_step(self, step: int, info: Dict[str, Any]):
        """记录一个step的信息"""
        step_info = {'step': step}
        step_info.update(info)
        self.current_episode_log['steps'].append(step_info)
    
    def end_episode(self, summary: Dict[str, Any]):
        """结束记录一个episode"""
        self.current_episode_log['end_time'] = datetime.now().isoformat()
        self.current_episode_log['summary'] = summary
        self.episode_logs.append(self.current_episode_log)
        
        # 保存到文件
        self._save_episode_log(self.current_episode_log)
    
    def _save_episode_log(self, log: Dict):
        """保存单个episode日志"""
        episode = log['episode']
        filename = os.path.join(self.log_dir, f'episode_{episode:05d}.json')
        with open(filename, 'w') as f:
            json.dump(log, f, indent=2)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取汇总统计"""
        if not self.episode_logs:
            return {}
        
        stats = {
            'total_episodes': len(self.episode_logs),
            'avg_reward': np.mean([ep['summary'].get('reward', 0) for ep in self.episode_logs]),
            'avg_satisfaction': np.mean([ep['summary'].get('satisfaction', 0) for ep in self.episode_logs]),
        }
        
        return stats


if __name__ == '__main__':
    # 测试监控器
    monitor = MAPPOTrainingMonitor('./test_logs')
    
    # 模拟一些数据
    for ep in range(100):
        metrics = {
            'reward': np.random.randn() * 10 + 50 + ep * 0.5,
            'satisfaction': 0.8 + np.random.randn() * 0.05 + ep * 0.001,
            'actor_loss': 0.1 * np.exp(-ep/50) + np.random.randn() * 0.01,
            'critic_loss': 0.5 * np.exp(-ep/50) + np.random.randn() * 0.05,
            'entropy': 1.0 - ep * 0.005 + np.random.randn() * 0.1,
            'kl_divergence': 0.01 + np.random.randn() * 0.005,
        }
        monitor.log_episode(ep, metrics)
    
    # 创建可视化
    monitor.create_comprehensive_visualization('test_visualization.png')
    print("测试完成!")
