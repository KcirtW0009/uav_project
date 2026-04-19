# -*- coding: utf-8 -*-
"""
MAPPO Training Monitor and Visualization System
Comprehensive monitoring of rewards, losses, entropy, gradients, and convergence
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent import MAPPOAgent


class TrainingMonitor:
    """Comprehensive training process monitor"""

    def __init__(self, save_dir='training_logs'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'total_losses': [],
            'actor_grad_norms': [],
            'critic_grad_norms': [],
            'value_mses': [],
            'approx_kls': [],
            'ratio_means': [],
            'advantage_means': [],
            'return_means': [],
            'action_distributions': [],
            'step_rewards': [],
        }

        self.episode_data = []
        self.start_time = None

    def log_episode(self, episode_num, reward, train_stats=None, action_counts=None):
        """Log metrics for a single episode"""
        self.metrics['episode_rewards'].append(reward)

        if train_stats:
            self.metrics['actor_losses'].append(train_stats.get('actor_loss', 0))
            self.metrics['critic_losses'].append(train_stats.get('critic_loss', 0))
            self.metrics['entropies'].append(train_stats.get('entropy', 0))
            self.metrics['total_losses'].append(train_stats.get('total_loss', 0))
            self.metrics['actor_grad_norms'].append(train_stats.get('actor_grad_norm', 0))
            self.metrics['critic_grad_norms'].append(train_stats.get('critic_grad_norm', 0))
            self.metrics['value_mses'].append(train_stats.get('value_mse', 0))
            self.metrics['approx_kls'].append(train_stats.get('approx_kl', 0))
            self.metrics['ratio_means'].append(train_stats.get('ratio_mean', 0))
            self.metrics['advantage_means'].append(train_stats.get('advantage_mean', 0))
            self.metrics['return_means'].append(train_stats.get('return_mean', 0))

        if action_counts:
            total = sum(action_counts.values())
            dist = {k: v / total for k, v in action_counts.items()}
            self.metrics['action_distributions'].append(dist)

    def compute_statistics(self):
        """Compute comprehensive statistics"""
        stats = {}

        if len(self.metrics['episode_rewards']) > 0:
            rewards = np.array(self.metrics['episode_rewards'])
            stats['reward'] = {
                'mean': float(rewards.mean()),
                'std': float(rewards.std()),
                'min': float(rewards.min()),
                'max': float(rewards.max()),
                'median': float(np.median(rewards)),
                'final_10_mean': float(rewards[-10:].mean()) if len(rewards) >= 10 else float(rewards.mean()),
                'trend': float(np.mean(rewards[-10:]) - np.mean(rewards[:10])) if len(rewards) > 10 else 0,
                'cv': float(rewards.std() / max(abs(rewards.mean()), 1e-8)),  # Coefficient of variation
            }

        if len(self.metrics['actor_losses']) > 0:
            actor_losses = np.array(self.metrics['actor_losses'])
            stats['actor_loss'] = {
                'mean': float(actor_losses.mean()),
                'std': float(actor_losses.std()),
                'is_nonzero': bool(np.abs(actor_losses).mean() > 1e-6),
            }

        if len(self.metrics['entropies']) > 0:
            entropies = np.array(self.metrics['entropies'])
            stats['entropy'] = {
                'mean': float(entropies.mean()),
                'std': float(entropies.std()),
                'final': float(entropies[-1]),
                'trend': float(np.mean(entropies[-10:]) - np.mean(entropies[:10])) if len(entropies) > 10 else 0,
            }

        if len(self.metrics['actor_grad_norms']) > 0:
            grad_norms = np.array(self.metrics['actor_grad_norms'])
            stats['gradient'] = {
                'mean': float(grad_norms.mean()),
                'max': float(grad_norms.max()),
                'is_stable': bool(grad_norms.max() < 100),  # No explosion
            }

        return stats

    def generate_visualization_report(self):
        """Generate comprehensive visualization report"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'MAPPO Training Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                     fontsize=16, fontweight='bold')

        episodes = range(1, len(self.metrics['episode_rewards']) + 1)

        # Plot 1: Episode Rewards with moving average
        ax1 = axes[0, 0]
        if len(self.metrics['episode_rewards']) > 0:
            rewards = np.array(self.metrics['episode_rewards'])
            ax1.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5, label='Episode Reward')
            # Moving average (window=10)
            window = min(10, len(rewards) // 2) if len(rewards) > 2 else 1
            if window > 1:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(rewards)+1), ma, 'r-', linewidth=2,
                        label=f'MA-{window}')
            ax1.axhline(y=np.mean(rewards), color='g', linestyle='--',
                       label=f'Overall Mean={np.mean(rewards):.1f}')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Reward Curve')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

        # Plot 2: Actor & Critic Losses
        ax2 = axes[0, 1]
        if len(self.metrics['actor_losses']) > 0:
            ax2.plot(range(1, len(self.metrics['actor_losses'])+1),
                    self.metrics['actor_losses'], 'b-', linewidth=1, label='Actor Loss')
            ax2.plot(range(1, len(self.metrics['critic_losses'])+1),
                    self.metrics['critic_losses'], 'r-', linewidth=1, label='Critic Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss Value')
            ax2.set_title('Policy & Value Loss Evolution')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')  # Log scale for better visualization

        # Plot 3: Policy Entropy
        ax3 = axes[0, 2]
        if len(self.metrics['entropies']) > 0:
            entropies = self.metrics['entropies']
            ax3.plot(range(1, len(entropies)+1), entropies, 'g-', linewidth=1.5)
            ax3.axhline(y=np.mean(entropies), color='r', linestyle='--',
                       label=f'Mean Entropy={np.mean(entropies):.3f}')
            ax3.fill_between(range(1, len(entropies)+1),
                            [e - 0.1 for e in entropies],
                            [e + 0.1 for e in entropies],
                            alpha=0.2, color='green')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Entropy')
            ax3.set_title('Policy Entropy (Exploration)')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(bottom=0)

        # Plot 4: Gradient Norms
        ax4 = axes[1, 0]
        if len(self.metrics['actor_grad_norms']) > 0:
            ax4.plot(range(1, len(self.metrics['actor_grad_norms'])+1),
                    self.metrics['actor_grad_norms'], 'b-', linewidth=1,
                    label='Actor Grad Norm')
            ax4.plot(range(1, len(self.metrics['critic_grad_norms'])+1),
                    self.metrics['critic_grad_norms'], 'r-', linewidth=1,
                    label='Critic Grad Norm')
            ax4.axhline(y=10, color='orange', linestyle=':', label='Warning Threshold (10)')
            ax4.axhline(y=50, color='red', linestyle=':', label='Explosion Threshold (50)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Gradient Norm')
            ax4.set_title('Gradient Stability Monitoring')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')

        # Plot 5: Value Function MSE
        ax5 = axes[1, 1]
        if len(self.metrics['value_mses']) > 0:
            vmses = self.metrics['value_mses']
            ax5.plot(range(1, len(vmses)+1), vmses, 'm-', linewidth=1.5)
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Value MSE')
            ax5.set_title('Value Function Estimation Error')
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')

        # Plot 6: Approximate KL Divergence
        ax6 = axes[1, 2]
        if len(self.metrics['approx_kls']) > 0:
            kls = self.metrics['approx_kls']
            ax6.plot(range(1, len(kls)+1), kls, 'c-', linewidth=1.5)
            ax6.axhline(y=0.02, color='orange', linestyle='--', label='Target KL (0.02)')
            ax6.axhline(y=0.05, color='red', linestyle='--', label='Early Stop KL (0.05)')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('KL Divergence')
            ax6.set_title('Policy Update Magnitude (KL)')
            ax6.legend(loc='best')
            ax6.grid(True, alpha=0.3)

        # Plot 7: Action Distribution Over Time (Stacked Area)
        ax7 = axes[2, 0]
        if len(self.metrics['action_distributions']) > 0:
            action_names = ['stay', 'best_sinr', 'best_capacity', 'sinr_capacity',
                           'predictive', 'biz_specific']
            distributions = []
            for dist in self.metrics['action_distributions']:
                row = [dist.get(i, 0) for i in range(len(action_names))]
                distributions.append(row)

            distributions = np.array(distributions).T
            x_axis = list(range(1, len(distributions[0]) + 1))

            ax7.stackplot(x_axis, distributions,
                         labels=action_names,
                         colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                '#9467bd', '#8c564b'],
                         alpha=0.8)
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Action Probability')
            ax7.set_title('Action Distribution Evolution')
            ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax7.set_ylim(0, 1)

        # Plot 8: Advantage & Return Statistics
        ax8 = axes[2, 1]
        if len(self.metrics['advantage_means']) > 0:
            ax8.plot(range(1, len(self.metrics['advantage_means'])+1),
                    self.metrics['advantage_means'], 'b-', linewidth=1,
                    label='Advantage Mean')
            ax8.plot(range(1, len(self.metrics['return_means'])+1),
                    self.metrics['return_means'], 'r-', linewidth=1,
                    label='Return Mean')
            ax8.set_xlabel('Episode')
            ax8.set_ylabel('Value')
            ax8.set_title('Advantage & Return Statistics')
            ax8.legend(loc='best')
            ax8.grid(True, alpha=0.3)

        # Plot 9: Summary Statistics Bar Chart
        ax9 = axes[2, 2]
        stats = self.compute_statistics()
        summary_metrics = ['Mean Reward', 'Final MA Reward', 'Mean Entropy',
                          'Mean Loss', 'Grad Stability']
        values = [
            stats.get('reward', {}).get('mean', 0),
            stats.get('reward', {}).get('final_10_mean', 0),
            stats.get('entropy', {}).get('mean', 0),
            stats.get('actor_loss', {}).get('mean', 0),
            1.0 if stats.get('gradient', {}).get('is_stable', False) else 0.0,
        ]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values[:-1]] + \
                 ['#2ecc71' if values[-1] == 1.0 else '#e74c3c']

        bars = ax9.bar(summary_metrics, values, color=colors, edgecolor='black')
        ax9.set_ylabel('Value')
        ax9.set_title('Training Health Summary')
        ax9.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax9.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.save_dir, f'training_report_{timestamp}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n[VISUALIZATION] Training report saved to: {output_path}")
        return output_path


def run_comprehensive_training_monitoring():
    """Run training with full monitoring"""
    print("=" * 80)
    print("MAPPO Comprehensive Training Monitoring System")
    print("=" * 80)

    set_global_seed(GLOBAL_SEED)

    env = MultiAgentHandoverEnv(
        num_bs=4, num_uav=10,
        max_steps=50, seed=GLOBAL_SEED,
        bs_capacity_range=(50, 100),
    )

    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.20,
        entropy_coef=0.10,
        use_hierarchical=True,
    )

    monitor = TrainingMonitor(save_dir='training_logs')

    num_episodes = 100
    print(f"\n[TRAINING] Starting {num_episodes} episodes with full monitoring...")

    for ep in range(num_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()
        biz_types = {i: env.env.uavs[i].true_business_type.value
                    for i in range(env.num_agents)}

        episode_reward = 0
        action_counts = defaultdict(int)

        for step in range(env.max_steps):
            actions, log_probs, values, pre_hidden = agent.select_actions(
                obs_dict, global_state, biz_types, training=True
            )
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

            for uid, a in actions.items():
                action_counts[a] += 1

            episode_reward += team_reward
            agent.insert_experience(
                step, obs_dict, global_state, actions,
                rewards, team_reward, done, log_probs, values,
                biz_types, pre_hidden
            )

            obs_dict = next_obs
            global_state = next_state

        train_stats = agent.train()
        monitor.log_episode(
            ep + 1, episode_reward, train_stats,
            dict(action_counts)
        )

        if (ep + 1) % 25 == 0:
            recent_rewards = monitor.metrics['episode_rewards'][-25:]
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Reward={episode_reward:.1f} "
                  f"(MA25={np.mean(recent_rewards):.1f}), "
                  f"Entropy={train_stats.get('entropy', 0):.3f}, "
                  f"Loss={train_stats.get('total_loss', 0):.4f}")

    # Generate report
    print("\n" + "-" * 80)
    print("[ANALYSIS] Computing final statistics...")
    stats = monitor.compute_statistics()

    print(f"\n  Final Training Statistics:")
    print(f"    Reward:")
    print(f"      Mean:   {stats['reward']['mean']:.2f}")
    print(f"      Std:    {stats['reward']['std']:.2f}")
    print(f"      Trend:  {'IMPROVING' if stats['reward']['trend'] > 0 else 'DECLINING'} ({stats['reward']['trend']:+.2f})")
    print(f"      CV:     {stats['reward']['cv']:.2%} (stability)")

    print(f"\n    Loss:")
    print(f"      Actor:  {stats['actor_loss']['mean']:.6f} (non-zero: {stats['actor_loss']['is_nonzero']})")

    print(f"\n    Entropy:")
    print(f"      Mean:   {stats['entropy']['mean']:.4f}")
    print(f"      Trend:  {'EXPLORING' if stats['entropy']['trend'] > 0 else 'EXPLOITING'} ({stats['entropy']['trend']:+.4f})")

    print(f"\n    Gradient:")
    print(f"      Stable: {'YES' if stats['gradient']['is_stable'] else 'NO - EXPLOSION RISK!'}")
    print(f"      Max:    {stats['gradient']['max']:.2f}")

    # Generate visualization
    print("\n[VISUALIZATION] Generating training report...")
    viz_path = monitor.generate_visualization_report()

    print("\n" + "=" * 80)
    print("TRAINING MONITORING COMPLETE")
    print("=" * 80)

    return monitor, stats


if __name__ == "__main__":
    run_comprehensive_training_monitoring()
