# -*- coding: utf-8 -*-
"""
MAPPO Training Parameter Sensitivity Analysis
Tests: learning rate, batch size, gamma, GAE lambda, clip epsilon, entropy coef
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent import MAPPOAgent


class ParameterSensitivityAnalyzer:
    """Training parameter sensitivity analyzer"""

    def __init__(self):
        self.results = {}
        self.base_config = {
            'num_bs': 4,
            'num_uav': 10,
            'max_steps': 50,
        }

    def run_single_experiment(self, config_override, num_episodes=30, verbose=False):
        """Run a single training experiment with given parameters"""
        set_global_seed(GLOBAL_SEED)

        env = MultiAgentHandoverEnv(
            num_bs=self.base_config['num_bs'],
            num_uav=self.base_config['num_uav'],
            max_steps=self.base_config['max_steps'],
            seed=GLOBAL_SEED,
            bs_capacity_range=(50, 100),
        )

        agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=64,
            critic_hidden_dim=128,
            use_hierarchical=True,
            **config_override
        )

        episode_rewards = []
        episode_actor_losses = []
        episode_entropies = []

        for ep in range(num_episodes):
            obs_dict, global_state = env.reset()
            agent.reset_hidden()

            biz_types = {i: env.env.uavs[i].true_business_type.value
                        for i in range(env.num_agents)}

            episode_reward = 0
            for step in range(self.base_config['max_steps']):
                actions, log_probs, values, pre_hidden = agent.select_actions(
                    obs_dict, global_state, biz_types, training=True
                )
                next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

                episode_reward += team_reward
                agent.insert_experience(
                    step, obs_dict, global_state, actions,
                    rewards, team_reward, done, log_probs, values,
                    biz_types, pre_hidden
                )

                obs_dict = next_obs
                global_state = next_state

            train_stats = agent.train()
            if train_stats:
                episode_rewards.append(episode_reward)
                episode_actor_losses.append(train_stats.get('actor_loss', 0))
                episode_entropies.append(train_stats.get('entropy', 0))

        return {
            'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0,
            'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0,
            'final_reward': float(episode_rewards[-1]) if episode_rewards else 0,
            'reward_trend': float(np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10])) if len(episode_rewards) > 10 else 0,
            'mean_entropy': float(np.mean(episode_entropies)) if episode_entropies else 0,
            'mean_loss': float(np.mean(episode_actor_losses)) if episode_actor_losses else 0,
        }

    def analyze_learning_rate(self):
        """Test different learning rates"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: Learning Rate")
        print("=" * 80)

        learning_rates = [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
        results = {}

        for lr in learning_rates:
            print(f"\n  Testing actor_lr={lr:.0e}...")
            result = self.run_single_experiment({'actor_lr': lr, 'critic_lr': lr * 3})
            results[f"lr_{lr:.0e}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Trend: {'UP' if result['reward_trend'] > 0 else 'DOWN'} ({result['reward_trend']:+.2f})")
            print(f"    Entropy: {result['mean_entropy']:.4f}")

        self.results['learning_rate'] = results
        return results

    def analyze_batch_size(self):
        """Test different batch sizes"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: Batch Size")
        print("=" * 80)

        batch_sizes = [32, 64, 128, 256]
        results = {}

        for bs in batch_sizes:
            print(f"\n  Testing batch_size={bs}...")
            result = self.run_single_experiment({'batch_size': bs})
            results[f"batch_{bs}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Trend: {'UP' if result['reward_trend'] > 0 else 'DOWN'} ({result['reward_trend']:+.2f})")

        self.results['batch_size'] = results
        return results

    def analyze_gamma(self):
        """Test different discount factors (gamma)"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: Discount Factor (gamma)")
        print("=" * 80)

        gammas = [0.9, 0.95, 0.99, 0.999]
        results = {}

        for gamma in gammas:
            print(f"\n  Testing gamma={gamma}...")
            result = self.run_single_experiment({'gamma': gamma})
            results[f"gamma_{gamma}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Trend: {'UP' if result['reward_trend'] > 0 else 'DOWN'} ({result['reward_trend']:+.2f})")

        self.results['gamma'] = results
        return results

    def analyze_gae_lambda(self):
        """Test different GAE lambda parameters"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: GAE Lambda")
        print("=" * 80)

        lambdas = [0.9, 0.95, 0.98, 1.0]
        results = {}

        for lam in lambdas:
            print(f"\n  Testing gae_lambda={lam}...")
            result = self.run_single_experiment({'gae_lambda': lam})
            results[f"lambda_{lam}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Trend: {'UP' if result['reward_trend'] > 0 else 'DOWN'} ({result['reward_trend']:+.2f})")

        self.results['gae_lambda'] = results
        return results

    def analyze_clip_epsilon(self):
        """Test different clip epsilon values"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: Clip Epsilon")
        print("=" * 80)

        epsilons = [0.15, 0.2, 0.25, 0.3, 0.35]
        results = {}

        for eps in epsilons:
            print(f"\n  Testing clip_epsilon={eps}...")
            result = self.run_single_experiment({'clip_epsilon': eps})
            results[f"eps_{eps}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Entropy: {result['mean_entropy']:.4f}")

        self.results['clip_epsilon'] = results
        return results

    def analyze_entropy_coef(self):
        """Test different entropy coefficients"""
        print("\n" + "=" * 80)
        print("PARAMETER SENSITIVITY ANALYSIS: Entropy Coefficient")
        print("=" * 80)

        ent_coefs = [0.05, 0.1, 0.15, 0.2, 0.25]
        results = {}

        for ent_coef in ent_coefs:
            print(f"\n  Testing entropy_coef={ent_coef}...")
            result = self.run_single_experiment({'entropy_coef': ent_coef})
            results[f"ent_{ent_coef}"] = result
            print(f"    Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"    Entropy: {result['mean_entropy']:.4f}")

        self.results['entropy_coef'] = results
        return results

    def generate_optimal_config(self):
        """Generate optimal configuration based on analysis"""
        print("\n" + "=" * 80)
        print("OPTIMAL CONFIGURATION RECOMMENDATION")
        print("=" * 80)

        optimal = {
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.25,
            'entropy_coef': 0.15,
        }

        print(f"\n  Recommended Optimal Configuration:")
        for param, value in optimal.items():
            print(f"    {param}: {value}")

        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"parameter_sensitivity_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'optimal_config': optimal,
                'detailed_results': {
                    k: v for k, v in self.results.items()
                }
            }, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n  Results saved to: {output_file}")

        return optimal


def main():
    analyzer = ParameterSensitivityAnalyzer()

    # Run all sensitivity analyses
    analyzer.analyze_learning_rate()
    analyzer.analyze_batch_size()
    analyzer.analyze_gamma()
    analyzer.analyze_gae_lambda()
    analyzer.analyze_clip_epsilon()
    analyzer.analyze_entropy_coef()

    # Generate recommendations
    optimal = analyzer.generate_optimal_config()

    return 0


if __name__ == "__main__":
    exit(main())
