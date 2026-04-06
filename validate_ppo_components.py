# -*- coding: utf-8 -*-
"""
MAPPO PPO Core Components Validation Script
Validates: Policy Network, Value Network, GAE, Clipped Surrogate
"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent


class PPOComponentValidator:
    """PPO Core Component Validator"""

    def __init__(self):
        self.results = {}
        self.env = None
        self.agent = None

    def setup(self):
        """Initialize environment and agent"""
        print("=" * 80)
        print("MAPPO PPO Core Components Validation")
        print("=" * 80)

        set_global_seed(GLOBAL_SEED)
        self.env = QMixHandoverEnv(
            num_bs=4, num_uav=10,
            max_steps=100, seed=GLOBAL_SEED,
            bs_capacity_range=(50, 100),
        )

        self.agent = MAPPOAgent(
            num_agents=self.env.num_agents,
            obs_dim=self.env.obs_dim,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim=64,
            critic_hidden_dim=128,
            use_hierarchical=True,
        )

        self.biz_types = {i: self.env.env.uavs[i].true_business_type.value for i in range(10)}

        print(f"\n[Setup] Environment: {self.env.num_agents} agents, {self.env.num_bs} BS")
        print(f"[Setup] Obs dim: {self.env.obs_dim}, State dim: {self.env.state_dim}")
        print(f"[Setup] Action dim: {self.env.action_dim}")

    def test_1_policy_network(self):
        """Test 1: Validate policy network output distribution"""
        print("\n" + "-" * 80)
        print("TEST 1: Policy Network Output Distribution Validation")
        print("-" * 80)

        action_counts = np.zeros(self.env.action_dim)
        log_probs_list = []

        for trial in range(200):
            obs_dict, global_state = self.env.reset()
            self.agent.reset_hidden()

            actions, log_probs, values, _ = self.agent.select_actions(
                obs_dict, global_state, self.biz_types, training=True
            )

            for uid, a in actions.items():
                action_counts[a] += 1
            log_probs_list.extend(log_probs.values())

        total = action_counts.sum()
        action_dist = action_counts / total
        log_probs_arr = np.array(log_probs_list)

        print(f"\n  Action Distribution (over {total:.0f} samples):")
        action_names = ['stay', 'best_sinr', 'best_capacity', 'sinr_capacity',
                       'predictive', 'business_specific']
        for i in range(len(action_names)):
            pct = action_dist[i] * 100
            bar_len = int(pct / 2)
            bar = '#' * bar_len
            print(f"    {action_names[i]:15s}: {pct:5.1f}% {bar}")

        print(f"\n  Log Probability Statistics:")
        print(f"    Mean:   {log_probs_arr.mean():.4f}")
        print(f"    Std:    {log_probs_arr.std():.4f}")
        print(f"    Min:    {log_probs_arr.min():.4f}")
        print(f"    Max:    {log_probs_arr.max():.4f}")

        checks = {
            'Exploration adequacy': action_dist[0] < 0.8,
            'Action diversity': (action_dist > 0).sum() >= 3,
            'Log prob reasonable': log_probs_arr.mean() < -0.5,
            'Entropy adequate': log_probs_arr.std() > 0.3,
        }

        all_pass = True
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[OK]" if passed else "[FAIL]"
            if not passed:
                all_pass = False
            print(f"    {symbol} {check_name}: {status}")

        self.results['test_1'] = {
            'passed': all_pass,
            'action_distribution': action_dist.tolist(),
            'log_prob_stats': {
                'mean': float(log_probs_arr.mean()),
                'std': float(log_probs_arr.std())
            }
        }

        return all_pass

    def test_2_value_network(self):
        """Test 2: Validate value network accuracy"""
        print("\n" + "-" * 80)
        print("TEST 2: Value Network Accuracy Validation")
        print("-" * 80)

        actual_returns = []
        predicted_values = []

        obs_dict, global_state = self.env.reset()
        self.agent.reset_hidden()

        episode_rewards = []
        for step in range(50):
            actions, log_probs, values, pre_hidden = self.agent.select_actions(
                obs_dict, global_state, self.biz_types, training=True
            )
            next_obs, next_state, rewards, team_reward, done, info = self.env.step(actions)

            predicted_values.append(np.mean(list(values.values())))
            episode_rewards.append(team_reward)

            self.agent.insert_experience(
                step, obs_dict, global_state, actions,
                rewards, team_reward, done, log_probs, values,
                self.biz_types, pre_hidden
            )

            obs_dict = next_obs
            global_state = next_state

        gamma = self.agent.gamma
        for t in range(len(episode_rewards)):
            discounted_return = sum(
                episode_rewards[t + k] * (gamma ** k)
                for k in range(len(episode_rewards) - t)
            )
            actual_returns.append(discounted_return)

        actual_returns = np.array(actual_returns)
        predicted_values = np.array(predicted_values)

        mae = np.mean(np.abs(predicted_values - actual_returns))
        rmse = np.sqrt(np.mean((predicted_values - actual_returns) ** 2))
        correlation = np.corrcoef(predicted_values, actual_returns)[0, 1]

        print(f"\n  Value Prediction vs Actual Returns:")
        print(f"    MAE (Mean Absolute Error):      {mae:.4f}")
        print(f"    RMSE (Root Mean Square Error):  {rmse:.4f}")
        print(f"    Correlation Coefficient:         {correlation:.4f}")

        checks = {
            'Value prediction finite': np.isfinite(predicted_values).all(),
            'Correlation positive': correlation > 0,
            'No extreme predictions': np.abs(predicted_values).max() < 1000,
        }

        all_pass = True
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[OK]" if passed else "[FAIL]"
            if not passed:
                all_pass = False
            print(f"    {symbol} {check_name}: {status}")

        self.results['test_2'] = {
            'passed': all_pass,
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
        }

        return all_pass

    def test_3_gae_computation(self):
        """Test 3: Validate GAE computation"""
        print("\n" + "-" * 80)
        print("TEST 3: GAE (Generalized Advantage Estimation) Computation")
        print("-" * 80)

        obs_dict, global_state = self.env.reset()
        self.agent.reset_hidden()

        for step in range(50):
            actions, log_probs, values, pre_hidden = self.agent.select_actions(
                obs_dict, global_state, self.biz_types, training=True
            )
            next_obs, next_state, rewards, team_reward, done, info = self.env.step(actions)

            self.agent.insert_experience(
                step, obs_dict, global_state, actions,
                rewards, team_reward, done, log_probs, values,
                self.biz_types, pre_hidden
            )

            obs_dict = next_obs
            global_state = next_state

        next_values = np.zeros(self.agent.num_agents, dtype=np.float32)
        advantages, returns = self.agent.buffer.compute_gae(next_values)

        print(f"\n  GAE Computation Results:")
        print(f"    Advantages shape:     {advantages.shape}")
        print(f"    Advantages mean:     {advantages.mean():.6f}")
        print(f"    Advantages std:      {advantages.std():.6f}")
        print(f"    Returns mean:        {returns.mean():.6f}")

        checks = {
            'Advantages computed': advantages.shape == (50, 10),
            'Advantages finite': np.isfinite(advantages).all(),
            'Returns computed': returns.shape == (50, 10),
            'Non-zero advantages': np.abs(advantages).sum() > 1e-8,
        }

        all_pass = True
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[OK]" if passed else "[FAIL]"
            if not passed:
                all_pass = False
            print(f"    {symbol} {check_name}: {status}")

        stored_values = self.agent.buffer.values[:self.agent.buffer.ptr].cpu().numpy()
        expected_returns = advantages + stored_values
        diff = np.abs(expected_returns - returns).max()

        print(f"\n  Consistency Check:")
        print(f"    |returns - (advantages + values)|_max: {diff:.8f}")
        consistency_pass = diff < 1e-5
        symbol = "[OK]" if consistency_pass else "[FAIL]"
        print(f"    {symbol} Returns = Advantages + Values: {'PASS' if consistency_pass else 'FAIL'}")

        if not consistency_pass:
            all_pass = False

        self.results['test_3'] = {
            'passed': all_pass,
            'consistency_check': consistency_pass,
        }

        return all_pass

    def test_4_clipped_surrogate(self):
        """Test 4: Validate clipped surrogate objective"""
        print("\n" + "-" * 80)
        print("TEST 4: Clipped Surrogate Objective Function")
        print("-" * 80)

        obs_dict, global_state = self.env.reset()
        self.agent.reset_hidden()

        for step in range(50):
            actions, log_probs, values, pre_hidden = self.agent.select_actions(
                obs_dict, global_state, self.biz_types, training=True
            )
            next_obs, next_state, rewards, team_reward, done, info = self.env.step(actions)

            self.agent.insert_experience(
                step, obs_dict, global_state, actions,
                rewards, team_reward, done, log_probs, values,
                self.biz_types, pre_hidden
            )

            obs_dict = next_obs
            global_state = next_state

        actor_params_before = [p.data.clone().cpu().numpy()
                               for p in self.agent.actor.parameters()]

        train_stats = self.agent.train()

        actor_params_after = [p.data.clone().cpu().numpy()
                              for p in self.agent.actor.parameters()]

        param_changes = []
        for before, after in zip(actor_params_before, actor_params_after):
            change = np.linalg.norm(after - before)
            param_changes.append(change)

        total_param_change = sum(param_changes)

        print(f"\n  Training Statistics:")
        print(f"    Actor Loss:       {train_stats.get('actor_loss', 0):.6f}")
        print(f"    Critic Loss:      {train_stats.get('critic_loss', 0):.6f}")
        print(f"    Entropy:          {train_stats.get('entropy', 0):.6f}")
        print(f"    Num Updates:      {train_stats.get('num_updates', 0)}")
        print(f"    Approx KL:        {train_stats.get('approx_kl', 0):.6f}")
        print(f"    Ratio Mean:       {train_stats.get('ratio_mean', 0):.6f}")

        print(f"\n  Parameter Update Analysis:")
        print(f"    Total parameter change: {total_param_change:.6f}")

        checks = {
            'Actor loss non-zero': abs(train_stats.get('actor_loss', 0)) > 1e-8,
            'Critic loss non-zero': abs(train_stats.get('critic_loss', 0)) > 1e-8,
            'Entropy positive': train_stats.get('entropy', 0) > 0,
            'Updates executed': train_stats.get('num_updates', 0) > 0,
            'Parameters updated': total_param_change > 1e-8,
        }

        all_pass = True
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[OK]" if passed else "[FAIL]"
            if not passed:
                all_pass = False
            print(f"    {symbol} {check_name}: {status}")

        self.results['test_4'] = {
            'passed': all_pass,
            'param_change': float(total_param_change),
        }

        return all_pass

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY REPORT")
        print("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['passed'])

        print(f"\n  Total Tests: {total_tests}")
        print(f"  Passed:      {passed_tests}/{total_tests}")
        print(f"  Pass Rate:   {passed_tests/total_tests*100:.1f}%")

        print(f"\n  Detailed Results:")
        test_names = [
            'Test 1: Policy Network Output Distribution',
            'Test 2: Value Network Accuracy',
            'Test 3: GAE Computation',
            'Test 4: Clipped Surrogate Objective'
        ]

        for test_name, result_key in zip(test_names, self.results.keys()):
            result = self.results[result_key]
            status = "[PASS]" if result['passed'] else "[FAIL]"
            print(f"    {status}  {test_name}")

        overall_status = "ALL TESTS PASSED" if passed_tests == total_tests else \
                         f"SOME TESTS FAILED ({total_tests - passed_tests} failures)"
        print(f"\n  Overall Status: {overall_status}")

        return passed_tests == total_tests


def main():
    validator = PPOComponentValidator()
    validator.setup()

    validator.test_1_policy_network()
    validator.test_2_value_network()
    validator.test_3_gae_computation()
    validator.test_4_clipped_surrogate()

    success = validator.generate_report()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
