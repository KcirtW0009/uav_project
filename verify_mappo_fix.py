"""
MAPPO 修复效果快速验证脚本
验证：网络初始化优化后，训练指标是否正常（不再全零）
"""

import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent

def test_initialization():
    """测试1: 验证网络初始化是否合理"""
    print("=" * 70)
    print("TEST 1: Network Initialization Check")
    print("=" * 70)

    set_global_seed(GLOBAL_SEED)

    env = QMixHandoverEnv(
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
        actor_lr=1e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.25,
        entropy_coef=0.15,
        use_hierarchical=True,
    )

    # 测试初始动作分布
    obs_dict, global_state = env.reset()
    agent.reset_hidden()

    biz_types = {}
    for uid in range(env.num_agents):
        uav = env.env.uavs[uid]
        biz_types[uid] = uav.true_business_type.value

    actions, log_probs, values, pre_hidden = agent.select_actions(
        obs_dict, global_state, biz_types, training=True
    )

    # 统计动作分布
    action_counts = {}
    for uid, a in actions.items():
        action_counts[a] = action_counts.get(a, 0) + 1

    total = sum(action_counts.values())
    print(f"\nInitial Action Distribution (after fix):")
    for a in sorted(action_counts.keys()):
        pct = action_counts[a] / total * 100
        print(f"  Action {a}: {action_counts[a]:3d} ({pct:5.1f}%)")

    # 检查log_probs和entropy
    log_prob_values = list(log_probs.values())
    value_values = list(values.values())

    print(f"\nLog Probabilities:")
    print(f"  Mean: {np.mean(log_prob_values):.4f}")
    print(f"  Std:  {np.std(log_prob_values):.4f}")
    print(f"  Min:  {np.min(log_prob_values):.4f}")
    print(f"  Max:  {np.max(log_prob_values):.4f}")

    print(f"\nValue Estimates:")
    print(f"  Mean: {np.mean(value_values):.4f}")
    print(f"  Std:  {np.std(value_values):.4f}")

    # 判断探索性
    stay_pct = action_counts.get(0, 0) / total * 100
    if stay_pct > 80:
        print(f"\n[WARN] Stay ratio too high ({stay_pct:.1f}%) - exploration may be insufficient")
    elif stay_pct < 40:
        print(f"\n[GOOD] Good exploration (stay={stay_pct:.1f}%)")
    else:
        print(f"\n[OK] Acceptable exploration (stay={stay_pct:.1f}%)")

    return agent, env, biz_types


def test_training_step(agent, env, biz_types):
    """测试2: 执行一个完整的训练step，检查loss是否正常"""
    print("\n" + "=" * 70)
    print("TEST 2: Training Step Validation")
    print("=" * 70)

    obs_dict, global_state = env.reset()
    agent.reset_hidden()

    # 收集一个episode的经验
    for step in range(50):
        actions, log_probs, values, pre_hidden = agent.select_actions(
            obs_dict, global_state, biz_types, training=True
        )
        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

        agent.insert_experience(
            step, obs_dict, global_state, actions,
            rewards, team_reward, done, log_probs, values,
            biz_types, pre_hidden
        )

        obs_dict = next_obs
        global_state = next_state

    # 执行PPO更新
    train_stats = agent.train()

    if not train_stats:
        print("\n[ERROR] train() returned empty dict - buffer may be empty!")
        return False

    print("\nTraining Statistics (should NOT be all zeros):")
    print(f"  Actor Loss:     {train_stats['actor_loss']:.6f}")
    print(f"  Critic Loss:    {train_stats['critic_loss']:.6f}")
    print(f"  Entropy:        {train_stats['entropy']:.6f}")
    print(f"  Total Loss:     {train_stats['total_loss']:.6f}")
    print(f"  Actor Grad:     {train_stats.get('actor_grad_norm', 0):.6f}")
    print(f"  Critic Grad:    {train_stats.get('critic_grad_norm', 0):.6f}")
    print(f"  Value MSE:      {train_stats.get('value_mse', 0):.6f}")

    if 'ratio_mean' in train_stats:
        print(f"  Ratio Mean:     {train_stats['ratio_mean']:.6f}")
    if 'advantage_mean' in train_stats:
        print(f"  Advantage Mean: {train_stats['advantage_mean']:.6f}")
    if 'return_mean' in train_stats:
        print(f"  Return Mean:    {train_stats['return_mean']:.6f}")
    if 'num_updates' in train_stats:
        print(f"  Num Updates:    {train_stats['num_updates']}")

    # 判断训练是否正常
    is_healthy = True
    issues = []

    if train_stats['actor_loss'] < 1e-8:
        issues.append("Actor loss ~0 (policy not updating)")
        is_healthy = False
    if train_stats['critic_loss'] < 1e-8:
        issues.append("Critic loss ~0 (value not learning)")
        is_healthy = False
    if train_stats['entropy'] < 0.3:
        issues.append(f"Entropy too low ({train_stats['entropy']:.4f} < 0.3)")
        is_healthy = False
    if train_stats.get('actor_grad_norm', 0) < 1e-8:
        issues.append("Actor grad ~0 (no gradient flow)")
        is_healthy = False

    if is_healthy:
        print("\n[PASS] Training looks HEALTHY!")
    else:
        print("\n[FAIL] Training has ISSUES:")
        for issue in issues:
            print(f"       - {issue}")

    return is_healthy


def main():
    print("\n" + "=" * 70)
    print("MAPPO Fix Verification Script")
    print("Verifying that training metrics are no longer all zeros")
    print("=" * 70 + "\n")

    try:
        # Test 1: Initialization
        agent, env, biz_types = test_initialization()

        # Test 2: Training Step
        success = test_training_step(agent, env, biz_types)

        print("\n" + "=" * 70)
        if success:
            print("OVERALL RESULT: ALL TESTS PASSED!")
            print("The MAPPO algorithm should now train correctly.")
        else:
            print("OVERALL RESULT: SOME TESTS FAILED")
            print("Further investigation needed.")
        print("=" * 70)

        return 0 if success else 1

    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
