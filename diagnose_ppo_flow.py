"""
深度诊断：追踪 PPO 训练数据流
验证：reward → normalize → buffer → compute_gae → get_batches → PPO loss 完整链路
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uav_system'))

from uav_system.qmix_environment import QMixHandoverEnv, RunningNormalizer
from uav_system.mappo_agent import MAPPOAgent, RolloutBuffer

def diagnose_data_flow():
    print("=" * 70)
    print("PPO 数据流深度诊断")
    print("=" * 70)

    # ---- 1. 创建环境 ----
    env_config = {
        'num_uav': 10,
        'num_bs': 4,
        'bs_capacity_range': (80, 180),
        'max_steps': 20,
        'seed': 42,
    }
    env = QMixHandoverEnv(**env_config)
    obs, state = env.reset()
    print(f"\n[1] 环境创建: {env.num_agents} UAVs, {env.action_dim} actions")

    # ---- 2. 检查 RunningNormalizer 行为 ----
    print(f"\n[2] RunningNormalizer 测试:")
    norm = RunningNormalizer(num_agents=10)

    test_rewards = [
        {i: float(-0.05 + (i % 3) * 0.1 - 0.02) for i in range(10)},  # 大部分是负的(stay)
        {i: float(3.0 if i < 3 else -0.05) for i in range(10)},       # 少数正的(good switch)
        {i: float(-0.03 + np.random.randn() * 0.1) for i in range(10)}, # 噪声
    ]

    for idx, rw in enumerate(test_rewards):
        normalized = norm.normalize(rw)
        raw_vals = np.array([rw[i] for i in range(10)])
        norm_vals = np.array([normalized[i] for i in range(10)])
        print(f"  Batch {idx}: raw mean={raw_vals.mean():.4f}, std={raw_vals.std():.4f} -> "
              f"norm mean={norm_vals.mean():.6f}, std={norm_vals.std():.4f}, "
              f"EMA_mean={norm.mean[0]:.4f}, EMA_var={norm.var[0]:.4f}")

    # ---- 3. 模拟一个 episode 的数据流 ----
    print(f"\n[3] 模拟 Episode 数据流:")

    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        rollout_length=50,
        gamma=0.95,
        gae_lambda=0.95,
        hidden_dim=64,
        use_hierarchical=True,
        use_biz_heads=True,
        use_attention_critic=True,
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=16,
        num_epochs=3,
    )

    obs_dict, global_state = env.reset()
    agent.reset_hidden()

    # 收集一个 episode 的数据
    for step in range(min(20, env.max_steps)):
        biz_types = {}
        for uid in range(env.num_agents):
            uav = env.env.uavs[uid]
            biz_types[uid] = uav.true_business_type.value

        actions, log_probs, values, pre_hidden = agent.select_actions(
            obs_dict, global_state, biz_types, training=True, env=env
        )

        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

        # 打印原始 reward vs 归一化后 reward（如果可以获取）
        if step < 3:
            print(f"  Step {step}: rewards sample={[f'{rewards.get(i,0):.4f}' for i in range(3)]}, "
                  f"values sample={[f'{values.get(i,0):.4f}' for i in range(3)]}")

        agent.insert_experience(step, obs_dict, global_state, actions,
                               rewards, team_reward, done, log_probs, values,
                               biz_types, pre_hidden)
        obs_dict = next_obs
        global_state = next_state

        if done:
            break

    print(f"\n  Buffer: ptr={agent.buffer.ptr}, rollout_len={agent.buffer.rollout_length}")

    # ---- 4. compute_gae 分析 ----
    print(f"\n[4] compute_gae 分析:")
    next_values = np.zeros(env.num_agents, dtype=np.float32)
    advantages, returns = agent.buffer.compute_gae(next_values)

    print(f"  advantages: shape={advantages.shape}")
    print(f"  adv mean={advantages.mean():.6f}, std={advantages.std():.6f}, "
          f"max={advantages.max():.6f}, min={advantages.min():.6f}")
    print(f"  returns: mean={returns.mean():.4f}, std={returns.std():.4f}")

    rewards_stored = agent.buffer.rewards[:agent.buffer.ptr].cpu().numpy()
    values_stored = agent.buffer.values[:agent.buffer.ptr].cpu().numpy()
    print(f"  stored rewards: mean={rewards_stored.mean():.6f}, std={rewards_stored.std():.6f}")
    print(f"  stored values:  mean={values_stored.mean():.6f}, std={values_stored.std():.6f}")

    # 计算 TD residuals
    for t in range(min(5, agent.buffer.ptr - 1)):
        next_val = float(values_stored[t + 1, 0]) if t < agent.buffer.ptr - 1 else 0.0
        delta = float(rewards_stored[t, 0]) + 0.95 * next_val - float(values_stored[t, 0])
        print(f"  TD residual[{t}]: r={rewards_stored[t,0]:.4f} + 0.95*V'={0.95*next_val:.4f} - V={values_stored[t,0]:.4f} = {delta:.6f}")

    # ---- 5. get_batches 分析 ----
    print(f"\n[5] get_batches 分析:")
    burn_in = min(5, agent.buffer.ptr // 3)
    print(f"  burn_in={burn_in}, batch_size={agent.batch_size}, num_epochs={agent.num_epochs}")

    batch_count = 0
    for batch in agent.buffer.get_batches(agent.batch_size, advantages, returns, agent.num_epochs, burn_in=burn_in):
        obs_b, obs_all_b, states_b, actions_b, lp_b, adv_b, ret_b, val_b, biz_b, hid_b = batch
        if batch_count == 0:
            print(f"  Batch-0: shape obs={obs_b.shape}, adv mean={adv_b.mean().item():.6f}, "
                  f"adv std={adv_b.std().item():.6f}, lp mean={lp_b.mean().item():.4f}")
        batch_count += 1

    print(f"  Total batches yielded: {batch_count}")

    # ---- 6. 执行 train() 并检查结果 ----
    print(f"\n[6] train() 执行结果:")
    stats = agent.train()
    print(f"  train() returned: {stats}")
    if stats:
        for k, v in stats.items():
            print(f"    {k}: {v:.6f}")
    else:
        print("  train() returned EMPTY dict!")

    # ---- 7. 关键问题诊断 ----
    print(f"\n{'='*70}")
    print("[7] 根因诊断结论:")
    print(f"{'='*70}")

    if abs(advantages.mean()) < 1e-6 and advantages.std() < 1e-6:
        print("  ❌ CRITICAL: advantages 全部接近零！")
        print("     → PPO policy gradient = ratio * advantage ≈ 0")
        print("     → actor_loss ≈ 0 无论策略如何更新")
        print("     可能原因:")
        print("     a) Reward 归一化后方差被压缩到接近0")
        print("     b) Value 函数完美预测了 return (TD error = 0)")
        print("     c) Rewards 本身全部相同/常数")
    elif abs(advantages.std()) < 0.01:
        print("  ⚠️ WARNING: advantages 方差极小 (<0.01)")
        print("     → PPO 更新信号极弱，学习效率极低")
    else:
        print("  ✅ advantages 有合理的非零值，PPO 应该能正常工作")

    if batch_count == 0:
        print("  ❌ CRITICAL: get_batches 没有产生任何 batch!")
        print("     → for 循环从未执行 → num_updates=0 → 所有 loss = 0")
        print("     可能原因: burn_in >= ptr 或 dataset_size = 0")
    else:
        print(f"  ✅ get_batches 正常产生了 {batch_count} 个 batch")

    if stats and abs(stats.get('actor_loss', 1.0)) < 1e-8:
        print("  ❌ actor_loss = 0 (精确零)")
        print("     这确认了 PPO 梯度信号完全消失")

    print(f"\n[建议]")
    if abs(advantages.std()) < 0.1:
        print("  → 主要问题：advantage 方差太小")
        print("  → 解决方案：禁用或减弱 Reward 归一化 (RunningNormalizer)")
        print("  → 或者：在 compute_gae 后不归一化 advantages")

if __name__ == '__main__':
    diagnose_data_flow()
