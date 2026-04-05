"""
完整训练流程诊断：模拟 experiments_mappo.py 的完整流程
包括：预训练 → 标准环境训练 → 检查 train() 输出
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uav_system'))

from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent
from uav_system.algorithms import EnhancedHandoverAlgorithm

def diagnose_full_training_flow():
    print("=" * 70)
    print("完整训练流程诊断（模拟 experiments_mappo.py）")
    print("=" * 70)

    # ---- 配置（与 experiments_mappo.py 一致）----
    num_uav = 10  # 用较小的规模加速测试
    num_bs = 4
    num_steps = 20  # 减少步数加速
    train_episodes = 5  # 只跑几个 episode

    # ---- 1. 创建环境 ----
    env = QMixHandoverEnv(
        num_bs=num_bs, num_uav=num_uav,
        max_steps=num_steps, seed=42,
        bs_capacity_range=(80, 180),
    )
    env.reset_normalizer()
    print(f"\n[1] 环境创建: {env.num_agents} UAVs")

    # ---- 2. 创建 Agent ----
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.95,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.05,
        value_coef=0.5,
        rollout_length=max(150, num_steps),
        num_epochs=5,
        batch_size=64,
        use_biz_heads=True,
        use_attention_critic=True,
        use_enhanced_algorithm=True,
        use_pretrain=True,
        use_hierarchical=True,
        use_transformer=False,
        use_data_augmentation=True,
    )
    print(f"[2] Agent 创建完成")

    # ---- 3. 设置增强算法 + 预训练 ----
    enhanced_algorithm = EnhancedHandoverAlgorithm(env.env)
    agent.set_enhanced_algorithm(enhanced_algorithm)

    print(f"\n[3] 开始预训练...")
    demonstrations = agent.collect_demonstrations(env, num_demos=100)
    agent.pretrain(demonstrations, epochs=10, batch_size=32)  # 减少epochs加速
    print(f"   预训练完成")

    # ---- 4. 设置 LR schedule ----
    agent._total_train_steps = train_episodes
    agent._current_train_step = 0

    # ---- 5. 主训练循环（模拟标准环境）----
    print(f"\n[4] 开始主训练循环 ({train_episodes} episodes)...")

    for ep in range(train_episodes):
        obs_dict, global_state = env.reset()
        agent.reset_hidden()

        for step in range(num_steps):
            biz_types = {}
            for uid in range(env.num_agents):
                uav = env.env.uavs[uid]
                biz_types[uid] = uav.true_business_type.value

            agent.update_enhanced_algorithm_prob(ep, train_episodes)

            actions, log_probs, values, pre_hidden = agent.select_actions(
                obs_dict, global_state, biz_types, training=True, env=env
            )

            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

            agent.insert_experience(
                step, obs_dict, global_state, actions,
                rewards, team_reward, done, log_probs, values,
                biz_types, pre_hidden
            )

            obs_dict = next_obs
            global_state = next_state

            if done:
                break

        # ---- 关键：调用 train() 并检查输出 ----
        print(f"\n  --- Episode {ep+1}: 调用 train() ---")
        train_stats = agent.train()

        if train_stats:
            print(f"  ✅ train() 返回非空:")
            for k, v in train_stats.items():
                print(f"     {k}: {v:.6f}")

            if abs(train_stats['actor_loss']) < 1e-8:
                print(f"  ❌❌❌ actor_loss = 0! PPO 训练无效!")
            else:
                print(f"  ✅✅✅ actor_loss != 0! PPO 训练正常!")
        else:
            print(f"  ❌❌❌ train() 返回空 dict!")

    # ---- 6. 最终结论 ----
    print(f"\n{'='*70}")
    print("[最终诊断结论]")
    print(f"{'='*70}")
    print("""
如果所有 episode 的 actor_loss 都为 0:
  → 问题在预训练后的状态或实验配置中
  → 需要对比有无预训练的差异

如果无预训练时正常、有预训练时异常:
  → pretrain() 方法修改了某些关键状态
  → 可能是 obs_normalizer、actor 权重、或其他内部状态

如果两者都正常:
  → 问题在更大规模的配置 (UAV=30, steps=100) 中
  → 可能是内存/数值问题
""")

if __name__ == '__main__':
    diagnose_full_training_flow()
