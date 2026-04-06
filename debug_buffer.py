"""
MAPPO Buffer Debug Script - 定位num_updates=0的原因
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent

def debug_buffer():
    """详细调试buffer数据流"""
    print("=" * 70)
    print("BUFFER DEBUG: Why num_updates = 0?")
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
        use_hierarchical=True,
    )

    obs_dict, global_state = env.reset()
    agent.reset_hidden()

    biz_types = {}
    for uid in range(env.num_agents):
        biz_types[uid] = env.env.uavs[uid].true_business_type.value

    # 收集经验
    print("\n[1] Collecting experiences...")
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

    # 检查buffer状态
    buffer = agent.buffer
    print(f"\n[2] Buffer State AFTER collection:")
    print(f"    buffer.ptr = {buffer.ptr}")
    print(f"    buffer.rollout_length = {buffer.rollout_length}")
    print(f"    buffer.obs.shape = {buffer.obs.shape}")
    print(f"    buffer.actions.shape = {buffer.actions.shape}")
    print(f"    buffer.log_probs.shape = {buffer.log_probs.shape}")
    print(f"    buffer.values.shape = {buffer.values.shape}")
    print(f"    buffer.hiddens.shape = {buffer.hiddens.shape}")

    # 计算GAE
    print(f"\n[3] Computing GAE...")
    next_values = np.zeros(agent.num_agents, dtype=np.float32)
    advantages, returns = buffer.compute_gae(next_values)

    print(f"    advantages.shape = {advantages.shape}")
    print(f"    returns.shape = {returns.shape}")
    print(f"    advantages mean = {advantages.mean():.6f}")
    print(f"    returns mean = {returns.mean():.6f}")

    # 模拟get_batches
    print(f"\n[4] Simulating get_batches()...")
    burn_in = min(5, buffer.ptr // 3)
    batch_size = 64

    print(f"    burn_in = {burn_in}")
    print(f"    batch_size = {batch_size}")

    ptr = buffer.ptr
    start_idx = min(burn_in, ptr)
    N = buffer.num_agents

    print(f"    ptr = {ptr}")
    print(f"    start_idx = {start_idx}")
    print(f"    N (num_agents) = {N}")

    try:
        obs_flat = buffer.obs[start_idx:ptr].reshape(-1, buffer.obs_dim)
        print(f"    obs_flat.shape = {obs_flat.shape}")

        actions_flat = buffer.actions[start_idx:ptr].reshape(-1)
        print(f"    actions_flat.shape = {actions_flat.shape}")

        actual_len = ptr - start_idx
        dataset_size = obs_flat.shape[0]
        print(f"    actual_len = {actual_len}")
        print(f"    dataset_size (should be > 0) = {dataset_size}")

        if dataset_size == 0:
            print(f"\n    [ERROR] dataset_size == 0! No batches will be generated!")
            print(f"           This is why num_updates = 0")
            if start_idx >= ptr:
                print(f"           Cause: start_idx ({start_idx}) >= ptr ({ptr})")
            return False

        # 尝试生成batch
        num_batches = 0
        for epoch in range(5):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]
                num_batches += 1
                if num_batches <= 2:  # 只打印前2个batch的信息
                    print(f"    Batch {num_batches}: idx shape = {idx.shape}")

        print(f"\n    Total batches generated: {num_batches}")
        print(f"    [SUCCESS] Batches are being generated correctly!")

        return True

    except Exception as e:
        print(f"\n    [ERROR] Exception during batch generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_buffer()
    sys.exit(0 if success else 1)
