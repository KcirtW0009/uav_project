"""
MAPPO Train Debug - 直接监控train()执行
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.mappo_agent import MAPPOAgent

def debug_train_directly():
    """直接调试train()方法"""
    print("=" * 70)
    print("DIRECT TRAIN DEBUG")
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

    print(f"\n[2] Buffer ptr before train(): {agent.buffer.ptr}")

    # Monkey-patch train to add debugging
    original_train = agent.train

    def debug_train():
        print(f"\n[3] Inside train() method:")
        print(f"    buffer.ptr = {agent.buffer.ptr}")

        if agent.buffer.ptr == 0:
            print("    [EARLY RETURN] buffer.ptr == 0!")
            return {}

        # Call original train but capture the process
        import io
        from contextlib import redirect_stdout, redirect_stderr

        f_stdout = io.StringIO()
        f_stderr = io.StringIO()

        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            result = original_train()

        stdout_output = f_stdout.getvalue()
        stderr_output = f_stderr.getvalue()

        if stdout_output:
            print(f"    STDOUT during train():\n{stdout_output}")
        if stderr_output:
            print(f"    STDERR during train():\n{stderr_output}")

        print(f"\n[4] train() returned:")
        if result:
            for k, v in result.items():
                print(f"    {k}: {v}")
        else:
            print("    EMPTY DICT (or None)")

        return result

    agent.train = debug_train

    # Now call train
    print("\n[5] Calling agent.train()...")
    train_stats = agent.train()

    return train_stats is not None and len(train_stats) > 0 and train_stats.get('actor_loss', 0) > 1e-8


if __name__ == "__main__":
    success = debug_train_directly()
    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Training is working!")
    else:
        print("FAILED: Training still broken")
    print("=" * 70)
    sys.exit(0 if success else 1)
