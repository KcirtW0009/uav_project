"""
终极调试 - 检查实际执行的代码
"""

import sys
import os
import numpy as np
import inspect

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
# 强制重新导入
if 'uav_system.mappo_agent' in sys.modules:
    del sys.modules['uav_system.mappo_agent']
from uav_system.mappo_agent import MAPPOAgent

def ultimate_debug():
    print("=" * 70)
    print("ULTIMATE DEBUG - Check actual code being executed")
    print("=" * 70)

    # 检查MAPPOAgent的来源
    print(f"\n[1] MAPPOAgent source file: {inspect.getfile(MAPPOAgent)}")

    # 检查train方法的源代码
    print(f"\n[2] train() method source (lines 1356-1375):")
    try:
        source_lines = inspect.getsourcelines(MAPPOAgent.train)[0]
        for i, line in enumerate(source_lines[25:40], start=1360):  # 偏移到try块附近
            print(f"    {i:4d}: {line}", end='')
    except Exception as e:
        print(f"    Error getting source: {e}")

    # 创建实例并测试
    set_global_seed(GLOBAL_SEED)
    env = QMixHandoverEnv(num_bs=4, num_uav=10, max_steps=50, seed=GLOBAL_SEED, bs_capacity_range=(50, 100))
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        use_hierarchical=True,
    )

    obs_dict, global_state = env.reset()
    agent.reset_hidden()
    biz_types = {uid: env.env.uavs[uid].true_business_type.value for uid in range(env.num_agents)}

    # 收集经验
    for step in range(50):
        actions, log_probs, values, pre_hidden = agent.select_actions(obs_dict, global_state, biz_types, training=True)
        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
        agent.insert_experience(step, obs_dict, global_state, actions, rewards, team_reward, done, log_probs, values, biz_types, pre_hidden)
        obs_dict = next_obs
        global_state = next_state

    print(f"\n[3] Buffer ptr: {agent.buffer.ptr}")

    # 直接调用train并监控
    print(f"\n[4] Calling agent.train() with detailed monitoring...")

    # 使用pdb风格的追踪
    import trace
    tracer = trace.Trace(count=False, trace=True)
    tracer.runfunc(agent.train)

    # 获取结果
    results = agent.train()  # 再次调用获取返回值
    print(f"\n[5] Final result from train():")
    for k, v in results.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    ultimate_debug()
