"""验证最终配置的快速测试"""
import numpy as np
from uav_system.config import GLOBAL_SEED, set_global_seed
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.parametric_algorithm import STRATEGY_CONFIGS

set_global_seed(GLOBAL_SEED)
print("=== Final config test: UAV=30, cap=(80,180), pos_range=3000 ===")
env = MultiAgentHandoverEnv(num_bs=8, num_uav=30, max_steps=100, seed=GLOBAL_SEED,
                       bs_capacity_range=(80, 180), pos_range=3000)
print(f'action_dim={env.action_dim}')

sinrs = env.env.sinr_matrix
print(f'SINR range: [{sinrs.min():.1f}, {sinrs.max():.1f}] dB')

for act_id, act_name in enumerate(['stay', 'conservative', 'balanced', 'rate_focus', 'stability', 'aggressive']):
    env.reset()
    sats = []
    for step in range(100):
        actions = {uid: act_id for uid in range(30)}
        obs, state, rewards, team_reward, done, info = env.step(actions)
        sats.append(info['avg_satisfaction'])
    ho = sum(env.env.uavs[uid].handover_count for uid in range(30))
    print(f'  action={act_id} ({act_name:15s}): sat={np.mean(sats):.4f}, ho={ho}')

print("\n=== UAV=50 ===")
env50 = MultiAgentHandoverEnv(num_bs=8, num_uav=50, max_steps=100, seed=GLOBAL_SEED,
                         bs_capacity_range=(80, 180), pos_range=3000)
for act_id, act_name in [(0, 'stay'), (3, 'rate_focus'), (5, 'aggressive')]:
    env50.reset()
    sats = []
    for step in range(100):
        actions = {uid: act_id for uid in range(50)}
        obs, state, rewards, team_reward, done, info = env50.step(actions)
        sats.append(info['avg_satisfaction'])
    ho = sum(env50.env.uavs[uid].handover_count for uid in range(50))
    print(f'  action={act_id} ({act_name:15s}): sat={np.mean(sats):.4f}, ho={ho}')
