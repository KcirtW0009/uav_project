"""Simple test"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force reload
if 'uav_system.mappo_agent' in sys.modules:
    del sys.modules['uav_system.mappo_agent']

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent import MAPPOAgent

set_global_seed(GLOBAL_SEED)
env = MultiAgentHandoverEnv(num_bs=4, num_uav=10, max_steps=20, seed=GLOBAL_SEED, bs_capacity_range=(50, 100))
agent = MAPPOAgent(num_agents=10, obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=env.action_dim, use_hierarchical=True)

obs_dict, global_state = env.reset()
agent.reset_hidden()
biz_types = {i: env.env.uavs[i].true_business_type.value for i in range(10)}

print("Collecting experiences...")
for step in range(20):
    actions, log_probs, values, pre_hidden = agent.select_actions(obs_dict, global_state, biz_types, training=True)
    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
    agent.insert_experience(step, obs_dict, global_state, actions, rewards, team_reward, done, log_probs, values, biz_types, pre_hidden)
    obs_dict = next_obs
    global_state = next_state

print(f"Buffer ptr: {agent.buffer.ptr}")
print("Calling train()...")
result = agent.train()
print(f"Result keys: {list(result.keys()) if result else 'EMPTY'}")
if result:
    for k in ['actor_loss', 'critic_loss', 'entropy', 'num_updates']:
        print(f"  {k}: {result.get(k, 'N/A')}")
