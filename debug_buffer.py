#!/usr/bin/env python3
"""快速诊断: 检查buffer数据流"""
import sys
sys.path.insert(0, r'f:\桌面\本科毕业论文\结题\uav_project')

from uav_system.mappo_agent import MAPPOAgent
from uav_system.qmix_environment import QMixHandoverEnv
import numpy as np

print('='*60)
print('Buffer 数据流诊断测试')
print('='*60)

env = QMixHandoverEnv(num_bs=4, num_uav=10, max_steps=20, seed=42)
obs_dict, global_state = env.reset()

agent = MAPPOAgent(
    num_agents=env.num_agents,
    obs_dim=env.obs_dim,
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    hidden_dim=32,
    critic_hidden_dim=64,
    use_biz_heads=True,
    use_attention_critic=False,
    use_hierarchical=True,
)

biz_types = {i: env.env.uavs[i].true_business_type.value for i in range(env.num_agents)}
agent.reset_hidden()

print('[1] 初始 buffer.ptr = %d' % agent.buffer.ptr)

actions, log_probs, values, pre_hidden = agent.select_actions(obs_dict, global_state, biz_types, training=True)
print('[2] 选择动作后 buffer.ptr = %d' % agent.buffer.ptr)

next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
agent.insert_experience(0, obs_dict, global_state, actions, rewards, team_reward, done, log_probs, values, biz_types, pre_hidden)
print('[3] insert_experience后 buffer.ptr = %d' % agent.buffer.ptr)

stats = agent.train()
print('[4] train()返回: %s' % str(stats))
print('[5] 返回是否为空: %s' % (len(stats) == 0 if isinstance(stats, dict) else True))

if stats:
    print('    actor_loss = %s' % stats.get('actor_loss', 'N/A'))
    print('    critic_loss = %s' % stats.get('critic_loss', 'N/A'))
    print('    entropy = %s' % stats.get('entropy', 'N/A'))
else:
    print('    [!!!] train()返回空字典 - PPO不会更新!')
    print('    [!!!] buffer实际ptr=%d' % agent.buffer.ptr)
