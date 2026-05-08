#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P0 Bug诊断脚本：精确定位 'numpy.float64' object has no attribute 'get' 错误
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[DIAG] Starting P0 bug diagnosis...", flush=True)

# 1. 测试导入
print("\n[STEP 1] Importing modules...", flush=True)
try:
    import finetune_multi_scenario
    import torch
    import numpy as np
    from uav_system.mappo_environment import MultiAgentHandoverEnv
    from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
    print("[OK] All modules imported", flush=True)
except Exception as e:
    print(f"[FAIL] Import error: {e}", flush=True)
    sys.exit(1)

# 2. 创建最小测试环境
print("\n[STEP 2] Creating test environment...", flush=True)
try:
    env = MultiAgentHandoverEnv(
        num_bs=8,
        num_uav=300,  # 使用较小的UAV数加速测试
        max_steps=50,  # 减少步数加速
        seed=42,
        scenario='industrial_inspection'
    )
    print(f"[OK] Environment created: {env.num_agents} agents", flush=True)
except Exception as e:
    print(f"[FAIL] Env creation error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 3. 创建Agent
print("\n[STEP 3] Creating agent...", flush=True)
try:
    model_path = os.path.join(
        'experiment_results', 'mappo_models',
        'mappo_8bs_300uav_best.pt'
    )

    if not os.path.exists(model_path):
        print(f"[WARN] Model not found: {model_path}")
        print("       Creating untrained agent for testing...")
        agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
        )
    else:
        agent = MAPPOAgent(
            num_agents=env.num_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
        ).load(model_path)
    
    print(f"[OK] Agent created/loaded", flush=True)
except Exception as e:
    print(f"[FAIL] Agent creation error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 4. 测试 env.step() 返回值
print("\n[STEP 4] Testing env.step() return values...", flush=True)
try:
    obs_dict, global_state = env.reset()
    print(f"  reset() -> obs_dict type: {type(obs_dict)}")
    print(f"            global_state type: {type(global_state)}")

    # 创建dummy actions
    dummy_actions = {uid: 0 for uid in range(env.num_agents)}
    print(f"  dummy_actions: {dummy_actions}")

    # 调用step()并检查返回值
    result = env.step(dummy_actions)
    print(f"\n  step() returned {len(result)} values:")
    for i, val in enumerate(result):
        print(f"    [{i}] type={type(val).__name__}, value_type={type(val)}")
        
        # 如果是dict，显示keys
        if isinstance(val, dict):
            print(f"         keys={list(val.keys())[:5]}...")
            # 检查每个value的类型
            for k, v in list(val.items())[:3]:
                print(f"           [{k}] = {v} (type: {type(v).__name__})")
        elif isinstance(val, (int, float, np.floating)):
            print(f"         value={val}")

    # 解包返回值（模拟_train_one_episode中的代码）
    print(f"\n  Unpacking (6 values expected)...")
    next_obs_dict, next_global_state, rewards, team_reward, done, info = result
    
    print(f"\n  Unpacked variables:")
    print(f"    next_obs_dict type: {type(next_obs_dict)}")
    print(f"    next_global_state type: {type(next_global_state)}")
    print(f"    rewards type: {type(rewards)}")
    if isinstance(rewards, dict):
        print(f"      rewards sample: {list(rewards.items())[:2]}")
    print(f"    team_reward type: {type(team_reward)}, value={team_reward}")
    print(f"    done type: {type(done)}, value={done}")
    print(f"    info type: {type(info)}")
    if isinstance(info, dict):
        print(f"      info keys: {list(info.keys())[:5]}")
    else:
        print(f"      [WARN] info is NOT a dict! value={info}")

    print(f"\n[OK] env.step() test completed", flush=True)

except Exception as e:
    print(f"\n[FAIL] env.step() error: {e}", flush=True)
    print(f"  Error type: {type(e).__name__}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 5. 测试 agent.select_actions()
print("\n[STEP 5] Testing agent.select_actions()...", flush=True)
try:
    biz_types = {
        uid: env.env.uavs[uid].true_business_type.value 
        for uid in range(env.num_agents)
    }
    
    actions, log_probs, values, pre_hiddens, obs_aug = \
        agent.select_actions(
            obs_dict, global_state,
            biz_types=biz_types,
            training=True,
            env=env
        )
    
    print(f"  select_actions() returned {len(values)} values:")
    print(f"    actions type: {type(actions)}")
    print(f"    log_probs type: {type(log_probs)}")
    print(f"    values type: {type(values)}")
    print(f"    pre_hiddens type: {type(pre_hiddens)}")
    print(f"    obs_aug type: {type(obs_aug)}")
    
    print(f"\n[OK] select_actions() test passed", flush=True)

except Exception as e:
    print(f"\n[FAIL] select_actions() error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 6. 测试完整的一个step循环
print("\n[STEP 6] Testing complete step loop (3 steps)...", flush=True)
try:
    obs_dict, global_state = env.reset()
    
    for step in range(3):
        biz_types = {
            uid: env.env.uavs[uid].true_business_type.value 
            for uid in range(env.num_agents)
        }
        
        actions, log_probs, values, pre_hiddens, obs_aug = \
            agent.select_actions(
                obs_dict, global_state,
                biz_types=biz_types,
                training=True,
                env=env
            )
        
        next_obs_dict, next_global_state, rewards, team_reward, done, info = \
            env.step(actions)
        
        # 测试reward scaling (关键!)
        reward_scale = 1.0 / np.sqrt(300)  # industrial_inspection的scale
        scaled_rewards = {
            uid: r * reward_scale for uid, r in rewards.items()
        }
        scaled_team_reward = team_reward * reward_scale
        
        print(f"  Step {step}: reward={team_reward:.2f}, "
              f"scaled={scaled_team_reward:.4f}, "
              f"done={done}")
        
        # 构建rollout buffer entry (模拟实际代码)
        rollout_entry = {
            'obs': obs_dict,
            'global_state': global_state,
            'actions': actions,
            'rewards': scaled_rewards,
            'log_probs': log_probs,
            'values': values,
            'hiddens': pre_hiddens,
            'dones': done,
            'biz_types': biz_types,
        }
        
        # 测试insert_experience
        agent.insert_experience(
            step=step,
            obs_dict=rollout_entry['obs'],
            state=rollout_entry['global_state'],
            actions=rollout_entry['actions'],
            rewards=rollout_entry['rewards'],
            team_reward=0.0,
            done=rollout_entry['dones'],
            log_probs=rollout_entry['log_probs'],
            values=rollout_entry['values'],
            biz_types=rollout_entry['biz_types'],
        )
        
        obs_dict = next_obs_dict
        global_state = next_global_state
        
        if done:
            print(f"  Episode ended at step {step}")
            break
    
    print(f"\n[OK] Step loop test passed", flush=True)

except Exception as e:
    print(f"\n[FAIL] Step loop error: {e}", flush=True)
    print(f"  Error type: {type(e).__name__}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 7. 测试 agent.train() 返回值
print("\n[STEP 7] Testing agent.train() return value...", flush=True)
try:
    loss_info = agent.train()
    print(f"  train() returned: type={type(loss_info)}")
    
    if loss_info is None:
        print("  [WARN] train() returned None (buffer too small?)")
    elif isinstance(loss_info, dict):
        print(f"  [OK] train() returned dict with {len(loss_info)} keys:")
        for key, val in loss_info.items():
            print(f"    [{key}] = {val} (type: {type(val).__name__})")
            
            # 关键检查: 这些值是否可以安全地使用 .get()? 
            # (它们不应该被调用 .get(), 因为它们已经是值而不是字典)
    else:
        print(f"  [WARN] train() returned unexpected type: {loss_info}")
    
    print(f"\n[OK] train() test passed", flush=True)

except Exception as e:
    print(f"\n[FAIL] train() error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("[SUCCESS] All diagnostic tests passed!")
print("="*60)
print("\nIf this script runs without errors, the bug might be in:")
print("  1. Edge cases with specific UAV numbers")
print("  2. Race conditions in multi-threaded execution")
print("  3. Specific episode scenarios that trigger the bug")
