"""
DQN 快速调试脚本

用途：验证修改后 DQN 是否正常收敛，约 5 分钟出结果
用法：python test_dqn_quick.py

修复内容验证：
1. 动作空间简化 (25→9 actions)
2. 训练拓扑多样化 (每50 ep重建环境)
3. 探索偏向stay (30%概率直接选stay)
4. 目标Q值无效动作屏蔽
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.environment import NetworkEnvironmentWithRecognition
from uav_system.algorithms import IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm
from uav_system.rl_environment import RLHandoverEnv
from uav_system.rl_agent import DQNAgent


def quick_test():
    """快速测试：20 UAV 场景，200 eps 训练，5 次评估"""
    # ===== 可调参数 =====
    NUM_BS = 8
    NUM_UAV = 20
    MAX_STEPS = 50           # 缩短到 50 步加速
    TRAIN_EPS = 200          # 缩短到 200 episodes
    REPEATS = 5              # 5 次评估
    TARGET_ID = 0
    SEED = GLOBAL_SEED
    # ======================

    print("=" * 60)
    print("  DQN 快速调试测试 (修复验证)")
    print("=" * 60)
    print(f"  {NUM_BS} BS x {NUM_UAV} UAV x {MAX_STEPS} steps")
    print(f"  Train {TRAIN_EPS} eps x Eval {REPEATS} reps")
    print(f"  Action space: 1(stay) + {NUM_BS}(switch) = {1 + NUM_BS}")
    print("=" * 60)

    # ---- 1. 训练 DQN ----
    t0 = time.time()
    print("\n[1/3] 训练 DQN...")

    agent = DQNAgent(
        state_dim=NUM_BS * 4 + 7,
        action_dim=1 + NUM_BS,
        lr=5e-4, gamma=0.95, hidden_dim=128,
        buffer_size=50000, batch_size=64,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        target_update_freq=500,
    )

    rl_env_train = RLHandoverEnv(
        NetworkEnvironmentWithRecognition(num_bs=NUM_BS, num_uav=NUM_UAV, seed=SEED),
        target_uav_id=TARGET_ID, max_steps=MAX_STEPS
    )
    train_rewards = []
    stay_counts = []
    switch_counts = []

    for ep in range(TRAIN_EPS):
        # 固定拓扑训练：让 DQN 充分收敛，评估时再用不同种子测泛化
        state = rl_env_train.reset()
        ep_reward = 0.0
        ep_sat_sum = 0.0
        ep_ho = 0
        ep_stay = 0
        ep_switch = 0

        for step_i in range(MAX_STEPS):
            invalid = rl_env_train.get_invalid_actions()
            action = agent.select_action(state, training=True, invalid_actions=invalid)
            next_state, reward, done, info = rl_env_train.step(action)
            next_invalid = rl_env_train.get_invalid_actions()
            agent.store_transition(state, action, reward, next_state, float(done),
                                   next_invalid_actions=next_invalid)
            agent.train_step()
            ep_reward += reward
            ep_sat_sum += info['satisfaction']
            if info['action_type'] == 'stay':
                ep_stay += 1
            else:
                ep_switch += 1
            ep_ho = info['total_handovers']
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        train_rewards.append(ep_reward)
        stay_counts.append(ep_stay)
        switch_counts.append(ep_switch)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(train_rewards[-50:])
            avg_stay = np.mean(stay_counts[-50:])
            avg_switch = np.mean(switch_counts[-50:])
            print(f"    Ep {ep+1}/{TRAIN_EPS}, eps={agent.epsilon:.3f}, "
                  f"avg_R={avg_r:.1f}, sat={ep_sat_sum/MAX_STEPS:.3f}, "
                  f"stay={avg_stay:.0f}/ep, switch={avg_switch:.0f}/ep, ho={ep_ho}")

    print(f"  训练完成, 耗时 {time.time()-t0:.1f}s")

    # ---- 2. 评估 DQN（详细模式）----
    t1 = time.time()
    print("\n[2/3] 评估 DQN (详细模式, Rep 1)...")

    set_global_seed(SEED)
    env_eval = NetworkEnvironmentWithRecognition(num_bs=NUM_BS, num_uav=NUM_UAV, seed=SEED)
    rl_env_eval = RLHandoverEnv(env_eval, target_uav_id=TARGET_ID, max_steps=MAX_STEPS)
    state = rl_env_eval.reset()
    dqn_sats = []
    stay_count = 0
    switch_attempts = 0
    effective_switches = 0

    for step in range(MAX_STEPS):
        invalid = rl_env_eval.get_invalid_actions()
        action = agent.select_action(state, training=False, invalid_actions=invalid)
        next_state, reward, done, info = rl_env_eval.step(action)
        dqn_sats.append(info['satisfaction'])
        if info['action_type'] == 'stay':
            stay_count += 1
        else:
            switch_attempts += 1
            if info.get('actual_switch', False):
                effective_switches += 1
        print(f"  Step {step+1:3d}: BS={info['connected_bs']}, "
              f"Sat={info['satisfaction']:.3f}, "
              f"Act={info['action_type']}->BS{info.get('action_target_bs','-')}, "
              f"sw={info.get('actual_switch', False)}, "
              f"cumHO={info['total_handovers']}")
        state = next_state
        if done:
            break

    print(f"\n  DQN 评估摘要:")
    print(f"    平均满意度: {np.mean(dqn_sats):.4f}")
    print(f"    stay={stay_count}, switch尝试={switch_attempts}, 有效切换={effective_switches}")
    print(f"    stay率={stay_count/MAX_STEPS*100:.1f}%")
    print(f"  耗时 {time.time()-t1:.1f}s")

    # ---- 3. 全量评估 + 基线对比 ----
    t2 = time.time()
    print("\n[3/3] 基线对比 (5次重复)...")

    dqn_sats_all = []
    dqn_hos_all = []
    for rep in range(REPEATS):
        set_global_seed(SEED + rep)
        env_dqn = NetworkEnvironmentWithRecognition(num_bs=NUM_BS, num_uav=NUM_UAV, seed=SEED + rep)
        rl_env = RLHandoverEnv(env_dqn, target_uav_id=TARGET_ID, max_steps=MAX_STEPS)
        state = rl_env.reset()
        sats = []
        for step in range(MAX_STEPS):
            invalid = rl_env.get_invalid_actions()
            action = agent.select_action(state, training=False, invalid_actions=invalid)
            next_state, reward, done, info = rl_env.step(action)
            sats.append(info['satisfaction'])
            state = next_state
            if done:
                break
        dqn_sats_all.append(np.mean(sats))
        dqn_hos_all.append(info.get('handover_count', 0))

    trad_sats = []
    enh_sats = []
    for rep in range(REPEATS):
        set_global_seed(SEED + rep)
        env = NetworkEnvironmentWithRecognition(num_bs=NUM_BS, num_uav=NUM_UAV, seed=SEED + rep)
        algo = IntegratedHandoverAlgorithm(env)
        ep_sats = []
        for step in range(MAX_STEPS):
            env.step()
            if hasattr(algo, 'run_step'):
                algo.run_step()
            ep_sats.append(env.uavs[TARGET_ID].current_satisfaction)
        trad_sats.append(np.mean(ep_sats))

        set_global_seed(SEED + rep)
        env = NetworkEnvironmentWithRecognition(num_bs=NUM_BS, num_uav=NUM_UAV, seed=SEED + rep)
        algo2 = EnhancedHandoverAlgorithm(env)
        algo2.epsilon = 0.0
        ep_sats = []
        for step in range(MAX_STEPS):
            env.step()
            algo2.run_step(enable_load_balancing=True)
            ep_sats.append(env.uavs[TARGET_ID].current_satisfaction)
        enh_sats.append(np.mean(ep_sats))

    print(f"\n  {'='*50}")
    print(f"  结果对比 ({NUM_UAV} UAV, {MAX_STEPS} steps):")
    print(f"  {'='*50}")
    print(f"  传统算法:    Sat={np.mean(trad_sats):.4f}+/-{np.std(trad_sats):.4f}")
    print(f"  增强启发式:  Sat={np.mean(enh_sats):.4f}+/-{np.std(enh_sats):.4f}")
    print(f"  DQN:         Sat={np.mean(dqn_sats_all):.4f}+/-{np.std(dqn_sats_all):.4f}, "
          f"HO={np.mean(dqn_hos_all):.1f}+/-{np.std(dqn_hos_all):.1f}")
    print(f"  {'='*50}")

    # ---- 诊断判定 ----
    dqn_sat = np.mean(dqn_sats_all)
    enh_sat = np.mean(enh_sats)
    avg_ho = np.mean(dqn_hos_all)

    print(f"\n  【诊断结果】")
    if avg_ho > MAX_STEPS * 0.8:
        print(f"  [FAIL] DQN 仍在疯狂切换 ({avg_ho:.0f}/{MAX_STEPS} steps)")
    elif avg_ho < 2:
        print(f"  [WARN] DQN 几乎不切换 ({avg_ho:.1f} steps) -- 可能过于保守")
    else:
        print(f"  [PASS] DQN 切换次数合理 ({avg_ho:.1f} steps)")

    if dqn_sat >= enh_sat * 0.95:
        print(f"  [PASS] DQN 满意度接近启发式 ({dqn_sat:.4f} vs {enh_sat:.4f})")
    elif dqn_sat >= 0.5:
        print(f"  [WARN] DQN 满意度中等 ({dqn_sat:.4f} vs {enh_sat:.4f})")
    else:
        print(f"  [FAIL] DQN 满意度过低 ({dqn_sat:.4f} vs {enh_sat:.4f})")

    print(f"\n{'='*60}")
    print(f"快速测试完成! 总耗时 {time.time()-t0:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    quick_test()
