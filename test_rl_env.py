"""
RL 环境包装器单元测试

验证 RLHandoverEnv 的 reset / step 接口正确性，包括：
1. 环境创建与维度
2. reset 返回正确形状的状态
3. step 执行正确且返回值格式正确
4. 动作映射表完整
5. 完整 episode 能跑通
6. 随机策略 episode 的基本统计
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_system.config import set_global_seed
from uav_system.environment import NetworkEnvironmentWithRecognition
from uav_system.rl_environment import RLHandoverEnv, create_rl_env


def test_env_creation():
    """测试1: 环境创建与维度检查"""
    print("=" * 60)
    print("测试1: 环境创建与维度检查")
    print("=" * 60)

    set_global_seed(42)
    env = NetworkEnvironmentWithRecognition(num_bs=8, num_uav=20, seed=42)
    rl_env = RLHandoverEnv(env, target_uav_id=0, max_steps=150)

    expected_state_dim = 8 * 2 + 3 + 3  # 22
    expected_action_dim = 1 + 8 * 3      # 25

    print(f"  基站数量: {env.num_bs}")
    print(f"  UAV 数量: {env.num_uav}")
    print(f"  状态维度: {rl_env.state_dim} (期望: {expected_state_dim})")
    print(f"  动作维度: {rl_env.action_dim} (期望: {expected_action_dim})")
    print(f"  动作映射数量: {len(rl_env.action_map)}")

    assert rl_env.state_dim == expected_state_dim, \
        f"状态维度不匹配: {rl_env.state_dim} != {expected_state_dim}"
    assert rl_env.action_dim == expected_action_dim, \
        f"动作维度不匹配: {rl_env.action_dim} != {expected_action_dim}"
    assert len(rl_env.action_map) == expected_action_dim, \
        f"动作映射数量不匹配: {len(rl_env.action_map)} != {expected_action_dim}"

    print("  [PASS] 环境创建与维度正确\n")
    return rl_env


def test_action_map():
    """测试2: 动作映射表完整性"""
    print("=" * 60)
    print("测试2: 动作映射表完整性")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20)

    # 动作0: 保持
    action0 = rl_env._decode_action(0)
    assert action0 == ('stay', None, 1.0), f"动作0映射错误: {action0}"
    print(f"  动作  0: {action0}")

    # 动作1: 切换到基站0, 全速率
    action1 = rl_env._decode_action(1)
    assert action1 == ('switch', 0, 1.0), f"动作1映射错误: {action1}"
    print(f"  动作  1: {action1}")

    # 动作2: 切换到基站0, 80%速率
    action2 = rl_env._decode_action(2)
    assert action2 == ('switch', 0, 0.8), f"动作2映射错误: {action2}"
    print(f"  动作  2: {action2}")

    # 动作3: 切换到基站0, 60%速率
    action3 = rl_env._decode_action(3)
    assert action3 == ('switch', 0, 0.6), f"动作3映射错误: {action3}"
    print(f"  动作  3: {action3}")

    # 动作4: 切换到基站1, 全速率
    action4 = rl_env._decode_action(4)
    assert action4 == ('switch', 1, 1.0), f"动作4映射错误: {action4}"
    print(f"  动作  4: {action4}")

    # 最后一个动作
    last_action = rl_env._decode_action(rl_env.action_dim - 1)
    expected = ('switch', 7, 0.6)
    assert last_action == expected, f"最后一个动作映射错误: {last_action} != {expected}"
    print(f"  动作{rl_env.action_dim - 1:2d}: {last_action}")

    # 越界动作
    out_action = rl_env._decode_action(999)
    assert out_action == ('stay', None, 1.0), f"越界动作应返回 stay"
    print(f"  越界动作: {out_action}")

    print("  [PASS] 动作映射表完整正确\n")


def test_reset():
    """测试3: reset 接口"""
    print("=" * 60)
    print("测试3: reset 接口")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20)

    state = rl_env.reset()

    assert isinstance(state, np.ndarray), f"state 类型错误: {type(state)}"
    assert state.shape == (rl_env.state_dim,), \
        f"state 形状错误: {state.shape} != ({rl_env.state_dim},)"
    assert np.all(np.isfinite(state)), "state 包含非有限值"
    assert np.all(state >= 0), f"state 应全部非负，但发现负值: min={state.min()}"

    # 检查各段状态范围
    n_bs = 8
    sinr = state[:n_bs]
    loads = state[n_bs:2*n_bs]
    biz = state[2*n_bs:2*n_bs+3]
    velocity = state[2*n_bs+3]
    satisfaction = state[2*n_bs+4]
    connected = state[2*n_bs+5]

    print(f"  SINR 范围: [{sinr.min():.4f}, {sinr.max():.4f}]")
    print(f"  负载率范围: [{loads.min():.4f}, {loads.max():.4f}]")
    print(f"  业务 one-hot: {biz} (sum={biz.sum():.1f})")
    print(f"  速度(归一化): {velocity:.4f}")
    print(f"  满意度: {satisfaction:.4f}")
    print(f"  连接状态: {connected:.1f}")

    assert np.all(sinr >= 0) and np.all(sinr <= 1), "SINR 归一化范围错误"
    assert np.all(loads >= 0) and np.all(loads <= 1), "负载率范围错误"
    assert abs(biz.sum() - 1.0) < 1e-6, "业务 one-hot 编码错误"
    assert satisfaction >= 0 and satisfaction <= 1, "满意度范围错误"
    assert connected in (0.0, 1.0), "连接状态应为 0 或 1"

    # 多次 reset 一致性（同种子应相同）
    set_global_seed(42)
    rl_env2 = create_rl_env(num_bs=8, num_uav=20)
    state2 = rl_env2.reset()
    assert np.allclose(state, state2), "同种子 reset 结果应相同"

    print("  [PASS] reset 接口正确\n")


def test_step():
    """测试4: step 接口"""
    print("=" * 60)
    print("测试4: step 接口")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20, max_steps=150)
    state = rl_env.reset()

    # 测试 stay 动作
    print("  --- 动作: stay ---")
    next_state, reward, done, info = rl_env.step(0)
    assert isinstance(next_state, np.ndarray), "next_state 类型错误"
    assert next_state.shape == (rl_env.state_dim,), "next_state 形状错误"
    assert isinstance(reward, float), f"reward 类型错误: {type(reward)}"
    assert isinstance(done, bool), f"done 类型错误: {type(done)}"
    assert isinstance(info, dict), f"info 类型错误: {type(info)}"
    assert done == False, "第1步不应结束"
    print(f"    reward={reward:.4f}, done={done}, satisfaction={info['satisfaction']:.4f}")

    # 测试切换动作（切换到基站0，全速率）
    print("  --- 动作: switch to BS0 @ 100% ---")
    next_state2, reward2, done2, info2 = rl_env.step(1)
    assert isinstance(next_state2, np.ndarray)
    assert isinstance(reward2, float)
    assert done2 == False
    print(f"    reward={reward2:.4f}, done={done2}, connected_bs={info2['connected_bs']}")

    # 测试切换动作（切换到基站1，80%速率）
    print("  --- 动作: switch to BS1 @ 80% ---")
    next_state3, reward3, done3, info3 = rl_env.step(5)  # 1 + 1*3 + 1 = 5
    assert isinstance(next_state3, np.ndarray)
    assert isinstance(reward3, float)
    print(f"    reward={reward3:.4f}, done={done3}, connected_bs={info3['connected_bs']}")

    # 检查 info 字段
    expected_keys = {'step', 'satisfaction', 'connected_bs', 'allocated_rate',
                     'handover_count', 'total_handovers', 'action_type'}
    assert expected_keys.issubset(set(info3.keys())), \
        f"info 缺少字段: {expected_keys - set(info3.keys())}"

    print("  [PASS] step 接口正确\n")


def test_full_episode():
    """测试5: 完整 episode 跑通"""
    print("=" * 60)
    print("测试5: 完整 episode (随机策略, 150步)")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20, max_steps=150)
    state = rl_env.reset()

    total_reward = 0.0
    rewards = []
    satisfactions = []
    handovers = 0

    for step_i in range(150):
        action = np.random.randint(0, rl_env.action_dim)
        state, reward, done, info = rl_env.step(action)
        total_reward += reward
        rewards.append(reward)
        satisfactions.append(info['satisfaction'])
        if step_i == 0 or step_i % 30 == 29:
            rl_env.render()

        if done:
            assert step_i == 149, f"应在第149步结束，实际在第{step_i}步"
            break

    assert done == True, "150步后应结束"
    assert len(rewards) == 150, f"奖励记录数错误: {len(rewards)}"

    print(f"\n  --- Episode 统计 ---")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  平均奖励: {np.mean(rewards):.4f}")
    print(f"  奖励标准差: {np.std(rewards):.4f}")
    print(f"  平均满意度: {np.mean(satisfactions):.4f}")
    print(f"  最终满意度: {satisfactions[-1]:.4f}")
    print(f"  切换次数: {info['total_handovers']}")
    print(f"  [PASS] 完整 episode 跑通\n")

    return total_reward, np.mean(satisfactions)


def test_multi_episode():
    """测试6: 多 episode 连续运行"""
    print("=" * 60)
    print("测试6: 多 episode 连续运行 (5 episodes)")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20, max_steps=50)

    episode_rewards = []
    episode_sats = []

    for ep in range(5):
        state = rl_env.reset()
        total_reward = 0.0
        sat_sum = 0.0

        for step_i in range(50):
            action = np.random.randint(0, rl_env.action_dim)
            state, reward, done, info = rl_env.step(action)
            total_reward += reward
            sat_sum += info['satisfaction']

            if done:
                break

        avg_sat = sat_sum / 50
        episode_rewards.append(total_reward)
        episode_sats.append(avg_sat)
        print(f"  Episode {ep}: reward={total_reward:.2f}, avg_sat={avg_sat:.4f}")

    print(f"\n  平均奖励: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  平均满意度: {np.mean(episode_sats):.4f} +/- {np.std(episode_sats):.4f}")
    print(f"  [PASS] 多 episode 运行稳定\n")


def test_reward_signs():
    """测试7: 奖励信号合理性"""
    print("=" * 60)
    print("测试7: 奖励信号合理性检查")
    print("=" * 60)

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20, max_steps=100)
    rl_env.reset()

    # 收集 stay 和 switch 动作的奖励
    stay_rewards = []
    switch_rewards = []

    for step_i in range(100):
        # 交替使用 stay 和 switch
        if step_i % 2 == 0:
            action = 0  # stay
        else:
            action = np.random.randint(1, rl_env.action_dim)  # switch
            action_type = rl_env._decode_action(action)[0]
            if action_type != 'switch':
                action = 1  # 强制 switch

        state, reward, done, info = rl_env.step(action)
        if done:
            break

        if action == 0:
            stay_rewards.append(reward)
        else:
            switch_rewards.append(reward)

    if stay_rewards:
        print(f"  Stay 动作 ({len(stay_rewards)}次): "
              f"均值={np.mean(stay_rewards):.4f}, 范围=[{np.min(stay_rewards):.4f}, {np.max(stay_rewards):.4f}]")
    if switch_rewards:
        print(f"  Switch 动作 ({len(switch_rewards)}次): "
              f"均值={np.mean(switch_rewards):.4f}, 范围=[{np.min(switch_rewards):.4f}, {np.max(switch_rewards):.4f}]")

    # switch 动作应有切换惩罚，平均奖励通常低于 stay
    if stay_rewards and switch_rewards:
        print(f"  Switch 平均奖励 < Stay 平均奖励: "
              f"{np.mean(switch_rewards) < np.mean(stay_rewards)}")
        print("  (这是预期的，因为 switch 有额外惩罚)")

    print(f"  [PASS] 奖励信号合理\n")


def test_speed():
    """测试8: 仿真速度基准"""
    print("=" * 60)
    print("测试8: 仿真速度基准")
    print("=" * 60)

    import time

    set_global_seed(42)
    rl_env = create_rl_env(num_bs=8, num_uav=20, max_steps=150, skip_recognition=True)
    rl_env.reset()

    # 预热
    for _ in range(10):
        rl_env.step(0)

    # 计时
    t_start = time.time()
    for step_i in range(150):
        action = np.random.randint(0, rl_env.action_dim)
        state, reward, done, info = rl_env.step(action)
        if done:
            break
    t_elapsed = time.time() - t_start

    print(f"  1 episode (150步) 耗时: {t_elapsed:.2f} 秒")
    print(f"  平均每步: {t_elapsed / 150 * 1000:.1f} ms")
    print(f"  3000 episodes 预估: {t_elapsed * 3000 / 60:.1f} 分钟")
    print(f"  5000 episodes 预估: {t_elapsed * 5000 / 60:.1f} 分钟")
    print(f"  [PASS] 速度测试完成\n")

    return t_elapsed


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  RLHandoverEnv 单元测试")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    tests = [
        ("环境创建与维度", test_env_creation),
        ("动作映射表", test_action_map),
        ("reset 接口", test_reset),
        ("step 接口", test_step),
        ("完整 episode", test_full_episode),
        ("多 episode 运行", test_multi_episode),
        ("奖励信号合理性", test_reward_signs),
        ("仿真速度基准", test_speed),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    print("=" * 60)
