# -*- coding: utf-8 -*-
"""
简单环境测试

验证环境基本功能和动作索引越界修复

Author: Simple Tester
Date: 2026-04-08
"""

import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.mappo_environment import MultiAgentHandoverEnv


def test_environment():
    """测试环境基本功能"""
    print("\n" + "="*60)
    print("SIMPLE ENVIRONMENT TEST")
    print("="*60)

    # 配置
    num_bs = 4
    num_uav = 10
    num_steps = 10

    # 创建环境
    env = MultiAgentHandoverEnv(
        num_bs=num_bs,
        num_uav=num_uav,
        max_steps=num_steps,
        pos_range=1000,
    )

    print("环境创建成功: %d BS, %d UAV" % (num_bs, num_uav))
    print("Action dim: %d" % env.action_dim)

    # 测试 reset
    print("\n测试 reset...")
    obs_dict, global_state = env.reset()
    print("reset 成功")
    print("观测维度: %d" % len(obs_dict[0]))
    print("全局状态维度: %d" % len(global_state))

    # 测试 step
    print("\n测试 step...")
    try:
        # 生成随机动作 (0-5)
        actions = {}
        for uid in range(env.num_agents):
            # 随机动作，确保在有效范围内
            action = np.random.randint(0, env.action_dim)
            actions[uid] = action

        print("生成动作: %s" % str(actions))

        # 执行动作
        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
        print("step 成功")
        print("团队奖励: %.2f" % team_reward)

        # 测试多次 step
        for step in range(1, num_steps):
            # 生成随机动作
            actions = {}
            for uid in range(env.num_agents):
                action = np.random.randint(0, env.action_dim)
                actions[uid] = action

            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            
            if step % 3 == 0:
                print("Step %d: 奖励=%.2f" % (step+1, team_reward))

        print("\n✓ 所有 step 测试成功")

        # 计算最终满意度
        total_sat = 0.0
        for uid in range(env.num_agents):
            uav = env.env.uavs[uid]
            total_sat += uav.current_satisfaction
        avg_sat = total_sat / env.num_agents

        print("最终平均满意度: %.3f" % avg_sat)

    except Exception as e:
        print("\n✗ 测试失败: %s" % str(e))

    print("\n测试完成！")


def main():
    """主函数"""
    set_global_seed(GLOBAL_SEED)
    test_environment()


if __name__ == "__main__":
    main()
