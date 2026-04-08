# -*- coding: utf-8 -*-
"""
动作索引越界修复测试

专注于修复动作索引越界的问题，确保small实验能够正常运行

Author: Action Index Fixer
Date: 2026-04-08
"""

import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED
from uav_system.qmix_environment import QMixHandoverEnv
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm


def test_action_index_fix():
    """测试动作索引越界修复"""
    print("\n" + "="*60)
    print("ACTION INDEX FIX TEST")
    print("="*60)

    # 配置
    num_bs = 4
    num_uav = 10
    num_steps = 50

    # 创建环境
    env = QMixHandoverEnv(
        num_bs=num_bs,
        num_uav=num_uav,
        max_steps=num_steps,
        pos_range=1000,
    )

    print(f"环境创建成功: {num_bs} BS, {num_uav} UAV")
    print(f"Action dim: {env.action_dim}")

    # 初始化算法
    traditional_algorithm = IntegratedHandoverAlgorithm(env.env)
    enhanced_algorithm = EnhancedHandoverAlgorithm(env.env)

    # 测试传统算法
    print("\n测试传统算法...")
    try:
        obs_dict, global_state = env.reset()
        for step in range(num_steps):
            # 使用传统算法
            traditional_algorithm.run_step(enable_load_balancing=True)
            actions = {}
            for uid in range(env.num_agents):
                uav = env.env.uavs[uid]
                best_bs = traditional_algorithm.get_best_base_station(uav)
                if best_bs == uav.connected_bs_id:
                    actions[uid] = 0  # stay
                else:
                    # 找到对应的动作索引
                    action = best_bs + 1  # 1-based
                    # 确保动作索引在有效范围内
                    if action >= env.action_dim:
                        action = env.action_dim - 1
                    actions[uid] = action

            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            obs_dict = next_obs

            if step % 10 == 0:
                print(f"  Step {step+1}/{num_steps}")

        print("✓ 传统算法测试成功")
    except Exception as e:
        print(f"✗ 传统算法测试失败: {e}")

    # 测试增强算法
    print("\n测试增强算法...")
    try:
        obs_dict, global_state = env.reset()
        for step in range(num_steps):
            # 使用增强算法
            enhanced_algorithm.run_step(enable_load_balancing=True)
            actions = {}
            for uid in range(env.num_agents):
                uav = env.env.uavs[uid]
                best_bs = enhanced_algorithm.get_best_base_station(uav)
                if best_bs == uav.connected_bs_id:
                    actions[uid] = 0  # stay
                else:
                    # 找到对应的动作索引
                    action = best_bs + 1  # 1-based
                    # 确保动作索引在有效范围内
                    if action >= env.action_dim:
                        action = env.action_dim - 1
                    actions[uid] = action

            # 执行动作
            next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
            obs_dict = next_obs

            if step % 10 == 0:
                print(f"  Step {step+1}/{num_steps}")

        print("✓ 增强算法测试成功")
    except Exception as e:
        print(f"✗ 增强算法测试失败: {e}")

    # 计算最终满意度
    total_sat = 0.0
    for uid in range(env.num_agents):
        uav = env.env.uavs[uid]
        total_sat += uav.current_satisfaction
    avg_sat = total_sat / env.num_agents

    print(f"\n最终平均满意度: {avg_sat:.3f}")
    print("\n测试完成！")


def main():
    """主函数"""
    set_global_seed(GLOBAL_SEED)
    test_action_index_fix()


if __name__ == "__main__":
    main()
