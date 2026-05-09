#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试 curriculum_learning.py 的关键功能
验证: 模型加载 + Agent动态重建
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """测试1: 验证模型文件存在且可加载"""
    print("="*70)
    print("  TEST 1: 模型文件检查")
    print("="*70)
    
    model_path = 'experiment_results/mappo_models/mappo_8bs_300uav_best.pt'
    
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"  [OK] 模型文件存在: {model_path}")
        print(f"       大小: {size_kb:.1f} KB")
        
        import torch
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"  [OK] 模型加载成功!")
            print(f"       Keys: {list(checkpoint.keys())[:5]}")
            
            if 'actor' in checkpoint:
                actor_params = sum(p.numel() for p in checkpoint['actor'].values())
                print(f"       Actor参数量: {actor_params:,}")
            
            return True
        except Exception as e:
            print(f"  [FAIL] 加载失败: {e}")
            return False
    else:
        print(f"  [FAIL] 模型不存在: {model_path}")
        return False


def test_agent_rebuild():
    """测试2: 验证Agent动态重建逻辑"""
    print("\n" + "="*70)
    print("  TEST 2: Agent重建逻辑 (模拟)")
    print("="*70)
    
    from uav_system.mappo_environment import MultiAgentHandoverEnv
    from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
    
    # 创建两个不同UAV数量的环境
    env_300 = MultiAgentHandoverEnv(
        num_bs=8, num_uav=300, max_steps=100,
        seed=42, bs_capacity_range=(500, 1000), pos_range=1000,
    )
    
    env_350 = MultiAgentHandoverEnv(
        num_bs=8, num_uav=350, max_steps=100,
        seed=43, bs_capacity_range=(500, 1000), pos_range=1000,
    )
    
    print(f"\n  环境1: {env_300.num_agents} UAVs")
    print(f"  环境2: {env_350.num_agents} UAVs")
    
    # 用环境1初始化Agent
    agent = MAPPOAgent(
        num_agents=env_300.num_agents,
        obs_dim=env_300.obs_dim,
        state_dim=env_300.state_dim,
        action_dim=env_300.action_dim,
        hidden_dim=64, critic_hidden_dim=128,
        actor_lr=3e-04, critic_lr=1e-03,
        gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
        entropy_coef=0.008, value_coef=0.5,
        rollout_length=100, num_epochs=5, batch_size=64,
        use_biz_heads=True, use_attention_critic=True,
        use_hierarchical=True, use_transformer=False,
        use_data_augmentation=True,
    )
    
    print(f"\n  Agent初始化: num_agents={agent.num_agents} (匹配环境1)")
    
    # 加载模型
    model_path = 'experiment_results/mappo_models/mappo_8bs_300uav_best.pt'
    if os.path.exists(model_path):
        agent.load(model_path, reset_optimizer=True)
        print(f"  [OK] 模型加载成功")
    
    # 测试在环境1中工作
    obs_dict, global_state = env_300.reset()
    agent.reset_hidden()
    
    try:
        biz_types = {uid: 0 for uid in range(env_300.num_agents)}
        actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types=biz_types)
        print(f"  [OK] 环境1(300UAV): select_actions成功! 动作数={len(actions)}")
    except Exception as e:
        print(f"  [FAIL] 环境1(300UAV): {e}")
        return False
    
    # 模拟切换到环境2 (需要重建)
    print(f"\n  [*] 模拟场景切换: 环境1(300) -> 环境2(350)")
    
    if agent.num_agents != env_350.num_agents:
        print(f"  [DETECT] UAV数量不匹配: agent({agent.num_agents}) != env({env_350.num_agents})")
        print(f"  [*] 触发Agent重建...")
        
        # 保存当前权重
        temp_path = 'test_temp.pt'
        agent.save(temp_path)
        
        # 重建agent
        new_agent = MAPPOAgent(
            num_agents=env_350.num_agents,
            obs_dim=env_350.obs_dim,
            state_dim=env_350.state_dim,
            action_dim=env_350.action_dim,
            hidden_dim=64, critic_hidden_dim=128,
            actor_lr=3e-04, critic_lr=1e-03,
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            entropy_coef=0.008, value_coef=0.5,
            rollout_length=100, num_epochs=5, batch_size=64,
            use_biz_heads=True, use_attention_critic=True,
            use_hierarchical=True, use_transformer=False,
            use_data_augmentation=True,
        )
        
        # 加载权重
        new_agent.load(temp_path, reset_optimizer=False)
        
        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        agent = new_agent
        print(f"  [OK] Agent重建完成: num_agents={agent.num_agents}")
    
    # 测试在环境2中工作
    obs_dict, global_state = env_350.reset()
    agent.reset_hidden()
    
    try:
        biz_types = {uid: 0 for uid in range(env_350.num_agents)}
        actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types=biz_types)
        print(f"  [OK] 环境2(350UAV): select_actions成功! 动作数={len(actions)}")
        return True
    except Exception as e:
        print(f"  [FAIL] 环境2(350UAV): {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("  Curriculum Learning v3.0 - 快速功能验证")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: 模型加载
    results.append(("模型加载", test_model_loading()))
    
    # Test 2: Agent重建
    results.append(("Agent重建", test_agent_rebuild()))
    
    # 汇总
    print("\n" + "="*70)
    print("  测试结果汇总")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + ("="*70))
    if all_passed:
        print("  [SUCCESS] 所有关键功能正常! 可以开始训练")
        print("\n  运行命令:")
        print("    .\\venv\\Scripts\\python.exe curriculum_learning.py --mode quick")
    else:
        print("  [FAIL] 存在问题，需要修复")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
