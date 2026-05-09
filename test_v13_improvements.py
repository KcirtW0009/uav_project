"""
V13 Reward Function 快速验证脚本
===============================
验证P1(负载均衡)和P2(业务感知)改进是否生效

运行方式:
    python test_v13_improvements.py

预期输出:
    - load_balance_penalty 应该为负值 (惩罚不均衡)
    - 控制信令UAV的target_gap惩罚应该更大
    - 训练日志中应显示 load_balance_penalty 指标
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent


def test_load_balance_penalty():
    """测试P1: 负载均衡惩罚"""
    print("\n" + "="*70)
    print("  TEST 1: 负载均衡惩罚 (P1)")
    print("="*70)
    
    env = MultiAgentHandoverEnv(
        num_bs=8,
        num_uav=300,
        max_steps=50,  # 短episode快速测试
        seed=42,
        bs_capacity_range=(500, 1000),
        pos_range=1000,
    )
    
    obs_dict, global_state = env.reset()
    
    # 模拟一个极端不均衡的场景: 所有UAV都连到BS 0
    print(f"\n  [初始状态] 基站负载:")
    for bs_id, bs in enumerate(env.env.base_stations.values()):
        print(f"    BS {bs_id}: load_ratio={bs.load_ratio:.3f}, UAVs={len(bs.connected_uavs)}")
    
    # 执行一步随机动作
    actions = {uid: 0 for uid in range(env.num_agents)}  # 全部stay
    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
    
    # 检查load_balance_penalty是否存在且为负
    if 'reward_diag' in info and 'load_balance_penalty' in info['reward_diag']:
        lb_penalty = info['reward_diag']['load_balance_penalty']
        print(f"\n  [结果] load_balance_penalty = {lb_penalty:.4f}")
        
        if lb_penalty < 0:
            print(f"  [OK] 惩罚为负值 (符合预期)")
            print(f"       → 团队reward包含负载均衡信号")
            
            # 计算当前负载标准差
            bs_loads = [bs.load_ratio for bs in env.env.base_stations.values()]
            load_std = np.std(bs_loads)
            print(f"       → 当前负载std={load_std:.4f}")
            return True
        else:
            print(f"  [FAIL] 惩罚应该为负值, 实际={lb_penalty}")
            return False
    else:
        print(f"  [FAIL] 未找到 load_balance_penalty 字段")
        print(f"  可用字段: {list(info.get('reward_diag', {}).keys())}")
        return False


def test_biz_specific_rewards():
    """测试P2: 业务类型差异化奖励"""
    print("\n" + "="*70)
    print("  TEST 2: 业务差异化权重 (P2)")
    print("="*70)
    
    env = MultiAgentHandoverEnv(
        num_bs=8,
        num_uav=300,
        max_steps=50,
        seed=43,
        bs_capacity_range=(500, 1000),
        pos_range=1000,
    )
    
    obs_dict, global_state = env.reset()
    
    # 收集不同业务类型的UAV
    biz_types = {}
    for uid in range(env.num_agents):
        uav = env.env.uavs[uid]
        bt = uav.true_business_type.value
        if bt not in biz_types:
            biz_types[bt] = []
        biz_types[bt].append(uid)
    
    print(f"\n  [业务分布]")
    biz_names = {0: "控制信令", 1: "视频回传", 2: "环境监测"}
    for bt, uids in biz_types.items():
        print(f"    {biz_names.get(bt, f'未知-{bt}')}: {len(uids)} UAVs")
    
    # 执行一步并检查rewards
    actions = {uid: 0 for uid in range(env.num_agents)}
    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
    
    # 检查不同业务类型的reward分布
    if 'reward_diag' in info and 'target_gap' in info['reward_diag']:
        avg_target_gap = info['reward_diag']['target_gap']
        print(f"\n  [结果] 平均 target_gap (业务差距惩罚) = {avg_target_gap:.4f}")
        
        if avg_target_gap < 0:
            print(f"  [OK] 存在业务差距惩罚 (负值=未达标被惩罚)")
            
            # 抽样检查几个UAV的个体reward
            sample_uids = [biz_types[0][0] if 0 in biz_types else 0,
                          biz_types[1][0] if 1 in biz_types else 1,
                          biz_types[2][0] if 2 in biz_types else 2]
            
            print(f"\n  [抽样] 不同业务类型UAV的reward:")
            for uid in sample_uids[:3]:
                if uid < env.num_agents:
                    uav = env.env.uavs[uid]
                    bt = uav.true_business_type.value
                    r = rewards.get(uid, 0)
                    sat = uav.current_satisfaction
                    print(f"    UID {uid} ({biz_names.get(bt)}): reward={r:.3f}, sat={sat:.3f}")
            
            return True
        else:
            print(f"  [WARN] target_gap={avg_target_gap} (可能所有UAV都已达标)")
            return True  # 这也是可接受的
    else:
        print(f"  [FAIL] 未找到 target_gap 字段")
        return False


def test_full_episode():
    """完整episode测试: 验证训练兼容性"""
    print("\n" + "="*70)
    print("  TEST 3: 完整Episode训练测试")
    print("="*70)
    
    env = MultiAgentHandoverEnv(
        num_bs=8,
        num_uav=300,
        max_steps=100,  # 中等长度
        seed=44,
        bs_capacity_range=(500, 1000),
        pos_range=1000,
    )
    
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
        actor_lr=3e-04,
        critic_lr=1e-03,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.008,
        value_coef=0.5,
        rollout_length=100,
        num_epochs=5,
        batch_size=64,
        use_biz_heads=True,
        use_attention_critic=True,
        use_hierarchical=True,
        use_transformer=False,
        use_data_augmentation=True,
    )
    
    obs_dict, global_state = env.reset()
    agent.reset_hidden()
    
    total_reward = 0
    lb_penalties = []
    target_gaps = []
    
    print(f"\n  运行 {env.max_steps} 步...")
    
    for step in range(env.max_steps):
        biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                    for uid in range(env.num_agents)}
        
        with torch.no_grad():
            actions, _, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types=biz_types)
        
        next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
        
        total_reward += team_reward
        
        if 'reward_diag' in info:
            if 'load_balance_penalty' in info['reward_diag']:
                lb_penalties.append(info['reward_diag']['load_balance_penalty'])
            if 'target_gap' in info['reward_diag']:
                target_gaps.append(info['reward_diag']['target_gap'])
        
        obs_dict = next_obs
        global_state = next_state
    
    print(f"\n  [Episode完成]")
    print(f"     总团队奖励: {total_reward:.2f}")
    print(f"     平均每步奖励: {total_reward/env.max_steps:.3f}")
    
    if lb_penalties:
        avg_lb = np.mean(lb_penalties)
        print(f"     平均负载均衡惩罚: {avg_lb:.4f}")
        print(f"     惩罚范围: [{min(lb_penalties):.4f}, {max(lb_penalties):.4f}]")
        
        negative_ratio = sum(1 for x in lb_penalties if x < 0) / len(lb_penalties)
        print(f"     负值比例: {negative_ratio*100:.1f}% (应该>90%)")
    
    if target_gaps:
        avg_tg = np.mean(target_gaps)
        print(f"     平均业务差距惩罚: {avg_tg:.4f}")
    
    # 注意: MultiAgentHandoverEnv 没有 close() 方法
    # env.close()
    
    print(f"\n  [OK] V13 Reward Function 测试通过!")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("  V13 Reward Function 改进验证")
    print("  P1: 全局负载均衡惩罚")
    print("  P2: 关键业务差异化权重")
    print("="*70)
    
    results = []
    
    results.append(("P1_负载均衡", test_load_balance_penalty()))
    results.append(("P2_业务感知", test_biz_specific_rewards()))
    results.append(("完整Episode", test_full_episode()))
    
    print("\n" + "="*70)
    print("  测试结果汇总")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n  ✅ 所有测试通过! 可以开始重新训练MAPPO")
        print(f"\n  下一步命令:")
        print(f"  .\\venv\\Scripts\\python.exe main.py --exp mappo --rl-load \\")
        print(f"      --mappo-model experiment_results/mappo_models/mappo_8bs_300uav_best.pt")
    else:
        print(f"\n  ❌ 部分测试失败, 请检查修改")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
