"""
快速修复脚本：修正MAPPO切换成功率从硬编码1.0改为真实计算值
===========================================================

问题定位：
  uav_system/experiments.py 第124行硬编码了 handover_success_rate = 1.0
  
环境层实际有完整的切换统计（通过info返回）：
  - switch_attempts: 切换尝试次数
  - switch_success: 成功切换次数（分配成功）
  - switch_rollback: 回滚成功次数（分配失败但保持原连接）
  - switch_disconnect: 断连次数（分配失败且回滚也失败）

正确公式（与传统算法一致）：
  HOSR = switch_success / switch_attempts
  
使用方法：
  python fix_mappo_hosr.py
  
输出：
  - 修正后的MAPPO 10次评估数据（打印到终端）
  - 更新 experiment_results/exp3_data.json
"""

import sys
import os
import json
import pickle
import numpy as np

# 确保项目目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 第一步：修改 evaluate_mappo_in_experiment 的核心逻辑
# ============================================================
# 我们不直接修改原文件（避免影响其他实验），
# 而是创建一个修正版的评估函数，只改HOSR计算部分

def patched_evaluate_mappo(num_bs=8, num_uav=300, num_steps=200,
                           recognition_model=None, scaler=None,
                           seed=42, scenario='default',
                           model_path=None):
    """
    修正版MAPPO评估函数：handover_success_rate 使用真实计算值而非硬编码1.0
    """
    from uav_system.mappo_environment import MultiAgentHandoverEnv
    from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
    from uav_system.config import RESULT_DIR
    import torch
    
    if model_path is None:
        model_path = os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav.pt')
    
    if not os.path.exists(model_path):
        print(f"  [MAPPO] 模型文件不存在: {model_path}")
        return None
    
    # 创建评估环境
    env = MultiAgentHandoverEnv(
        num_bs=num_bs, num_uav=num_uav,
        max_steps=num_steps, seed=seed,
        bs_capacity_range=(500, 1000), pos_range=1000,
        recognition_model=recognition_model,
        scaler=scaler,
        event_probability=0.05,
    )
    obs_dict, global_state = env.reset()
    
    # 初始化agent并加载模型
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
    )
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
    except Exception as e:
        print(f"  [MAPPO] 模型加载失败: {e}")
        return None
    
    # ========== 关键修改：收集切换统计 ==========
    all_sats = []
    all_connected_rates = []
    
    # 新增：累计切换统计
    total_switch_attempts = 0
    total_switch_success = 0
    total_switch_rollback = 0
    total_switch_disconnect = 0
    
    for step in range(num_steps):
        biz_types = {uid: env.env.uavs[uid].true_business_type.value 
                     for uid in range(env.num_agents)}
        actions, _, _, _, _ = agent.select_actions(
            obs_dict, global_state, biz_types=biz_types, training=False)
        obs_dict, global_state, rewards, team_reward, done, info = env.step(actions)
        
        all_sats.append(info['avg_satisfaction'])
        all_connected_rates.append(info['connected_rate'])
        
        # 收集每步的切换统计（注意：嵌套在 reward_diag 字典中！）
        diag = info.get('reward_diag', {})
        total_switch_attempts += diag.get('switch_attempts', 0)
        total_switch_success += diag.get('switch_success', 0)
        total_switch_rollback += diag.get('switch_rollback', 0)
        total_switch_disconnect += diag.get('switch_disconnect', 0)
    
    # 构建结果字典
    final_sats = [env.env.uavs[uid].current_satisfaction 
                  for uid in range(env.num_agents)]
    connected_count = sum(1 for uid in range(env.num_agents) 
                          if env.env.uavs[uid].connected_bs_id is not None)
    total_ho = sum(env.env.uavs[uid].handover_count 
                   for uid in range(env.num_agents))
    
    # ========== 核心修正：真实HOSR计算 ==========
    if total_switch_attempts > 0:
        real_hosr = total_switch_success / total_switch_attempts
    else:
        real_hosr = 1.0  # 无切换尝试则默认完美
    
    stats = {
        'avg_satisfaction': np.mean(final_sats),
        'critical_satisfaction': np.mean([s for i, s in enumerate(final_sats)
                                          if env.env.uavs[i].true_business_type.value == 0]),
        'weighted_satisfaction': np.mean(final_sats),
        'connected_count': connected_count,
        'connected_ratio': connected_count / max(env.num_agents, 1),
        'total_throughput': sum(env.env.uavs[uid].current_allocated_rate 
                               for uid in range(env.num_agents)
                               if env.env.uavs[uid].connected_bs_id is not None),
        'handover_success_rate': real_hosr,  # ← 修正点！
        'avg_switching_latency_ms': np.mean(env._communication_metrics.get('handover_latencies', [0])),
        'load_variance': np.var([bs.load_ratio for bs in env.env.base_stations.values()]),
        'avg_sinr': np.mean(env.env.sinr_matrix[:env.num_agents, :num_bs]),
        '_algorithm': 'MAPPO',
        # 额外保存原始统计数据供审查
        '_switch_attempts': total_switch_attempts,
        '_switch_success': total_switch_success,
        '_switch_rollback': total_switch_rollback,
        '_switch_disconnect': total_switch_disconnect,
    }
    
    print(f"  [MAPPO seed={seed}] 满足率={stats['avg_satisfaction']:.3f}, "
          f"连接率={stats['connected_ratio']*100:.1f}%, "
          f"HOSR={real_hosr*100:.1f}% "
          f"(成功={total_switch_success}/尝试={total_switch_attempts}, "
          f"回滚={total_switch_rollback}, 断连={total_switch_disconnect})")
    
    return stats


def main():
    print("=" * 70)
    print("  MAPPO切换成功率修复脚本")
    print("  问题: experiments.py 第124行硬编码 handover_success_rate=1.0")
    print("  修复: 从环境的switch统计真实计算")
    print("=" * 70)
    
    # 原始实验3使用的种子（必须一致才能对比）
    exp3_seeds = [42, 123, 456, 789, 1024, 
                  2024, 3030, 4050, 5060, 6070]
    
    print(f"\n开始重跑MAPPO评估 ({len(exp3_seeds)}次，种子: {exp3_seeds})...\n")
    
    mappo_results = []
    for i, seed in enumerate(exp3_seeds):
        result = patched_evaluate_mappo(
            num_bs=8, num_uav=300, num_steps=200,
            seed=seed, scenario='exp3_main'
        )
        if result:
            mappo_results.append(result)
        print(f"  进度: {i+1}/{len(exp3_seeds)} 完成\n")
    
    if not mappo_results:
        print("错误：所有MAPPO评估都失败了！")
        return
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("  修正后MAPPO结果汇总")
    print("=" * 70)
    
    hosrs = [r['handover_success_rate'] for r in mappo_results]
    sats = [r['avg_satisfaction'] for r in mappo_results]
    conn_rates = [r['connected_ratio'] for r in mappo_results]
    
    print(f"\n  切换成功率(HOSR):")
    print(f"    原始(硬编码): 100.0% +/- 0.0%  [假值!]")
    print(f"    修正后(真实):   {np.mean(hosrs)*100:.1f}% +/- {np.std(hosrs)*100:.1f}%  [OK]")
    print(f"    范围: [{min(hosrs)*100:.1f}%, {max(hosrs)*100:.1f}%]")
    
    print(f"\n  各次详细HOSR: {[f'{h*100:.1f}%' for h in hosrs]}")
    
    print(f"\n  其他指标（应与原始数据一致）:")
    print(f"    平均满意度: {np.mean(sats):.3f} ± {np.std(sats):.3f}")
    print(f"    连接率:      {np.mean(conn_rates)*100:.1f}% ± {np.std(conn_rates)*100:.1f}%")
    
    # 显示切换统计详情
    attempts = [r['_switch_attempts'] for r in mappo_results]
    successes = [r['_switch_success'] for r in mappo_results]
    rollbacks = [r['_switch_rollback'] for r in mappo_results]
    disconnects = [r['_switch_disconnect'] for r in mappo_results]
    
    print(f"\n  切换统计明细:")
    print(f"    总尝试次数(10次合计): {sum(attempts)}")
    print(f"    总成功次数(10次合计): {sum(successes)}")
    print(f"    总回滚次数(10次合计): {sum(rollbacks)}")
    print(f"    总断连次数(10次合计): {sum(disconnects)}")
    
    if sum(attempts) > 0:
        overall_hosr = sum(successes) / sum(attempts)
        print(f"    综合HOSR(合并计算):  {overall_hosr*100:.1f}%")
    
    # ============================================================
    # 第二步：更新 exp3_data.json
    # ============================================================
    exp3_json_path = os.path.join('experiment_results', 'exp3_data.json')
    
    if os.path.exists(exp3_json_path):
        with open(exp3_json_path, 'r', encoding='utf-8') as f:
            exp3_data = json.load(f)
        
        # 备份原始数据
        backup_path = exp3_json_path + '.bak_before_hosr_fix'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(exp3_data, f, ensure_ascii=False, indent=2)
        print(f"\n  已备份原始数据到: {backup_path}")
        
        # 更新MAPPO的handover_success_rate
        old_hosr = 1.0
        new_hosr_mean = float(np.mean(hosrs))
        new_hosr_std = float(np.std(hosrs))
        
        # 遍历所有seed的数据更新HOSR
        updated_count = 0
        for seed_str, seed_data in exp3_data.items():
            if 'MAPPO' in seed_data:
                # 找到对应的结果索引
                idx = exp3_seeds.index(int(seed_str)) if int(seed_str) in exp3_seeds else None
                if idx is not None and idx < len(mappo_results):
                    old_val = seed_data['MAPPO'].get('handover_success_rate', 'N/A')
                    seed_data['MAPPO']['handover_success_rate'] = mappo_results[idx]['handover_success_rate']
                    # 同时更新辅助字段
                    seed_data['MAPPO']['_switch_attempts'] = mappo_results[idx]['_switch_attempts']
                    seed_data['MAPPO']['_switch_success'] = mappo_results[idx]['_switch_success']
                    seed_data['MAPPO']['_switch_rollback'] = mappo_results[idx]['_switch_rollback']
                    seed_data['MAPPO']['_switch_disconnect'] = mappo_results[idx]['_switch_disconnect']
                    updated_count += 1
        
        # 保存更新后的数据
        with open(exp3_json_path, 'w', encoding='utf-8') as f:
            json.dump(exp3_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n  [OK] 已更新 exp3_data.json:")
        print(f"     MAPPO HOSR: {old_hosr*100:.0f}% → {new_hosr_mean*100:.1f}% ± {new_hosr_std*100:.1f}%")
        print(f"     更新了 {updated_count} 个seed的数据")
    else:
        print(f"\n  [WARN] 未找到 {exp3_json_path}，跳过文件更新")
    
    # ============================================================
    # 第三步：同时更新 pkl 文件（如果存在）
    # ============================================================
    exp3_pkl_path = os.path.join('experiment_results', 'exp3_data.pkl')
    if os.path.exists(exp3_pkl_path):
        with open(exp3_pkl_path, 'rb') as f:
            exp3_pkl_data = pickle.load(f)
        
        # 同样更新
        for key in exp3_pkl_data:
            if isinstance(exp3_pkl_data[key], dict) and 'MAPPO' in exp3_pkl_data[key]:
                idx = None
                try:
                    int_key = int(key)
                    if int_key in exp3_seeds:
                        idx = exp3_seeds.index(int_key)
                except:
                    pass
                
                if idx is not None and idx < len(mappo_results):
                    exp3_pkl_data[key]['MAPPO']['handover_success_rate'] = \
                        mappo_results[idx]['handover_success_rate']
        
        with open(exp3_pkl_path, 'wb') as f:
            pickle.dump(exp3_pkl_data, f)
        print(f"  [OK] 已同步更新 exp3_data.pkl")
    
    print("\n" + "=" * 70)
    print("  修复完成！")
    print(f"  MAPPO真实切换成功率: {np.mean(hosrs)*100:.1f}% ± {np.std(hosrs)*100:.1f}%")
    print("=" * 70)
    
    # 返回结果供后续使用
    return {
        'hosr_mean': np.mean(hosrs),
        'hosr_std': np.std(hosrs),
        'hosrs': hosrs,
        'results': mappo_results
    }


if __name__ == '__main__':
    result = main()
