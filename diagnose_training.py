"""
终极诊断脚本: 验证 train() 是否真的在更新参数

用途:
1. 加载基线模型
2. 调用一次 train()
3. 对比train()前后的权重
4. 检查梯度、KL、loss等详细信息

运行方式:
    python diagnose_training.py
    
预计耗时: 30秒~1分钟
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.mappo_environment import MultiAgentHandoverEnv
from uav_system.mappo_agent_v2 import MAPPOAgent


def get_weight_snapshot(agent: MAPPOAgent) -> dict:
    """获取当前权重的快照"""
    snapshot = {}
    for name, param in agent.actor.named_parameters():
        snapshot[f"actor_{name}"] = {
            'data': param.data.clone(),
            'norm': torch.norm(param.data).item(),
        }
    for name, param in agent.critic.named_parameters():
        snapshot[f"critic_{name}"] = {
            'data': param.data.clone(),
            'norm': torch.norm(param.data).item(),
        }
    return snapshot


def compare_snapshots(before: dict, after: dict, label: str = "") -> list:
    """对比两个快照，返回变化报告"""
    changes = []
    
    print(f"\n{'='*70}")
    print(f"[COMPARE] {label}")
    print(f"{'='*70}")
    print(f"\n  {'层名称':<40s} │ {'Train前 norm':>12s} │ {'Train后 norm':>12s} │ {'变化%':>8s} │ {'状态'}")
    print(f"  {'-'*40}-┼-{'-'*12}-┼-{'-'*12}-┼-{'-'*8}-┼------")
    
    max_change = 0.0
    total_params_changed = 0
    total_layers = len(before)
    
    for key in sorted(before.keys()):
        if key not in after:
            continue
        
        before_norm = before[key]['norm']
        after_norm = after[key]['norm']
        
        if before_norm > 1e-8:
            change_pct = (after_norm - before_norm) / before_norm * 100
        else:
            change_pct = 0.0
        
        abs_change = abs(change_pct)
        max_change = max(max_change, abs_change)
        
        if abs_change > 1.0:
            status = "✅ 已更新"
            total_params_changed += 1
        elif abs_change > 0.1:
            status = "~ 微小变化"
        else:
            status = "❌ 未变"
        
        layer_name = key.replace("actor_", "").replace("critic_", "")
        print(f"  {layer_name:<40s} │ {before_norm:>12.4f} │ {after_norm:>12.4f} │ {change_pct:>+7.2f}% │ {status}")
        
        changes.append({
            'layer': key,
            'before': before_norm,
            'after': after_norm,
            'change_pct': change_pct,
        })
    
    print(f"\n  [SUMMARY]")
    print(f"     总层数: {total_layers}")
    print(f"     有变化的层 (>1%): {total_params_changed}/{total_layers}")
    print(f"     最大变化幅度: {max_change:.2f}%")
    
    if max_change < 1.0:
        print(f"\n  [🚨 致命] 所有层的变化都 <1%，train()没有真正更新权重!")
    elif max_change < 10.0:
        print(f"\n  [⚠️ 警告] 变化较小 ({max_change:.2f}%)，可能学习率过低")
    else:
        print(f"\n  [✅ 正常] 权重有明显更新 (最大变化 {max_change:.2f}%)")
    
    return changes, max_change


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" " * 25 + "Train() 功能验证工具")
    print(" " * 15 + "(检查训练是否真的更新参数)")
    print("="*70)
    
    # Step 1: 创建环境和Agent
    print("\n[STEP 1] 初始化环境...")
    env = MultiAgentHandoverEnv(
        num_bs=8, num_uav=300,
        max_steps=250,  # 足够长以收集200步数据
        seed=42,
        scenario='industrial_inspection',
    )
    
    obs_dict, global_state = env.reset()
    
    print("[STEP 2] 创建Agent并加载基线模型...")
    agent = MAPPOAgent(
        num_agents=env.num_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=64,
        critic_hidden_dim=128,
    )
    
    # 加载基线模型 (使用默认的 reset_optimizer=True)
    base_model_path = r"experiment_results\mappo_models\mappo_8bs_300uav_best.pt"
    print("\n  [🔧 FIX] 使用修复后的load() (reset_optimizer=True)")
    agent.load(base_model_path)
    
    # 验证学习率是否正确
    current_actor_lr = agent.actor_optimizer.param_groups[0]['lr']
    current_critic_lr = agent.critic_optimizer.param_groups[0]['lr']
    print(f"  [VERIFY] 当前Actor LR: {current_actor_lr:.2e} (应该接近3e-4)")
    print(f"  [VERIFY] 当前Critic LR: {current_critic_lr:.2e} (应该接近1e-3)")
    
    # Step 2: 记录训练前的权重
    print("\n[STEP 3] 记录训练前权重快照...")
    snapshot_before = get_weight_snapshot(agent)
    print(f"  [OK] 已记录 {len(snapshot_before)} 个参数层的快照")
    
    # Step 3: 收集足够的经验数据 (必须 >= rollout_length=150)
    print("\n[STEP 4] 收集训练数据 (运行200步以满足rollout_length要求)...")
    for step in range(200):
        biz_types = {
            uid: env.env.uavs[uid].true_business_type.value
            for uid in range(env.num_agents)
        }
        
        actions, log_probs, values, _, _ = agent.select_actions(
            obs_dict, global_state,
            biz_types=biz_types, training=True
        )
        
        next_obs_dict, next_global_state, rewards, team_reward, done, info = \
            env.step(actions)
        
        # 存储到buffer
        scaled_rewards = {uid: r * 1.0 for uid, r in rewards.items()}
        
        agent.insert_experience(
            step=step,
            obs_dict=obs_dict,
            state=global_state,
            actions=actions,
            rewards=scaled_rewards,
            team_reward=team_reward,
            done=done,
            log_probs=log_probs,
            values=values,
            biz_types=biz_types,
        )
        
        obs_dict = next_obs_dict
        global_state = next_global_state
        
        if done:
            break
    
    buffer_size = len(agent.buffer['obs'])
    print(f"  [OK] 已收集 {buffer_size} 步经验数据")
    
    # Step 4: 调用train()并监控
    print("\n[STEP 5] 调用 agent.train() 并监控...")
    print(f"  Buffer大小: {buffer_size}")
    print(f"  Rollout长度要求: {agent.rollout_length}")
    print(f"  Num epochs: {agent.num_epochs}")
    
    # [KEY] 在train()前后记录详细状态
    print(f"\n  [TRAINING DETAILS]")
    print(f"     Actor优化器学习率: {agent.actor_optimizer.param_groups[0]['lr']:.2e}")
    print(f"     Critic优化器学习率: {agent.critic_optimizer.param_groups[0]['lr']:.2e}")
    
    # 检查是否有梯度历史
    has_grad_history = hasattr(agent, '_current_train_step')
    print(f"     当前训练步数: {agent._current_train_step if has_grad_history else 0}")
    
    # 调用train()
    loss_info = agent.train()
    
    if loss_info is None:
        print(f"\n  [FAIL] train() 返回 None (buffer不足?)")
        print(f"         Buffer大小={buffer_size}, 要求={agent.rollout_length}")
        return
    elif not isinstance(loss_info, dict):
        print(f"\n  [WARN] train() 返回非字典类型: {type(loss_info).__name__}")
        return
    
    print(f"\n  [TRAIN RESULT] train() 成功返回:")
    print(f"     Actor Loss:  {loss_info.get('actor_loss', 'N/A')}")
    print(f"     Critic Loss: {loss_info.get('critic_loss', 'N/A')}")
    print(f"     Entropy:     {loss_info.get('entropy', 'N/A')}")
    print(f"     KL Div:      {loss_info.get('kl_divergence', 'N/A')}")
    print(f"     Value MSE:   {loss_info.get('value_mse', 'N/A')}")
    
    # Step 5: 对比权重
    print("\n[STEP 6] 对比训练前后权重...")
    snapshot_after = get_weight_snapshot(agent)
    
    changes, max_change = compare_snapshots(
        snapshot_before, snapshot_after,
        label="一次 train() 调用的效果"
    )
    
    # Step 6: 最终判断
    print(f"\n\n{'='*70}")
    print(f"[FINAL CONCLUSION]")
    print(f"{'='*70}")
    
    if max_change < 1.0:
        print(f"""
  🚨🚨🚨 确认了! train() 函数存在严重bug!

  证据链:
  1. ✅ train() 返回了正常的Loss值
  2. ❌ 但权重几乎没变 (<1% 变化)
  3. → 结论: Loss计算和参数更新完全脱节!

  可能原因 (按概率排序):
  A. KL Early Stopping 导致所有batch被跳过
     - 检查: approx_kl 是否总是 > kl_threshold?
     
  B. 梯度为None或接近零
     - 检查: actor_grads/critic_grads 是否为空列表?
     
  C. optimizer.step() 未执行
     - 检查: 是否有条件判断跳过了step()?
     
  D. detach() 导致计算图断裂
     - 检查: 数据流中是否有不必要的detach?
  
  下一步行动:
  → 我会在 train() 中添加详细日志，定位具体原因
""")
    elif max_change < 10.0:
        print(f"""
  ⚠️ train() 有效但效果微弱
  
  最大权重变化: {max_change:.2f}%
  
  可能原因:
  - 学习率过小 (当前: actor_lr=3e-4)
  - 梯度裁剪过于激进 (clip_norm=0.5)
  - rollout_length不够长 (当前: 150)
  
  建议:
  - 尝试提高学习率 5-10倍
  - 放宽梯度裁剪到 1.0-2.0
  - 增加rollout_length到300-500
""")
    else:
        print(f"""
  ✅ train() 正常工作!
  
  最大权重变化: {max_change:.2f}%
  
  这说明问题不在train()本身，而可能在:
  - _run_phase_v2() 的调用方式
  - 或者save/load的时机
""")

    print(f"\n")


if __name__ == "__main__":
    main()
