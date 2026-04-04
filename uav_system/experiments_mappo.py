"""
BA-MAPPO 多智能体强化学习实验

基于 MAPPO (Multi-Agent PPO, Yu et al. 2022 NeurIPS) 的 UAV 切换决策实验，
并引入 Business-Aware Actor 和 Attention-Enhanced Critic 两项改进。

实验内容:
1. Phase 1 — 训练收敛分析
   - 在不同 UAV 数量 (10, 20) 下训练 BA-MAPPO
   - 绘制训练曲线 (reward, actor_loss, critic_loss, entropy)

2. Phase 2 — 对比评估
   - BA-MAPPO-优化参数 vs 人工固定参数 (5种策略) vs 增强算法 vs 传统算法
   - 多维度对比: 满意率、切换成功率、资源利用率、关键业务满足率

3. Phase 3 — 多场景泛化
   - 训练场景 vs 5个测试场景

4. 可视化
   - 训练收敛曲线
   - 策略选择热力图
   - 场景泛化雷达图

与 QMIX 的主要区别:
- On-policy (rollout + PPO update) 替代 Off-policy (replay buffer + TD)
- 策略梯度替代值分解
- BA Actor 支持业务类型异构
- Attention Critic 替代简单 Mixing Network
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS
from .qmix_environment import QMixHandoverEnv
from .mappo_agent import MAPPOAgent
from .parametric_algorithm import ParametricEnhancedAlgorithm, STRATEGY_CONFIGS, NUM_STRATEGIES
from .algorithms import EnhancedHandoverAlgorithm
from .business import BusinessType


class ExperimentBAMAPPO:
    """BA-MAPPO 多智能体强化学习实验"""

    MODEL_DIR = os.path.join(RESULT_DIR, 'mappo_models')
    RESULT_FILE = os.path.join(RESULT_DIR, 'mappo_experiment_data.pkl')

    @staticmethod
    def run(num_uav_list=(10, 20, 30, 40),
            num_bs=8, num_steps=150,
            train_episodes=1000, eval_episodes=5,
            bs_capacity_range=(400, 800),
            pos_range=1000,
            load_models=False, phase='both',
            verbose=True,
            # BA-MAPPO 配置开关
            use_biz_heads=True,
            use_attention_critic=True,
            rollout_length=100,
            actor_lr=3e-4, critic_lr=5e-4,
            hidden_dim=64, critic_hidden_dim=128):
        """
        运行 BA-MAPPO 实验

        Args:
            num_uav_list: 要测试的 UAV 数量列表
            num_bs: 基站数量
            num_steps: 每个 episode 的步数
            train_episodes: 训练 episodes 数
            eval_episodes: 评估重复次数
            bs_capacity_range: 基站容量范围
            load_models: 是否加载已有模型
            phase: 运行阶段 'both' / 'phase1' / 'phase2'
            verbose: 是否打印详细信息
            use_biz_heads: 是否启用 BA Actor
            use_attention_critic: 是否启用 Attention Critic
            rollout_length: rollout 长度
            actor_lr: Actor 学习率
            critic_lr: Critic 学习率
            hidden_dim: Actor 隐藏层维度
            critic_hidden_dim: Critic 隐藏层维度
        """
        mode_name = "BA-MAPPO" if use_biz_heads and use_attention_critic else \
                    "MAPPO+BA" if use_biz_heads else \
                    "MAPPO+Attn" if use_attention_critic else "MAPPO"

        print("\n" + "=" * 80)
        print(f"{mode_name} 多智能体强化学习实验")
        print("=" * 80)
        print(f"  基站数量: {num_bs}")
        print(f"  UAV 数量列表: {num_uav_list}")
        print(f"  训练 episodes: {train_episodes}")
        print(f"  评估重复次数: {eval_episodes}")
        print(f"  容量范围: {bs_capacity_range}")
        print(f"  地图范围: {pos_range}m")
        print(f"  BA Actor: {use_biz_heads}")
        print(f"  Attention Critic: {use_attention_critic}")
        print(f"  Rollout 长度: {rollout_length}")
        print(f"  Actor LR: {actor_lr}, Critic LR: {critic_lr}")
        print("=" * 80)

        os.makedirs(ExperimentBAMAPPO.MODEL_DIR, exist_ok=True)

        all_results = {
            'config': {
                'mode_name': mode_name,
                'use_biz_heads': use_biz_heads,
                'use_attention_critic': use_attention_critic,
                'rollout_length': rollout_length,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
            }
        }

        if phase in ('both', 'phase1'):
            training_results = ExperimentBAMAPPO._phase1_training(
                num_uav_list, num_bs, num_steps, train_episodes,
                bs_capacity_range, pos_range, load_models, verbose,
                use_biz_heads, use_attention_critic,
                rollout_length, actor_lr, critic_lr,
                hidden_dim, critic_hidden_dim,
            )
            all_results['training'] = training_results

        if phase in ('both', 'phase2'):
            eval_results = ExperimentBAMAPPO._phase2_evaluation(
                num_uav_list, num_bs, num_steps, eval_episodes,
                bs_capacity_range, pos_range, load_models, verbose,
                use_biz_heads, use_attention_critic,
                hidden_dim, critic_hidden_dim,
            )
            all_results['evaluation'] = eval_results

            scenario_results = ExperimentBAMAPPO._phase3_scenarios(
                num_bs=8, num_uav=20, num_steps=100, repeats=10,
                bs_capacity_range=None, pos_range=None, verbose=verbose,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
                hidden_dim=hidden_dim, critic_hidden_dim=critic_hidden_dim,
                trained_uav_list=num_uav_list,
            )
            all_results['scenarios'] = scenario_results

        # 可视化
        ExperimentBAMAPPO._plot_all(all_results, num_uav_list)

        # 保存结果
        with open(ExperimentBAMAPPO.RESULT_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n  实验数据已保存: {ExperimentBAMAPPO.RESULT_FILE}")

        return all_results

    # ==================== Phase 1: 训练 ====================

    @staticmethod
    def _phase1_training(num_uav_list, num_bs, num_steps, train_episodes,
                         bs_capacity_range, pos_range, load_models, verbose,
                         use_biz_heads, use_attention_critic,
                         rollout_length, actor_lr, critic_lr,
                         hidden_dim, critic_hidden_dim):
        """Phase 1: 训练收敛分析"""
        print("\n" + "-" * 60)
        print("Phase 1: BA-MAPPO 训练收敛分析")
        print("-" * 60)

        training_results = {}

        for num_uav in num_uav_list:
            print(f"\n>>> 训练 UAV={num_uav} <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 100)
            model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                      f'mappo_{num_bs}bs_{num_uav}uav.pt')

            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 100,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
            )
            env.reset_normalizer()  # 训练前重置 normalizer

            agent = MAPPOAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                rollout_length=max(rollout_length, num_steps),
                num_epochs=3,
                batch_size=64,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
            )

            if load_models and os.path.exists(model_path):
                agent.load(model_path)
                training_results[num_uav] = {'loaded': True}
                print(f"  已加载模型: {model_path}")
                continue

            episode_rewards = []
            episode_satisfactions = []
            episode_actor_losses = []
            episode_critic_losses = []
            episode_entropies = []
            best_reward = float('-inf')
            best_sat = float('-inf')  # satisfaction-based 模型选择
            save_interval = 50
            # 分离 best 和 latest 模型路径
            best_model_path = model_path
            latest_model_path = model_path.replace('.pt', '_latest.pt')
            # ---- Early stopping 参数 (基于 satisfaction 而非 reward) ----
            early_stop_patience = train_episodes // 5       # 200 轮无改善则停止
            early_stop_min_delta = 0.002                    # 最小改善幅度
            early_stop_warmup = train_episodes // 5         # 前 20% 不计入 best_sat 追踪
            no_improve_count = 0
            early_stopped = False
            # 早期健康检查: 在 10% 训练进度时检查 reward 是否在正增长
            health_check_ep = max(10, train_episodes // 10)

            # 设置 LR schedule 的总步数
            agent._total_train_steps = train_episodes
            agent._current_train_step = 0

            for ep in range(train_episodes):
                obs_dict, global_state = env.reset()
                agent.reset_hidden()
                episode_reward = 0.0
                episode_sat = []
                # ---- 诊断: reward 组分分解 ----
                ep_action_counts = {'stay': 0, 'switch': 0}
                ep_per_action = np.zeros(env.action_dim)
                ep_switch_success = 0
                ep_switch_fail = 0
                ep_disconnected_steps = 0
                ep_reward_diag = {'delta_sum': 0, 'stay_bonus': 0, 'switch_cost': 0,
                                  'disconnect_pen': 0, 'good_switch': 0, 'bad_switch': 0,
                                  'raw_mean': 0, 'norm_mean': 0, 'count': 0}

                for step in range(num_steps):
                    # 获取 biz_types
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    # 选择动作 + 获取 log_probs, values, pre-step hidden
                    actions, log_probs, values, pre_hidden = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True
                    )

                    # 诊断: 统计 action 分布
                    for a in actions.values():
                        ep_per_action[a] += 1
                        if a == 0:
                            ep_action_counts['stay'] += 1
                        else:
                            ep_action_counts['switch'] += 1

                    # 执行动作
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)

                    # 诊断: 断连率 + reward 组分
                    if info['connected_rate'] < 1.0:
                        ep_disconnected_steps += 1
                    if 'reward_diag' in info:
                        rd = info['reward_diag']
                        for k in ep_reward_diag:
                            if k == 'count':
                                ep_reward_diag[k] += 1
                            elif k in ('good_switch', 'bad_switch'):
                                ep_reward_diag[k] += rd.get(k, 0)
                            else:
                                ep_reward_diag[k] += rd.get(k, 0)

                    # 存储经验 (传入 biz_types + hidden state 供训练时使用)
                    agent.insert_experience(
                        step, obs_dict, global_state, actions,
                        rewards, team_reward, done, log_probs, values,
                        biz_types, pre_hidden
                    )

                    obs_dict = next_obs
                    global_state = next_state
                    episode_reward += team_reward
                    episode_sat.append(info['avg_satisfaction'])

                episode_rewards.append(episode_reward)
                episode_satisfactions.append(np.mean(episode_sat))

                # PPO 更新 (每个 episode 结束后，包括第一个 episode)
                train_stats = agent.train()
                if train_stats:
                    episode_actor_losses.append(train_stats['actor_loss'])
                    episode_critic_losses.append(train_stats['critic_loss'])
                    episode_entropies.append(train_stats['entropy'])

                # ---- 早期健康检查 ----
                if ep == health_check_ep and verbose:
                    recent_avg = np.mean(episode_rewards[-health_check_ep:])
                    early_avg = np.mean(episode_rewards[:max(1, health_check_ep // 2)])
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    reward_std = np.std(episode_rewards[-health_check_ep:])
                    # per-action 分布 (3-action: stay/best_sinr/best_capacity)
                    action_total = max(ep_per_action.sum(), 1)
                    action_str = ', '.join(
                        [f'stay={ep_per_action[0]/action_total:.0%}',
                         f'best_sinr={ep_per_action[1]/action_total:.0%}',
                         f'best_cap={ep_per_action[2]/action_total:.0%}']
                    )
                    print(f"\n  {'='*60}")
                    print(f"  诊断报告 [Episode {ep+1}]")
                    print(f"  {'='*60}")
                    print(f"  Reward: 均值={np.mean(episode_rewards):.2f}, "
                          f"标准差={reward_std:.2f}, "
                          f"变异系数={reward_std/max(abs(np.mean(episode_rewards)),0.01)*100:.1f}%")
                    print(f"  Satisfaction: {np.mean(episode_satisfactions):.3f}")
                    print(f"  Action 分布: {action_str}")
                    if recent_avg <= early_avg and len(episode_rewards) > health_check_ep // 2:
                        print(f"  ⚠ Reward 无上升趋势 (前期={early_avg:.2f} → 近期={recent_avg:.2f})")
                    else:
                        print(f"  ✓ Reward 趋势正常 (前期={early_avg:.2f} → 近期={recent_avg:.2f})")
                    if reward_std / max(abs(np.mean(episode_rewards)), 0.01) > 0.3:
                        print(f"  ⚠ Reward 变异系数 > 30%，PPO 难以收敛 — 考虑 reward 归一化")
                    # Reward 组分
                    if ep_reward_diag['count'] > 0:
                        n = ep_reward_diag['count']
                        print(f"  Reward 组分 (avg/step): "
                              f"delta={ep_reward_diag['delta_sum']/n:.3f}, "
                              f"stay_bonus={ep_reward_diag['stay_bonus']/n:.3f}, "
                              f"switch_cost={ep_reward_diag['switch_cost']/n:.3f}, "
                              f"disc_pen={ep_reward_diag['disconnect_pen']/n:.3f}")
                        print(f"  Switch 质量: good={ep_reward_diag['good_switch']}, "
                              f"bad={ep_reward_diag['bad_switch']}, "
                              f"ratio={ep_reward_diag['good_switch']/max(ep_reward_diag['good_switch']+ep_reward_diag['bad_switch'],1):.1%}")
                    # 切换尝试/成功率
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    if sa > 0:
                        print(f"  切换统计: 尝试={sa}, 成功={ss}({ss/sa:.0%}), "
                              f"回滚={sr}({sr/sa:.0%}), 断连={sd}({sd/sa:.0%})")
                    else:
                        print(f"  切换统计: 无切换尝试")
                    print(f"  {'='*60}\n")

                if verbose and (ep + 1) % 30 == 0:
                    avg_al = np.mean(episode_actor_losses[-20:]) if episode_actor_losses else 0
                    avg_cl = np.mean(episode_critic_losses[-20:]) if episode_critic_losses else 0
                    avg_ent = np.mean(episode_entropies[-20:]) if episode_entropies else 0
                    recent_rews = episode_rewards[-30:]
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    # reward 组分 + 切换诊断摘要
                    n = max(ep_reward_diag['count'], 1)
                    rd_str = (f"Δs={ep_reward_diag['delta_sum']/n:.2f} "
                              f"sb={ep_reward_diag['stay_bonus']/n:.2f} "
                              f"sc={ep_reward_diag['switch_cost']/n:.2f}")
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    sw_str = f"sw={sa}(ok={ss},rb={sr},dc={sd})" if sa > 0 else "sw=0"
                    print(f"  Episode {ep+1}/{train_episodes}: "
                          f"reward={episode_reward:.1f}(μ={np.mean(recent_rews):.1f},σ={np.std(recent_rews):.1f}), "
                          f"sat={np.mean(episode_sat):.3f}, "
                          f"stay={stay_pct:.0f}%, {sw_str}, "
                          f"a_loss={avg_al:.4f}, c_loss={avg_cl:.2f}, "
                          f"H={avg_ent:.3f} | {rd_str}")

                # ---- Early stopping 判断 (基于 satisfaction) ----
                ep_sat = np.mean(episode_sat)
                is_best_sat = ep_sat > best_sat + early_stop_min_delta
                is_best_reward = episode_reward > best_reward + early_stop_min_delta
                if is_best_sat:
                    best_sat = ep_sat
                if is_best_reward:
                    best_reward = episode_reward

                # warmup 期间: 记录 best_sat 但不计入 no_improve
                if ep < early_stop_warmup:
                    no_improve_count = 0
                elif is_best_sat:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # 模型保存: best 模型仅在 satisfaction 新高时保存, latest 每 save_interval 保存
                if is_best_sat:
                    agent.save(best_model_path)
                if (ep + 1) % save_interval == 0:
                    agent.save(latest_model_path)

                if no_improve_count >= early_stop_patience and ep >= health_check_ep * 2:
                    if verbose:
                        print(f"\n  ⏹ Early stopping [Episode {ep+1}]: "
                              f"连续 {early_stop_patience} 轮 satisfaction 无改善 "
                              f"(best_sat={best_sat:.4f}, best_reward={best_reward:.3f})")
                    early_stopped = True
                    break

            # 训练结束: 确保 model_path 指向 best 模型
            import shutil
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_path)
                if verbose:
                    print(f"  ✓ 最终模型已更新为 best_sat={best_sat:.4f} 的版本")

                training_results[num_uav] = {
                    'rewards': episode_rewards,
                    'satisfactions': episode_satisfactions,
                    'actor_losses': episode_actor_losses,
                    'critic_losses': episode_critic_losses,
                    'entropies': episode_entropies,
                    'final_avg_sat': np.mean(episode_satisfactions[-50:]),
                    'early_stopped': early_stopped,
                }

        return training_results

    # ==================== Phase 2: 对比评估 ====================

    @staticmethod
    def _phase2_evaluation(num_uav_list, num_bs, num_steps, eval_episodes,
                           bs_capacity_range, pos_range, load_models, verbose,
                           use_biz_heads, use_attention_critic,
                           hidden_dim, critic_hidden_dim):
        """Phase 2: BA-MAPPO vs 人工固定参数对比"""
        print("\n" + "-" * 60)
        print("Phase 2: BA-MAPPO-优化参数 vs 人工固定参数对比")
        print("-" * 60)

        eval_results = {}

        for num_uav in num_uav_list:
            print(f"\n>>> 评估 UAV={num_uav} <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 200)
            model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                      f'mappo_{num_bs}bs_{num_uav}uav.pt')

            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 200,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
            )

            agent = MAPPOAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
            )

            if os.path.exists(model_path):
                agent.load(model_path)
                print(f"  已加载模型: {model_path}")
            elif load_models:
                print(f"  警告: 未找到模型 {model_path}, 使用 random 策略")

            # 评估 BA-MAPPO
            mappo_sats = []
            # 初始化 action 分布计数 (stay, best_sinr, best_capacity)
            action_labels = ['stay', 'best_sinr', 'best_capacity']
            strategy_counts = {name: 0 for name in action_labels}

            for rep in range(eval_episodes):
                obs_dict, global_state = env.reset()
                agent.reset_hidden()
                ep_sats = []
                for step in range(num_steps):
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    actions, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=False)
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    obs_dict = next_obs
                    global_state = next_state

                    ep_sats.append(info['avg_satisfaction'])
                    for sname, count in info['strategy_distribution'].items():
                        strategy_counts[sname] = strategy_counts.get(sname, 0) + count

                mappo_sats.append(np.mean(ep_sats))

            # 评估固定动作基线（使用统一环境 QMixHandoverEnv）
            strategy_results = {}
            # Baseline 1: stay (不切换)
            rep_sats = []
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 300 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=num_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=bs_capacity_range,
                    pos_range=pos_range,
                )
                eval_env.reset()
                for step in range(num_steps):
                    actions = {uid: 0 for uid in range(eval_env.num_agents)}
                    eval_env.step(actions)
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                rep_sats.append(avg_sat)
            strategy_results['stay'] = {
                'satisfaction': (np.mean(rep_sats), np.std(rep_sats)),
            }
            # Baseline 2: best-SINR (每步选择 SINR 最高的 BS)
            rep_sats = []
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 350 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=num_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=bs_capacity_range,
                    pos_range=pos_range,
                )
                eval_env.reset()
                for step in range(num_steps):
                    actions = {uid: 1 for uid in range(eval_env.num_agents)}  # action=1 = best_sinr
                    eval_env.step(actions)
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                rep_sats.append(avg_sat)
            strategy_results['best_sinr'] = {
                'satisfaction': (np.mean(rep_sats), np.std(rep_sats)),
            }
            # Baseline 3: random BS
            rep_sats = []
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 400 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=num_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=bs_capacity_range,
                    pos_range=pos_range,
                )
                eval_env.reset()
                for step in range(num_steps):
                    actions = {uid: np.random.randint(0, eval_env.action_dim)
                               for uid in range(eval_env.num_agents)}
                    eval_env.step(actions)
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                rep_sats.append(avg_sat)
            strategy_results['random_bs'] = {
                'satisfaction': (np.mean(rep_sats), np.std(rep_sats)),
            }

            # 增强算法基线（使用统一环境 QMixHandoverEnv）
            enhanced_sats = []
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 500 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=num_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=bs_capacity_range,
                    pos_range=pos_range,
                )
                eval_env.reset()
                algo = EnhancedHandoverAlgorithm(eval_env.env)
                for step in range(num_steps):
                    algo.run_step(enable_load_balancing=True)
                    eval_env.advance_env_only()
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                enhanced_sats.append(avg_sat)

            # 传统算法基线（使用统一环境 QMixHandoverEnv）
            traditional_sats = []
            from .algorithms import IntegratedHandoverAlgorithm
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 600 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=num_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=bs_capacity_range,
                    pos_range=pos_range,
                )
                eval_env.reset()
                algo = IntegratedHandoverAlgorithm(eval_env.env)
                for step in range(num_steps):
                    algo.run_step()
                    eval_env.advance_env_only()
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                traditional_sats.append(avg_sat)

            mappo_mean = np.mean(mappo_sats) if mappo_sats else 0
            mappo_std = np.std(mappo_sats) if mappo_sats else 0

            eval_results[num_uav] = {
                'mappo': {'satisfaction': (mappo_mean, mappo_std)},
                'enhanced': {'satisfaction': (np.mean(enhanced_sats), np.std(enhanced_sats))},
                'traditional': {'satisfaction': (np.mean(traditional_sats), np.std(traditional_sats))},
                'strategies': strategy_results,
                'strategy_distribution': strategy_counts,
            }

            if verbose:
                print(f"\n  UAV={num_uav} 对比结果:")
                print(f"    BA-MAPPO:   {mappo_mean:.4f} +/- {mappo_std:.4f}")
                print(f"    增强算法:   {np.mean(enhanced_sats):.4f} +/- {np.std(enhanced_sats):.4f}")
                print(f"    传统算法:   {np.mean(traditional_sats):.4f} +/- {np.std(traditional_sats):.4f}")
                for sn, sr in strategy_results.items():
                    print(f"    基线-{sn}:  {sr['satisfaction'][0]:.4f} +/- {sr['satisfaction'][1]:.4f}")

        return eval_results

    # ==================== Phase 3: 多场景泛化 ====================

    @staticmethod
    def _phase3_scenarios(num_bs, num_uav, num_steps, repeats,
                          bs_capacity_range, pos_range, verbose,
                          use_biz_heads, use_attention_critic,
                          hidden_dim, critic_hidden_dim,
                          trained_uav_list=None):
        """Phase 3: 多场景泛化验证"""
        print("\n" + "-" * 60)
        print("Phase 3: 多场景泛化验证")
        print("-" * 60)

        scenarios = {
            'default':               {'num_uav': 50, 'bs_capacity_range': (500, 1000)},
            'smart_city':            {'num_uav': 400, 'bs_capacity_range': (1500, 2400)},
            'industrial_inspection': {'num_uav': 300, 'bs_capacity_range': (1400, 2300)},
            'emergency_rescue':      {'num_uav': 300, 'bs_capacity_range': (900, 1200)},
            'logistics_delivery':    {'num_uav': 500, 'bs_capacity_range': (1200, 2100)},
        }

        scenario_names_cn = {
            'default': '默认场景',
            'smart_city': '城市监控',
            'industrial_inspection': '工业巡检',
            'emergency_rescue': '应急救援',
            'logistics_delivery': '物流配送',
        }

        scenario_results = {}

        # 加载模型: 按场景 UAV 数量精确匹配 (Critic 依赖 num_agents)
        trained_uav_set = set(trained_uav_list) if trained_uav_list else set()
        if verbose:
            print(f"  已训练模型 UAV 数量: {trained_uav_list}")

        for scenario_name, scenario_cfg in scenarios.items():
            print(f"\n>>> 场景: {scenario_name} ({scenario_names_cn.get(scenario_name, scenario_name)}) <<<")

            s_uav = scenario_cfg['num_uav']
            s_cap = scenario_cfg['bs_capacity_range']

            strategy_sats = {}
            enhanced_sats = []
            traditional_sats = []
            mappo_sats = []

            # 寻找与当前场景 UAV 数量精确匹配的训练模型
            matching_uav = s_uav if s_uav in trained_uav_set else None

            if matching_uav is not None:
                model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                          f'mappo_{num_bs}bs_{matching_uav}uav.pt')
                if os.path.exists(model_path):
                    print(f"  加载 BA-MAPPO 模型 (UAV={matching_uav}) 评估 {scenario_name}")
                    mappo_env = QMixHandoverEnv(
                        num_bs=num_bs, num_uav=matching_uav,
                        max_steps=num_steps, seed=GLOBAL_SEED + 9999,
                        bs_capacity_range=s_cap,
                    )
                    mappo_agent = MAPPOAgent(
                        num_agents=mappo_env.num_agents,
                        obs_dim=mappo_env.obs_dim,
                        state_dim=mappo_env.state_dim,
                        action_dim=mappo_env.action_dim,
                        hidden_dim=hidden_dim,
                        critic_hidden_dim=critic_hidden_dim,
                        use_biz_heads=use_biz_heads,
                        use_attention_critic=use_attention_critic,
                    )
                    mappo_agent.load(model_path)

                    for rep in range(repeats):
                        seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                        set_global_seed(seed)
                        obs_dict, global_state = mappo_env.reset()
                        mappo_agent.reset_hidden()
                        ep_sats = []
                        for step in range(num_steps):
                            biz_types = {}
                            for uid in range(mappo_env.num_agents):
                                uav = mappo_env.env.uavs[uid]
                                biz_types[uid] = uav.true_business_type.value
                            actions, _, _, _ = mappo_agent.select_actions(
                                obs_dict, global_state, biz_types, training=False)
                            next_obs, next_state, rewards, team_reward, done, info = mappo_env.step(actions)
                            obs_dict = next_obs
                            global_state = next_state
                            ep_sats.append(info['avg_satisfaction'])
                        mappo_sats.append(np.mean(ep_sats))
                elif verbose:
                    print(f"  跳过 MAPPO: 模型文件不存在 {model_path}")
            elif verbose:
                print(f"  跳过 MAPPO: 无 UAV={s_uav} 的训练模型 "
                      f"(已有: {trained_uav_list})")

            # 固定动作基线
            for baseline_name, baseline_fn in [('stay', lambda env, uid, step: 0),
                                                ('best_sinr', lambda env, uid, step: 1)]:
                rep_sats = []
                for rep in range(repeats):
                    seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000 + hash(baseline_name) % 100
                    set_global_seed(seed)
                    eval_env = QMixHandoverEnv(
                        num_bs=num_bs, num_uav=s_uav,
                        max_steps=num_steps, seed=seed,
                        bs_capacity_range=s_cap,
                    )
                    eval_env.reset()
                    for step in range(num_steps):
                        actions = {uid: baseline_fn(eval_env, uid, step)
                                   for uid in range(eval_env.num_agents)}
                        eval_env.step(actions)
                    avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                       for uid in range(eval_env.num_agents)])
                    rep_sats.append(avg_sat)
                strategy_sats[baseline_name] = (np.mean(rep_sats), np.std(rep_sats))

            for rep in range(repeats):
                seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=s_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=s_cap,
                )
                eval_env.reset()
                algo = EnhancedHandoverAlgorithm(eval_env.env)
                for step in range(num_steps):
                    algo.run_step(enable_load_balancing=True)
                    eval_env.advance_env_only()
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                enhanced_sats.append(avg_sat)

            from .algorithms import IntegratedHandoverAlgorithm
            for rep in range(repeats):
                seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                set_global_seed(seed)
                eval_env = QMixHandoverEnv(
                    num_bs=num_bs, num_uav=s_uav,
                    max_steps=num_steps, seed=seed,
                    bs_capacity_range=s_cap,
                )
                eval_env.reset()
                algo = IntegratedHandoverAlgorithm(eval_env.env)
                for step in range(num_steps):
                    algo.run_step()
                    eval_env.advance_env_only()
                avg_sat = np.mean([eval_env.env.uavs[uid].current_satisfaction
                                   for uid in range(eval_env.num_agents)])
                traditional_sats.append(avg_sat)

            best_strat = max(strategy_sats, key=lambda k: strategy_sats[k][0])

            scenario_results[scenario_name] = {
                'name_cn': scenario_names_cn.get(scenario_name, scenario_name),
                'strategies': strategy_sats,
                'best_strategy': best_strat,
                'enhanced': (np.mean(enhanced_sats), np.std(enhanced_sats)),
                'traditional': (np.mean(traditional_sats), np.std(traditional_sats)),
            }
            if mappo_sats:
                scenario_results[scenario_name]['mappo'] = (np.mean(mappo_sats), np.std(mappo_sats))

            if verbose:
                print(f"    最优策略: {best_strat} ({strategy_sats[best_strat][0]:.4f})")
                print(f"    增强算法: {np.mean(enhanced_sats):.4f}")
                print(f"    传统算法: {np.mean(traditional_sats):.4f}")
                if mappo_sats:
                    print(f"    BA-MAPPO: {np.mean(mappo_sats):.4f} +/- {np.std(mappo_sats):.4f}")

        return scenario_results

    # ==================== 可视化 ====================

    @staticmethod
    def _plot_all(all_results, num_uav_list):
        """生成所有可视化图表"""
        mode_name = all_results.get('config', {}).get('mode_name', 'BA-MAPPO')
        print(f"\n生成 {mode_name} 实验可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{mode_name} 实验结果', fontsize=16, fontweight='bold')

        # 图1: 训练收敛曲线
        ax = axes[0, 0]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'rewards' in tr:
                        rewards = np.array(tr['rewards'])
                        window = max(1, len(rewards) // 20)
                        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                        ax.plot(smoothed, label=f'UAV={num_uav}', alpha=0.8)
            ax.set_xlabel('Episode')
            ax.set_ylabel('团队奖励')
            ax.set_title('(a) 训练收敛曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 图2: 训练期间满意度变化
        ax = axes[0, 1]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'satisfactions' in tr:
                        sats = np.array(tr['satisfactions'])
                        window = max(1, len(sats) // 20)
                        smoothed = np.convolve(sats, np.ones(window)/window, mode='valid')
                        ax.plot(smoothed, label=f'UAV={num_uav}', alpha=0.8)
            ax.set_xlabel('Episode')
            ax.set_ylabel('平均满意度')
            ax.set_title('(b) 训练期间满意度变化')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 图3: 对比评估柱状图
        ax = axes[0, 2]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                x = np.arange(len(uav_to_show))
                width = 0.2
                mappo_vals = [all_results['evaluation'][u]['mappo']['satisfaction'][0] for u in uav_to_show]
                enh_vals = [all_results['evaluation'][u]['enhanced']['satisfaction'][0] for u in uav_to_show]
                trad_vals = [all_results['evaluation'][u]['traditional']['satisfaction'][0] for u in uav_to_show]
                ax.bar(x - width, mappo_vals, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
                ax.bar(x, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax.bar(x + width, trad_vals, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_ylabel('平均满意度')
                ax.set_title('(c) 算法对比')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

        # 图4: 策略分布热力图
        ax = axes[1, 0]
        if 'evaluation' in all_results:
            # 使用实际的 action 标签 (stay, BS0, ..., BS{n-1}) 而非策略配置名
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                # 从第一个 UAV 的 strategy_distribution 中提取实际使用的 action 标签
                sample_dist = all_results['evaluation'][uav_to_show[0]].get('strategy_distribution', {})
                strat_names = list(sample_dist.keys()) if sample_dist else ['stay']
                matrix = np.zeros((len(uav_to_show), len(strat_names)))
                for i, u in enumerate(uav_to_show):
                    total = sum(all_results['evaluation'][u].get('strategy_distribution', {}).values()) or 1
                    for j, sn in enumerate(strat_names):
                        matrix[i, j] = all_results['evaluation'][u].get('strategy_distribution', {}).get(sn, 0) / total
                im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
                ax.set_xticks(range(len(strat_names)))
                ax.set_xticklabels(strat_names, rotation=30, ha='right', fontsize=8)
                ax.set_yticks(range(len(uav_to_show)))
                ax.set_yticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_title(f'(d) {mode_name} 动作分布')
                plt.colorbar(im, ax=ax)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

        # 图5: 多场景泛化对比
        ax = axes[1, 1]
        ax.axis('off')
        if 'scenarios' in all_results and all_results['scenarios']:
            ax2 = fig.add_subplot(2, 3, 6)
            scenarios_data = all_results['scenarios']
            s_names = list(scenarios_data.keys())
            s_names_cn = [scenarios_data[s].get('name_cn', s) for s in s_names]

            best_strat_sats = [scenarios_data[s]['strategies'][scenarios_data[s]['best_strategy']][0] for s in s_names]
            enhanced_sats = [scenarios_data[s]['enhanced'][0] for s in s_names]
            traditional_sats = [scenarios_data[s]['traditional'][0] for s in s_names]
            mappo_scenario_sats = [scenarios_data[s].get('mappo', (0, 0))[0] if 'mappo' in scenarios_data[s] else 0 for s in s_names]

            x = np.arange(len(s_names))
            width = 0.2
            has_mappo = any('mappo' in scenarios_data[s] for s in s_names)
            if has_mappo:
                offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
                ax2.bar(x + offsets[0], best_strat_sats, width, label='最优策略', color=COLORS['success'], alpha=0.8)
                ax2.bar(x + offsets[1], enhanced_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax2.bar(x + offsets[2], traditional_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + offsets[3], mappo_scenario_sats, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
            else:
                ax2.bar(x - width, best_strat_sats, width, label='最优策略', color=COLORS['success'], alpha=0.8)
                ax2.bar(x, enhanced_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax2.bar(x + width, traditional_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(s_names_cn, rotation=20, ha='right', fontsize=8)
            ax2.set_ylabel('平均满意度')
            ax2.set_title('(f) 多场景泛化对比')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

        # 图6: 关键结果文本
        ax = axes[1, 2]
        ax.axis('off')
        text_lines = [f'【{mode_name} 关键发现】\n']
        if 'evaluation' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['evaluation']:
                    ev = all_results['evaluation'][num_uav]
                    mp = ev['mappo']['satisfaction'][0]
                    en = ev['enhanced']['satisfaction'][0]
                    tr = ev['traditional']['satisfaction'][0]
                    improvement_vs_trad = (mp - tr) / max(tr, 0.001) * 100
                    text_lines.append(f'UAV={num_uav}:')
                    text_lines.append(f'  {mode_name} vs 传统: +{improvement_vs_trad:.1f}%')
                    text_lines.append(f'  {mode_name}={mp:.4f}, 增强={en:.4f}, 传统={tr:.4f}')
                    text_lines.append('')
        if 'scenarios' in all_results:
            text_lines.append('【场景最优策略】\n')
            for sn, sd in all_results['scenarios'].items():
                line = f'  {sd["name_cn"]}: {sd["best_strategy"]}'
                if 'mappo' in sd:
                    line += f' ({mode_name}={sd["mappo"][0]:.4f})'
                text_lines.append(line)

        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'mappo_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  可视化已保存: {save_path}")
