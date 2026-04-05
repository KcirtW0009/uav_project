"""
BA-MAPPO 多智能体强化学习实验

基于 MAPPO (Multi-Agent PPO, Yu et al. 2022 NeurIPS) 的 UAV 切换决策实验，
并引入 Business-Aware Actor 和 Attention-Enhanced Critic 两项改进。

研究脉络:
  实验1: 识别准确率对系统性能的影响
  实验2/2b/2c: 切换算法设计（传统 3GPP A3 → 增强算法）
  实验3: 识别与切换联动系统
  实验4 (本实验): BA-MAPPO 多智能体协同决策
    → 目标: 通过多智能体协同超越增强算法的单步贪心策略

基线设计逻辑:
  - 传统算法 (3GPP A3): 论文核心对比对象，代表工业界基线
  - 增强算法: 本文前序工作，BA-MAPPO 的直接改进对象
  - stay/best_sinr: 仅作参考，不作为核心对比对象
  - 容量设计: 控制负载率在 65%~85%，确保资源竞争真实存在

改进:
  1. Business-Aware Actor: 业务类型特定的独立输出头，差异化策略
  2. Attention-Enhanced Critic: 多头注意力聚合全局信息
  3. 分层策略网络: 高层(是否切换) + 底层(目标基站)
  4. 模仿学习预训练: 从增强算法示范中冷启动
  5. Domain Randomization: 训练时随机化环境参数提升泛化

实验内容:
1. Phase 1 — 训练收敛分析
2. Phase 2 — 算法对比评估 (传统 → 增强 → BA-MAPPO，分业务类型统计)
3. Phase 3 — 多场景泛化验证
4. 可视化 — 训练曲线 / 算法对比 / 动作分布 / 场景泛化
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
from .algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
from .business import BusinessType

# 业务类型中文名称（用于统计输出）
BIZ_TYPE_NAMES = {
    0: '控制信令', 1: '视频回传', 2: '环境监测'
}
BIZ_TYPE_KEYS = [BusinessType.CONTROL_SIGNAL, BusinessType.VIDEO_STREAMING, BusinessType.ENVIRONMENT_MONITORING]


def _collect_biz_satisfaction(env_or_env_wrapper):
    """
    按业务类型统计满意度（统一接口）。

    Args:
        env_or_env_wrapper: QMixHandoverEnv 实例（env.env 为底层网络环境）

    Returns:
        dict: {
            'avg': float,  # 全局平均满意度
            'per_biz': {0: (mean, std), 1: (mean, std), 2: (mean, std)},
            'critical_sat': float,  # 关键业务(控制信令+视频)平均满意度
        }
    """
    inner_env = env_or_env_wrapper.env if hasattr(env_or_env_wrapper, 'env') else env_or_env_wrapper
    biz_sats = {}
    for bt_val in [0, 1, 2]:
        sats = [uav.current_satisfaction for uav in inner_env.uavs.values()
                if uav.true_business_type.value == bt_val]
        biz_sats[bt_val] = (np.mean(sats), np.std(sats)) if sats else (0.0, 0.0)

    all_sats = [uav.current_satisfaction for uav in inner_env.uavs.values()]
    avg_sat = np.mean(all_sats) if all_sats else 0.0

    critical_sats = [uav.current_satisfaction for uav in inner_env.uavs.values()
                     if uav.true_business_type.value in (0, 1)]
    critical_sat = np.mean(critical_sats) if critical_sats else 0.0

    return {
        'avg': avg_sat,
        'per_biz': biz_sats,
        'critical_sat': critical_sat,
    }


def _run_fixed_action_baseline(env, num_steps, action=0):
    """运行固定动作基线 (action=0: stay, action=1: best_sinr 等)"""
    for step in range(num_steps):
        actions = {uid: action for uid in range(env.num_agents)}
        env.step(actions)


def _run_algo_baseline(env, num_steps, algo_class, enable_lb=False):
    """运行启发式算法基线 (传统/增强算法)"""
    algo = algo_class(env.env)
    for step in range(num_steps):
        kwargs = {}
        if enable_lb and algo_class == EnhancedHandoverAlgorithm:
            kwargs['enable_load_balancing'] = True
        algo.run_step(**kwargs)
        env.advance_env_only()


class ExperimentBAMAPPO:
    """BA-MAPPO 多智能体强化学习实验"""

    MODEL_DIR = os.path.join(RESULT_DIR, 'mappo_models')
    RESULT_FILE = os.path.join(RESULT_DIR, 'mappo_experiment_data.pkl')

    @staticmethod
    def run(num_uav_list=(10, 20, 30, 40),
            num_bs=8, num_steps=150,
            train_episodes=100, eval_episodes=5,
            bs_capacity_range=(400, 800),
            pos_range=1000,
            load_models=False, phase='both',
            verbose=True,
            # BA-MAPPO 配置开关
            use_biz_heads=True,
            use_attention_critic=True,
            rollout_length=150,
            actor_lr=1e-4, critic_lr=3e-4,
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

            # 直接在标准环境中训练，确保模型能够适应真实场景
            # 移除简单环境，避免环境变化过于剧烈导致模型不适应
            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 100,
                bs_capacity_range=bs_capacity_range,
                pos_range=pos_range,
            )
            env.reset_normalizer()  # 训练前重置 normalizer
            
            # 为了保持代码兼容性，将 env 赋值给 simple_env
            simple_env = env

            # 初始化智能体
            agent = MAPPOAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=hidden_dim,
                critic_hidden_dim=critic_hidden_dim,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=0.95,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                entropy_coef=0.05,
                value_coef=0.5,
                rollout_length=max(rollout_length, num_steps),
                num_epochs=5,
                batch_size=64,
                use_biz_heads=use_biz_heads,
                use_attention_critic=use_attention_critic,
                use_enhanced_algorithm=True,
                use_pretrain=True,
                use_hierarchical=True,
                use_transformer=False,
                use_data_augmentation=True,
            )
            
            # 初始化增强算法
            enhanced_algorithm = EnhancedHandoverAlgorithm(env.env)
            agent.set_enhanced_algorithm(enhanced_algorithm)
            
            # 执行模仿学习预训练
            if agent.use_pretrain and not (load_models and os.path.exists(model_path)):
                # 收集示范数据
                demonstrations = agent.collect_demonstrations(simple_env, num_demos=1000)
                # 预训练
                agent.pretrain(demonstrations, epochs=50, batch_size=64)

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
            best_model_path = model_path.replace('.pt', '_best.pt')
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

            # 直接在标准环境中训练，确保模型能够适应真实场景
            print("\n  开始训练：标准环境 + Domain Randomization")
            simple_episodes = 0  # 移除简单环境训练
            for ep in range(simple_episodes):
                obs_dict, global_state = simple_env.reset()
                agent.reset_hidden()
                episode_reward = 0.0
                episode_sat = []
                # ---- 诊断: reward 组分分解 ----
                ep_action_counts = {'stay': 0, 'switch': 0}
                ep_per_action = np.zeros(simple_env.action_dim)
                ep_switch_success = 0
                ep_switch_fail = 0
                ep_disconnected_steps = 0
                ep_reward_diag = {'delta_sum': 0, 'value_reward': 0, 'biz_reward': 0, 'action_reward': 0,
                                  'connect_reward': 0, 'good_switch': 0, 'bad_switch': 0,
                                  'raw_mean': 0, 'norm_mean': 0, 'count': 0,
                                  'switch_attempts': 0, 'switch_success': 0, 'switch_rollback': 0, 'switch_disconnect': 0}
                # 业务类型统计
                ep_biz_stats = {}
                for bt in range(3):
                    ep_biz_stats[bt] = {'stay': 0, 'switch': 0, 'satisfaction': [], 'reward': []}

                for step in range(num_steps):
                    # 获取 biz_types
                    biz_types = {}
                    for uid in range(simple_env.num_agents):
                        uav = simple_env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    # 更新增强算法的使用概率
                    agent.update_enhanced_algorithm_prob(ep, simple_episodes)

                    # 选择动作 + 获取 log_probs, values, pre-step hidden
                    actions, log_probs, values, pre_hidden = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True, env=simple_env
                    )

                    # 诊断: 统计 action 分布
                    for uid, a in actions.items():
                        ep_per_action[a] += 1
                        if a == 0:
                            ep_action_counts['stay'] += 1
                        else:
                            ep_action_counts['switch'] += 1
                        # 更新业务类型统计
                        biz_type = biz_types[uid]
                        if a == 0:
                            ep_biz_stats[biz_type]['stay'] += 1
                        else:
                            ep_biz_stats[biz_type]['switch'] += 1

                    # 执行动作
                    next_obs, next_state, rewards, team_reward, done, info = simple_env.step(actions)

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
                    
                    # 更新业务类型满意度和奖励统计
                    for uid in range(simple_env.num_agents):
                        biz_type = biz_types[uid]
                        uav = simple_env.env.uavs[uid]
                        ep_biz_stats[biz_type]['satisfaction'].append(uav.current_satisfaction)
                        ep_biz_stats[biz_type]['reward'].append(rewards.get(uid, 0))

                episode_rewards.append(episode_reward)
                episode_satisfactions.append(np.mean(episode_sat))

                # PPO 更新 (每个 episode 结束后，包括第一个 episode)
                train_stats = agent.train()
                if train_stats:
                    episode_actor_losses.append(train_stats['actor_loss'])
                    episode_critic_losses.append(train_stats['critic_loss'])
                    episode_entropies.append(train_stats['entropy'])

                if verbose and (ep + 1) % 30 == 0:
                    avg_al = np.mean(episode_actor_losses[-20:]) if episode_actor_losses else 0
                    avg_cl = np.mean(episode_critic_losses[-20:]) if episode_critic_losses else 0
                    avg_ent = np.mean(episode_entropies[-20:]) if episode_entropies else 0
                    recent_rews = episode_rewards[-30:]
                    stay_pct = ep_action_counts['stay'] / max(sum(ep_action_counts.values()), 1) * 100
                    # reward 组分 + 切换诊断摘要
                    n = max(ep_reward_diag['count'], 1)
                    rd_str = (f"Δs={ep_reward_diag['delta_sum']/n:.2f} "
                              f"biz={ep_reward_diag['biz_reward']/n:.2f} "
                              f"act={ep_reward_diag['action_reward']/n:.2f} "
                              f"conn={ep_reward_diag['connect_reward']/n:.2f}")
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    sw_str = f"sw={sa}(ok={ss},rb={sr},dc={sd})" if sa > 0 else "sw=0"
                    print(f"  简单环境 Episode {ep+1}/{simple_episodes}: "
                          f"reward={episode_reward:.1f}(μ={np.mean(recent_rews):.1f},σ={np.std(recent_rews):.1f}), "
                          f"sat={np.mean(episode_sat):.3f}, "
                          f"stay={stay_pct:.0f}%, {sw_str}, "
                          f"a_loss={avg_al:.4f}, c_loss={avg_cl:.2f}, "
                          f"H={avg_ent:.3f} | {rd_str}")

            # 直接在标准环境中训练 + Domain Randomization
            for ep in range(train_episodes):
                # Domain Randomization: 随机化环境参数
                random_capacity_range = (
                    int(bs_capacity_range[0] * (0.8 + 0.4 * np.random.rand())),
                    int(bs_capacity_range[1] * (0.8 + 0.4 * np.random.rand()))
                )
                
                # 重置环境，使用随机化的容量范围
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
                ep_reward_diag = {'delta_sum': 0, 'value_reward': 0, 'biz_reward': 0, 'action_reward': 0,
                                  'connect_reward': 0, 'good_switch': 0, 'bad_switch': 0,
                                  'raw_mean': 0, 'norm_mean': 0, 'count': 0,
                                  'switch_attempts': 0, 'switch_success': 0, 'switch_rollback': 0, 'switch_disconnect': 0}
                # 业务类型统计
                ep_biz_stats = {}
                for bt in range(3):
                    ep_biz_stats[bt] = {'stay': 0, 'switch': 0, 'satisfaction': [], 'reward': []}

                for step in range(num_steps):
                    # 获取 biz_types
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value

                    # 更新增强算法的使用概率
                    agent.update_enhanced_algorithm_prob(ep - simple_episodes, train_episodes - simple_episodes)

                    # 选择动作 + 获取 log_probs, values, pre-step hidden
                    actions, log_probs, values, pre_hidden = agent.select_actions(
                        obs_dict, global_state, biz_types, training=True, env=env
                    )

                    # 诊断: 统计 action 分布
                    for uid, a in actions.items():
                        ep_per_action[a] += 1
                        if a == 0:
                            ep_action_counts['stay'] += 1
                        else:
                            ep_action_counts['switch'] += 1
                        # 更新业务类型统计
                        biz_type = biz_types[uid]
                        if a == 0:
                            ep_biz_stats[biz_type]['stay'] += 1
                        else:
                            ep_biz_stats[biz_type]['switch'] += 1

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
                    
                    # 更新业务类型满意度和奖励统计
                    for uid in range(env.num_agents):
                        biz_type = biz_types[uid]
                        uav = env.env.uavs[uid]
                        ep_biz_stats[biz_type]['satisfaction'].append(uav.current_satisfaction)
                        ep_biz_stats[biz_type]['reward'].append(rewards.get(uid, 0))

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
                    # per-action 分布
                    action_total = max(ep_per_action.sum(), 1)
                    action_str = ', '.join(
                        [f'stay={ep_per_action[0]/action_total:.0%}',
                         f'best_sinr={ep_per_action[1]/action_total:.0%}',
                         f'best_cap={ep_per_action[2]/action_total:.0%}',
                         f'sinr_cap={ep_per_action[3]/action_total:.0%}',
                         f'predict={ep_per_action[4]/action_total:.0%}',
                         f'biz_spec={ep_per_action[5]/action_total:.0%}']
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
                              f"value={ep_reward_diag['value_reward']/n:.3f}, "
                              f"biz={ep_reward_diag['biz_reward']/n:.3f}, "
                              f"action={ep_reward_diag['action_reward']/n:.3f}, "
                              f"connect={ep_reward_diag['connect_reward']/n:.3f}")
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
                    # 业务类型统计
                    print(f"  业务类型统计:")
                    for bt in range(3):
                        stats = ep_biz_stats[bt]
                        total = stats['stay'] + stats['switch']
                        if total > 0:
                            stay_pct = stats['stay'] / total * 100
                            switch_pct = stats['switch'] / total * 100
                            avg_sat = np.mean(stats['satisfaction']) if stats['satisfaction'] else 0
                            avg_reward = np.mean(stats['reward']) if stats['reward'] else 0
                            print(f"    业务类型 {bt}: stay={stay_pct:.1f}%, switch={switch_pct:.1f}%, sat={avg_sat:.3f}, reward={avg_reward:.3f}")
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
                              f"biz={ep_reward_diag['biz_reward']/n:.2f} "
                              f"act={ep_reward_diag['action_reward']/n:.2f} "
                              f"conn={ep_reward_diag['connect_reward']/n:.2f}")
                    sa = ep_reward_diag.get('switch_attempts', 0)
                    ss = ep_reward_diag.get('switch_success', 0)
                    sr = ep_reward_diag.get('switch_rollback', 0)
                    sd = ep_reward_diag.get('switch_disconnect', 0)
                    sw_str = f"sw={sa}(ok={ss},rb={sr},dc={sd})" if sa > 0 else "sw=0"
                    print(f"  标准环境 Episode {ep+1}/{train_episodes}: "
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
                        print(f"\n  [STOP] Early stopping [Episode {ep+1}]: "
                              f"连续 {early_stop_patience} 轮 satisfaction 无改善 "
                              f"(best_sat={best_sat:.4f}, best_reward={best_reward:.3f})")
                    early_stopped = True
                    break

            # 训练结束: 确保 model_path 指向 best 模型
            import shutil
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_path)
                if verbose:
                    print(f"  [OK] 最终模型已更新为 best_sat={best_sat:.4f} 的版本")

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
        """
        Phase 2: 算法对比评估

        研究脉络:
          实验1→2→2b→3 验证了增强算法相对于传统算法(3GPP A3)的优势；
          BA-MAPPO 的核心目标是: 通过多智能体协同决策，超越增强算法（单步贪心）。
          因此基线优先级: 传统算法(3GPP) < 增强算法(本文) < BA-MAPPO(本文)。
          简单基线(stay/best_sinr)仅作参考，不作为核心对比对象。

        统计维度:
          - 整体平均满意度
          - 关键业务满意度（控制信令+视频回传）
          - 分业务类型满意度
        """
        print("\n" + "-" * 60)
        print("Phase 2: 算法对比评估 (传统 → 增强 → BA-MAPPO)")
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
                use_hierarchical=True,
                use_transformer=False,
                use_data_augmentation=True,
            )

            if os.path.exists(model_path):
                agent.load(model_path)
                print(f"  已加载模型: {model_path}")
            elif load_models:
                print(f"  警告: 未找到模型 {model_path}, 使用随机策略")

            # ---- 评估 BA-MAPPO ----
            mappo_avg_sats = []
            mappo_biz_sats = {0: [], 1: [], 2: []}
            mappo_critical_sats = []
            strategy_counts = {}

            for rep in range(eval_episodes):
                obs_dict, global_state = env.reset()
                agent.reset_hidden()
                for step in range(num_steps):
                    biz_types = {}
                    for uid in range(env.num_agents):
                        uav = env.env.uavs[uid]
                        biz_types[uid] = uav.true_business_type.value
                    actions, _, _, _ = agent.select_actions(obs_dict, global_state, biz_types, training=False)
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    obs_dict = next_obs
                    global_state = next_state
                    for sname, count in info['strategy_distribution'].items():
                        strategy_counts[sname] = strategy_counts.get(sname, 0) + count

                stats = _collect_biz_satisfaction(env)
                mappo_avg_sats.append(stats['avg'])
                mappo_critical_sats.append(stats['critical_sat'])
                for bt in range(3):
                    mappo_biz_sats[bt].append(stats['per_biz'][bt][0])

            # ---- 评估基线算法 ----
            def _eval_baseline(baseline_name, run_fn):
                """通用基线评估函数，返回 {avg, critical, per_biz} 的多次重复均值"""
                avg_list, critical_list = [], []
                biz_lists = {0: [], 1: [], 2: []}
                for rep in range(eval_episodes):
                    seed = GLOBAL_SEED + num_uav * 300 + rep * 1000 + hash(baseline_name) % 10000
                    set_global_seed(seed)
                    eval_env = QMixHandoverEnv(
                        num_bs=num_bs, num_uav=num_uav,
                        max_steps=num_steps, seed=seed,
                        bs_capacity_range=bs_capacity_range,
                        pos_range=pos_range,
                    )
                    eval_env.reset()
                    run_fn(eval_env, num_steps)
                    stats = _collect_biz_satisfaction(eval_env)
                    avg_list.append(stats['avg'])
                    critical_list.append(stats['critical_sat'])
                    for bt in range(3):
                        biz_lists[bt].append(stats['per_biz'][bt][0])
                return {
                    'avg': (np.mean(avg_list), np.std(avg_list)),
                    'critical': (np.mean(critical_list), np.std(critical_list)),
                    'per_biz': {bt: (np.mean(biz_lists[bt]), np.std(biz_lists[bt])) for bt in range(3)},
                }

            # 传统算法基线 (3GPP A3 — 论文核心对比对象)
            traditional_results = _eval_baseline('traditional',
                lambda e, steps: _run_algo_baseline(e, steps, IntegratedHandoverAlgorithm))

            # 增强算法基线 (本文前序工作 — BA-MAPPO 的直接改进对象)
            enhanced_results = _eval_baseline('enhanced',
                lambda e, steps: _run_algo_baseline(e, steps, EnhancedHandoverAlgorithm, enable_lb=True))

            # 简单参考基线 (仅作参考，不作为核心对比)
            stay_results = _eval_baseline('stay',
                lambda e, steps: _run_fixed_action_baseline(e, steps, action=0))
            best_sinr_results = _eval_baseline('best_sinr',
                lambda e, steps: _run_fixed_action_baseline(e, steps, action=1))

            # ---- 结果汇总 ----
            eval_results[num_uav] = {
                'mappo': {
                    'avg': (np.mean(mappo_avg_sats), np.std(mappo_avg_sats)),
                    'critical': (np.mean(mappo_critical_sats), np.std(mappo_critical_sats)),
                    'per_biz': {bt: (np.mean(mappo_biz_sats[bt]), np.std(mappo_biz_sats[bt]))
                                for bt in range(3)},
                },
                'enhanced': enhanced_results,
                'traditional': traditional_results,
                'ref_baselines': {
                    'stay': stay_results['avg'],
                    'best_sinr': best_sinr_results['avg'],
                },
                'strategy_distribution': strategy_counts,
            }

            if verbose:
                print(f"\n  UAV={num_uav} 对比结果:")
                print(f"  {'算法':<16} {'平均满意度':>12} {'关键业务':>12} {'控制信令':>10} {'视频回传':>10} {'环境监测':>10}")
                print(f"  {'-'*70}")
                for name, data in [
                    ('传统算法(3GPP)', traditional_results),
                    ('增强算法(本文)', enhanced_results),
                    ('BA-MAPPO(本文)', eval_results[num_uav]['mappo']),
                ]:
                    avg = data['avg']
                    cri = data['critical']
                    biz0 = data['per_biz'][0]
                    biz1 = data['per_biz'][1]
                    biz2 = data['per_biz'][2]
                    print(f"  {name:<16} {avg[0]:>8.4f}+/-{avg[1]:.3f} {cri[0]:>8.4f}+/-{cri[1]:.3f} "
                          f"{biz0[0]:>8.4f} {biz1[0]:>8.4f} {biz2[0]:>8.4f}")
                print(f"\n  [参考基线] stay={stay_results['avg'][0]:.4f}, "
                      f"best_sinr={best_sinr_results['avg'][0]:.4f}")

                # 计算相对提升
                trad_avg = traditional_results['avg'][0]
                enh_avg = enhanced_results['avg'][0]
                mappo_avg = eval_results[num_uav]['mappo']['avg'][0]
                if trad_avg > 0.001:
                    print(f"\n  BA-MAPPO 相对提升:")
                    print(f"    vs 传统算法: {(mappo_avg - trad_avg)/trad_avg*100:+.1f}%")
                    print(f"    vs 增强算法: {(mappo_avg - enh_avg)/max(enh_avg,0.001)*100:+.1f}%")

        return eval_results

    # ==================== Phase 3: 多场景泛化 ====================

    @staticmethod
    def _phase3_scenarios(num_bs, num_uav, num_steps, repeats,
                          bs_capacity_range, pos_range, verbose,
                          use_biz_heads, use_attention_critic,
                          hidden_dim, critic_hidden_dim,
                          trained_uav_list=None):
        """
        Phase 3: 多场景泛化验证

        评估 BA-MAPPO 在不同业务场景下的泛化能力。
        基线: 传统算法 + 增强算法 (与 Phase 2 一致，保持连贯性)。
        """
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
        trained_uav_set = set(trained_uav_list) if trained_uav_list else set()
        if verbose:
            print(f"  已训练模型 UAV 数量: {trained_uav_list}")

        for scenario_name, scenario_cfg in scenarios.items():
            print(f"\n>>> 场景: {scenario_name} ({scenario_names_cn.get(scenario_name, scenario_name)}) <<<")

            s_uav = scenario_cfg['num_uav']
            s_cap = scenario_cfg['bs_capacity_range']

            # ---- 评估 BA-MAPPO ----
            mappo_avg_sats = []
            matching_uav = s_uav if s_uav in trained_uav_set else None

            if matching_uav is not None:
                model_path = os.path.join(ExperimentBAMAPPO.MODEL_DIR,
                                          f'mappo_{num_bs}bs_{matching_uav}uav.pt')
                if os.path.exists(model_path):
                    print(f"  加载 BA-MAPPO 模型 (UAV={matching_uav})")
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
                        use_hierarchical=True,
                    )
                    mappo_agent.load(model_path)

                    for rep in range(repeats):
                        seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                        set_global_seed(seed)
                        obs_dict, global_state = mappo_env.reset()
                        mappo_agent.reset_hidden()
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
                        mappo_avg_sats.append(_collect_biz_satisfaction(mappo_env)['avg'])
                elif verbose:
                    print(f"  跳过 MAPPO: 模型文件不存在 {model_path}")
            elif verbose:
                print(f"  跳过 MAPPO: 无 UAV={s_uav} 的训练模型 (已有: {trained_uav_list})")

            # ---- 评估基线算法 (与 Phase 2 一致) ----
            def _eval_scenario_baseline(baseline_name, run_fn):
                avg_list = []
                for rep in range(repeats):
                    seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000 + hash(baseline_name) % 10000
                    set_global_seed(seed)
                    eval_env = QMixHandoverEnv(
                        num_bs=num_bs, num_uav=s_uav,
                        max_steps=num_steps, seed=seed,
                        bs_capacity_range=s_cap,
                    )
                    eval_env.reset()
                    run_fn(eval_env, num_steps)
                    avg_list.append(_collect_biz_satisfaction(eval_env)['avg'])
                return (np.mean(avg_list), np.std(avg_list))

            traditional_sat = _eval_scenario_baseline('traditional',
                lambda e, steps: _run_algo_baseline(e, steps, IntegratedHandoverAlgorithm))
            enhanced_sat = _eval_scenario_baseline('enhanced',
                lambda e, steps: _run_algo_baseline(e, steps, EnhancedHandoverAlgorithm, enable_lb=True))
            stay_sat = _eval_scenario_baseline('stay',
                lambda e, steps: _run_fixed_action_baseline(e, steps, action=0))

            scenario_results[scenario_name] = {
                'name_cn': scenario_names_cn.get(scenario_name, scenario_name),
                'traditional': traditional_sat,
                'enhanced': enhanced_sat,
                'ref_baseline_stay': stay_sat,
            }
            if mappo_avg_sats:
                scenario_results[scenario_name]['mappo'] = (np.mean(mappo_avg_sats), np.std(mappo_avg_sats))

            if verbose:
                print(f"    传统算法:   {traditional_sat[0]:.4f}")
                print(f"    增强算法:   {enhanced_sat[0]:.4f}")
                print(f"    [参考] stay: {stay_sat[0]:.4f}")
                if mappo_avg_sats:
                    m = scenario_results[scenario_name]['mappo']
                    print(f"    BA-MAPPO:   {m[0]:.4f} +/- {m[1]:.4f}")
                    trad = traditional_sat[0]
                    if trad > 0.001:
                        print(f"    BA-MAPPO vs 传统: {(m[0]-trad)/trad*100:+.1f}%")

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

        # 图3: 对比评估柱状图 (核心对比: 传统 → 增强 → BA-MAPPO)
        ax = axes[0, 2]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
                x = np.arange(len(uav_to_show))
                width = 0.25
                mappo_vals = [all_results['evaluation'][u]['mappo']['avg'][0] for u in uav_to_show]
                enh_vals = [all_results['evaluation'][u]['enhanced']['avg'][0] for u in uav_to_show]
                trad_vals = [all_results['evaluation'][u]['traditional']['avg'][0] for u in uav_to_show]
                ax.bar(x - width, trad_vals, width, label='传统算法(3GPP)', color=COLORS['danger'], alpha=0.8)
                ax.bar(x, enh_vals, width, label='增强算法(本文)', color=COLORS['primary'], alpha=0.8)
                ax.bar(x + width, mappo_vals, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_ylabel('平均满意度')
                ax.set_title('(c) 算法对比: 传统→增强→BA-MAPPO')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')

        # 图4: 策略分布热力图
        ax = axes[1, 0]
        if 'evaluation' in all_results:
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
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

        # 图5: 多场景泛化对比 (仅核心算法: 传统+增强+BA-MAPPO)
        ax = axes[1, 1]
        ax.axis('off')
        if 'scenarios' in all_results and all_results['scenarios']:
            ax2 = fig.add_subplot(2, 3, 6)
            scenarios_data = all_results['scenarios']
            s_names = list(scenarios_data.keys())
            s_names_cn = [scenarios_data[s].get('name_cn', s) for s in s_names]

            trad_sats = [scenarios_data[s]['traditional'][0] for s in s_names]
            enh_sats = [scenarios_data[s]['enhanced'][0] for s in s_names]
            mappo_scenario_sats = [scenarios_data[s].get('mappo', (0, 0))[0] if 'mappo' in scenarios_data[s] else 0 for s in s_names]

            x = np.arange(len(s_names))
            width = 0.25
            has_mappo = any('mappo' in scenarios_data[s] for s in s_names)
            if has_mappo:
                offsets = [-width, 0, width]
                ax2.bar(x + offsets[0], trad_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + offsets[1], enh_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax2.bar(x + offsets[2], mappo_scenario_sats, width, label=mode_name, color=COLORS['warning'], alpha=0.8)
            else:
                ax2.bar(x - width/2, trad_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + width/2, enh_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
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
                    mp = ev['mappo']['avg'][0]
                    en = ev['enhanced']['avg'][0]
                    tr = ev['traditional']['avg'][0]
                    improvement_vs_trad = (mp - tr) / max(tr, 0.001) * 100
                    improvement_vs_enh = (mp - en) / max(en, 0.001) * 100
                    text_lines.append(f'UAV={num_uav}:')
                    text_lines.append(f'  {mode_name} vs 传统: {improvement_vs_trad:+.1f}%')
                    text_lines.append(f'  {mode_name} vs 增强: {improvement_vs_enh:+.1f}%')
                    text_lines.append(f'  {mode_name}={mp:.4f}, 增强={en:.4f}, 传统={tr:.4f}')
                    text_lines.append('')
        if 'scenarios' in all_results:
            text_lines.append('【场景泛化】\n')
            for sn, sd in all_results['scenarios'].items():
                line = f'  {sd["name_cn"]}: 传统={sd["traditional"][0]:.4f}, 增强={sd["enhanced"][0]:.4f}'
                if 'mappo' in sd:
                    line += f', MAPPO={sd["mappo"][0]:.4f}'
                text_lines.append(line)

        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'mappo_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  可视化已保存: {save_path}")
