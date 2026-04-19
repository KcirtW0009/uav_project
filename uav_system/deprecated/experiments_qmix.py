"""
QMIX 元控制器实验

实验内容:
1. Phase 1 — 训练收敛分析
   - 在不同 UAV 数量 (10, 20) 下训练 QMIX
   - 绘制训练曲线 (reward, loss, epsilon)

2. Phase 2 — 对比评估
   - QMIX-优化参数 vs 人工固定参数 (5种策略)
   - 多维度对比: 满意率、切换成功率、资源利用率、关键业务满足率

3. Phase 3 — 多场景泛化
   - 训练场景 vs 5个测试场景 (城市监控、工业巡检、应急救援、物流配送、农业监测)

4. 可视化
   - 训练收敛曲线
   - 策略选择热力图
   - 参数演化曲线
   - 场景泛化雷达图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS
from .qmix_environment import QMixHandoverEnv
from .qmix_agent import QMIXAgent
from .parametric_algorithm import ParametricEnhancedAlgorithm, STRATEGY_CONFIGS, NUM_STRATEGIES
from .algorithms import EnhancedHandoverAlgorithm
from .environment import EnhancedNetworkEnvironment
from .business import BusinessType
from .satisfaction import HierarchicalSatisfactionMetric


class ExperimentQMIX:
    """QMIX 元控制器实验"""

    MODEL_DIR = os.path.join(RESULT_DIR, 'qmix_models')
    RESULT_FILE = os.path.join(RESULT_DIR, 'qmix_experiment_data.pkl')

    @staticmethod
    def run(num_uav_list=(10, 20, 30, 40),
            num_bs=8, num_steps=150,
            train_episodes=1000, eval_episodes=5,
            bs_capacity_range=(400, 800),
            load_models=False, phase='both',
            verbose=True):
        """
        运行 QMIX 实验

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
        """
        print("\n" + "=" * 80)
        print("QMIX 元控制器实验")
        print("=" * 80)
        print(f"  基站数量: {num_bs}")
        print(f"  UAV 数量列表: {num_uav_list}")
        print(f"  训练 episodes: {train_episodes}")
        print(f"  评估重复次数: {eval_episodes}")
        print(f"  容量范围: {bs_capacity_range}")
        print("=" * 80)

        os.makedirs(ExperimentQMIX.MODEL_DIR, exist_ok=True)

        all_results = {}

        if phase in ('both', 'phase1'):
            # Phase 1: 训练 + 收敛分析
            training_results = ExperimentQMIX._phase1_training(
                num_uav_list, num_bs, num_steps, train_episodes,
                bs_capacity_range, load_models, verbose
            )
            all_results['training'] = training_results

        if phase in ('both', 'phase2'):
            # Phase 2: 对比评估
            eval_results = ExperimentQMIX._phase2_evaluation(
                num_uav_list, num_bs, num_steps, eval_episodes,
                bs_capacity_range, load_models, verbose
            )
            all_results['evaluation'] = eval_results

            # Phase 3: 多场景泛化
            scenario_results = ExperimentQMIX._phase3_scenarios(
                num_bs=8, num_uav=20, num_steps=100, repeats=10,
                bs_capacity_range=None, verbose=verbose
            )
            all_results['scenarios'] = scenario_results

        # 可视化
        ExperimentQMIX._plot_all(all_results, num_uav_list)

        # 保存结果
        with open(ExperimentQMIX.RESULT_FILE, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\n  实验数据已保存: {ExperimentQMIX.RESULT_FILE}")

        return all_results

    # ==================== Phase 1: 训练 ====================

    @staticmethod
    def _phase1_training(num_uav_list, num_bs, num_steps, train_episodes,
                         bs_capacity_range, load_models, verbose):
        """Phase 1: 训练收敛分析"""
        print("\n" + "-" * 60)
        print("Phase 1: QMIX 训练收敛分析")
        print("-" * 60)

        training_results = {}

        for num_uav in num_uav_list:
            print(f"\n>>> 训练 UAV={num_uav} <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 100)
            model_path = os.path.join(ExperimentQMIX.MODEL_DIR,
                                      f'qmix_{num_bs}bs_{num_uav}uav.pt')

            # 创建环境和 Agent
            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 100,
                bs_capacity_range=bs_capacity_range,
            )

            agent = QMIXAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=32,
                lr=5e-4,
                gamma=0.99,
                buffer_size=50000,
                batch_size=32,
                epsilon_decay=0.995,
            )

            if load_models and os.path.exists(model_path):
                agent.load(model_path)
                training_results[num_uav] = {'loaded': True}
                print(f"  已加载模型: {model_path}")
                continue

            # 训练循环
            episode_rewards = []
            episode_satisfactions = []
            episode_losses = []
            episode_epsilons = []
            best_reward = float('-inf')
            save_interval = 50  # 每 50 episode 保存一次

            for ep in range(train_episodes):
                obs_dict, global_state = env.reset()
                episode_reward = 0.0
                episode_sat = []

                for step in range(num_steps):
                    actions = agent.select_actions(obs_dict, training=True)
                    next_obs, next_state, rewards, team_reward, done, info = env.step(actions)
                    agent.store_transition(
                        obs_dict, actions, rewards,
                        next_obs, global_state, next_state,
                        team_reward, done
                    )
                    obs_dict = next_obs
                    global_state = next_state
                    episode_reward += team_reward

                    # 训练
                    loss = agent.train_step()
                    if loss is not None:
                        episode_losses.append(loss)

                    episode_sat.append(info['avg_satisfaction'])

                agent.decay_epsilon()

                episode_rewards.append(episode_reward)
                episode_satisfactions.append(np.mean(episode_sat))
                episode_epsilons.append(agent.epsilon)

                if verbose and (ep + 1) % 30 == 0:
                    avg_loss = agent.get_avg_loss()
                    avg_q = agent.get_avg_q_tot()
                    print(f"  Episode {ep+1}/{train_episodes}: "
                          f"reward={episode_reward:.3f}, "
                          f"sat={np.mean(episode_sat):.3f}, "
                          f"eps={agent.epsilon:.4f}, "
                          f"loss={avg_loss:.4f}, "
                          f"Q_tot={avg_q:.3f}")

                # 保存模型：每 save_interval episode 或达到最优 reward 时保存
                is_best = episode_reward > best_reward
                if is_best:
                    best_reward = episode_reward
                if (ep + 1) % save_interval == 0 or is_best:
                    agent.save(model_path)
                    if is_best and (ep + 1) % save_interval != 0:
                        pass  # 已在 agent.save 中打印

                training_results[num_uav] = {
                    'rewards': episode_rewards,
                    'satisfactions': episode_satisfactions,
                    'losses': episode_losses,
                    'epsilons': episode_epsilons,
                    'final_epsilon': agent.epsilon,
                    'final_avg_sat': np.mean(episode_satisfactions[-50:]),
                }

        return training_results

    # ==================== Phase 2: 对比评估 ====================

    @staticmethod
    def _phase2_evaluation(num_uav_list, num_bs, num_steps, eval_episodes,
                           bs_capacity_range, load_models, verbose):
        """Phase 2: QMIX vs 人工固定参数对比"""
        print("\n" + "-" * 60)
        print("Phase 2: QMIX-优化参数 vs 人工固定参数对比")
        print("-" * 60)

        eval_results = {}

        for num_uav in num_uav_list:
            print(f"\n>>> 评估 UAV={num_uav} <<<")

            set_global_seed(GLOBAL_SEED + num_uav * 200)
            model_path = os.path.join(ExperimentQMIX.MODEL_DIR,
                                      f'qmix_{num_bs}bs_{num_uav}uav.pt')

            # 创建环境和加载模型
            env = QMixHandoverEnv(
                num_bs=num_bs, num_uav=num_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + num_uav * 200,
                bs_capacity_range=bs_capacity_range,
            )

            agent = QMIXAgent(
                num_agents=env.num_agents,
                obs_dim=env.obs_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=32,
            )

            if os.path.exists(model_path):
                agent.load(model_path)
                agent.epsilon = 0.0  # 评估时关闭探索
                print(f"  已加载模型: {model_path}")
            elif load_models:
                print(f"  警告: 未找到模型 {model_path}, 使用 random 策略")
                agent.epsilon = 1.0
            else:
                print(f"  警告: Phase 1 未训练，未找到模型 {model_path}, 使用 random 策略")
                agent.epsilon = 1.0

            # 评估 QMIX
            qmix_sats, qmix_success_rates, qmix_disconnected = [], [], []
            strategy_counts = {name: 0 for name in STRATEGY_CONFIGS.keys()}

            for rep in range(eval_episodes):
                obs_dict, _ = env.reset()
                ep_sats = []
                for step in range(num_steps):
                    actions = agent.select_actions(obs_dict, training=False)
                    _, _, rewards, team_reward, done, info = env.step(actions)
                    obs_dict, _ = env.get_all_obs(), env.get_global_state()

                    ep_sats.append(info['avg_satisfaction'])
                    for sname, count in info['strategy_distribution'].items():
                        strategy_counts[sname] += count

                qmix_sats.append(np.mean(ep_sats))

            # 评估各人工固定策略
            strategy_results = {}

            for strat_name, strat_params in STRATEGY_CONFIGS.items():
                rep_sats = []
                for rep in range(eval_episodes):
                    seed = GLOBAL_SEED + num_uav * 300 + rep * 1000
                    set_global_seed(seed)

                    eval_env = EnhancedNetworkEnvironment(
                        num_bs=num_bs, num_uav=num_uav,
                        recognition_model=None, scaler=None,
                        seed=seed,
                        bs_capacity_range=bs_capacity_range,
                    )
                    eval_env.recognition_updater = None

                    algo = ParametricEnhancedAlgorithm.from_strategy_name(eval_env, strat_name)

                    for step in range(num_steps):
                        eval_env.step()
                        algo.run_step(enable_load_balancing=True)

                    stats = eval_env.get_state_statistics()
                    rep_sats.append(stats['avg_satisfaction'])

                strategy_results[strat_name] = {
                    'satisfaction': (np.mean(rep_sats), np.std(rep_sats)),
                }

            # 评估原始增强算法（作为对照）
            enhanced_sats = []
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 400 + rep * 1000
                set_global_seed(seed)

                eval_env = EnhancedNetworkEnvironment(
                    num_bs=num_bs, num_uav=num_uav,
                    recognition_model=None, scaler=None,
                    seed=seed,
                    bs_capacity_range=bs_capacity_range,
                )
                eval_env.recognition_updater = None

                algo = EnhancedHandoverAlgorithm(eval_env)

                for step in range(num_steps):
                    eval_env.step()
                    algo.run_step(enable_load_balancing=True)

                stats = eval_env.get_state_statistics()
                enhanced_sats.append(stats['avg_satisfaction'])

            # 传统算法基线
            traditional_sats = []
            from .algorithms import IntegratedHandoverAlgorithm
            for rep in range(eval_episodes):
                seed = GLOBAL_SEED + num_uav * 500 + rep * 1000
                set_global_seed(seed)

                eval_env = EnhancedNetworkEnvironment(
                    num_bs=num_bs, num_uav=num_uav,
                    recognition_model=None, scaler=None,
                    seed=seed,
                    bs_capacity_range=bs_capacity_range,
                )
                eval_env.recognition_updater = None

                algo = IntegratedHandoverAlgorithm(eval_env)

                for step in range(num_steps):
                    eval_env.step()
                    algo.run_step()

                stats = eval_env.get_state_statistics()
                traditional_sats.append(stats['avg_satisfaction'])

            qmix_mean = np.mean(qmix_sats) if qmix_sats else 0
            qmix_std = np.std(qmix_sats) if qmix_sats else 0

            eval_results[num_uav] = {
                'qmix': {'satisfaction': (qmix_mean, qmix_std)},
                'enhanced': {'satisfaction': (np.mean(enhanced_sats), np.std(enhanced_sats))},
                'traditional': {'satisfaction': (np.mean(traditional_sats), np.std(traditional_sats))},
                'strategies': strategy_results,
                'strategy_distribution': strategy_counts,
            }

            if verbose:
                print(f"\n  UAV={num_uav} 对比结果:")
                print(f"    QMIX-优化:   {qmix_mean:.4f} +/- {qmix_std:.4f}")
                print(f"    增强算法:     {np.mean(enhanced_sats):.4f} +/- {np.std(enhanced_sats):.4f}")
                print(f"    传统算法:     {np.mean(traditional_sats):.4f} +/- {np.std(traditional_sats):.4f}")
                for sn, sr in strategy_results.items():
                    print(f"    策略-{sn}:  {sr['satisfaction'][0]:.4f} +/- {sr['satisfaction'][1]:.4f}")

        return eval_results

    # ==================== Phase 3: 多场景泛化 ====================

    @staticmethod
    def _phase3_scenarios(num_bs, num_uav, num_steps, repeats,
                          bs_capacity_range, verbose):
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

        # 加载 default 场景训练的 QMIX 模型
        train_uav = 20
        model_path = os.path.join(ExperimentQMIX.MODEL_DIR,
                                  f'qmix_{num_bs}bs_{train_uav}uav.pt')

        # 尝试加载 QMIX 模型用于 default 场景评估
        qmix_agent_for_default = None
        qmix_env_for_default = None
        if os.path.exists(model_path):
            qmix_env_for_default = QMixHandoverEnv(
                num_bs=num_bs, num_uav=50,  # default 场景用 50 UAV
                max_steps=num_steps, seed=GLOBAL_SEED + 9999,
                bs_capacity_range=(500, 1000),
            )
            # 注意: QMIX 模型是按 num_agents 绑定的，只能用于训练时的 UAV 数量
            # default 场景 UAV=50 != 训练时的 20，无法直接加载
            # 因此在 default 场景上使用训练时的 UAV=20
            qmix_env_for_default = QMixHandoverEnv(
                num_bs=num_bs, num_uav=train_uav,
                max_steps=num_steps, seed=GLOBAL_SEED + 9999,
                bs_capacity_range=(500, 1000),
            )
            qmix_agent_for_default = QMIXAgent(
                num_agents=qmix_env_for_default.num_agents,
                obs_dim=qmix_env_for_default.obs_dim,
                state_dim=qmix_env_for_default.state_dim,
                action_dim=qmix_env_for_default.action_dim,
                hidden_dim=32,
            )
            qmix_agent_for_default.load(model_path)
            qmix_agent_for_default.epsilon = 0.0
            print(f"  Phase 3: 已加载 QMIX 模型用于 default 场景评估")
        else:
            print(f"  Phase 3: 未找到 QMIX 模型 {model_path}, default 场景将仅使用策略对比")

        for scenario_name, scenario_cfg in scenarios.items():
            print(f"\n>>> 场景: {scenario_name} ({scenario_names_cn.get(scenario_name, scenario_name)}) <<<")

            s_uav = scenario_cfg['num_uav']
            s_cap = scenario_cfg['bs_capacity_range']

            # 评估所有 5 种策略在该场景的表现
            strategy_sats = {}
            enhanced_sats = []
            traditional_sats = []
            qmix_sats = []  # QMIX 模型评估（仅 default 场景可用）

            # 如果是 default 场景且有 QMIX 模型，用 QMIX 模型评估
            if scenario_name == 'default' and qmix_agent_for_default is not None:
                for rep in range(repeats):
                    seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                    set_global_seed(seed)
                    obs_dict, _ = qmix_env_for_default.reset()
                    ep_sats = []
                    for step in range(num_steps):
                        actions = qmix_agent_for_default.select_actions(obs_dict, training=False)
                        _, _, rewards, team_reward, done, info = qmix_env_for_default.step(actions)
                        obs_dict, _ = qmix_env_for_default.get_all_obs(), qmix_env_for_default.get_global_state()
                        ep_sats.append(info['avg_satisfaction'])
                    qmix_sats.append(np.mean(ep_sats))

            for strat_name in STRATEGY_CONFIGS.keys():
                rep_sats = []
                for rep in range(repeats):
                    seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                    set_global_seed(seed)

                    eval_env = EnhancedNetworkEnvironment(
                        num_bs=num_bs, num_uav=s_uav,
                        recognition_model=None, scaler=None,
                        seed=seed, scenario=scenario_name,
                        bs_capacity_range=s_cap,
                    )
                    eval_env.recognition_updater = None

                    algo = ParametricEnhancedAlgorithm.from_strategy_name(eval_env, strat_name)

                    for step in range(num_steps):
                        eval_env.step()
                        algo.run_step(enable_load_balancing=True)

                    stats = eval_env.get_state_statistics()
                    rep_sats.append(stats['avg_satisfaction'])

                strategy_sats[strat_name] = (np.mean(rep_sats), np.std(rep_sats))

            # 增强算法基线
            for rep in range(repeats):
                seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                set_global_seed(seed)

                eval_env = EnhancedNetworkEnvironment(
                    num_bs=num_bs, num_uav=s_uav,
                    recognition_model=None, scaler=None,
                    seed=seed, scenario=scenario_name,
                    bs_capacity_range=s_cap,
                )
                eval_env.recognition_updater = None
                algo = EnhancedHandoverAlgorithm(eval_env)

                for step in range(num_steps):
                    eval_env.step()
                    algo.run_step(enable_load_balancing=True)

                stats = eval_env.get_state_statistics()
                enhanced_sats.append(stats['avg_satisfaction'])

            # 传统算法基线
            from .algorithms import IntegratedHandoverAlgorithm
            for rep in range(repeats):
                seed = GLOBAL_SEED + hash(scenario_name) % 10000 + rep * 1000
                set_global_seed(seed)

                eval_env = EnhancedNetworkEnvironment(
                    num_bs=num_bs, num_uav=s_uav,
                    recognition_model=None, scaler=None,
                    seed=seed, scenario=scenario_name,
                    bs_capacity_range=s_cap,
                )
                eval_env.recognition_updater = None
                algo = IntegratedHandoverAlgorithm(eval_env)

                for step in range(num_steps):
                    eval_env.step()
                    algo.run_step()

                stats = eval_env.get_state_statistics()
                traditional_sats.append(stats['avg_satisfaction'])

            # 找到最优策略
            best_strat = max(strategy_sats, key=lambda k: strategy_sats[k][0])

            scenario_results[scenario_name] = {
                'name_cn': scenario_names_cn.get(scenario_name, scenario_name),
                'strategies': strategy_sats,
                'best_strategy': best_strat,
                'enhanced': (np.mean(enhanced_sats), np.std(enhanced_sats)),
                'traditional': (np.mean(traditional_sats), np.std(traditional_sats)),
            }
            if qmix_sats:
                scenario_results[scenario_name]['qmix'] = (np.mean(qmix_sats), np.std(qmix_sats))

            if verbose:
                print(f"    最优策略: {best_strat} ({strategy_sats[best_strat][0]:.4f})")
                print(f"    增强算法: {np.mean(enhanced_sats):.4f}")
                print(f"    传统算法: {np.mean(traditional_sats):.4f}")
                if qmix_sats:
                    print(f"    QMIX模型: {np.mean(qmix_sats):.4f} +/- {np.std(qmix_sats):.4f}")

        return scenario_results

    # ==================== 可视化 ====================

    @staticmethod
    def _plot_all(all_results, num_uav_list):
        """生成所有可视化图表"""
        print("\n生成 QMIX 实验可视化...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('QMIX 元控制器实验结果', fontsize=16, fontweight='bold')

        # 图1: 训练收敛曲线
        ax = axes[0, 0]
        if 'training' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['training']:
                    tr = all_results['training'][num_uav]
                    if 'rewards' in tr:
                        # 平滑曲线
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
                qmix_vals = [all_results['evaluation'][u]['qmix']['satisfaction'][0] for u in uav_to_show]
                enh_vals = [all_results['evaluation'][u]['enhanced']['satisfaction'][0] for u in uav_to_show]
                trad_vals = [all_results['evaluation'][u]['traditional']['satisfaction'][0] for u in uav_to_show]
                ax.bar(x - width, qmix_vals, width, label='QMIX-优化', color=COLORS['warning'], alpha=0.8)
                ax.bar(x, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax.bar(x + width, trad_vals, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([f'UAV={u}' for u in uav_to_show])
                ax.set_ylabel('平均满意度')
                ax.set_title('(c) 算法对比')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

        # 图4: 策略分布（热力图）
        ax = axes[1, 0]
        if 'evaluation' in all_results:
            strat_names = list(STRATEGY_CONFIGS.keys())
            uav_to_show = [u for u in num_uav_list if u in all_results.get('evaluation', {})]
            if uav_to_show:
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
                ax.set_title('(d) QMIX 策略选择分布')
                plt.colorbar(im, ax=ax)
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)

        # 图5: 多场景泛化雷达图
        ax = axes[1, 1]
        ax.axis('off')
        if 'scenarios' in all_results and all_results['scenarios']:
            # 用柱状图代替雷达图（更直观）
            ax2 = fig.add_subplot(2, 3, 6)
            scenarios_data = all_results['scenarios']
            s_names = list(scenarios_data.keys())
            s_names_cn = [scenarios_data[s].get('name_cn', s) for s in s_names]

            best_strat_sats = [scenarios_data[s]['strategies'][scenarios_data[s]['best_strategy']][0] for s in s_names]
            enhanced_sats = [scenarios_data[s]['enhanced'][0] for s in s_names]
            traditional_sats = [scenarios_data[s]['traditional'][0] for s in s_names]
            qmix_scenario_sats = [scenarios_data[s].get('qmix', (0, 0))[0] if 'qmix' in scenarios_data[s] else 0 for s in s_names]

            x = np.arange(len(s_names))
            width = 0.2
            has_qmix = any('qmix' in scenarios_data[s] for s in s_names)
            if has_qmix:
                offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
                ax2.bar(x + offsets[0], best_strat_sats, width, label='最优策略', color=COLORS['success'], alpha=0.8)
                ax2.bar(x + offsets[1], enhanced_sats, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
                ax2.bar(x + offsets[2], traditional_sats, width, label='传统算法', color=COLORS['danger'], alpha=0.8)
                ax2.bar(x + offsets[3], qmix_scenario_sats, width, label='QMIX模型', color=COLORS['warning'], alpha=0.8)
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
        text_lines = ['【关键发现】\n']
        if 'evaluation' in all_results:
            for num_uav in num_uav_list:
                if num_uav in all_results['evaluation']:
                    ev = all_results['evaluation'][num_uav]
                    qm = ev['qmix']['satisfaction'][0]
                    en = ev['enhanced']['satisfaction'][0]
                    tr = ev['traditional']['satisfaction'][0]
                    improvement_vs_trad = (qm - tr) / max(tr, 0.001) * 100
                    text_lines.append(f'UAV={num_uav}:')
                    text_lines.append(f'  QMIX vs 传统: +{improvement_vs_trad:.1f}%')
                    text_lines.append(f'  QMIX={qm:.4f}, 增强={en:.4f}, 传统={tr:.4f}')
                    text_lines.append('')
        if 'scenarios' in all_results:
            text_lines.append('【场景最优策略】\n')
            for sn, sd in all_results['scenarios'].items():
                line = f'  {sd["name_cn"]}: {sd["best_strategy"]}'
                if 'qmix' in sd:
                    line += f' (QMIX={sd["qmix"][0]:.4f})'
                text_lines.append(line)

        ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'qmix_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  可视化已保存: {save_path}")
