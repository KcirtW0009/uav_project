"""
实验5：RL 辅助决策 vs 启发式算法对比实验

三种策略对比：
1. 传统算法（3GPP A3 基线）
2. 增强启发式算法（业务感知 + 多机制协同）
3. DQN 强化学习辅助决策

对比维度：满意度、切换成功率、切换次数、吞吐量、连接保持率等。
包含统计显著性检验（t-test / Mann-Whitney U）。
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR, COLORS
from uav_system.environment import NetworkEnvironmentWithRecognition
from uav_system.algorithms import IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm
from uav_system.rl_environment import RLHandoverEnv
from uav_system.rl_agent import DQNAgent


matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class Experiment5:
    """
    实验5：RL 辅助决策 vs 启发式算法对比

    对三种切换策略进行全面对比：
    - 传统算法 (3GPP A3)
    - 增强启发式算法
    - DQN 强化学习（训练好的模型 或 先训练再评估）
    """

    ALGO_NAMES = {
        'traditional': '传统算法(3GPP A3)',
        'enhanced': '增强启发式算法',
        'dqn': 'DQN强化学习',
    }

    # 对比指标
    METRICS = {
        'avg_satisfaction': '平均满意度',
        'final_satisfaction': '最终满意度',
        'min_satisfaction': '最低满意度',
        'handover_count': '切换次数',
        'avg_satisfaction_per_step': '步均满意度',
        'connected_ratio': '连接保持率',
        'avg_allocated_rate': '平均分配速率(Mbps)',
    }

    @staticmethod
    def run(num_steps=150, repeats=10, num_bs=8, num_uav=20,
             target_uav_id=0, dqn_train_episodes=1000,
             load_model=False, model_path=None, verbose=False):
        """
        运行实验5

        Args:
            num_steps: 每个 episode 的步数
            repeats: 重复实验次数（不同随机种子）
            num_bs: 基站数量
            num_uav: UAV 总数
            target_uav_id: RL 控制的目标 UAV
            dqn_train_episodes: DQN 训练 episodes
            load_model: 是否加载已有模型
            model_path: 模型路径
            verbose: 是否打印详细调试信息
        """
        print("=" * 80)
        print("实验5：RL 辅助决策 vs 启发式算法对比")
        print("=" * 80)
        print(f"\n配置: {num_bs} 基站 × {num_uav} UAV × {num_steps} 步 × {repeats} 次重复")
        print(f"目标 UAV: {target_uav_id}")
        print(f"DQN 训练: {'加载模型 ' + model_path if load_model else str(dqn_train_episodes) + ' episodes'}")

        # ---- Step 1: 训练或加载 DQN 模型 ----
        print("\n--- Step 1: DQN 模型准备 ---")
        t0 = time.time()

        rl_env_template = RLHandoverEnv(
            NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED),
            target_uav_id=target_uav_id, max_steps=num_steps
        )

        agent = DQNAgent(
            state_dim=rl_env_template.state_dim,
            action_dim=rl_env_template.action_dim,
            lr=5e-4, gamma=0.95, hidden_dim=128,
            buffer_size=50000, batch_size=64,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
            target_update_freq=500,
        )

        if load_model and model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  已加载模型: {model_path}")
        else:
            print(f"  训练 DQN ({dqn_train_episodes} episodes)...")
            rl_env_train = RLHandoverEnv(
                NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED),
                target_uav_id=target_uav_id, max_steps=num_steps
            )
            # 训练过程数据收集
            train_rewards = []
            train_losses = []
            train_sats = []
            train_hos = []
            epsilon_history = []
            for ep in range(dqn_train_episodes):
                # 多样化训练种子：每个 episode 使用不同种子，让 DQN 见过各种拓扑
                set_global_seed(GLOBAL_SEED + ep % 20)
                state = rl_env_train.reset()
                set_global_seed(GLOBAL_SEED + ep % 20)  # reset 会重置种子，再设一次
                ep_reward = 0.0
                ep_sat_sum = 0.0
                ep_ho = 0
                ep_losses = []
                for step_i in range(num_steps):
                    invalid = rl_env_train.get_invalid_actions()
                    action = agent.select_action(state, training=True, invalid_actions=invalid)
                    next_state, reward, done, info = rl_env_train.step(action)
                    next_invalid = rl_env_train.get_invalid_actions()
                    agent.store_transition(state, action, reward, next_state, float(done),
                                           next_invalid_actions=next_invalid)
                    loss = agent.train_step()
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_reward += reward
                    ep_sat_sum += info['satisfaction']
                    ep_ho = info['total_handovers']
                    state = next_state
                    if done:
                        break
                agent.decay_epsilon()

                # 记录训练数据
                train_rewards.append(ep_reward)
                train_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
                train_sats.append(ep_sat_sum / num_steps)
                train_hos.append(ep_ho)
                epsilon_history.append(agent.epsilon)

                if (ep + 1) % 100 == 0:
                    avg_r = np.mean(train_rewards[-100:])
                    avg_loss = np.mean(train_losses[-100:])
                    print(f"    Ep {ep+1}/{dqn_train_episodes}, eps={agent.epsilon:.3f}, "
                          f"avg_R={avg_r:.1f}, avg_sat={np.mean(train_sats[-100:]):.3f}, "
                          f"avg_ho={np.mean(train_hos[-100:]):.1f}, avg_loss={avg_loss:.4f}")

            # 保存训练好的模型
            save_path = os.path.join(RESULT_DIR, 'dqn_exp5_model.pt')
            agent.save(save_path)

            # 保存训练数据
            train_data_path = os.path.join(RESULT_DIR, 'dqn_exp5_training_data.npz')
            np.savez(train_data_path,
                     episode_rewards=train_rewards, episode_losses=train_losses,
                     epsilon_history=epsilon_history, episode_satisfactions=train_sats,
                     episode_handovers=train_hos)

            # 绘制训练曲线
            Experiment5._plot_single_training_curves(
                {num_uav: {
                    'rewards': train_rewards, 'losses': train_losses,
                    'epsilon': epsilon_history, 'sats': train_sats, 'hos': train_hos,
                }}, dqn_train_episodes,
                save_name='dqn_training_curves_exp5.png'
            )

        print(f"  模型准备完成, 耗时 {time.time()-t0:.1f}s")

        # ---- Step 2: 三种算法对比实验 ----
        print("\n--- Step 2: 对比实验 ---")
        results = {
            'traditional': [],
            'enhanced': [],
            'dqn': [],
        }

        for rep in range(repeats):
            print(f"\n  重复 {rep+1}/{repeats}")
            seed = GLOBAL_SEED + rep

            # --- 传统算法 ---
            set_global_seed(seed)
            env_trad = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed)
            algo_trad = IntegratedHandoverAlgorithm(env_trad)
            trad_result = Experiment5._run_single(env_trad, algo_trad, num_steps, target_uav_id)
            results['traditional'].append(trad_result)

            # --- 增强启发式算法 ---
            set_global_seed(seed)
            env_enh = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed)
            algo_enh = EnhancedHandoverAlgorithm(env_enh)
            algo_enh.epsilon = 0.0  # 关闭探索以公平对比
            enh_result = Experiment5._run_single(env_enh, algo_enh, num_steps, target_uav_id)
            results['enhanced'].append(enh_result)

            # --- DQN ---
            set_global_seed(seed)
            env_dqn = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed)
            rl_env = RLHandoverEnv(env_dqn, target_uav_id=target_uav_id, max_steps=num_steps)
            dqn_result = Experiment5._run_dqn_eval(rl_env, agent, num_steps, target_uav_id,
                                                     verbose=(verbose and rep == repeats - 1))
            results['dqn'].append(dqn_result)

            print(f"    传统: Sat={trad_result['avg_satisfaction']:.3f}, "
                  f"HO={trad_result['handover_count']:.0f}")
            print(f"    增强: Sat={enh_result['avg_satisfaction']:.3f}, "
                  f"HO={enh_result['handover_count']:.0f}")
            print(f"    DQN:  Sat={dqn_result['avg_satisfaction']:.3f}, "
                  f"HO={dqn_result['handover_count']:.0f}, "
                  f"有效切换={dqn_result.get('effective_switch_count', 0):.0f}/{dqn_result.get('switch_attempts', 0):.0f}")

        # ---- Step 3: 统计分析 ----
        summary = Experiment5._summarize(results)
        Experiment5._print_results_table(summary)
        Experiment5._statistical_tests(results)
        Experiment5._plot(summary, results)

        # 保存数据
        import pickle, json
        from datetime import datetime
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        with open(os.path.join(RESULT_DIR, 'exp5_data.pkl'), 'wb') as f:
            pickle.dump(summary, f)
        with open(os.path.join(RESULT_DIR, 'exp5_data.json'), 'w', encoding='utf-8') as f:
            json.dump(_convert(summary), f, ensure_ascii=False, indent=2)
        print(f"  数据已保存: exp5_data.pkl / exp5_data.json")

        return summary

    @staticmethod
    def _plot_single_training_curves(all_training_data, num_episodes, save_name='dqn_training_curves.png'):
        """
        绘制 DQN 训练曲线（单场景或多场景）

        Args:
            all_training_data: {num_uav: {rewards, losses, epsilon, sats, hos}}
            num_episodes: 训练总 episodes
            save_name: 保存文件名
        """
        uav_counts = sorted(all_training_data.keys())
        n = len(uav_counts)

        fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('DQN 训练过程曲线', fontsize=16, fontweight='bold', y=1.01)

        for row_idx, num_uav in enumerate(uav_counts):
            data = all_training_data[num_uav]
            rewards = data['rewards']
            losses = data['losses']
            epsilon = data['epsilon']
            sats = data['sats']
            hos = data['hos']

            episodes = range(1, len(rewards) + 1)
            title_prefix = f'{num_uav} UAV'
            color_r = '#667eea'
            color_s = '#4ECDC4'
            color_l = '#FF6B6B'
            color_e = '#764ba2'

            # 1. 奖励曲线
            ax = axes[row_idx, 0]
            ax.plot(list(episodes), rewards, alpha=0.3, color=color_r, linewidth=0.8)
            window = min(20, len(rewards))
            if window >= 5:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(rewards) + 1), ma,
                        color=color_r, linewidth=2, label=f'MA(w={window})')
            ax.set_title(f'{title_prefix} - 奖励曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('总奖励')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 2. 满意度曲线
            ax = axes[row_idx, 1]
            ax.plot(list(episodes), sats, alpha=0.3, color=color_s, linewidth=0.8)
            if window >= 5:
                ma = np.convolve(sats, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(sats) + 1), ma,
                        color=color_s, linewidth=2, label=f'MA(w={window})')
            ax.set_title(f'{title_prefix} - 满意度曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('平均满意度')
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 3. Loss 曲线
            ax = axes[row_idx, 2]
            valid_losses = [l for l in losses if l > 0]
            if valid_losses:
                ax.plot(range(1, len(valid_losses) + 1), valid_losses,
                        alpha=0.4, color=color_l, linewidth=0.8)
                loss_window = min(50, len(valid_losses))
                if loss_window >= 5:
                    ma = np.convolve(valid_losses, np.ones(loss_window)/loss_window, mode='valid')
                    ax.plot(range(loss_window, len(valid_losses) + 1), ma,
                            color=color_l, linewidth=2, label=f'MA(w={loss_window})')
            ax.set_title(f'{title_prefix} - Loss 曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 4. Epsilon + 切换次数
            ax = axes[row_idx, 3]
            ax_twin = ax.twinx()
            ax.plot(list(episodes), epsilon, color=color_e, linewidth=1.5, label='ε (探索率)')
            ax_twin.plot(list(episodes), hos, alpha=0.4, color='#fbbf24',
                         linewidth=0.8, label='切换次数')
            ax.set_xlabel('Episode')
            ax.set_ylabel('ε', color=color_e)
            ax_twin.set_ylabel('切换次数', color='#fbbf24')
            ax.set_title(f'{title_prefix} - 探索率 & 切换次数', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  训练曲线已保存: {save_path}")

    @staticmethod
    def _run_single(env, algo, num_steps, target_uav_id):
        """运行单个仿真 episode，返回指标"""
        satisfaction_history = []
        handover_count = 0
        connected_count = 0
        rate_history = []

        for step in range(num_steps):
            env.step()
            if hasattr(algo, 'run_step'):
                if isinstance(algo, EnhancedHandoverAlgorithm):
                    algo.run_step(enable_load_balancing=False)
                else:
                    algo.run_step()

            uav = env.uavs[target_uav_id]
            satisfaction_history.append(uav.current_satisfaction)
            if uav.connected_bs_id is not None:
                connected_count += 1
                rate_history.append(uav.current_allocated_rate)

        uav = env.uavs[target_uav_id]

        return {
            'avg_satisfaction': np.mean(satisfaction_history),
            'final_satisfaction': satisfaction_history[-1],
            'min_satisfaction': np.min(satisfaction_history),
            'handover_count': uav.handover_count,
            'avg_satisfaction_per_step': np.mean(satisfaction_history),
            'connected_ratio': connected_count / num_steps,
            'avg_allocated_rate': np.mean(rate_history) if rate_history else 0.0,
            'satisfaction_history': satisfaction_history,
        }

    @staticmethod
    def _run_dqn_eval(rl_env, agent, num_steps, target_uav_id, verbose=False):
        """
        运行 DQN 评估 episode，收集详细调试信息

        Args:
            rl_env: RL 环境
            agent: DQN Agent
            num_steps: 步数
            target_uav_id: 目标 UAV ID
            verbose: 是否打印每10步详细状态
        """
        state = rl_env.reset()
        satisfaction_history = []
        connected_count = 0
        rate_history = []

        # 调试统计
        action_types = []        # 'stay' / 'switch'
        switch_attempts = 0      # RL 选择 switch 的次数
        effective_switches = 0   # 实际执行成功的次数
        skipped_same_bs = 0      # 跳过（同基站）
        skipped_failed = 0       # 跳过（分配失败+回滚失败）

        for step in range(num_steps):
            invalid = rl_env.get_invalid_actions()
            action = agent.select_action(state, training=False, invalid_actions=invalid)
            next_state, reward, done, info = rl_env.step(action)
            satisfaction_history.append(info['satisfaction'])
            if info['connected_bs'] is not None:
                connected_count += 1
                rate_history.append(info['allocated_rate'])

            # 收集调试信息
            action_types.append(info['action_type'])
            if info['action_type'] == 'switch':
                switch_attempts += 1
                if info.get('actual_switch', False):
                    effective_switches += 1

            # 详细打印（每 10 步）
            if verbose and (step + 1) % 10 == 0:
                uav = rl_env.env.uavs[target_uav_id]
                print(f"      Step {step+1:3d}: BS={info['connected_bs']}, "
                      f"Sat={info['satisfaction']:.3f}, "
                      f"Rate={info['allocated_rate']:.1f}, "
                      f"Action={info['action_type']}→BS{info.get('action_target_bs','-')}, "
                      f"实际切换={info.get('actual_switch', False)}, "
                      f"累计切换={info['total_handovers']}")

            state = next_state
            if done:
                break

        if verbose:
            stay_count = action_types.count('stay')
            print(f"\n    === DQN 评估调试摘要 ===")
            print(f"    总步数: {len(action_types)}")
            print(f"    动作分布: stay={stay_count}, switch={switch_attempts} "
                  f"(stay率={stay_count/len(action_types)*100:.1f}%)")
            print(f"    切换效率: 有效切换={effective_switches}/{switch_attempts} "
                  f"(有效率={effective_switches/max(switch_attempts,1)*100:.1f}%)")
            print(f"    环境层 handover_count={info['handover_count']}, "
                  f"RL层 total_handovers={info['total_handovers']}")

        result = {
            'avg_satisfaction': np.mean(satisfaction_history),
            'final_satisfaction': satisfaction_history[-1],
            'min_satisfaction': np.min(satisfaction_history),
            'handover_count': info.get('handover_count', 0),
            'avg_satisfaction_per_step': np.mean(satisfaction_history),
            'connected_ratio': connected_count / num_steps,
            'avg_allocated_rate': np.mean(rate_history) if rate_history else 0.0,
            'satisfaction_history': satisfaction_history,
            'switch_attempts': switch_attempts,
            'effective_switch_count': effective_switches,
        }

        return result

    @staticmethod
    def _summarize(results):
        """汇总结果"""
        summary = {}
        for algo_name in ['traditional', 'enhanced', 'dqn']:
            data_list = results[algo_name]
            summary[algo_name] = {}
            for key in Experiment5.METRICS.keys():
                if key == 'satisfaction_history':
                    continue
                vals = [d[key] for d in data_list]
                summary[algo_name][key] = (np.mean(vals), np.std(vals))
            # 满意度时序数据
            summary[algo_name]['satisfaction_history'] = data_list[0]['satisfaction_history']
            # DQN 特有的切换效率
            if algo_name == 'dqn':
                summary[algo_name]['switch_attempts'] = (np.mean([d['switch_attempts'] for d in data_list]),
                                                          np.std([d['switch_attempts'] for d in data_list]))
                summary[algo_name]['effective_switch_count'] = (np.mean([d['effective_switch_count'] for d in data_list]),
                                                                  np.std([d['effective_switch_count'] for d in data_list]))
        return summary

    @staticmethod
    def _print_results_table(summary):
        """打印结果表格"""
        headers = ["指标", "传统算法", "增强启发式", "DQN", "DQN vs 增强"]

        print("\n" + "=" * 100)
        print("实验5结果：RL 辅助决策 vs 启发式算法对比")
        print("=" * 100)

        rows_with_improve = []
        for key, name in Experiment5.METRICS.items():
            row = [name]
            for algo in ['traditional', 'enhanced', 'dqn']:
                if key in summary[algo]:
                    mean, std = summary[algo][key]
                    row.append(f"{mean:.4f}+/-{std:.4f}")
                else:
                    row.append("N/A")
            # DQN vs 增强
            if key in summary['enhanced'] and key in summary['dqn']:
                enh = summary['enhanced'][key][0]
                dqn = summary['dqn'][key][0]
                improvement = (dqn - enh) / abs(enh) * 100 if enh != 0 else 0
                row.append(f"{improvement:+.1f}%")
            else:
                row.append("-")
            rows_with_improve.append(row)

        # 打印表格
        col_widths = [max(len(str(item)) for item in col) + 2 for col in zip(*([headers] + rows_with_improve))]
        header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        print(f"\n  {header_line}")
        print(f"  {'-' * len(header_line)}")
        for row in rows_with_improve:
            line = "  ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths))
            print(f"  {line}")

        # DQN 切换效率
        if 'switch_attempts' in summary.get('dqn', {}):
            sa = summary['dqn']['switch_attempts'][0]
            esc = summary['dqn']['effective_switch_count'][0]
            print(f"\n  DQN 切换效率: 尝试={sa:.1f}次, 有效={esc:.1f}次 "
                  f"(有效率={esc/max(sa,1)*100:.1f}%)")

        # 关键发现
        print(f"\n【关键发现】")
        enh_sat = summary['enhanced']['avg_satisfaction'][0]
        dqn_sat = summary['dqn']['avg_satisfaction'][0]
        trad_sat = summary['traditional']['avg_satisfaction'][0]

        print(f"  - DQN 平均满意度: {dqn_sat:.4f}")
        print(f"  - 增强启发式满意度: {enh_sat:.4f}")
        print(f"  - 传统算法满意度: {trad_sat:.4f}")

        if dqn_sat > enh_sat:
            print(f"  - DQN 相对增强启发式提升: {(dqn_sat-enh_sat)/enh_sat*100:+.1f}%")
        elif dqn_sat > enh_sat * 0.95:
            print(f"  - DQN 达到增强启发式 {(dqn_sat/enh_sat)*100:.1f}% 的性能水平")
        else:
            print(f"  - DQN 性能低于增强启发式，需要分析原因")

        enh_ho = summary['enhanced']['handover_count'][0]
        dqn_ho = summary['dqn']['handover_count'][0]
        print(f"  - DQN 切换次数: {dqn_ho:.1f} (增强: {enh_ho:.1f}, "
              f"{'更少切换' if dqn_ho < enh_ho else '更多切换'})")

        print("=" * 100)

    @staticmethod
    def _statistical_tests(results):
        """统计显著性检验"""
        print("\n" + "=" * 80)
        print("统计显著性检验")
        print("=" * 80)

        metrics_to_test = ['avg_satisfaction', 'final_satisfaction', 'min_satisfaction',
                           'connected_ratio', 'handover_count', 'avg_allocated_rate']
        pairs = [('dqn', 'enhanced', 'DQN', '增强启发式'),
                 ('dqn', 'traditional', 'DQN', '传统算法'),
                 ('enhanced', 'traditional', '增强启发式', '传统算法')]

        for m in metrics_to_test:
            print(f"\n  【{Experiment5.METRICS.get(m, m)}】")
            for algo1, algo2, name1, name2 in pairs:
                group1 = [r[m] for r in results[algo1]]
                group2 = [r[m] for r in results[algo2]]

                # 选择检验方法
                _, p1 = stats.shapiro(group1) if len(group1) >= 3 else (1, 1)
                _, p2 = stats.shapiro(group2) if len(group2) >= 3 else (1, 1)
                test_method = 'ttest' if (p1 > 0.05 and p2 > 0.05) else 'mannwhitney'

                if test_method == 'ttest':
                    statistic, p_value = stats.ttest_ind(group1, group2)
                else:
                    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                direction = "↑" if np.mean(group1) > np.mean(group2) else "↓"
                print(f"    {name1} vs {name2}: "
                      f"{np.mean(group1):.4f} {direction} {np.mean(group2):.4f}, "
                      f"p={p_value:.4f} {significant}")

    @staticmethod
    def _plot(summary, results):
        """绘制对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验5：RL 辅助决策 vs 启发式算法对比', fontsize=15, fontweight='bold')

        algos = ['traditional', 'enhanced', 'dqn']
        algo_labels = [Experiment5.ALGO_NAMES[a] for a in algos]
        algo_colors = [COLORS['neutral'], COLORS['primary'], COLORS['success']]

        # 1. 平均满意度对比柱状图
        ax = axes[0, 0]
        sats = [summary[a]['avg_satisfaction'][0] for a in algos]
        stds = [summary[a]['avg_satisfaction'][1] for a in algos]
        bars = ax.bar(algo_labels, sats, yerr=stds, color=algo_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, sats):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('平均满意度')
        ax.set_title('目标 UAV 平均满意度对比', fontweight='bold')
        ax.set_ylim(0, max(sats) * 1.3)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. 连接保持率
        ax = axes[0, 1]
        conn = [summary[a]['connected_ratio'][0] for a in algos]
        conn_std = [summary[a]['connected_ratio'][1] for a in algos]
        bars = ax.bar(algo_labels, conn, yerr=conn_std, color=algo_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, conn):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('连接保持率')
        ax.set_title('目标 UAV 连接保持率对比', fontweight='bold')
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 切换次数
        ax = axes[0, 2]
        ho = [summary[a]['handover_count'][0] for a in algos]
        ho_std = [summary[a]['handover_count'][1] for a in algos]
        bars = ax.bar(algo_labels, ho, yerr=ho_std, color=algo_colors, alpha=0.8,
                      edgecolor='white', linewidth=1.5, capsize=5)
        for bar, val in zip(bars, ho):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('切换次数')
        ax.set_title('目标 UAV 切换次数对比', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. 满意度时序曲线（取最后一次重复）
        ax = axes[1, 0]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            history = results[algo][-1]['satisfaction_history']
            window = min(10, len(history) // 5) if len(history) > 20 else 1
            if window > 1:
                smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
                x_smooth = range(window-1, len(history))
                ax.plot(x_smooth, smoothed, color=color, linewidth=2, label=label, alpha=0.9)
            else:
                ax.plot(history, color=color, linewidth=2, label=label, alpha=0.9)
        ax.set_xlabel('仿真步')
        ax.set_ylabel('满意度')
        ax.set_title('满意度变化时序（单次典型运行）', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

        # 5. 雷达图
        ax = fig.add_subplot(2, 3, 5, projection='polar')
        categories = ['平均满意度', '最终满意度', '连接保持率', '分配速率', '切换效率']
        metric_keys = ['avg_satisfaction', 'final_satisfaction', 'connected_ratio',
                       'avg_allocated_rate', 'handover_count']

        for algo, label, color in zip(algos, algo_labels, algo_colors):
            vals = []
            for mk in metric_keys:
                if mk in summary[algo]:
                    v = summary[algo][mk][0]
                    if mk == 'avg_allocated_rate':
                        max_val = max(summary[a]['avg_allocated_rate'][0] for a in algos)
                        v = v / max_val if max_val > 0 else 0
                    elif mk == 'handover_count':
                        max_ho = max(summary[a]['handover_count'][0] for a in algos)
                        v = 1 - (v / max_ho) if max_ho > 0 else 0
                    vals.append(v)
                else:
                    vals.append(0)
            vals += vals[:1]
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=label, color=color, markersize=5)
            ax.fill(angles, vals, alpha=0.1, color=color)

        ax.set_xticks(np.linspace(0, 2*np.pi, len(categories), endpoint=False))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title('综合性能雷达图', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

        # 6. 箱线图
        ax = axes[1, 1]
        bp_data_trad = [r['avg_satisfaction'] for r in results['traditional']]
        bp_data_enh = [r['avg_satisfaction'] for r in results['enhanced']]
        bp_data_dqn = [r['avg_satisfaction'] for r in results['dqn']]
        bp = ax.boxplot([bp_data_trad, bp_data_enh, bp_data_dqn],
                        tick_labels=algo_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], algo_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('平均满意度')
        ax.set_title('满意度分布（{0}次重复）'.format(len(bp_data_trad)), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 7. 文本摘要
        ax = axes[1, 2]
        ax.axis('off')

        dqn_sat = summary['dqn']['avg_satisfaction'][0]
        enh_sat = summary['enhanced']['avg_satisfaction'][0]
        trad_sat = summary['traditional']['avg_satisfaction'][0]
        dqn_ho = summary['dqn']['handover_count'][0]
        enh_ho = summary['enhanced']['handover_count'][0]
        dqn_conn = summary['dqn']['connected_ratio'][0]
        enh_conn = summary['enhanced']['connected_ratio'][0]

        text = "【实验5 关键发现】\n\n"
        text += f"1. 满意度对比:\n"
        text += f"   DQN: {dqn_sat:.4f}\n"
        text += f"   增强启发式: {enh_sat:.4f}\n"
        text += f"   传统算法: {trad_sat:.4f}\n\n"

        if dqn_sat >= enh_sat:
            text += f"   DQN 满意度 >= 增强启发式\n"
            text += f"   RL 辅助决策有效\n\n"
        else:
            text += f"   DQN 满意度 < 增强启发式\n"
            text += f"   原因分析:\n"
            text += f"   (a) 训练量不足\n"
            text += f"   (b) 启发式利用了先验领域知识\n"
            text += f"   (c) 状态空间信息不完整\n\n"

        text += f"2. 切换效率:\n"
        text += f"   DQN: {dqn_ho:.1f} 次\n"
        text += f"   增强: {enh_ho:.1f} 次\n\n"
        text += f"3. 连接保持率:\n"
        text += f"   DQN: {dqn_conn*100:.1f}%\n"
        text += f"   增强: {enh_conn*100:.1f}%\n\n"
        text += f"4. 结论:\n"
        if dqn_sat >= enh_sat * 0.9:
            text += f"   DQN 达到启发式算法的性能水平,\n"
            text += f"   验证了 RL 在切换决策中的可行性。"
        else:
            text += f"   DQN 性能不如启发式, 但作为\n"
            text += f"   端到端学习方案, 具备自适应\n"
            text += f"   和泛化的潜力。"

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp5_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  图表已保存: {os.path.join(RESULT_DIR, 'exp5_results.png')}")


class Experiment5b:
    """
    实验5b：不同 UAV 密度下的多场景对比

    目标：验证在资源充裕 / 紧张场景下，各算法的表现差异
    场景：8BS × 10/20/30/40 UAV
    """

    ALGO_NAMES = {
        'traditional': '传统算法(3GPP A3)',
        'enhanced': '增强启发式算法',
        'dqn': 'DQN强化学习',
    }

    @staticmethod
    def run(num_steps=150, repeats=10, num_bs=8,
             uav_counts=(10, 20, 30, 40), target_uav_id=0,
             dqn_train_episodes=1000, verbose=False,
             bs_capacity_range=(250, 450)):
        """
        运行实验5b

        Args:
            num_steps: 每个 episode 的步数
            repeats: 重复实验次数
            num_bs: 基站数量
            uav_counts: UAV 数量列表
            target_uav_id: RL 控制的目标 UAV
            dqn_train_episodes: DQN 训练 episodes
            verbose: 是否打印详细调试信息
            bs_capacity_range: 基站容量范围 (low, high) Mbps，
                默认 (250, 450) 使得 30+ UAV 出现资源竞争
                (8 BS 总容量 ~2800 Mbps, UAV 平均需求 ~104 Mbps:
                 10UAV=37%, 20UAV=74%, 30UAV=111%, 40UAV=149%)
        """
        print("=" * 80)
        print("实验5b：不同 UAV 密度下的多场景对比")
        print("=" * 80)
        print(f"\n配置: {num_bs} 基站 × UAV数量 {uav_counts}")
        print(f"BS 容量范围: {bs_capacity_range[0]}-{bs_capacity_range[1]} Mbps/BS "
              f"(总计 ~{num_bs * (bs_capacity_range[0]+bs_capacity_range[1])/2:.0f} Mbps)")
        print(f"每个场景: {num_steps} 步 × {repeats} 次重复")
        print(f"DQN 训练: {dqn_train_episodes} episodes / 场景")

        all_results = {}  # {num_uav: {'traditional': [...], 'enhanced': [...], 'dqn': [...]}}
        all_training_data = {}  # {num_uav: 训练曲线数据}

        for num_uav in uav_counts:
            print(f"\n{'='*60}")
            print(f"  场景: {num_bs} BS × {num_uav} UAV "
                  f"(每BS平均 {num_uav/num_bs:.1f} UAV)")
            print(f"{'='*60}")

            # ---- 训练 DQN ----
            print(f"\n  [训练 DQN - {num_uav} UAV 场景]")
            t0 = time.time()

            rl_env_template = RLHandoverEnv(
                NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED,
                                                   bs_capacity_range=bs_capacity_range),
                target_uav_id=target_uav_id, max_steps=num_steps
            )

            agent = DQNAgent(
                state_dim=rl_env_template.state_dim,
                action_dim=rl_env_template.action_dim,
                lr=5e-4, gamma=0.95, hidden_dim=128,
                buffer_size=50000, batch_size=64,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                target_update_freq=500,
            )

            rl_env_train = RLHandoverEnv(
                NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED,
                                                   bs_capacity_range=bs_capacity_range),
                target_uav_id=target_uav_id, max_steps=num_steps
            )
            # 训练过程数据收集
            train_rewards = []
            train_losses = []
            train_sats = []
            train_hos = []
            epsilon_history = []
            for ep in range(dqn_train_episodes):
                # 多样化训练种子：每个 episode 使用不同种子，让 DQN 见过各种拓扑
                set_global_seed(GLOBAL_SEED + ep % 20)
                state = rl_env_train.reset()
                set_global_seed(GLOBAL_SEED + ep % 20)  # reset 会重置种子，再设一次
                ep_reward = 0.0
                ep_sat_sum = 0.0
                ep_ho = 0
                ep_losses = []
                for step_i in range(num_steps):
                    invalid = rl_env_train.get_invalid_actions()
                    action = agent.select_action(state, training=True, invalid_actions=invalid)
                    next_state, reward, done, info = rl_env_train.step(action)
                    next_invalid = rl_env_train.get_invalid_actions()
                    agent.store_transition(state, action, reward, next_state, float(done),
                                           next_invalid_actions=next_invalid)
                    loss = agent.train_step()
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_reward += reward
                    ep_sat_sum += info['satisfaction']
                    ep_ho = info['total_handovers']
                    state = next_state
                    if done:
                        break
                agent.decay_epsilon()

                # 记录训练数据
                train_rewards.append(ep_reward)
                train_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
                train_sats.append(ep_sat_sum / num_steps)
                train_hos.append(ep_ho)
                epsilon_history.append(agent.epsilon)

                if (ep + 1) % 100 == 0:
                    avg_r = np.mean(train_rewards[-100:])
                    avg_loss = np.mean(train_losses[-100:])
                    print(f"    Ep {ep+1}/{dqn_train_episodes}, eps={agent.epsilon:.3f}, "
                          f"avg_R={avg_r:.1f}, avg_sat={np.mean(train_sats[-100:]):.3f}, "
                          f"avg_ho={np.mean(train_hos[-100:]):.1f}, avg_loss={avg_loss:.4f}")

            # 保存模型
            model_save_path = os.path.join(RESULT_DIR, f'dqn_exp5b_{num_uav}uav_model.pt')
            agent.save(model_save_path)
            print(f"  模型训练完成, 耗时 {time.time()-t0:.1f}s, 已保存至 {model_save_path}")

            # 保存训练数据到 npz
            train_data_path = os.path.join(RESULT_DIR, f'dqn_exp5b_{num_uav}uav_training_data.npz')
            np.savez(train_data_path,
                     episode_rewards=train_rewards, episode_losses=train_losses,
                     epsilon_history=epsilon_history, episode_satisfactions=train_sats,
                     episode_handovers=train_hos)
            all_training_data[num_uav] = {
                'rewards': train_rewards, 'losses': train_losses,
                'epsilon': epsilon_history, 'sats': train_sats, 'hos': train_hos,
            }

            # ---- 对比实验 ----
            scene_results = {'traditional': [], 'enhanced': [], 'dqn': []}

            for rep in range(repeats):
                seed = GLOBAL_SEED + rep

                # 传统算法
                set_global_seed(seed)
                env_trad = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed,
                                                              bs_capacity_range=bs_capacity_range)
                algo_trad = IntegratedHandoverAlgorithm(env_trad)
                trad_result = Experiment5._run_single(env_trad, algo_trad, num_steps, target_uav_id)
                scene_results['traditional'].append(trad_result)

                # 增强启发式
                set_global_seed(seed)
                env_enh = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed,
                                                             bs_capacity_range=bs_capacity_range)
                algo_enh = EnhancedHandoverAlgorithm(env_enh)
                algo_enh.epsilon = 0.0
                enh_result = Experiment5._run_single(env_enh, algo_enh, num_steps, target_uav_id)
                scene_results['enhanced'].append(enh_result)

                # DQN
                set_global_seed(seed)
                env_dqn = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed,
                                                             bs_capacity_range=bs_capacity_range)
                rl_env = RLHandoverEnv(env_dqn, target_uav_id=target_uav_id, max_steps=num_steps)
                dqn_result = Experiment5._run_dqn_eval(rl_env, agent, num_steps, target_uav_id,
                                                         verbose=(verbose and rep == repeats - 1))
                scene_results['dqn'].append(dqn_result)

                print(f"    Rep {rep+1}: "
                      f"传统={trad_result['avg_satisfaction']:.3f}, "
                      f"增强={enh_result['avg_satisfaction']:.3f}, "
                      f"DQN={dqn_result['avg_satisfaction']:.3f} "
                      f"(切换={dqn_result.get('effective_switch_count', 0):.0f}"
                      f"/{dqn_result.get('switch_attempts', 0):.0f})")

            all_results[num_uav] = scene_results

            # 该场景摘要
            for algo_name in ['traditional', 'enhanced', 'dqn']:
                sats = [r['avg_satisfaction'] for r in scene_results[algo_name]]
                hos = [r['handover_count'] for r in scene_results[algo_name]]
                print(f"  [{Experiment5b.ALGO_NAMES[algo_name]}] "
                      f"Sat={np.mean(sats):.4f}+/-{np.std(sats):.4f}, "
                      f"HO={np.mean(hos):.1f}+/-{np.std(hos):.1f}")

        # ---- 汇总对比 ----
        Experiment5b._plot_comparison(all_results, uav_counts, num_bs)

        # ---- 训练曲线可视化 ----
        Experiment5b._plot_training_curves(all_training_data, dqn_train_episodes)

        return all_results

    @staticmethod
    def _plot_training_curves(all_training_data, num_episodes):
        """
        绘制多场景 DQN 训练曲线

        Args:
            all_training_data: {num_uav: {rewards, losses, epsilon, sats, hos}}
            num_episodes: 训练总 episodes
        """
        uav_counts = sorted(all_training_data.keys())
        n = len(uav_counts)

        fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('DQN 训练过程曲线（多场景对比）', fontsize=16, fontweight='bold', y=1.01)

        for row_idx, num_uav in enumerate(uav_counts):
            data = all_training_data[num_uav]
            rewards = data['rewards']
            losses = data['losses']
            epsilon = data['epsilon']
            sats = data['sats']
            hos = data['hos']

            episodes = range(1, len(rewards) + 1)
            title_prefix = f'{num_uav} UAV'
            color_r = '#667eea'
            color_s = '#4ECDC4'
            color_l = '#FF6B6B'
            color_e = '#764ba2'

            # 1. 奖励曲线
            ax = axes[row_idx, 0]
            ax.plot(list(episodes), rewards, alpha=0.3, color=color_r, linewidth=0.8)
            window = min(20, len(rewards))
            if window >= 5:
                ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(rewards) + 1), ma,
                        color=color_r, linewidth=2, label=f'MA(w={window})')
            ax.set_title(f'{title_prefix} - 奖励曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('总奖励')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 2. 满意度曲线
            ax = axes[row_idx, 1]
            ax.plot(list(episodes), sats, alpha=0.3, color=color_s, linewidth=0.8)
            if window >= 5:
                ma = np.convolve(sats, np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(sats) + 1), ma,
                        color=color_s, linewidth=2, label=f'MA(w={window})')
            ax.set_title(f'{title_prefix} - 满意度曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('平均满意度')
            ax.set_ylim(-0.05, 1.1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 3. Loss 曲线
            ax = axes[row_idx, 2]
            valid_losses = [l for l in losses if l > 0]
            if valid_losses:
                ax.plot(range(1, len(valid_losses) + 1), valid_losses,
                        alpha=0.4, color=color_l, linewidth=0.8)
                loss_window = min(50, len(valid_losses))
                if loss_window >= 5:
                    ma = np.convolve(valid_losses, np.ones(loss_window)/loss_window, mode='valid')
                    ax.plot(range(loss_window, len(valid_losses) + 1), ma,
                            color=color_l, linewidth=2, label=f'MA(w={loss_window})')
            ax.set_title(f'{title_prefix} - Loss 曲线', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # 4. Epsilon + 切换次数
            ax = axes[row_idx, 3]
            ax_twin = ax.twinx()
            ax.plot(list(episodes), epsilon, color=color_e, linewidth=1.5, label='ε (探索率)')
            ax_twin.plot(list(episodes), hos, alpha=0.4, color='#fbbf24',
                         linewidth=0.8, label='切换次数')
            ax.set_xlabel('Episode')
            ax.set_ylabel('ε', color=color_e)
            ax_twin.set_ylabel('切换次数', color='#fbbf24')
            ax.set_title(f'{title_prefix} - 探索率 & 切换次数', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'dqn_training_curves_5b.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  训练曲线已保存: {save_path}")

    @staticmethod
    def _plot_comparison(all_results, uav_counts, num_bs):
        """绘制多场景对比图"""
        algos = ['traditional', 'enhanced', 'dqn']
        algo_labels = [Experiment5b.ALGO_NAMES[a] for a in algos]
        algo_colors = [COLORS['neutral'], COLORS['primary'], COLORS['success']]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'实验5b：不同 UAV 密度下的算法对比（{num_bs} 基站）',
                     fontsize=15, fontweight='bold')

        x = np.arange(len(uav_counts))
        width = 0.25

        # 1. 满意度 vs UAV 数量
        ax = axes[0, 0]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['avg_satisfaction'] for r in all_results[n][algo]]) for n in uav_counts]
            stds = [np.std([r['avg_satisfaction'] for r in all_results[n][algo]]) for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('平均满意度')
        ax.set_title('平均满意度 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.15)

        # 2. 切换次数 vs UAV 数量
        ax = axes[0, 1]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['handover_count'] for r in all_results[n][algo]]) for n in uav_counts]
            stds = [np.std([r['handover_count'] for r in all_results[n][algo]]) for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('切换次数')
        ax.set_title('切换次数 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 连接保持率 vs UAV 数量
        ax = axes[1, 0]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['connected_ratio'] for r in all_results[n][algo]]) * 100 for n in uav_counts]
            stds = [np.std([r['connected_ratio'] for r in all_results[n][algo]]) * 100 for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('连接保持率 (%)')
        ax.set_title('连接保持率 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(50, 105)

        # 4. 折线图 - 满意度趋势
        ax = axes[1, 1]
        for algo, label, color in zip(algos, algo_labels, algo_colors):
            means = [np.mean([r['avg_satisfaction'] for r in all_results[n][algo]]) for n in uav_counts]
            stds = [np.std([r['avg_satisfaction'] for r in all_results[n][algo]]) for n in uav_counts]
            ax.errorbar(uav_counts, means, yerr=stds, marker='o', linewidth=2,
                       capsize=4, label=label, color=color)

        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('平均满意度')
        ax.set_title('满意度随 UAV 密度的变化趋势', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # 标注资源紧张区域
        ax.axvspan(25, 45, alpha=0.08, color='red', label='资源紧张区域')
        ax.text(32, 0.03, '资源紧张区域', fontsize=9, color='red', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp5b_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  图表已保存: {os.path.join(RESULT_DIR, 'exp5b_results.png')}")

        # 打印汇总表
        print(f"\n{'='*80}")
        print(f"实验5b 汇总对比表")
        print(f"{'='*80}")
        header = f"{'场景':^20s} | {'指标':^16s} | {'传统算法':^14s} | {'增强启发式':^14s} | {'DQN':^14s}"
        print(f"  {header}")
        print(f"  {'-'*len(header)}")

        for num_uav in uav_counts:
            ratio_str = f"{num_uav} UAV ({num_uav/num_bs:.1f}/BS)"
            for metric_name, metric_key in [('平均满意度', 'avg_satisfaction'),
                                             ('切换次数', 'handover_count'),
                                             ('连接保持率', 'connected_ratio')]:
                vals = []
                for algo in algos:
                    v = np.mean([r[metric_key] for r in all_results[num_uav][algo]])
                    if metric_key == 'connected_ratio':
                        vals.append(f"{v*100:.1f}%")
                    else:
                        vals.append(f"{v:.4f}" if metric_key == 'avg_satisfaction' else f"{v:.1f}")
                print(f"  {ratio_str:^20s} | {metric_name:^16s} | {vals[0]:^14s} | {vals[1]:^14s} | {vals[2]:^14s}")


if __name__ == '__main__':
    set_global_seed(GLOBAL_SEED)
    Experiment5.run(
        num_steps=150,
        repeats=10,
        num_bs=8,
        num_uav=20,
        target_uav_id=0,
        dqn_train_episodes=500,
        load_model=False,
        verbose=True,
    )


class Experiment5c:
    """
    实验5c：DQN 特征消融实验（已合并到 Experiment5Unified）

    保留此类仅作向后兼容，实际运行请使用 Experiment5Unified。
    """
    pass


# ============================================================================
# 统一 RL 实验：合并原实验 5 / 5b / 5c
# ============================================================================

def _create_dqn_agent(state_dim, action_dim):
    """创建标准 DQN Agent（统一超参数）"""
    return DQNAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=5e-4, gamma=0.95, hidden_dim=128,
        buffer_size=50000, batch_size=64,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
        target_update_freq=500,
    )


def _train_dqn(agent, rl_env, num_episodes, num_steps, print_every=100):
    """
    标准 DQN 训练循环

    Returns:
        dict: {'rewards': [], 'losses': [], 'sats': [], 'hos': [], 'epsilon': []}
    """
    train_rewards = []
    train_losses = []
    train_sats = []
    train_hos = []
    epsilon_history = []

    for ep in range(num_episodes):
        set_global_seed(GLOBAL_SEED + ep % 20)
        state = rl_env.reset()
        set_global_seed(GLOBAL_SEED + ep % 20)

        ep_reward = 0.0
        ep_sat_sum = 0.0
        ep_ho = 0
        ep_losses = []

        for step_i in range(num_steps):
            invalid = rl_env.get_invalid_actions()
            action = agent.select_action(state, training=True, invalid_actions=invalid)
            next_state, reward, done, info = rl_env.step(action)
            next_invalid = rl_env.get_invalid_actions()
            agent.store_transition(state, action, reward, next_state, float(done),
                                   next_invalid_actions=next_invalid)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            ep_reward += reward
            ep_sat_sum += info['satisfaction']
            ep_ho = info['total_handovers']
            state = next_state
            if done:
                break

        agent.decay_epsilon()

        train_rewards.append(ep_reward)
        train_losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        train_sats.append(ep_sat_sum / num_steps)
        train_hos.append(ep_ho)
        epsilon_history.append(agent.epsilon)

        if (ep + 1) % print_every == 0:
            avg_r = np.mean(train_rewards[-print_every:])
            avg_loss = np.mean(train_losses[-print_every:])
            print(f"    Ep {ep+1}/{num_episodes}, eps={agent.epsilon:.3f}, "
                  f"avg_R={avg_r:.1f}, avg_sat={np.mean(train_sats[-print_every:]):.3f}, "
                  f"avg_ho={np.mean(train_hos[-print_every:]):.1f}, avg_loss={avg_loss:.4f}")

    return {
        'rewards': train_rewards, 'losses': train_losses,
        'sats': train_sats, 'hos': train_hos, 'epsilon': epsilon_history,
    }


def _eval_three_algorithms(agent, num_bs, num_uav, num_steps, target_uav_id,
                            repeats, bs_capacity_range, verbose=False):
    """
    三种算法对比评估

    Returns:
        dict: {'traditional': [...], 'enhanced': [...], 'dqn': [...]}
    """
    results = {'traditional': [], 'enhanced': [], 'dqn': []}

    for rep in range(repeats):
        seed = GLOBAL_SEED + rep

        # 传统算法
        set_global_seed(seed)
        env_trad = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav, seed=seed,
            bs_capacity_range=bs_capacity_range)
        algo_trad = IntegratedHandoverAlgorithm(env_trad)
        trad_result = Experiment5._run_single(env_trad, algo_trad, num_steps, target_uav_id)
        results['traditional'].append(trad_result)

        # 增强启发式
        set_global_seed(seed)
        env_enh = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav, seed=seed,
            bs_capacity_range=bs_capacity_range)
        algo_enh = EnhancedHandoverAlgorithm(env_enh)
        algo_enh.epsilon = 0.0
        enh_result = Experiment5._run_single(env_enh, algo_enh, num_steps, target_uav_id)
        results['enhanced'].append(enh_result)

        # DQN
        set_global_seed(seed)
        env_dqn = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav, seed=seed,
            bs_capacity_range=bs_capacity_range)
        rl_env = RLHandoverEnv(env_dqn, target_uav_id=target_uav_id, max_steps=num_steps)
        dqn_result = Experiment5._run_dqn_eval(rl_env, agent, num_steps, target_uav_id,
                                                 verbose=(verbose and rep == repeats - 1))
        results['dqn'].append(dqn_result)

        print(f"    Rep {rep+1}: "
              f"传统={trad_result['avg_satisfaction']:.3f}, "
              f"增强={enh_result['avg_satisfaction']:.3f}, "
              f"DQN={dqn_result['avg_satisfaction']:.3f} "
              f"(切换={dqn_result.get('effective_switch_count', 0):.0f}"
              f"/{dqn_result.get('switch_attempts', 0):.0f})")

    return results


def _eval_dqn_agent(agent, num_bs, num_uav, num_steps, target_uav_id,
                     repeats, bs_capacity_range,
                     use_capacity_ratios=True, use_global_tension=True,
                     adaptive_penalty=True, verbose=False):
    """
    单独评估一个 DQN agent

    Returns:
        list: [result_dict, ...]
    """
    eval_results = []
    for rep in range(repeats):
        seed = GLOBAL_SEED + rep
        set_global_seed(seed)
        env_eval = NetworkEnvironmentWithRecognition(
            num_bs=num_bs, num_uav=num_uav, seed=seed,
            bs_capacity_range=bs_capacity_range)
        rl_env_eval = RLHandoverEnv(
            env_eval, target_uav_id=target_uav_id, max_steps=num_steps,
            use_capacity_ratios=use_capacity_ratios,
            use_global_tension=use_global_tension,
            adaptive_penalty=adaptive_penalty,
        )
        result = Experiment5._run_dqn_eval(rl_env_eval, agent, num_steps, target_uav_id,
                                            verbose=(verbose and rep == 0))
        eval_results.append(result)
    return eval_results


class Experiment5Unified:
    """
    统一 RL 实验：合并原实验 5 / 5b / 5c

    Phase 1: 多场景 DQN 训练 + 三种算法对比（原 5 + 5b）
    Phase 2: 30 UAV 特征消融实验（原 5c，复用 Phase 1 的 30 UAV Full 模型）

    改进点：
    - 消融训练量 500ep → 1000ep（与 Phase 1 一致）
    - 消融评估次数 5 → 10（统计意义更强）
    - Full 消融组直接复用 Phase 1 训练好的 30 UAV 模型，无需重复训练
    - 消除代码重复，提取公共工具函数
    """

    # 消融配置（不包含 Full，Full 复用 Phase 1 模型）
    ABLATION_CONFIGS = {
        '-capacity_ratios': {
            'use_capacity_ratios': False, 'use_global_tension': True,
            'adaptive_penalty': True,
        },
        '-global_tension': {
            'use_capacity_ratios': True, 'use_global_tension': False,
            'adaptive_penalty': True,
        },
        '-adaptive_penalty': {
            'use_capacity_ratios': True, 'use_global_tension': True,
            'adaptive_penalty': False,
        },
        '-全部 (旧版)': {
            'use_capacity_ratios': False, 'use_global_tension': False,
            'adaptive_penalty': False,
        },
    }

    ALGO_NAMES = {
        'traditional': '传统算法(3GPP A3)',
        'enhanced': '增强启发式算法',
        'dqn': 'DQN强化学习',
    }

    @staticmethod
    def run(uav_counts=(10, 20, 30, 40),
            num_steps=150, repeats=10, num_bs=8,
            target_uav_id=0, dqn_train_episodes=1000,
            ablation_train_episodes=None,
            bs_capacity_range=(250, 450),
            verbose=False, demo=False,
            load_models=False, phase='both'):
        """
        运行统一 RL 实验

        Args:
            uav_counts: Phase 1 场景 UAV 数量列表
            num_steps: 每个 episode 的步数
            repeats: 评估重复次数
            num_bs: 基站数量
            target_uav_id: RL 控制的目标 UAV
            dqn_train_episodes: Phase 1 DQN 训练 episodes（默认 1000）
            ablation_train_episodes: Phase 2 消融训练 episodes
                默认 None 表示与 dqn_train_episodes 相同，
                设为 500 可节省约一半 Phase 2 时间
            bs_capacity_range: 基站容量范围
            verbose: 是否打印详细调试信息
            demo: 快速验证模式（50ep 训练 + 2 次评估）
            load_models: 是否加载已有模型（跳过训练）
            phase: 运行哪些阶段，'both' / 'phase1' / 'phase2'
        """
        if demo:
            dqn_train_episodes = 50
            ablation_train_episodes = 50
            repeats = 2
            verbose = False

        if ablation_train_episodes is None:
            ablation_train_episodes = dqn_train_episodes

        print("=" * 80)
        print("统一 RL 实验 (原实验 5 + 5b + 5c)")
        print("=" * 80)
        print(f"\nPhase 1: {num_bs} BS × UAV {uav_counts} 场景 "
              f"({'加载已有模型' if load_models else f'{dqn_train_episodes}ep 训练'})")
        print(f"Phase 2: 30 UAV 消融实验 ({len(Experiment5Unified.ABLATION_CONFIGS)} 组, "
              f"{ablation_train_episodes}ep 训练)")
        print(f"公共参数: {num_steps} 步/episode, {repeats} 次评估")
        print(f"运行阶段: {phase}")
        if demo:
            print("*** DEMO 模式: 快速验证 ***")
        if load_models:
            print("*** 加载模式: 跳过训练，直接评估 ***")

        total_t0 = time.time()

        # ========== Phase 1: 多场景训练 + 算法对比 ==========
        scene_agents = {}
        scene_results = {}
        training_data = {}

        if phase in ('both', 'phase1'):
            print(f"\n{'='*80}")
            print(f"Phase 1: 多场景 DQN 训练 + 算法对比")
            print(f"{'='*80}")

            for num_uav in uav_counts:
                print(f"\n{'—'*60}")
                print(f"  场景: {num_bs} BS × {num_uav} UAV "
                      f"(每BS平均 {num_uav/num_bs:.1f} UAV)")
                print(f"{'—'*60}")

                t0 = time.time()

                rl_env_template = RLHandoverEnv(
                    NetworkEnvironmentWithRecognition(
                        num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED,
                        bs_capacity_range=bs_capacity_range),
                    target_uav_id=target_uav_id, max_steps=num_steps)

                agent = _create_dqn_agent(rl_env_template.state_dim,
                                          rl_env_template.action_dim)

                model_path = os.path.join(RESULT_DIR,
                                          f'dqn_unified_{num_uav}uav_model.pt')

                if load_models and os.path.exists(model_path):
                    agent.load(model_path)
                    print(f"  已加载已有模型: {model_path}")
                    training_data[num_uav] = None
                else:
                    print(f"  [训练 DQN - {num_uav} UAV, {dqn_train_episodes}ep]")
                    rl_env_train = RLHandoverEnv(
                        NetworkEnvironmentWithRecognition(
                            num_bs=num_bs, num_uav=num_uav, seed=GLOBAL_SEED,
                            bs_capacity_range=bs_capacity_range),
                        target_uav_id=target_uav_id, max_steps=num_steps)
                    td = _train_dqn(agent, rl_env_train, dqn_train_episodes, num_steps)
                    training_data[num_uav] = td
                    agent.save(model_path)

                print(f"  场景准备完成, 耗时 {time.time()-t0:.1f}s")
                scene_agents[num_uav] = agent

                # 评估
                print(f"  [评估 - 三种算法 × {repeats} 次]")
                scene_results[num_uav] = _eval_three_algorithms(
                    agent, num_bs, num_uav, num_steps, target_uav_id,
                    repeats, bs_capacity_range, verbose=verbose)

                for algo_name in ['traditional', 'enhanced', 'dqn']:
                    sats = [r['avg_satisfaction'] for r in scene_results[num_uav][algo_name]]
                    hos = [r['handover_count'] for r in scene_results[num_uav][algo_name]]
                    print(f"  [{Experiment5Unified.ALGO_NAMES[algo_name]}] "
                          f"Sat={np.mean(sats):.4f}+/-{np.std(sats):.4f}, "
                          f"HO={np.mean(hos):.1f}+/-{np.std(hos):.1f}")

            # Phase 1 绘图
            Experiment5Unified._plot_comparison(scene_results, uav_counts, num_bs)
            # 训练曲线仅在有训练数据时绘制
            avail_td = {k: v for k, v in training_data.items() if v is not None}
            if avail_td:
                max_ep = max(dqn_train_episodes,
                             max(len(v['rewards']) for v in avail_td.values()))
                Experiment5._plot_single_training_curves(
                    avail_td, max_ep, save_name='dqn_unified_training.png')

        # ========== Phase 2: 30 UAV 消融实验 ==========
        ablation_results = {}

        if phase in ('both', 'phase2'):
            print(f"\n{'='*80}")
            print(f"Phase 2: 30 UAV 特征消融实验")
            print(f"{'='*80}")

            # Full 组：复用 Phase 1 的 30 UAV 模型
            if 30 in scene_agents:
                print(f"\n  消融组: Full (完整) [复用 Phase 1 模型]")
                full_eval = _eval_dqn_agent(
                    scene_agents[30], num_bs, 30, num_steps, target_uav_id,
                    repeats, bs_capacity_range,
                    use_capacity_ratios=True, use_global_tension=True,
                    adaptive_penalty=True, verbose=verbose)
                ablation_results['Full (完整)'] = full_eval
                sats = [r['avg_satisfaction'] for r in full_eval]
                hos = [r['handover_count'] for r in full_eval]
                conns = [r['connected_ratio'] for r in full_eval]
                print(f"  [Full (完整)] Sat={np.mean(sats):.4f}+/-{np.std(sats):.4f}, "
                      f"HO={np.mean(hos):.1f}+/-{np.std(hos):.1f}, "
                      f"连接保持率={np.mean(conns)*100:.1f}%")
            elif phase == 'phase2':
                # 独立运行 phase2 时，需要训练 Full 模型
                print(f"\n  消融组: Full (完整) [独立训练]")
                t0 = time.time()
                rl_env_template = RLHandoverEnv(
                    NetworkEnvironmentWithRecognition(
                        num_bs=num_bs, num_uav=30, seed=GLOBAL_SEED,
                        bs_capacity_range=bs_capacity_range),
                    target_uav_id=target_uav_id, max_steps=num_steps)
                agent = _create_dqn_agent(rl_env_template.state_dim,
                                          rl_env_template.action_dim)
                rl_env_train = RLHandoverEnv(
                    NetworkEnvironmentWithRecognition(
                        num_bs=num_bs, num_uav=30, seed=GLOBAL_SEED,
                        bs_capacity_range=bs_capacity_range),
                    target_uav_id=target_uav_id, max_steps=num_steps)
                _train_dqn(agent, rl_env_train, ablation_train_episodes, num_steps)
                model_path = os.path.join(RESULT_DIR, 'dqn_unified_30uav_model.pt')
                agent.save(model_path)
                print(f"  Full 训练完成, 耗时 {time.time()-t0:.1f}s")
                full_eval = _eval_dqn_agent(
                    agent, num_bs, 30, num_steps, target_uav_id,
                    repeats, bs_capacity_range,
                    use_capacity_ratios=True, use_global_tension=True,
                    adaptive_penalty=True, verbose=verbose)
                ablation_results['Full (完整)'] = full_eval
                sats = [r['avg_satisfaction'] for r in full_eval]
                print(f"  [Full (完整)] Sat={np.mean(sats):.4f}+/-{np.std(sats):.4f}")

            # 消融组训练 + 评估
            for config_name, config in Experiment5Unified.ABLATION_CONFIGS.items():
                print(f"\n{'—'*60}")
                print(f"  消融组: {config_name}")
                print(f"  配置: capacity_ratios={config['use_capacity_ratios']}, "
                      f"global_tension={config['use_global_tension']}, "
                      f"adaptive_penalty={config['adaptive_penalty']}")
                print(f"{'—'*60}")

                t0 = time.time()

                rl_env_template = RLHandoverEnv(
                    NetworkEnvironmentWithRecognition(
                        num_bs=num_bs, num_uav=30, seed=GLOBAL_SEED,
                        bs_capacity_range=bs_capacity_range),
                    target_uav_id=target_uav_id, max_steps=num_steps,
                    use_capacity_ratios=config['use_capacity_ratios'],
                    use_global_tension=config['use_global_tension'],
                    adaptive_penalty=config['adaptive_penalty'])

                agent = _create_dqn_agent(rl_env_template.state_dim,
                                          rl_env_template.action_dim)

                safe_name = config_name.replace(" ", "_")
                model_path = os.path.join(
                    RESULT_DIR,
                    f'dqn_unified_30uav_{safe_name}_model.pt')

                if load_models and os.path.exists(model_path):
                    agent.load(model_path)
                    print(f"  已加载已有模型: {model_path}")
                else:
                    print(f"  [训练消融 DQN, {ablation_train_episodes}ep]")
                    rl_env_train = RLHandoverEnv(
                        NetworkEnvironmentWithRecognition(
                            num_bs=num_bs, num_uav=30, seed=GLOBAL_SEED,
                            bs_capacity_range=bs_capacity_range),
                        target_uav_id=target_uav_id, max_steps=num_steps,
                        use_capacity_ratios=config['use_capacity_ratios'],
                        use_global_tension=config['use_global_tension'],
                        adaptive_penalty=config['adaptive_penalty'])
                    _train_dqn(agent, rl_env_train, ablation_train_episodes, num_steps)
                    agent.save(model_path)

                print(f"  消融组准备完成, 耗时 {time.time()-t0:.1f}s")

                # 评估
                print(f"  [评估 × {repeats} 次]")
                eval_results = _eval_dqn_agent(
                    agent, num_bs, 30, num_steps, target_uav_id,
                    repeats, bs_capacity_range,
                    use_capacity_ratios=config['use_capacity_ratios'],
                    use_global_tension=config['use_global_tension'],
                    adaptive_penalty=config['adaptive_penalty'],
                    verbose=verbose)
                ablation_results[config_name] = eval_results

                sats = [r['avg_satisfaction'] for r in eval_results]
                hos = [r['handover_count'] for r in eval_results]
                conns = [r['connected_ratio'] for r in eval_results]
                print(f"  [{config_name}] Sat={np.mean(sats):.4f}+/-{np.std(sats):.4f}, "
                      f"HO={np.mean(hos):.1f}+/-{np.std(hos):.1f}, "
                      f"连接保持率={np.mean(conns)*100:.1f}%")

            # Phase 2 汇总 + 绘图
            if ablation_results:
                Experiment5Unified._summarize_ablation(ablation_results)
                Experiment5Unified._plot_ablation(ablation_results)

        # ========== 统计显著性检验 ==========
        if 30 in scene_results:
            Experiment5._statistical_tests(scene_results[30])

        # ========== 总耗时 ==========
        print(f"\n{'='*80}")
        print(f"统一实验全部完成, 总耗时 {time.time()-total_t0:.1f}s "
              f"({(time.time()-total_t0)/60:.1f}min)")
        print(f"结果保存在: {os.path.abspath(RESULT_DIR)}")
        print(f"{'='*80}")

        # ========== 保存实验数据到文件 ==========
        import pickle, json
        from datetime import datetime

        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        # 保存场景对比结果
        if scene_results:
            scene_path = os.path.join(RESULT_DIR, 'exp5_scene_data.pkl')
            with open(scene_path, 'wb') as f:
                pickle.dump(scene_results, f)
            print(f"  场景数据已保存: {scene_path}")

            # JSON 格式（人类可读）
            json_path = os.path.join(RESULT_DIR, 'exp5_scene_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(_convert(scene_results), f, ensure_ascii=False, indent=2)
            print(f"  场景数据已保存: {json_path}")

        # 保存消融实验结果
        if ablation_results:
            abl_path = os.path.join(RESULT_DIR, 'exp5_ablation_data.pkl')
            with open(abl_path, 'wb') as f:
                pickle.dump(ablation_results, f)
            print(f"  消融数据已保存: {abl_path}")

        return {
            'scene_results': scene_results,
            'ablation_results': ablation_results,
            'training_data': training_data,
        }

    # ==================== 绘图方法 ====================

    @staticmethod
    def _plot_comparison(all_results, uav_counts, num_bs):
        """绘制多场景算法对比图（Phase 1）"""
        algos = ['traditional', 'enhanced', 'dqn']
        algo_labels = [Experiment5Unified.ALGO_NAMES[a] for a in algos]
        algo_colors = [COLORS['neutral'], COLORS['primary'], COLORS['success']]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'统一实验5：不同 UAV 密度下的算法对比（{num_bs} 基站）',
                     fontsize=15, fontweight='bold')

        x = np.arange(len(uav_counts))
        width = 0.25

        # 1. 满意度 vs UAV 数量
        ax = axes[0, 0]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['avg_satisfaction'] for r in all_results[n][algo]])
                     for n in uav_counts]
            stds = [np.std([r['avg_satisfaction'] for r in all_results[n][algo]])
                    for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('平均满意度')
        ax.set_title('平均满意度 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.15)

        # 2. 切换次数 vs UAV 数量
        ax = axes[0, 1]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['handover_count'] for r in all_results[n][algo]])
                     for n in uav_counts]
            stds = [np.std([r['handover_count'] for r in all_results[n][algo]])
                    for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('切换次数')
        ax.set_title('切换次数 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 连接保持率 vs UAV 数量
        ax = axes[1, 0]
        for i, (algo, label, color) in enumerate(zip(algos, algo_labels, algo_colors)):
            means = [np.mean([r['connected_ratio'] for r in all_results[n][algo]]) * 100
                     for n in uav_counts]
            stds = [np.std([r['connected_ratio'] for r in all_results[n][algo]]) * 100
                    for n in uav_counts]
            bars = ax.bar(x + i * width, means, width, yerr=stds, label=label,
                         color=color, alpha=0.8, capsize=4)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('连接保持率 (%)')
        ax.set_title('连接保持率 vs UAV 密度', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{n} UAV\n({n/num_bs:.1f}/BS)' for n in uav_counts], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(50, 105)

        # 4. 折线图 - 满意度趋势
        ax = axes[1, 1]
        for algo, label, color in zip(algos, algo_labels, algo_colors):
            means = [np.mean([r['avg_satisfaction'] for r in all_results[n][algo]])
                     for n in uav_counts]
            stds = [np.std([r['avg_satisfaction'] for r in all_results[n][algo]])
                    for n in uav_counts]
            ax.errorbar(uav_counts, means, yerr=stds, marker='o', linewidth=2,
                       capsize=4, label=label, color=color)
        ax.set_xlabel('UAV 数量')
        ax.set_ylabel('平均满意度')
        ax.set_title('满意度随 UAV 密度的变化趋势', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.axvspan(25, 45, alpha=0.08, color='red', label='资源紧张区域')
        ax.text(32, 0.03, '资源紧张区域', fontsize=9, color='red', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'exp5_unified_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  算法对比图已保存: {save_path}")

        # 打印汇总表
        print(f"\n{'='*80}")
        print(f"Phase 1 汇总对比表")
        print(f"{'='*80}")
        header = (f"{'场景':^20s} | {'指标':^16s} | {'传统算法':^14s} | "
                  f"{'增强启发式':^14s} | {'DQN':^14s}")
        print(f"  {header}")
        print(f"  {'-'*len(header)}")
        for num_uav in uav_counts:
            ratio_str = f"{num_uav} UAV ({num_uav/num_bs:.1f}/BS)"
            for metric_name, metric_key in [('平均满意度', 'avg_satisfaction'),
                                             ('切换次数', 'handover_count'),
                                             ('连接保持率', 'connected_ratio')]:
                vals = []
                for algo in algos:
                    v = np.mean([r[metric_key] for r in all_results[num_uav][algo]])
                    if metric_key == 'connected_ratio':
                        vals.append(f"{v*100:.1f}%")
                    else:
                        vals.append(f"{v:.4f}" if metric_key == 'avg_satisfaction'
                                    else f"{v:.1f}")
                print(f"  {ratio_str:^20s} | {metric_name:^16s} | "
                      f"{vals[0]:^14s} | {vals[1]:^14s} | {vals[2]:^14s}")

    @staticmethod
    def _summarize_ablation(ablation_results):
        """打印消融实验汇总表"""
        print(f"\n{'='*80}")
        print(f"Phase 2 消融实验汇总表 (30 UAV 资源紧张场景)")
        print(f"{'='*80}")

        # 确保顺序: Full 在前
        ordered_names = []
        if 'Full (完整)' in ablation_results:
            ordered_names.append('Full (完整)')
        for name in Experiment5Unified.ABLATION_CONFIGS:
            if name in ablation_results:
                ordered_names.append(name)

        full_sat = np.mean([r['avg_satisfaction']
                            for r in ablation_results[ordered_names[0]]])

        header = (f"{'消融组':^20s} | {'平均满意度':^12s} | "
                  f"{'切换次数':^10s} | {'连接保持率':^10s} | {'vs Full':^10s}")
        print(f"  {header}")
        print(f"  {'-'*len(header)}")

        for name in ordered_names:
            results = ablation_results[name]
            sat = np.mean([r['avg_satisfaction'] for r in results])
            sat_std = np.std([r['avg_satisfaction'] for r in results])
            ho = np.mean([r['handover_count'] for r in results])
            conn = np.mean([r['connected_ratio'] for r in results])
            diff = (sat - full_sat) / full_sat * 100 if full_sat != 0 else 0
            diff_str = f"{diff:+.1f}%" if name != ordered_names[0] else "基线"
            print(f"  {name:^20s} | {sat:.4f}+/-{sat_std:.3f} | "
                  f"{ho:^10.1f} | {conn*100:>6.1f}%    | {diff_str:^10s}")

        print(f"\n【消融结论】")
        for name in ordered_names[1:]:
            sat = np.mean([r['avg_satisfaction'] for r in ablation_results[name]])
            diff = sat - full_sat
            impact = "显著" if abs(diff) > 0.02 else "轻微" if abs(diff) > 0.005 else "微小"
            direction = "下降" if diff < 0 else "上升"
            print(f"  {name}: 满意度 {direction} {abs(diff):.4f} ({impact}影响)")

    @staticmethod
    def _plot_ablation(ablation_results):
        """绘制消融实验对比图"""
        ordered_names = []
        if 'Full (完整)' in ablation_results:
            ordered_names.append('Full (完整)')
        for name in Experiment5Unified.ABLATION_CONFIGS:
            if name in ablation_results:
                ordered_names.append(name)
        n_configs = len(ordered_names)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('统一实验5：DQN 特征消融实验（30 UAV 资源紧张场景）',
                     fontsize=14, fontweight='bold')

        # 颜色：基线绿色，消融组红/橙渐变
        colors = ['#2ecc71'] + ['#e74c3c', '#e67e22', '#f39c12', '#d35400'][:n_configs-1]

        # 1. 满意度
        ax = axes[0]
        sats = [np.mean([r['avg_satisfaction'] for r in ablation_results[n]])
                for n in ordered_names]
        sat_stds = [np.std([r['avg_satisfaction'] for r in ablation_results[n]])
                    for n in ordered_names]
        bars = ax.bar(range(n_configs), sats, yerr=sat_stds, color=colors, alpha=0.8,
                      capsize=4, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, sats):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(ordered_names, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('平均满意度')
        ax.set_title('满意度对比', fontweight='bold')
        ax.set_ylim(min(sats) - 0.05, max(sats) + 0.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=sats[0], color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)

        # 2. 切换次数
        ax = axes[1]
        hos = [np.mean([r['handover_count'] for r in ablation_results[n]])
               for n in ordered_names]
        ho_stds = [np.std([r['handover_count'] for r in ablation_results[n]])
                   for n in ordered_names]
        bars = ax.bar(range(n_configs), hos, yerr=ho_stds, color=colors, alpha=0.8,
                      capsize=4, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, hos):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(ordered_names, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('切换次数')
        ax.set_title('切换次数对比', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=hos[0], color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)

        # 3. 连接保持率
        ax = axes[2]
        conns = [np.mean([r['connected_ratio'] for r in ablation_results[n]]) * 100
                 for n in ordered_names]
        conn_stds = [np.std([r['connected_ratio'] for r in ablation_results[n]]) * 100
                     for n in ordered_names]
        bars = ax.bar(range(n_configs), conns, yerr=conn_stds, color=colors, alpha=0.8,
                      capsize=4, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, conns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(ordered_names, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('连接保持率 (%)')
        ax.set_title('连接保持率对比', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=conns[0], color='#2ecc71', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(min(conns) - 5, max(conns) + 5)

        plt.tight_layout()
        save_path = os.path.join(RESULT_DIR, 'exp5_unified_ablation.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  消融对比图已保存: {save_path}")
