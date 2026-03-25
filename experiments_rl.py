"""
实验6：RL 辅助决策 vs 启发式算法对比实验

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


class Experiment6:
    """
    实验6：RL 辅助决策 vs 启发式算法对比

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
             target_uav_id=0, dqn_train_episodes=500,
             load_model=False, model_path=None):
        """
        运行实验6

        Args:
            num_steps: 每个 episode 的步数
            repeats: 重复实验次数（不同随机种子）
            num_bs: 基站数量
            num_uav: UAV 总数
            target_uav_id: RL 控制的目标 UAV
            dqn_train_episodes: DQN 训练 episodes
            load_model: 是否加载已有模型
            model_path: 模型路径
        """
        print("=" * 80)
        print("实验6：RL 辅助决策 vs 启发式算法对比")
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
            lr=1e-3, gamma=0.99, hidden_dim=128,
            buffer_size=10000, batch_size=64,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
            target_update_freq=100,
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
            for ep in range(dqn_train_episodes):
                state = rl_env_train.reset()
                for step_i in range(num_steps):
                    action = agent.select_action(state, training=True)
                    next_state, reward, done, info = rl_env_train.step(action)
                    agent.store_transition(state, action, reward, next_state, float(done))
                    loss = agent.train_step()
                    state = next_state
                    if done:
                        break
                agent.decay_epsilon()
                if (ep + 1) % 100 == 0:
                    print(f"    Ep {ep+1}/{dqn_train_episodes}, eps={agent.epsilon:.3f}")

            # 保存训练好的模型
            save_path = os.path.join(RESULT_DIR, 'dqn_exp6_model.pt')
            agent.save(save_path)

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
            trad_result = Experiment6._run_single(env_trad, algo_trad, num_steps, target_uav_id)
            results['traditional'].append(trad_result)

            # --- 增强启发式算法 ---
            set_global_seed(seed)
            env_enh = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed)
            algo_enh = EnhancedHandoverAlgorithm(env_enh)
            algo_enh.epsilon = 0.0  # 关闭探索以公平对比
            enh_result = Experiment6._run_single(env_enh, algo_enh, num_steps, target_uav_id)
            results['enhanced'].append(enh_result)

            # --- DQN ---
            set_global_seed(seed)
            env_dqn = NetworkEnvironmentWithRecognition(num_bs=num_bs, num_uav=num_uav, seed=seed)
            rl_env = RLHandoverEnv(env_dqn, target_uav_id=target_uav_id, max_steps=num_steps)
            dqn_result = Experiment6._run_dqn_eval(rl_env, agent, num_steps, target_uav_id)
            results['dqn'].append(dqn_result)

            print(f"    传统: Sat={trad_result['avg_satisfaction']:.3f}, "
                  f"HO={trad_result['handover_count']:.0f}")
            print(f"    增强: Sat={enh_result['avg_satisfaction']:.3f}, "
                  f"HO={enh_result['handover_count']:.0f}")
            print(f"    DQN:  Sat={dqn_result['avg_satisfaction']:.3f}, "
                  f"HO={dqn_result['handover_count']:.0f}")

        # ---- Step 3: 统计分析 ----
        summary = Experiment6._summarize(results)
        Experiment6._print_results_table(summary)
        Experiment6._statistical_tests(results)
        Experiment6._plot(summary, results)

        return summary

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
                    algo.run_step(enable_load_balancing=True)
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
    def _run_dqn_eval(rl_env, agent, num_steps, target_uav_id):
        """运行 DQN 评估 episode"""
        state = rl_env.reset()
        satisfaction_history = []
        connected_count = 0
        rate_history = []

        for step in range(num_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = rl_env.step(action)
            satisfaction_history.append(info['satisfaction'])
            if info['connected_bs'] is not None:
                connected_count += 1
                rate_history.append(info['allocated_rate'])
            state = next_state
            if done:
                break

        return {
            'avg_satisfaction': np.mean(satisfaction_history),
            'final_satisfaction': satisfaction_history[-1],
            'min_satisfaction': np.min(satisfaction_history),
            'handover_count': info.get('total_handovers', 0),
            'avg_satisfaction_per_step': np.mean(satisfaction_history),
            'connected_ratio': connected_count / num_steps,
            'avg_allocated_rate': np.mean(rate_history) if rate_history else 0.0,
            'satisfaction_history': satisfaction_history,
        }

    @staticmethod
    def _summarize(results):
        """汇总结果"""
        summary = {}
        for algo_name in ['traditional', 'enhanced', 'dqn']:
            data_list = results[algo_name]
            summary[algo_name] = {}
            for key in Experiment6.METRICS.keys():
                if key == 'satisfaction_history':
                    continue
                vals = [d[key] for d in data_list]
                summary[algo_name][key] = (np.mean(vals), np.std(vals))
            # 满意度时序数据
            summary[algo_name]['satisfaction_history'] = data_list[0]['satisfaction_history']
        return summary

    @staticmethod
    def _print_results_table(summary):
        """打印结果表格"""
        headers = ["指标"] + [Experiment6.ALGO_NAMES[k] for k in ['traditional', 'enhanced', 'dqn']]
        rows = []

        for key, name in Experiment6.METRICS.items():
            row = [name]
            for algo in ['traditional', 'enhanced', 'dqn']:
                if key in summary[algo]:
                    mean, std = summary[algo][key]
                    row.append(f"{mean:.4f}+/-{std:.4f}")
                else:
                    row.append("N/A")
            rows.append(row)

        # 计算提升（DQN 相对增强算法）
        for key, name in Experiment6.METRICS.items():
            if key in summary['enhanced'] and key in summary['dqn']:
                enh = summary['enhanced'][key][0]
                dqn = summary['dqn'][key][0]
                # 对切换次数和连接率，越高越好；其他指标也越高越好
                improvement = (dqn - enh) / abs(enh) * 100 if enh != 0 else 0
                rows[-1 if key == list(Experiment6.METRICS.keys())[-1] else
                      list(Experiment6.METRICS.keys()).index(key)].append(
                    f"{improvement:+.1f}%"
                ) if rows else None

        # 更新 headers 加入提升列
        headers = ["指标", "传统算法", "增强启发式", "DQN", "DQN vs 增强"]

        print("\n" + "=" * 100)
        print("实验6结果：RL 辅助决策 vs 启发式算法对比")
        print("=" * 100)

        # 重新构建带提升的行
        rows_with_improve = []
        for key, name in Experiment6.METRICS.items():
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
            print(f"\n  【{Experiment6.METRICS.get(m, m)}】")
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
        fig.suptitle('实验6：RL 辅助决策 vs 启发式算法对比', fontsize=15, fontweight='bold')

        algos = ['traditional', 'enhanced', 'dqn']
        algo_labels = [Experiment6.ALGO_NAMES[a] for a in algos]
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
            # 平滑
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
                    # 归一化
                    if mk == 'avg_allocated_rate':
                        max_val = max(summary[a]['avg_allocated_rate'][0] for a in algos)
                        v = v / max_val if max_val > 0 else 0
                    elif mk == 'handover_count':
                        # 切换次数越少越好，用倒数
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

        # 6. 多指标对比（箱线图用到的原始数据）
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

        text = "【实验6 关键发现】\n\n"
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
        plt.savefig(os.path.join(RESULT_DIR, 'exp6_results.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  图表已保存: {os.path.join(RESULT_DIR, 'exp6_results.png')}")


if __name__ == '__main__':
    set_global_seed(GLOBAL_SEED)
    Experiment6.run(
        num_steps=150,
        repeats=10,
        num_bs=8,
        num_uav=20,
        target_uav_id=0,
        dqn_train_episodes=500,
        load_model=False,
    )
