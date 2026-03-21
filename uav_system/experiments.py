import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS, CMAP_PRIMARY, CMAP_SUCCESS, CMAP_WARNING
from .business import BusinessType, QoSProfile, QOS_PROFILES
from .satisfaction import HierarchicalSatisfactionMetric
from .recognition import AdaptiveRecognitionUpdater, BusinessRecognitionModel, train_or_load_recognition_model
from .environment import EnhancedNetworkEnvironment
from .algorithms import IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm
from .visualization import VisualizationHelper

# -------------------- 实验1 --------------------
class Experiment1:
    UNIFIED_QOS = QoSProfile(
        business_type=BusinessType.ENVIRONMENT_MONITORING,
        min_rate=50, ideal_rate=100, max_delay=50, max_loss_rate=0.05,
        priority=0.5, downgrade_tolerance=0.3, criticality=0.5, latency_sensitivity=0.5
    )

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=5):
        print("\n" + "="*80)
        print("实验1：业务感知机制有效性验证")
        print(" - 动态识别：识别模型 + 差异化QoS")
        print(" - 无差异化：真实类型 + 统一QoS（证明差异化处理的价值）")
        print(" - 完美识别：真实类型 + 差异化QoS（理论上界）")
        print("="*80)

        dynamic_results, uniform_results, oracle_results = [], [], []

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            # 动态识别环境
            env_dynamic = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=50,
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep
            )
            algo_dynamic = EnhancedHandoverAlgorithm(env_dynamic)

            # 无差异化处理环境
            env_uniform = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=50,
                recognition_model=None, scaler=None,
                seed=GLOBAL_SEED + rep
            )
            for uav in env_uniform.uavs.values():
                uav.business_type = uav.true_business_type
                uav.qos_profile = Experiment1.UNIFIED_QOS
                uav.recognition_confidence = 1.0
            env_uniform.recognition_updater = None
            algo_uniform = EnhancedHandoverAlgorithm(env_uniform)

            # 完美识别环境
            env_oracle = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=50,
                recognition_model=None, scaler=None,
                seed=GLOBAL_SEED + rep
            )
            for uav in env_oracle.uavs.values():
                uav.business_type = uav.true_business_type
                uav.qos_profile = QOS_PROFILES[uav.true_business_type]
                uav.recognition_confidence = 1.0
            env_oracle.recognition_updater = None
            algo_oracle = EnhancedHandoverAlgorithm(env_oracle)

            # 运行仿真
            for step in range(num_steps):
                env_dynamic.step()
                algo_dynamic.run_step(enable_load_balancing=True)
                env_uniform.step()
                algo_uniform.run_step(enable_load_balancing=True)
                env_oracle.step()
                algo_oracle.run_step(enable_load_balancing=True)

            # 收集结果
            dynamic_stats = env_dynamic.get_state_statistics()
            dynamic_stats.update(algo_dynamic.get_detailed_stats())
            dynamic_results.append(dynamic_stats)

            uniform_stats = env_uniform.get_state_statistics()
            uniform_stats.update(algo_uniform.get_detailed_stats())
            uniform_results.append(uniform_stats)

            oracle_stats = env_oracle.get_state_statistics()
            oracle_stats.update(algo_oracle.get_detailed_stats())
            oracle_results.append(oracle_stats)

            print(f" 动态识别 - 满足率: {dynamic_stats['avg_satisfaction']:.3f}, "
                  f"识别准确率: {dynamic_stats['recognition_accuracy']:.1f}%")
            print(f" 无差异化 - 满足率: {uniform_stats['avg_satisfaction']:.3f}")
            print(f" 完美识别 - 满足率: {oracle_stats['avg_satisfaction']:.3f}")

        result = Experiment1._summarize_results(dynamic_results, uniform_results, oracle_results)
        Experiment1._print_results_table(result)
        Experiment1._plot(result)
        return result

    @staticmethod
    def _summarize_results(dynamic_results, uniform_results, oracle_results):
        def avg_std(key, results):
            values = [r[key] for r in results]
            return np.mean(values), np.std(values)

        return {
            'dynamic': {
                'satisfaction': avg_std('avg_satisfaction', dynamic_results),
                'recognition_accuracy': avg_std('recognition_accuracy', dynamic_results),
                'handover_success': avg_std('handover_success_rate', dynamic_results),
                'throughput': avg_std('total_load', dynamic_results),
                'critical_sat': avg_std('critical_satisfaction', dynamic_results),
                'weighted_sat': avg_std('weighted_satisfaction', dynamic_results),
            },
            'uniform': {
                'satisfaction': avg_std('avg_satisfaction', uniform_results),
                'handover_success': avg_std('handover_success_rate', uniform_results),
                'throughput': avg_std('total_load', uniform_results),
                'critical_sat': avg_std('critical_satisfaction', uniform_results),
                'weighted_sat': avg_std('weighted_satisfaction', uniform_results),
            },
            'oracle': {
                'satisfaction': avg_std('avg_satisfaction', oracle_results),
                'handover_success': avg_std('handover_success_rate', oracle_results),
                'throughput': avg_std('total_load', oracle_results),
                'critical_sat': avg_std('critical_satisfaction', oracle_results),
                'weighted_sat': avg_std('weighted_satisfaction', oracle_results),
            }
        }

    @staticmethod
    def _print_results_table(result):
        headers = ["指标", "动态识别", "无差异化", "完美识别", "差异化增益", "识别差距"]
        rows = [
            ["整体满足率",
             f"{result['dynamic']['satisfaction'][0]:.3f}±{result['dynamic']['satisfaction'][1]:.3f}",
             f"{result['uniform']['satisfaction'][0]:.3f}±{result['uniform']['satisfaction'][1]:.3f}",
             f"{result['oracle']['satisfaction'][0]:.3f}±{result['oracle']['satisfaction'][1]:.3f}",
             f"+{(result['dynamic']['satisfaction'][0]/result['uniform']['satisfaction'][0]-1)*100:.1f}%",
             f"{(result['oracle']['satisfaction'][0]-result['dynamic']['satisfaction'][0])*100:.2f}%点"],
            ["关键业务满足率",
             f"{result['dynamic']['critical_sat'][0]:.3f}±{result['dynamic']['critical_sat'][1]:.3f}",
             f"{result['uniform']['critical_sat'][0]:.3f}±{result['uniform']['critical_sat'][1]:.3f}",
             f"{result['oracle']['critical_sat'][0]:.3f}±{result['oracle']['critical_sat'][1]:.3f}",
             f"+{(result['dynamic']['critical_sat'][0]/result['uniform']['critical_sat'][0]-1)*100:.1f}%",
             f"{(result['oracle']['critical_sat'][0]-result['dynamic']['critical_sat'][0])*100:.2f}%点"],
            ["切换成功率",
             f"{result['dynamic']['handover_success'][0]*100:.1f}%",
             f"{result['uniform']['handover_success'][0]*100:.1f}%",
             f"{result['oracle']['handover_success'][0]*100:.1f}%",
             f"+{(result['dynamic']['handover_success'][0]/max(result['uniform']['handover_success'][0],0.01)-1)*100:.1f}%",
             f"{(result['oracle']['handover_success'][0]-result['dynamic']['handover_success'][0])*100:.1f}%点"],
            ["系统吞吐量(Mbps)",
             f"{result['dynamic']['throughput'][0]:.1f}±{result['dynamic']['throughput'][1]:.1f}",
             f"{result['uniform']['throughput'][0]:.1f}±{result['uniform']['throughput'][1]:.1f}",
             f"{result['oracle']['throughput'][0]:.1f}±{result['oracle']['throughput'][1]:.1f}",
             f"+{(result['dynamic']['throughput'][0]/result['uniform']['throughput'][0]-1)*100:.1f}%",
             f"{(result['dynamic']['throughput'][0]/result['oracle']['throughput'][0]-1)*100:.1f}%"],
        ]
        VisualizationHelper.print_data_table("实验1数据表", headers, rows)

    @staticmethod
    def _plot(result):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验1：业务感知机制有效性验证\n动态识别 vs 无差异化处理 vs 完美识别',
                     fontsize=14, fontweight='bold')

        metrics = ['satisfaction', 'critical_sat', 'weighted_sat', 'handover_success', 'throughput']
        titles = ['整体满足率', '关键业务满足率', '加权满足率', '切换成功率', '系统吞吐量']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]
            dyn_val = result['dynamic'][metric][0]
            dyn_std = result['dynamic'][metric][1]
            uni_val = result['uniform'][metric][0]
            uni_std = result['uniform'][metric][1]
            ora_val = result['oracle'][metric][0]
            ora_std = result['oracle'][metric][1]

            heights = [dyn_val, uni_val, ora_val]
            errors = [dyn_std, uni_std, ora_std]
            colors = [COLORS['primary'], COLORS['neutral'], COLORS['success']]
            labels = ['动态识别', '无差异化', '完美识别']

            bars = ax.bar(labels, heights, yerr=errors, capsize=5, color=colors,
                          alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('数值')
            for bar, height in zip(bars, heights):
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 差异化增益分析
        ax = axes[1, 2]
        categories = ['满足率', '关键业务', '切换成功率']
        gains = [
            (result['dynamic']['satisfaction'][0]/result['uniform']['satisfaction'][0]-1)*100,
            (result['dynamic']['critical_sat'][0]/result['uniform']['critical_sat'][0]-1)*100,
            (result['dynamic']['handover_success'][0]/max(result['uniform']['handover_success'][0],0.01)-1)*100
        ]
        gaps = [
            (result['oracle']['satisfaction'][0]-result['dynamic']['satisfaction'][0])*100,
            (result['oracle']['critical_sat'][0]-result['dynamic']['critical_sat'][0])*100,
            (result['oracle']['handover_success'][0]-result['dynamic']['handover_success'][0])*100
        ]
        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax.bar(x - width/2, gains, width, label='差异化增益(%)', color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, gaps, width, label='与完美的差距(%点)', color=COLORS['warning'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title('差异化增益与识别差距分析', fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp1_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验2 --------------------
class Experiment2:
    @staticmethod
    def run(recognition_model, scaler, num_steps=200, repeats=10):
        print("\n" + "="*80)
        print("实验2：业务识别与切换算法耦合监测")
        print("="*80)

        X_test, y_test = BusinessRecognitionModel.generate_business_data(num_samples_per_class=500, seed=GLOBAL_SEED+999)
        acc, f1, report = recognition_model.evaluate_on_test(X_test, y_test)
        print(f"\n模型基准准确率: {acc*100:.2f}%, F1-score: {f1:.3f}")

        all_results = []
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            env = EnhancedNetworkEnvironment(
                num_bs=8, num_uav=50,
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep, event_probability=0.05
            )
            algo = EnhancedHandoverAlgorithm(env)

            step_data = {
                'recognition_accuracy': [],
                'handover_success_rate': [],
                'avg_satisfaction': [],
                'critical_satisfaction': [],
                'weighted_satisfaction': [],
                'load_variance': [],
                'total_throughput': []
            }

            for step in range(num_steps):
                env.step()
                handovers, _ = algo.run_step(enable_load_balancing=True)
                stats = env.get_state_statistics()
                step_data['recognition_accuracy'].append(stats['recognition_accuracy'])
                step_data['avg_satisfaction'].append(stats['avg_satisfaction'])
                step_data['critical_satisfaction'].append(stats['critical_satisfaction'])
                step_data['weighted_satisfaction'].append(stats['weighted_satisfaction'])
                step_data['load_variance'].append(stats['load_variance'])
                step_data['total_throughput'].append(stats['total_load'])
                if algo.handover_attempts > 0:
                    cum_success_rate = algo.handover_successes / algo.handover_attempts
                else:
                    cum_success_rate = 1.0
                step_data['handover_success_rate'].append(cum_success_rate)

            corr_recog_sat = np.corrcoef(step_data['recognition_accuracy'], step_data['avg_satisfaction'])[0,1]
            corr_recog_success = np.corrcoef(step_data['recognition_accuracy'], step_data['handover_success_rate'])[0,1]

            result = {
                'avg_recognition_accuracy': np.mean(step_data['recognition_accuracy']),
                'final_handover_success': step_data['handover_success_rate'][-1],
                'avg_satisfaction': np.mean(step_data['avg_satisfaction']),
                'avg_critical_sat': np.mean(step_data['critical_satisfaction']),
                'avg_weighted_sat': np.mean(step_data['weighted_satisfaction']),
                'corr_recog_sat': corr_recog_sat,
                'corr_recog_success': corr_recog_success,
                'step_data': step_data,
                'updater_stats': env.recognition_updater.get_stats()
            }
            all_results.append(result)

            print(f" 平均识别准确率: {result['avg_recognition_accuracy']:.2f}%")
            print(f" 最终切换成功率: {result['final_handover_success']*100:.2f}%")
            print(f" 识别-满足率相关系数: {corr_recog_sat:.3f}")
            print(f" 识别-成功率相关系数: {corr_recog_success:.3f}")
            print(f" 自适应更新器: 更新{result['updater_stats']['update_count']}次, "
                  f"跳过{result['updater_stats']['skip_count']}次, "
                  f"漂移检测={result['updater_stats']['drift_detected']}")

        summary = Experiment2._summarize(all_results)
        Experiment2._print_results_table(summary, all_results)
        Experiment2._plot(summary, all_results)
        return summary

    @staticmethod
    def _summarize(all_results):
        return {
            'avg_recognition_accuracy': (np.mean([r['avg_recognition_accuracy'] for r in all_results]),
                                         np.std([r['avg_recognition_accuracy'] for r in all_results])),
            'final_handover_success': (np.mean([r['final_handover_success'] for r in all_results]),
                                       np.std([r['final_handover_success'] for r in all_results])),
            'avg_satisfaction': (np.mean([r['avg_satisfaction'] for r in all_results]),
                                 np.std([r['avg_satisfaction'] for r in all_results])),
            'avg_critical_sat': (np.mean([r['avg_critical_sat'] for r in all_results]),
                                 np.std([r['avg_critical_sat'] for r in all_results])),
            'corr_recog_sat': (np.mean([r['corr_recog_sat'] for r in all_results]),
                               np.std([r['corr_recog_sat'] for r in all_results])),
            'corr_recog_success': (np.mean([r['corr_recog_success'] for r in all_results]),
                                   np.std([r['corr_recog_success'] for r in all_results])),
        }

    @staticmethod
    def _print_results_table(summary, all_results):
        headers = ["指标", "均值±std", "说明"]
        rows = [
            ["识别准确率(%)", f"{summary['avg_recognition_accuracy'][0]:.2f}±{summary['avg_recognition_accuracy'][1]:.2f}", "业务识别模块精度"],
            ["切换成功率(%)", f"{summary['final_handover_success'][0]*100:.2f}±{summary['final_handover_success'][1]*100:.2f}", "切换算法有效性"],
            ["整体满足率", f"{summary['avg_satisfaction'][0]:.3f}±{summary['avg_satisfaction'][1]:.3f}", "QoE评估指标"],
            ["关键业务满足率", f"{summary['avg_critical_sat'][0]:.3f}±{summary['avg_critical_sat'][1]:.3f}", "控制信令等关键业务"],
            ["识别-满足率相关系数", f"{summary['corr_recog_sat'][0]:.3f}±{summary['corr_recog_sat'][1]:.3f}", "识别与QoE的关联度"],
            ["识别-成功率相关系数", f"{summary['corr_recog_success'][0]:.3f}±{summary['corr_recog_success'][1]:.3f}", "识别与切换性能的关联度"],
        ]
        VisualizationHelper.print_data_table("实验2结果：识别与切换耦合监测", headers, rows)

        print("\n自适应识别更新器统计:")
        total_updates = sum(r['updater_stats']['update_count'] for r in all_results)
        total_skips = sum(r['updater_stats']['skip_count'] for r in all_results)
        drift_alerts = sum(r['updater_stats']['drift_alerts'] for r in all_results)
        print(f" 总更新次数: {total_updates}")
        print(f" 总跳过次数: {total_skips}")
        print(f" 更新比例: {total_updates/(total_updates+total_skips)*100:.1f}%")
        print(f" 漂移警报次数: {drift_alerts}")

    @staticmethod
    def _plot(summary, all_results):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验2：业务识别与切换算法耦合监测', fontsize=14, fontweight='bold')

        rep_data = all_results[0]['step_data']
        steps = range(len(rep_data['recognition_accuracy']))

        # 识别准确率与满足率
        ax = axes[0,0]
        ax.plot(steps, rep_data['recognition_accuracy'], label='识别准确率', color=COLORS['primary'], linewidth=2)
        ax.plot(steps, [s*100 for s in rep_data['avg_satisfaction']], label='满足率(×100)', color=COLORS['success'], linewidth=2)
        ax.fill_between(steps, rep_data['recognition_accuracy'], alpha=0.2, color=COLORS['primary'])
        ax.set_xlabel('时间步')
        ax.set_ylabel('百分比(%)')
        ax.set_title('识别准确率与满足率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 识别准确率与切换成功率
        ax = axes[0,1]
        ax.plot(steps, rep_data['recognition_accuracy'], label='识别准确率', color=COLORS['primary'], linewidth=2)
        ax.plot(steps, [s*100 for s in rep_data['handover_success_rate']], label='切换成功率(×100)', color=COLORS['warning'], linewidth=2)
        ax.fill_between(steps, [s*100 for s in rep_data['handover_success_rate']], alpha=0.2, color=COLORS['warning'])
        ax.set_xlabel('时间步')
        ax.set_ylabel('百分比(%)')
        ax.set_title('识别准确率与切换成功率')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 分层满足率
        ax = axes[0,2]
        ax.plot(steps, rep_data['avg_satisfaction'], label='整体满足率', color=COLORS['primary'], linewidth=2)
        ax.plot(steps, rep_data['critical_satisfaction'], label='关键业务满足率', color=COLORS['danger'], linewidth=2)
        ax.plot(steps, rep_data['weighted_satisfaction'], label='加权满足率', color=COLORS['info'], linewidth=2)
        ax.fill_between(steps, rep_data['critical_satisfaction'], alpha=0.2, color=COLORS['danger'])
        ax.set_xlabel('时间步')
        ax.set_ylabel('满足率')
        ax.set_title('分层满足率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 散点图：识别准确率 vs 满足率
        ax = axes[1,0]
        for r in all_results:
            ax.scatter(r['avg_recognition_accuracy'], r['avg_satisfaction'], alpha=0.6, s=100, color=COLORS['primary'])
        z = np.polyfit([r['avg_recognition_accuracy'] for r in all_results],
                       [r['avg_satisfaction'] for r in all_results], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(r['avg_recognition_accuracy'] for r in all_results),
                             max(r['avg_recognition_accuracy'] for r in all_results), 50)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'拟合线 (斜率={z[0]:.4f})')
        ax.set_xlabel('识别准确率(%)')
        ax.set_ylabel('平均满足率')
        ax.set_title(f'识别准确率 vs 满足率 (r={summary["corr_recog_sat"][0]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 散点图：识别准确率 vs 切换成功率
        ax = axes[1,1]
        for r in all_results:
            ax.scatter(r['avg_recognition_accuracy'], r['final_handover_success']*100, alpha=0.6, s=100, color=COLORS['warning'])
        z = np.polyfit([r['avg_recognition_accuracy'] for r in all_results],
                       [r['final_handover_success']*100 for r in all_results], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(r['avg_recognition_accuracy'] for r in all_results),
                             max(r['avg_recognition_accuracy'] for r in all_results), 50)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'拟合线 (斜率={z[0]:.4f})')
        ax.set_xlabel('识别准确率(%)')
        ax.set_ylabel('切换成功率(%)')
        ax.set_title(f'识别准确率 vs 切换成功率 (r={summary["corr_recog_success"][0]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 系统吞吐量与负载方差
        ax = axes[1,2]
        ax_twin = ax.twinx()
        line1 = ax.plot(steps, rep_data['total_throughput'], label='吞吐量', color=COLORS['success'], linewidth=2)
        line2 = ax_twin.plot(steps, rep_data['load_variance'], label='负载方差', color=COLORS['accent'], linewidth=2, linestyle='--')
        ax.fill_between(steps, rep_data['total_throughput'], alpha=0.2, color=COLORS['success'])
        ax.set_xlabel('时间步')
        ax.set_ylabel('吞吐量(Mbps)', color=COLORS['success'])
        ax_twin.set_ylabel('负载方差', color=COLORS['accent'])
        ax.set_title('系统吞吐量与负载均衡')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp2_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验3 --------------------
class Experiment3:
    METRICS = {
        'handover_success_rate': '切换成功率',
        'avg_switching_latency_ms': '平均切换时延(ms)',
        'max_switching_latency_ms': '最大切换时延(ms)',
        'avg_decision_time_ms': '平均决策时间(ms)',
        'missed_opportunity_rate': '错失机会率',
        'avg_satisfaction': '整体满足率',
        'critical_satisfaction': '关键业务满足率',
        'weighted_satisfaction': '加权满足率',
        'latency_satisfaction': '时延满足率',
        'rate_satisfaction': '速率满足率',
        'total_throughput': '系统吞吐量(Mbps)',
        'load_variance': '负载方差',
        'avg_sinr': '平均SINR(dB)',
        'recognition_accuracy': '识别准确率(%)',
        'migration_success_rate': '迁移成功率',
        'connected_ratio': '连接保持率',
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=200, repeats=5):
        print("\n" + "="*80)
        print("实验3：增强算法 vs 传统算法（全面对比）")
        print("="*80)

        enhanced_results, traditional_results = [], []
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            # 增强算法
            env_enh = EnhancedNetworkEnvironment(
                num_bs=10, num_uav=80,
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep, event_probability=0.05
            )
            algo_enh = EnhancedHandoverAlgorithm(env_enh)

            # 传统算法
            env_trad = EnhancedNetworkEnvironment(
                num_bs=10, num_uav=80,
                recognition_model=recognition_model, scaler=scaler,
                seed=GLOBAL_SEED + rep, event_probability=0.05
            )
            algo_trad = IntegratedHandoverAlgorithm(env_trad)

            for step in range(num_steps):
                env_enh.step()
                algo_enh.run_step(enable_load_balancing=True)
                env_trad.step()
                algo_trad.run_step()

            enh_stats = env_enh.get_state_statistics()
            enh_stats.update(algo_enh.get_detailed_stats())
            enh_stats['connected_ratio'] = enh_stats['connected_count'] / env_enh.num_uav
            enhanced_results.append(enh_stats)

            trad_stats = env_trad.get_state_statistics()
            trad_stats.update(algo_trad.get_detailed_stats())
            trad_stats['connected_ratio'] = trad_stats['connected_count'] / env_trad.num_uav
            traditional_results.append(trad_stats)

            print(f" 增强算法 - 满足率: {enh_stats['avg_satisfaction']:.3f}, "
                  f"切换成功率: {enh_stats['handover_success_rate']*100:.1f}%, "
                  f"吞吐量: {enh_stats['total_load']:.1f} Mbps")
            print(f" 传统算法 - 满足率: {trad_stats['avg_satisfaction']:.3f}, "
                  f"切换成功率: {trad_stats['handover_success_rate']*100:.1f}%, "
                  f"吞吐量: {trad_stats['total_load']:.1f} Mbps")

        summary = Experiment3._summarize(enhanced_results, traditional_results)
        Experiment3._print_results_table(summary)
        Experiment3._plot(summary)
        return summary

    @staticmethod
    def _summarize(enhanced_results, traditional_results):
        summary = {'enhanced': {}, 'traditional': {}, 'improvement': {}}
        for key in Experiment3.METRICS.keys():
            if key in enhanced_results[0]:
                enh_vals = [r[key] for r in enhanced_results]
                summary['enhanced'][key] = (np.mean(enh_vals), np.std(enh_vals))
            if key in traditional_results[0]:
                trad_vals = [r[key] for r in traditional_results]
                summary['traditional'][key] = (np.mean(trad_vals), np.std(trad_vals))
                if np.mean(trad_vals) != 0:
                    improvement = (np.mean(enh_vals) - np.mean(trad_vals)) / abs(np.mean(trad_vals)) * 100
                else:
                    improvement = 0
                summary['improvement'][key] = improvement
            else:
                summary['traditional'][key] = (0,0)
                summary['improvement'][key] = 0
        return summary

    @staticmethod
    def _print_results_table(summary):
        headers = ["指标", "增强算法(均值±std)", "传统算法(均值±std)", "提升"]
        rows = []
        for key, name in Experiment3.METRICS.items():
            if key in summary['enhanced']:
                enh_mean, enh_std = summary['enhanced'][key]
                trad_mean, trad_std = summary['traditional'][key]
                imp = summary['improvement'][key]
                rows.append([name, f"{enh_mean:.3f}±{enh_std:.3f}", f"{trad_mean:.3f}±{trad_std:.3f}", f"{imp:+.1f}%"])
        VisualizationHelper.print_data_table("实验3结果：增强算法 vs 传统算法", headers, rows)

    @staticmethod
    def _plot(summary):
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('实验3：增强算法 vs 传统算法（全面对比）', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        def plot_bars(ax, metrics, labels, title):
            x = np.arange(len(labels))
            width = 0.35
            enh_vals = [summary['enhanced'][m][0] if m in summary['enhanced'] else 0 for m in metrics]
            trad_vals = [summary['traditional'][m][0] if m in summary['traditional'] else 0 for m in metrics]
            colors_enh = CMAP_PRIMARY(np.linspace(0.4, 0.8, len(labels)))
            colors_trad = plt.cm.Greys(np.linspace(0.4, 0.7, len(labels)))
            bars1 = ax.bar(x - width/2, enh_vals, width, label='增强算法', color=colors_enh)
            bars2 = ax.bar(x + width/2, trad_vals, width, label='传统算法', color=colors_trad)
            ax.set_ylabel('数值')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.legend()
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=7)

        # 子图
        ax = fig.add_subplot(gs[0,0])
        plot_bars(ax, ['handover_success_rate', 'avg_switching_latency_ms', 'max_switching_latency_ms'],
                  ['成功率', '平均时延', '最大时延'], '切换性能指标')

        ax = fig.add_subplot(gs[0,1])
        plot_bars(ax, ['avg_decision_time_ms', 'missed_opportunity_rate'],
                  ['决策时间', '错失率'], '决策性能指标')

        ax = fig.add_subplot(gs[0,2])
        plot_bars(ax, ['avg_satisfaction', 'critical_satisfaction', 'weighted_satisfaction'],
                  ['整体', '关键业务', '加权'], 'QoS满足率指标')

        ax = fig.add_subplot(gs[0,3])
        plot_bars(ax, ['total_throughput', 'load_variance', 'avg_sinr'],
                  ['吞吐量', '负载方差', 'SINR'], '网络性能指标')

        # 雷达图
        ax = fig.add_subplot(gs[1,:2], projection='polar')
        categories = ['切换成功率', '整体满足率', '关键业务满足率', '吞吐量', '连接保持率']
        metrics_map = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                       'total_throughput', 'connected_ratio']
        enh_vals, trad_vals = [], []
        for m in metrics_map:
            if m in summary['enhanced']:
                enh_val = summary['enhanced'][m][0]
                trad_val = summary['traditional'][m][0]
                if m == 'total_throughput':
                    enh_val = min(enh_val / 1000, 1.0)
                    trad_val = min(trad_val / 1000, 1.0)
                enh_vals.append(enh_val)
                trad_vals.append(trad_val)
            else:
                enh_vals.append(0); trad_vals.append(0)
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        enh_vals += enh_vals[:1]; trad_vals += trad_vals[:1]; angles += angles[:1]
        ax.plot(angles, enh_vals, 'o-', linewidth=2, label='增强算法', color=COLORS['primary'])
        ax.fill(angles, enh_vals, alpha=0.25, color=COLORS['primary'])
        ax.plot(angles, trad_vals, 'o-', linewidth=2, label='传统算法', color=COLORS['neutral'])
        ax.fill(angles, trad_vals, alpha=0.15, color=COLORS['neutral'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0,1)
        ax.set_title('综合性能雷达图', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))

        # 提升百分比
        ax = fig.add_subplot(gs[1,2:])
        improvements = [(k,v) for k,v in summary['improvement'].items() if abs(v)>0.1 and k in Experiment3.METRICS]
        improvements.sort(key=lambda x: abs(x[1]), reverse=True)
        if len(improvements) > 10:
            improvements = improvements[:10]
        names = [Experiment3.METRICS[k] for k,_ in improvements]
        values = [v for _,v in improvements]
        colors = [COLORS['success'] if v>0 else COLORS['danger'] for v in values]
        bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('提升百分比(%)')
        ax.set_title('关键指标提升对比', fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%',
                    ha='left' if val>0 else 'right', va='center', fontsize=8)

        # 热力图
        ax = fig.add_subplot(gs[2,:2])
        metrics_subset = ['handover_success_rate', 'avg_satisfaction', 'critical_satisfaction',
                          'latency_satisfaction', 'rate_satisfaction', 'connected_ratio']
        data = np.array([
            [summary['enhanced'][m][0] if m in summary['enhanced'] else 0 for m in metrics_subset],
            [summary['traditional'][m][0] if m in summary['traditional'] else 0 for m in metrics_subset]
        ])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics_subset)))
        ax.set_xticklabels([Experiment3.METRICS[m] for m in metrics_subset], rotation=45, ha='right')
        ax.set_yticks([0,1])
        ax.set_yticklabels(['增强算法', '传统算法'])
        ax.set_title('性能指标热力图', fontweight='bold')
        for i in range(2):
            for j in range(len(metrics_subset)):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax)

        # 关键指标分布对比
        ax = fig.add_subplot(gs[2,2:])
        metrics = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction']
        x_pos = np.arange(len(metrics))
        for i, m in enumerate(metrics):
            if m in summary['enhanced']:
                enh_mean, enh_std = summary['enhanced'][m]
                trad_mean, trad_std = summary['traditional'][m]
                ax.errorbar(i-0.15, enh_mean, yerr=enh_std, fmt='o', color=COLORS['primary'],
                            markersize=10, capsize=5, label='增强算法' if i==0 else '')
                ax.errorbar(i+0.15, trad_mean, yerr=trad_std, fmt='s', color=COLORS['neutral'],
                            markersize=10, capsize=5, label='传统算法' if i==0 else '')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([Experiment3.METRICS[m] for m in metrics], rotation=15, ha='right')
        ax.set_ylabel('数值')
        ax.set_title('关键指标分布对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 文本摘要
        ax = fig.add_subplot(gs[3,:])
        ax.axis('off')
        text = "【实验3关键发现】\n\n"
        key_findings = [
            ("切换成功率", 'handover_success_rate', "%", 100),
            ("整体满足率", 'avg_satisfaction', "", 1),
            ("关键业务满足率", 'critical_satisfaction', "", 1),
            ("系统吞吐量", 'total_throughput', " Mbps", 1),
            ("平均切换时延", 'avg_switching_latency_ms', " ms", 1),
        ]
        for name, key, unit, scale in key_findings:
            if key in summary['enhanced']:
                enh_val = summary['enhanced'][key][0] * scale
                trad_val = summary['traditional'][key][0] * scale
                improvement = summary['improvement'][key]
                text += f"• {name}: 增强算法 {enh_val:.2f}{unit} vs 传统算法 {trad_val:.2f}{unit} ({improvement:+.1f}%)\n"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(os.path.join(RESULT_DIR, 'exp3_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验4 --------------------
class Experiment4:
    MECHANISMS = {
        'full': '完整增强算法',
        'no_dynamic_threshold': '禁用动态阈值',
        'no_business_weights': '禁用业务特化权重',
        'no_epsilon_greedy': '禁用ε-greedy探索',
        'no_load_balance': '禁用负载均衡',
        'no_adaptive_recognition': '禁用自适应识别更新',
        'traditional': '传统算法（基线）'
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=3):
        print("\n" + "="*80)
        print("实验4：增强算法各机制有效性验证")
        print("="*80)

        results = {key: [] for key in Experiment4.MECHANISMS.keys()}
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)
            for mechanism in Experiment4.MECHANISMS.keys():
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=50,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, event_probability=0.05
                )
                if mechanism == 'traditional':
                    algo = IntegratedHandoverAlgorithm(env)
                else:
                    algo = EnhancedHandoverAlgorithm(env)
                    if mechanism == 'no_dynamic_threshold':
                        algo.base_threshold = 0.005
                        algo.calculate_dynamic_threshold = lambda uav: 0.005
                    elif mechanism == 'no_business_weights':
                        for bt in BusinessType:
                            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}
                    elif mechanism == 'no_epsilon_greedy':
                        algo.epsilon = 0.0
                    elif mechanism == 'no_adaptive_recognition':
                        env.recognition_updater = AdaptiveRecognitionUpdater(min_update_interval=999)

                for step in range(num_steps):
                    env.step()
                    if mechanism == 'traditional':
                        algo.run_step()
                    elif mechanism == 'no_load_balance':
                        algo.run_step(enable_load_balancing=False)
                    else:
                        algo.run_step(enable_load_balancing=True)

                stats = env.get_state_statistics()
                if hasattr(algo, 'get_detailed_stats'):
                    stats.update(algo.get_detailed_stats())
                results[mechanism].append(stats)
                print(f" {Experiment4.MECHANISMS[mechanism]}: "
                      f"满足率={stats['avg_satisfaction']:.3f}, "
                      f"切换成功率={stats.get('handover_success_rate',0)*100:.1f}%")

        summary = Experiment4._summarize(results)
        Experiment4._print_results_table(summary)
        Experiment4._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for mechanism, data_list in results.items():
            summary[mechanism] = {}
            for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                        'weighted_satisfaction', 'total_load', 'load_variance']:
                if key in data_list[0]:
                    vals = [d[key] for d in data_list]
                    summary[mechanism][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        headers = ["机制配置", "整体满足率", "切换成功率", "关键业务满足率", "吞吐量", "负载方差"]
        rows = []
        for mechanism, name in Experiment4.MECHANISMS.items():
            if mechanism in summary:
                data = summary[mechanism]
                row = [name]
                for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'total_load', 'load_variance']:
                    if key in data:
                        mean, std = data[key]
                        if key == 'handover_success_rate':
                            row.append(f"{mean*100:.1f}%±{std*100:.1f}%")
                        else:
                            row.append(f"{mean:.3f}±{std:.3f}")
                    else:
                        row.append("N/A")
                rows.append(row)
        VisualizationHelper.print_data_table("实验4结果：机制有效性验证", headers, rows)

        # 贡献分析
        print("\n各机制贡献分析（相对于完整算法）:")
        if 'full' in summary:
            full_sat = summary['full']['avg_satisfaction'][0]
            for mechanism in ['no_dynamic_threshold', 'no_business_weights', 'no_epsilon_greedy',
                              'no_load_balance', 'no_adaptive_recognition']:
                if mechanism in summary:
                    no_sat = summary[mechanism]['avg_satisfaction'][0]
                    contribution = full_sat - no_sat
                    print(f" {Experiment4.MECHANISMS[mechanism]}: 贡献 = {contribution:.4f} ({contribution/full_sat*100:+.1f}%)")

    @staticmethod
    def _plot(summary):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验4：增强算法各机制有效性验证', fontsize=14, fontweight='bold')
        mechanisms = list(Experiment4.MECHANISMS.keys())
        names = list(Experiment4.MECHANISMS.values())

        def plot_hbar(ax, key, title, xlabel):
            vals = [summary[m][key][0] if m in summary else 0 for m in mechanisms]
            errs = [summary[m][key][1] if m in summary else 0 for m in mechanisms]
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mechanisms)))
            bars = ax.barh(names, vals, xerr=errs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontweight='bold')
            for bar, val in zip(bars, vals):
                ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=9)

        plot_hbar(axes[0,0], 'avg_satisfaction', '整体满足率对比', '整体满足率')
        plot_hbar(axes[0,1], 'handover_success_rate', '切换成功率对比', '切换成功率')
        plot_hbar(axes[0,2], 'critical_satisfaction', '关键业务满足率对比', '关键业务满足率')

        # 贡献瀑布图
        ax = axes[1,0]
        if 'full' in summary:
            contributions = []
            contrib_names = []
            for mechanism in ['no_dynamic_threshold', 'no_business_weights', 'no_epsilon_greedy',
                              'no_load_balance', 'no_adaptive_recognition']:
                if mechanism in summary:
                    contrib = summary['full']['avg_satisfaction'][0] - summary[mechanism]['avg_satisfaction'][0]
                    contributions.append(contrib)
                    contrib_names.append(Experiment4.MECHANISMS[mechanism].replace('禁用',''))
            colors = [COLORS['success'] if c>0 else COLORS['danger'] for c in contributions]
            bars = ax.bar(contrib_names, contributions, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('贡献值')
            ax.set_title('各机制对满足率的贡献', fontweight='bold')
            ax.set_xticklabels(contrib_names, rotation=30, ha='right')

        # 核心指标对比
        ax = axes[1,1]
        if 'full' in summary and 'traditional' in summary:
            metrics = ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'weighted_satisfaction']
            metric_names = ['整体满足率', '切换成功率', '关键业务满足率', '加权满足率']
            full_vals = [summary['full'][m][0] for m in metrics]
            trad_vals = [summary['traditional'][m][0] for m in metrics]
            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, full_vals, width, label='完整增强算法', color=COLORS['primary'], alpha=0.8)
            ax.bar(x + width/2, trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, rotation=15, ha='right')
            ax.set_ylabel('数值')
            ax.set_title('核心指标对比', fontweight='bold')
            ax.legend()

        # 综合评分
        ax = axes[1,2]
        scores = []
        score_names = []
        for mechanism in mechanisms:
            if mechanism in summary:
                score = (0.4 * summary[mechanism]['avg_satisfaction'][0] +
                         0.3 * summary[mechanism]['handover_success_rate'][0] +
                         0.3 * summary[mechanism]['critical_satisfaction'][0])
                scores.append(score)
                score_names.append(Experiment4.MECHANISMS[mechanism])
        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        bars = ax.barh(score_names, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('综合评分')
        ax.set_title('机制综合评分', fontweight='bold')
        for bar, val in zip(bars, scores):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp4_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验5 --------------------
class Experiment5:
    SCENARIOS = {
        'default': {'name': '默认场景', 'desc': '标准仿真环境'},
        'urban': {'name': '城市物流', 'desc': '密集部署，障碍物多，时延敏感'},
        'emergency': {'name': '应急救援', 'desc': '高容量需求，低时延容忍，重视视频回传'},
        'agriculture': {'name': '农田监测', 'desc': '稀疏部署，大范围覆盖，周期性数据'}
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=3):
        print("\n" + "="*80)
        print("实验5：多场景对比实验")
        print("="*80)

        results = {scenario: {'enhanced': [], 'traditional': []} for scenario in Experiment5.SCENARIOS.keys()}
        for scenario, info in Experiment5.SCENARIOS.items():
            print(f"\n{'='*60}")
            print(f"场景: {info['name']} - {info['desc']}")
            print('='*60)
            for rep in range(repeats):
                print(f"\n 重复 {rep+1}/{repeats}")
                set_global_seed(GLOBAL_SEED + rep)

                env_enh = EnhancedNetworkEnvironment(
                    num_bs=10, num_uav=80,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, scenario=scenario, event_probability=0.05
                )
                algo_enh = EnhancedHandoverAlgorithm(env_enh)

                env_trad = EnhancedNetworkEnvironment(
                    num_bs=10, num_uav=80,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, scenario=scenario, event_probability=0.05
                )
                algo_trad = IntegratedHandoverAlgorithm(env_trad)

                for step in range(num_steps):
                    env_enh.step()
                    algo_enh.run_step(enable_load_balancing=True)
                    env_trad.step()
                    algo_trad.run_step()

                enh_stats = env_enh.get_state_statistics()
                enh_stats.update(algo_enh.get_detailed_stats())
                enh_stats['business_stats'] = env_enh.get_business_type_stats()
                results[scenario]['enhanced'].append(enh_stats)

                trad_stats = env_trad.get_state_statistics()
                trad_stats.update(algo_trad.get_detailed_stats())
                trad_stats['business_stats'] = env_trad.get_business_type_stats()
                results[scenario]['traditional'].append(trad_stats)

                print(f" 增强算法 - 满足率: {enh_stats['avg_satisfaction']:.3f}, "
                      f"关键业务: {enh_stats['critical_satisfaction']:.3f}")
                print(f" 传统算法 - 满足率: {trad_stats['avg_satisfaction']:.3f}, "
                      f"关键业务: {trad_stats['critical_satisfaction']:.3f}")

        summary = Experiment5._summarize(results)
        Experiment5._print_results_table(summary)
        Experiment5._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for scenario in Experiment5.SCENARIOS.keys():
            summary[scenario] = {'enhanced': {}, 'traditional': {}}
            for algo_type in ['enhanced', 'traditional']:
                data_list = results[scenario][algo_type]
                for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                            'weighted_satisfaction', 'total_load', 'avg_sinr', 'load_variance']:
                    if key in data_list[0]:
                        vals = [d[key] for d in data_list]
                        summary[scenario][algo_type][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        for scenario, info in Experiment5.SCENARIOS.items():
            print(f"\n【{info['name']}】{info['desc']}")
            headers = ["算法", "整体满足率", "切换成功率", "关键业务满足率", "吞吐量(Mbps)", "SINR(dB)"]
            rows = []
            for algo_type, algo_name in [('enhanced', '增强算法'), ('traditional', '传统算法')]:
                if algo_type in summary[scenario]:
                    data = summary[scenario][algo_type]
                    row = [algo_name]
                    for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction', 'total_load', 'avg_sinr']:
                        if key in data:
                            mean, std = data[key]
                            if key == 'handover_success_rate':
                                row.append(f"{mean*100:.1f}%±{std*100:.1f}%")
                            elif key == 'total_load':
                                row.append(f"{mean:.1f}±{std:.1f}")
                            elif key == 'avg_sinr':
                                row.append(f"{mean:.1f}±{std:.1f}")
                            else:
                                row.append(f"{mean:.3f}±{std:.3f}")
                        else:
                            row.append("N/A")
                    rows.append(row)
            # 计算提升
            if 'enhanced' in summary[scenario] and 'traditional' in summary[scenario]:
                enh_sat = summary[scenario]['enhanced']['avg_satisfaction'][0]
                trad_sat = summary[scenario]['traditional']['avg_satisfaction'][0]
                improvement = (enh_sat - trad_sat) / trad_sat * 100
                print(f" 满足率提升: {improvement:+.1f}%")
            VisualizationHelper.print_data_table(f"{info['name']}详细结果", headers, rows)

    @staticmethod
    def _plot(summary):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验5：多场景对比实验', fontsize=14, fontweight='bold')
        scenarios = list(Experiment5.SCENARIOS.keys())
        scenario_names = [Experiment5.SCENARIOS[s]['name'] for s in scenarios]
        x = np.arange(len(scenarios))
        width = 0.35

        # 满足率
        ax = axes[0,0]
        enh_vals = [summary[s]['enhanced']['avg_satisfaction'][0] if 'enhanced' in summary[s] else 0 for s in scenarios]
        trad_vals = [summary[s]['traditional']['avg_satisfaction'][0] if 'traditional' in summary[s] else 0 for s in scenarios]
        ax.bar(x - width/2, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('整体满足率')
        ax.set_title('各场景整体满足率对比', fontweight='bold')
        ax.legend()

        # 切换成功率
        ax = axes[0,1]
        enh_vals = [summary[s]['enhanced']['handover_success_rate'][0]*100 if 'enhanced' in summary[s] else 0 for s in scenarios]
        trad_vals = [summary[s]['traditional']['handover_success_rate'][0]*100 if 'traditional' in summary[s] else 0 for s in scenarios]
        ax.bar(x - width/2, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('切换成功率(%)')
        ax.set_title('各场景切换成功率对比', fontweight='bold')
        ax.legend()

        # 关键业务满足率
        ax = axes[0,2]
        enh_vals = [summary[s]['enhanced']['critical_satisfaction'][0] if 'enhanced' in summary[s] else 0 for s in scenarios]
        trad_vals = [summary[s]['traditional']['critical_satisfaction'][0] if 'traditional' in summary[s] else 0 for s in scenarios]
        ax.bar(x - width/2, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('关键业务满足率')
        ax.set_title('各场景关键业务满足率对比', fontweight='bold')
        ax.legend()

        # 吞吐量
        ax = axes[1,0]
        enh_vals = [summary[s]['enhanced']['total_load'][0] if 'enhanced' in summary[s] else 0 for s in scenarios]
        trad_vals = [summary[s]['traditional']['total_load'][0] if 'traditional' in summary[s] else 0 for s in scenarios]
        ax.bar(x - width/2, enh_vals, width, label='增强算法', color=COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, trad_vals, width, label='传统算法', color=COLORS['neutral'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        ax.set_ylabel('吞吐量(Mbps)')
        ax.set_title('各场景吞吐量对比', fontweight='bold')
        ax.legend()

        # 提升百分比
        ax = axes[1,1]
        improvements = []
        for s in scenarios:
            if 'enhanced' in summary[s] and 'traditional' in summary[s]:
                enh = summary[s]['enhanced']['avg_satisfaction'][0]
                trad = summary[s]['traditional']['avg_satisfaction'][0]
                improvements.append((enh - trad) / trad * 100)
            else:
                improvements.append(0)
        colors = [COLORS['success'] if i>0 else COLORS['danger'] for i in improvements]
        bars = ax.bar(scenario_names, improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('提升百分比(%)')
        ax.set_title('增强算法在各场景的满足率提升', fontweight='bold')
        ax.set_xticklabels(scenario_names, rotation=15, ha='right')
        for bar, val in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.1f}%',
                    ha='center', va='bottom' if val>0 else 'top', fontsize=9, fontweight='bold')

        # 热力图
        ax = axes[1,2]
        data = np.array([
            [summary[s]['enhanced']['avg_satisfaction'][0] if 'enhanced' in summary[s] else 0 for s in scenarios],
            [summary[s]['traditional']['avg_satisfaction'][0] if 'traditional' in summary[s] else 0 for s in scenarios],
            [summary[s]['enhanced']['handover_success_rate'][0] if 'enhanced' in summary[s] else 0 for s in scenarios],
            [summary[s]['traditional']['handover_success_rate'][0] if 'traditional' in summary[s] else 0 for s in scenarios],
        ])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_names, rotation=30, ha='right')
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(['增强-满足率', '传统-满足率', '增强-成功率', '传统-成功率'])
        ax.set_title('场景适应性热力图', fontweight='bold')
        for i in range(4):
            for j in range(len(scenarios)):
                val = data[i,j]
                text = ax.text(j, i, f'{val*100:.0f}%' if i>=2 else f'{val:.2f}',
                               ha='center', va='center',
                               color='white' if val<0.5 else 'black', fontsize=9, fontweight='bold')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp5_results.png'), dpi=200, bbox_inches='tight')
        plt.show()