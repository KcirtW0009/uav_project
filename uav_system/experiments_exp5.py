"""
实验5：鲁棒性测试实验

目的：测试增强算法在极端和突发情况下的性能

测试场景：
1. 正常基准（baseline）
2. 超高负载150%（overload_150）
3. 超高负载180%（overload_180）
4. 流量突发（burst_50）：50步内突增50% UAV
5. 基站故障（bs_failure）：基站故障50步
6. 混合压力（mixed_stress）：高负载+突发+故障

关键指标：
- 系统稳定性：是否崩溃/死锁
- 关键业务满足率：压力下的保障能力
- 恢复时间：故障后恢复到正常水平的步数
- 性能下降幅度：相对于正常情况的性能损失
- 存活率：完成仿真且未崩溃的比例
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS
from .business import BusinessType
from .recognition import train_or_load_recognition_model
from .environment import EnhancedNetworkEnvironment
from .algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
from .visualization import VisualizationHelper


class Experiment5:
    """实验5：鲁棒性测试实验"""

    SCENARIOS = {
        'baseline': {
            'name': '正常基准',
            'desc': '标准仿真环境（负载率约100%）',
            'params': {'num_bs': 8, 'num_uav': 180, 'event_probability': 0.05}
        },
        'overload_150': {
            'name': '超高负载150%',
            'desc': '系统负载率150%，测试极限性能',
            'params': {'num_bs': 8, 'num_uav': 270, 'event_probability': 0.05}
        },
        'overload_180': {
            'name': '超高负载180%',
            'desc': '系统负载率180%，测试崩溃阈值',
            'params': {'num_bs': 8, 'num_uav': 324, 'event_probability': 0.05}
        },
        'burst_50': {
            'name': '流量突发',
            'desc': '第100步突增90台UAV（50%）',
            'params': {'num_bs': 8, 'num_uav': 180, 'event_probability': 0.05,
                      'burst_config': {'burst_step': 100, 'burst_add': 90}}
        },
        'bs_failure': {
            'name': '基站故障',
            'desc': '第150步随机基站故障50步',
            'params': {'num_bs': 8, 'num_uav': 180, 'event_probability': 0.05,
                      'failure_config': {'failure_step': 150, 'failure_duration': 50}}
        },
        'mixed_stress': {
            'name': '混合压力',
            'desc': '高负载+突发+故障的综合压力测试',
            'params': {'num_bs': 8, 'num_uav': 270, 'event_probability': 0.10,
                      'burst_config': {'burst_step': 80, 'burst_add': 45},
                      'failure_config': {'failure_step': 180, 'failure_duration': 40}}
        }
    }

    METRICS = {
        'avg_satisfaction': '整体满足率',
        'critical_satisfaction': '关键业务满足率',
        'connected_ratio': '连接保持率',
        'handover_success_rate': '切换成功率',
        'total_load': '系统吞吐量(Mbps)',
        'avg_sinr': '平均SINR(dB)',
        'load_variance': '负载方差',
        'survival_rate': '系统存活率(%)',
        'performance_drop': '性能下降幅度(%)',
        'recovery_time': '平均恢复时间(步)',
        'crash_count': '崩溃次数',
        'decision_time_ms': '平均决策时间(ms)',
        'switching_latency_ms': '平均切换时延(ms)'
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=500, repeats=1):
        """
        运行鲁棒性测试实验

        Args:
            recognition_model: 业务识别模型
            scaler: 标准化器
            num_steps: 仿真步数
            repeats: 重复次数（快速测试：1次）
        """
        print("\n" + "="*80)
        print("实验5：鲁棒性测试实验")
        print("="*80)
        print("\n实验目的：测试增强算法在极端和突发情况下的性能")
        print("\n测试场景：")
        for scenario_id, info in Experiment5.SCENARIOS.items():
            print(f"  {scenario_id:15s}: {info['name']} - {info['desc']}")
        print("="*80)

        # 存储结果
        results = {
            scenario_id: {'enhanced': [], 'traditional': []}
            for scenario_id in Experiment5.SCENARIOS.keys()
        }

        # 先获取基线性能作为参考
        baseline_enhanced = []
        baseline_traditional = []
        for rep in range(repeats):
            print(f"\n--- 基线测试 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            env_enh, stats_enh = Experiment5._run_single_scenario(
                'baseline', rep, recognition_model, scaler, num_steps, algorithm_type='enhanced'
            )
            env_trad, stats_trad = Experiment5._run_single_scenario(
                'baseline', rep, recognition_model, scaler, num_steps, algorithm_type='traditional'
            )

            baseline_enhanced.append(stats_enh)
            baseline_traditional.append(stats_trad)

        # 计算基线平均性能
        baseline_avg_enh = {
            'avg_satisfaction': np.mean([s['avg_satisfaction'] for s in baseline_enhanced]),
            'critical_satisfaction': np.mean([s['critical_satisfaction'] for s in baseline_enhanced]),
            'connected_ratio': np.mean([s['connected_ratio'] for s in baseline_enhanced])
        }
        baseline_avg_trad = {
            'avg_satisfaction': np.mean([s['avg_satisfaction'] for s in baseline_traditional]),
            'critical_satisfaction': np.mean([s['critical_satisfaction'] for s in baseline_traditional]),
            'connected_ratio': np.mean([s['connected_ratio'] for s in baseline_traditional])
        }

        print(f"\n基线性能参考（增强算法）：")
        print(f"  整体满足率: {baseline_avg_enh['avg_satisfaction']:.4f}")
        print(f"  关键业务满足率: {baseline_avg_enh['critical_satisfaction']:.4f}")
        print(f"  连接保持率: {baseline_avg_enh['connected_ratio']:.4f}")

        # 运行其他场景
        for scenario_id, scenario_info in Experiment5.SCENARIOS.items():
            if scenario_id == 'baseline':
                continue  # 基线已测试

            print(f"\n{'='*80}")
            print(f"场景: {scenario_info['name']}")
            print(f"描述: {scenario_info['desc']}")
            print('='*80)

            for rep in range(repeats):
                print(f"\n--- 重复 {rep+1}/{repeats} ---")
                set_global_seed(GLOBAL_SEED + rep + 1000)  # 使用不同种子

                # 增强算法
                env_enh, stats_enh = Experiment5._run_single_scenario(
                    scenario_id, rep, recognition_model, scaler, num_steps, algorithm_type='enhanced'
                )
                # 传统算法
                env_trad, stats_trad = Experiment5._run_single_scenario(
                    scenario_id, rep, recognition_model, scaler, num_steps, algorithm_type='traditional'
                )

                # 计算性能下降幅度（相对于基线）
                # 修复：限制在合理范围（0-100%）
                drop_enh = (
                    (baseline_avg_enh['avg_satisfaction'] - stats_enh['avg_satisfaction']) /
                    baseline_avg_enh['avg_satisfaction'] * 100
                )
                drop_trad = (
                    (baseline_avg_trad['avg_satisfaction'] - stats_trad['avg_satisfaction']) /
                    baseline_avg_trad['avg_satisfaction'] * 100
                )
                stats_enh['performance_drop'] = max(0.0, min(drop_enh, 100.0))
                stats_trad['performance_drop'] = max(0.0, min(drop_trad, 100.0))

                # 计算恢复时间
                stats_enh['recovery_time'] = Experiment5._calculate_recovery_time(env_enh, scenario_id)
                stats_trad['recovery_time'] = Experiment5._calculate_recovery_time(env_trad, scenario_id)

                results[scenario_id]['enhanced'].append(stats_enh)
                results[scenario_id]['traditional'].append(stats_trad)

                print(f" 增强算法 - 满足率: {stats_enh['avg_satisfaction']:.3f}, "
                      f"关键业务: {stats_enh['critical_satisfaction']:.3f}, "
                      f"性能下降: {stats_enh['performance_drop']:.2f}%, "
                      f"恢复时间: {stats_enh['recovery_time']:.1f}步")
                print(f" 传统算法 - 满足率: {stats_trad['avg_satisfaction']:.3f}, "
                      f"关键业务: {stats_trad['critical_satisfaction']:.3f}, "
                      f"性能下降: {stats_trad['performance_drop']:.2f}%, "
                      f"恢复时间: {stats_trad['recovery_time']:.1f}步")

        # 汇总基线结果
        results['baseline']['enhanced'] = baseline_enhanced
        results['baseline']['traditional'] = baseline_traditional

        # 汇总和分析
        summary = Experiment5._summarize_results(results, baseline_avg_enh, baseline_avg_trad)
        Experiment5._print_results_table(summary)
        Experiment5._plot(summary)

        return summary

    @staticmethod
    def _run_single_scenario(scenario_id, rep, recognition_model, scaler,
                           num_steps, algorithm_type='enhanced'):
        """
        运行单个场景的单次实验

        Args:
            scenario_id: 场景ID
            rep: 重复次数
            recognition_model: 业务识别模型
            scaler: 标准化器
            num_steps: 仿真步数
            algorithm_type: 算法类型 ('enhanced' 或 'traditional')

        Returns:
            (env, stats): 环境对象和统计结果
        """
        scenario_params = Experiment5.SCENARIOS[scenario_id]['params'].copy()

        # 创建环境
        env = EnhancedNetworkEnvironment(
            num_bs=scenario_params['num_bs'],
            num_uav=scenario_params['num_uav'],
            recognition_model=recognition_model,
            scaler=scaler,
            seed=GLOBAL_SEED + rep + 1000,
            scenario='default',
            event_probability=scenario_params.get('event_probability', 0.05)
        )

        # 记录初始连接数
        initial_connected = sum(1 for uav in env.uavs.values() if uav.connected_bs_id is not None)

        # 创建算法
        if algorithm_type == 'enhanced':
            algo = EnhancedHandoverAlgorithm(env)
        else:
            algo = IntegratedHandoverAlgorithm(env)

        # 处理突发流量配置
        burst_config = scenario_params.get('burst_config', None)
        if burst_config:
            env.burst_step = burst_config['burst_step']
            env.burst_add = burst_config['burst_add']

        # 处理故障配置
        failure_config = scenario_params.get('failure_config', None)
        if failure_config:
            env.forced_failure_step = failure_config['failure_step']
            env.forced_failure_duration = failure_config['failure_duration']
        scenario_params = Experiment5.SCENARIOS[scenario_id]['params'].copy()

        # 创建环境
        env = EnhancedNetworkEnvironment(
            num_bs=scenario_params['num_bs'],
            num_uav=scenario_params['num_uav'],
            recognition_model=recognition_model,
            scaler=scaler,
            seed=GLOBAL_SEED + rep + 1000,
            scenario='default',
            event_probability=scenario_params.get('event_probability', 0.05)
        )

        # 创建算法
        if algorithm_type == 'enhanced':
            algo = EnhancedHandoverAlgorithm(env)
        else:
            algo = IntegratedHandoverAlgorithm(env)

        # 处理突发流量配置
        burst_config = scenario_params.get('burst_config', None)
        if burst_config:
            env.burst_step = burst_config['burst_step']
            env.burst_add = burst_config['burst_add']

        # 处理故障配置
        failure_config = scenario_params.get('failure_config', None)
        if failure_config:
            env.forced_failure_step = failure_config['failure_step']
            env.forced_failure_duration = failure_config['failure_duration']

        # 运行仿真
        crash_count = 0
        for step in range(num_steps):
            try:
                # 触发突发流量
                if hasattr(env, 'burst_step') and step == env.burst_step:
                    Experiment5._trigger_burst_traffic(env, env.burst_add)

                # 触发强制故障
                if hasattr(env, 'forced_failure_step') and step == env.forced_failure_step:
                    Experiment5._trigger_forced_failure(env)

                env.step()
                if algorithm_type == 'enhanced':
                    algo.run_step(enable_load_balancing=True)
                else:
                    algo.run_step()

            except Exception as e:
                print(f"  [警告] 步骤 {step} 发生异常: {e}")
                crash_count += 1
                break

        # 收集统计信息
        stats = env.get_state_statistics()
        stats.update(algo.get_detailed_stats())

        # 修复：正确的存活率计算
        stats['crash_count'] = crash_count
        stats['survival_rate'] = 100.0 if crash_count == 0 else 0.0

        # 修复：连接保持率 = 当前连接数 / 总UAV数
        stats['connected_ratio'] = stats['connected_count'] / env.num_uav

        # 调试输出
        if 'handover_attempts' in stats:
            print(f"    [调试] 切换尝试: {stats['handover_attempts']}, 切换成功: {stats['handover_successes']}")
            print(f"    [调试] 切换成功率: {stats['handover_success_rate']*100:.2f}%")
        if 'decision_calls' in stats:
            print(f"    [调试] 决策调用: {stats['decision_calls']}, 过滤次数: {stats.get('missed_opportunity', 0)}")
        if 'decision_filters' in stats and stats['decision_filters']:
            print(f"    [调试] 决策过滤原因: {stats['decision_filters']}")

        return env, stats

    @staticmethod
    def _trigger_burst_traffic(env, num_add):
        """触发突发流量，添加新的UAV"""
        print(f"  [突发流量] 添加 {num_add} 台UAV")
        for i in range(num_add):
            new_id = len(env.uavs)
            biz_type = np.random.choice(list(BusinessType))
            from .entities import UAV
            env.uavs[new_id] = UAV(new_id, business_type=biz_type)

        # 扩展矩阵
        env.num_uav = len(env.uavs)
        env.connection_matrix = np.vstack([
            env.connection_matrix,
            np.zeros((num_add, env.num_bs))
        ])
        env.sinr_matrix = np.vstack([
            env.sinr_matrix,
            np.zeros((num_add, env.num_bs))
        ])

        # 初始化连接
        for uav_id in env.uavs.keys():
            if env.uavs[uav_id].connected_bs_id is None:
                best_bs = None
                best_sinr = -999
                for bs_id, bs in env.base_stations.items():
                    sinr = env._calculate_sinr(uav_id, bs_id)
                    if sinr > best_sinr:
                        best_sinr = sinr
                        best_bs = bs_id
                if best_bs is not None:
                    if env.base_stations[best_bs].allocate(uav_id, env.uavs[uav_id].required_rate):
                        env.uavs[uav_id].connected_bs_id = best_bs
                        env.connection_matrix[uav_id, best_bs] = 1

        # 更新SINR矩阵
        env._update_sinr_matrix()

    @staticmethod
    def _trigger_forced_failure(env):
        """触发强制基站故障"""
        import random
        bs_id = random.choice(list(env.base_stations.keys()))
        bs = env.base_stations[bs_id]
        if not bs.failure_state:
            print(f"  [基站故障] 基站 {bs_id} 故障")
            bs.set_failure(True)
            env.event_stats['bs_failure'] += 1
            env.event_id_counter += 1
            event_id = f"failure_{env.event_id_counter}"
            affected_uavs = [uav_id for uav_id, uav in env.uavs.items()
                           if uav.connected_bs_id == bs_id]
            env.event_history.append({
                'type': 'bs_failure', 'bs_id': bs_id, 'old_capacity': bs.capacity,
                'step': env.current_step, 'event_id': event_id, 'affected_uavs': affected_uavs
            })
            env.active_failures[event_id] = {
                'type': 'bs_failure', 'step': env.current_step,
                'affected_uavs': affected_uavs, 'bs_id': bs_id
            }

    @staticmethod
    def _calculate_recovery_time(env, scenario_id):
        """
        计算恢复时间

        对于故障场景：从故障发生到满足率恢复到故障前80%的步数
        对于突发场景：从突发发生到满足率恢复到突发前85%的步数
        """
        if len(env.stats_history['step']) < 50:
            return 0.0

        steps = env.stats_history['step']
        satisfactions = env.stats_history['avg_satisfaction']

        # 确定事件发生点
        if 'failure' in scenario_id:
            event_step = Experiment5.SCENARIOS[scenario_id]['params'].get(
                'failure_config', {}).get('failure_step', 150)
            target_ratio = 0.8
        elif 'burst' in scenario_id:
            event_step = Experiment5.SCENARIOS[scenario_id]['params'].get(
                'burst_config', {}).get('burst_step', 100)
            target_ratio = 0.85
        else:
            return 0.0

        # 找到事件前的平均满足率
        pre_event_sats = [s for step, s in zip(steps, satisfactions) if step < event_step]
        if len(pre_event_sats) == 0:
            return 0.0
        pre_event_avg = np.mean(pre_event_sats[-10:])  # 取前10步平均

        # 计算目标满足率
        target_sat = pre_event_avg * target_ratio

        # 从事件发生后查找恢复时间
        post_event_steps = [step for step in steps if step >= event_step]
        post_event_sats = [s for step, s in zip(steps, satisfactions) if step >= event_step]

        for step, sat in zip(post_event_steps, post_event_sats):
            if sat >= target_sat:
                return step - event_step

        # 未恢复
        return len(post_event_steps) if post_event_steps else 0.0

    @staticmethod
    def _summarize_results(results, baseline_enh, baseline_trad):
        """汇总实验结果"""
        summary = {}
        for scenario_id in Experiment5.SCENARIOS.keys():
            summary[scenario_id] = {
                'enhanced': {},
                'traditional': {},
                'improvement': {}
            }

            for algo_type in ['enhanced', 'traditional']:
                data_list = results[scenario_id][algo_type]
                for key in ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio',
                          'handover_success_rate', 'total_load', 'avg_sinr', 'load_variance',
                          'performance_drop', 'recovery_time', 'survival_rate']:
                    if key in data_list[0]:
                        vals = [d[key] for d in data_list]
                        summary[scenario_id][algo_type][key] = (np.mean(vals), np.std(vals))

            # 计算改进幅度
            for key in ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio',
                      'handover_success_rate', 'performance_drop', 'recovery_time']:
                enh_mean = summary[scenario_id]['enhanced'].get(key, (0, 0))[0]
                trad_mean = summary[scenario_id]['traditional'].get(key, (0, 0))[0]

                if key in ['performance_drop', 'recovery_time']:
                    # 越小越好
                    if trad_mean != 0:
                        improvement = (trad_mean - enh_mean) / abs(trad_mean) * 100
                    else:
                        improvement = 0
                else:
                    # 越大越好
                    if trad_mean != 0:
                        improvement = (enh_mean - trad_mean) / abs(trad_mean) * 100
                    else:
                        improvement = 0

                summary[scenario_id]['improvement'][key] = improvement

        return summary

    @staticmethod
    def _print_results_table(summary):
        """打印结果表格"""
        print("\n" + "="*120)
        print("实验5结果：鲁棒性测试")
        print("="*120)

        for scenario_id, scenario_info in Experiment5.SCENARIOS.items():
            print(f"\n【{scenario_info['name']}】{scenario_info['desc']}")

            headers = ["指标", "增强算法", "传统算法", "改进"]
            rows = []

            metrics_to_show = [
                ('avg_satisfaction', '整体满足率', ''),
                ('critical_satisfaction', '关键业务满足率', ''),
                ('connected_ratio', '连接保持率', ''),
                ('handover_success_rate', '切换成功率', '%'),
                ('performance_drop', '性能下降幅度', '%'),
                ('recovery_time', '平均恢复时间', '步'),
                ('survival_rate', '系统存活率', '%')
            ]

            for key, name, unit in metrics_to_show:
                enh_mean, enh_std = summary[scenario_id]['enhanced'].get(key, (0, 0))
                trad_mean, trad_std = summary[scenario_id]['traditional'].get(key, (0, 0))
                imp = summary[scenario_id]['improvement'].get(key, 0)

                if key == 'handover_success_rate' or key == 'survival_rate' or unit == '%':
                    enh_str = f"{enh_mean*100:.1f}%±{enh_std*100:.1f}%"
                    trad_str = f"{trad_mean*100:.1f}%±{trad_std*100:.1f}%"
                else:
                    enh_str = f"{enh_mean:.4f}±{enh_std:.4f}"
                    trad_str = f"{trad_mean:.4f}±{trad_std:.4f}"

                row = [name, enh_str, trad_str, f"{imp:+.1f}%"]
                rows.append(row)

            VisualizationHelper.print_data_table(
                f"{scenario_info['name']}详细结果",
                headers,
                rows
            )

        # 汇总关键发现
        print("\n" + "="*120)
        print("【鲁棒性关键发现】")
        print("="*120)

        high_load_scenarios = ['overload_150', 'overload_180']
        stress_scenarios = ['burst_50', 'bs_failure', 'mixed_stress']

        print("\n高负载场景表现：")
        for sc in high_load_scenarios:
            if sc in summary:
                enh_sat = summary[sc]['enhanced']['avg_satisfaction'][0]
                trad_sat = summary[sc]['traditional']['avg_satisfaction'][0]
                enh_surv = summary[sc]['enhanced']['survival_rate'][0]
                trad_surv = summary[sc]['traditional']['survival_rate'][0]
                imp = summary[sc]['improvement']['avg_satisfaction']
                print(f"  {Experiment5.SCENARIOS[sc]['name']}: ")
                print(f"    满足率: 增强={enh_sat:.4f} vs 传统={trad_sat:.4f} ({imp:+.1f}%)")
                print(f"    存活率: 增强={enh_surv:.1f}% vs 传统={trad_surv:.1f}%")

        print("\n压力场景表现：")
        for sc in stress_scenarios:
            if sc in summary:
                enh_drop = summary[sc]['enhanced']['performance_drop'][0]
                trad_drop = summary[sc]['traditional']['performance_drop'][0]
                enh_recov = summary[sc]['enhanced']['recovery_time'][0]
                trad_recov = summary[sc]['traditional']['recovery_time'][0]
                recov_imp = summary[sc]['improvement']['recovery_time']
                print(f"  {Experiment5.SCENARIOS[sc]['name']}: ")
                print(f"    性能下降: 增强={enh_drop:.2f}% vs 传统={trad_drop:.2f}%")
                print(f"    恢复时间: 增强={enh_recov:.1f}步 vs 传统={trad_recov:.1f}步 ({recov_imp:+.1f}%)")

        print("="*120)

    @staticmethod
    def _plot(summary):
        """绘制结果图表"""
        fig = plt.figure(figsize=(22, 16))
        fig.suptitle('实验5：鲁棒性测试实验', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        scenarios = list(Experiment5.SCENARIOS.keys())
        scenario_names = [Experiment5.SCENARIOS[s]['name'] for s in scenarios]

        # 1. 整体满足率对比
        ax = fig.add_subplot(gs[0, 0])
        x = np.arange(len(scenarios))
        width = 0.35
        enh_sats = [summary[s]['enhanced'].get('avg_satisfaction', (0,0))[0] for s in scenarios]
        trad_sats = [summary[s]['traditional'].get('avg_satisfaction', (0,0))[0] for s in scenarios]
        bars1 = ax.bar(x - width/2, enh_sats, width, label='增强算法',
                      color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, trad_sats, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('整体满足率')
        ax.set_title('各场景整体满足率对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 2. 关键业务满足率对比
        ax = fig.add_subplot(gs[0, 1])
        crit_enh = [summary[s]['enhanced'].get('critical_satisfaction', (0,0))[0] for s in scenarios]
        crit_trad = [summary[s]['traditional'].get('critical_satisfaction', (0,0))[0] for s in scenarios]
        bars1 = ax.bar(x - width/2, crit_enh, width, label='增强算法',
                      color=COLORS['danger'], alpha=0.8)
        bars2 = ax.bar(x + width/2, crit_trad, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('关键业务满足率')
        ax.set_title('各场景关键业务满足率对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. 性能下降幅度
        ax = fig.add_subplot(gs[0, 2])
        drops_enh = [summary[s]['enhanced'].get('performance_drop', (0,0))[0] for s in scenarios]
        drops_trad = [summary[s]['traditional'].get('performance_drop', (0,0))[0] for s in scenarios]
        bars1 = ax.bar(x - width/2, drops_enh, width, label='增强算法',
                      color=COLORS['warning'], alpha=0.8)
        bars2 = ax.bar(x + width/2, drops_trad, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('性能下降幅度 (%)')
        ax.set_title('各场景性能下降幅度对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # 4. 恢复时间对比
        ax = fig.add_subplot(gs[0, 3])
        recov_enh = [summary[s]['enhanced'].get('recovery_time', (0,0))[0] for s in scenarios]
        recov_trad = [summary[s]['traditional'].get('recovery_time', (0,0))[0] for s in scenarios]
        bars1 = ax.bar(x - width/2, recov_enh, width, label='增强算法',
                      color=COLORS['success'], alpha=0.8)
        bars2 = ax.bar(x + width/2, recov_trad, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('恢复时间 (步)')
        ax.set_title('各场景恢复时间对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 5. 系统存活率
        ax = fig.add_subplot(gs[1, 0])
        surv_enh = [summary[s]['enhanced'].get('survival_rate', (0,0))[0] for s in scenarios]
        surv_trad = [summary[s]['traditional'].get('survival_rate', (0,0))[0] for s in scenarios]
        colors_enh = [COLORS['success'] if v >= 90 else COLORS['danger'] for v in surv_enh]
        colors_trad = [plt.cm.Greys(v/100) for v in surv_trad]
        bars1 = ax.bar(x - width/2, surv_enh, width, label='增强算法',
                      color=colors_enh, alpha=0.8, edgecolor='white', linewidth=1.5)
        bars2 = ax.bar(x + width/2, surv_trad, width, label='传统算法',
                      color=colors_trad, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('系统存活率 (%)')
        ax.set_title('各场景系统存活率对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])

        # 6. 切换成功率
        ax = fig.add_subplot(gs[1, 1])
        success_enh = [summary[s]['enhanced'].get('handover_success_rate', (0,0))[0]*100 for s in scenarios]
        success_trad = [summary[s]['traditional'].get('handover_success_rate', (0,0))[0]*100 for s in scenarios]
        bars1 = ax.bar(x - width/2, success_enh, width, label='增强算法',
                      color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, success_trad, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('切换成功率 (%)')
        ax.set_title('各场景切换成功率对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])

        # 7. 连接保持率
        ax = fig.add_subplot(gs[1, 2])
        conn_enh = [summary[s]['enhanced'].get('connected_ratio', (0,0))[0] for s in scenarios]
        conn_trad = [summary[s]['traditional'].get('connected_ratio', (0,0))[0] for s in scenarios]
        bars1 = ax.bar(x - width/2, conn_enh, width, label='增强算法',
                      color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, conn_trad, width, label='传统算法',
                      color=COLORS['neutral'], alpha=0.8)
        ax.set_ylabel('连接保持率')
        ax.set_title('各场景连接保持率对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 8. 改进幅度（性能下降）
        ax = fig.add_subplot(gs[1, 3])
        imp_drop = [summary[s]['improvement'].get('performance_drop', 0) for s in scenarios]
        colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in imp_drop]
        bars = ax.bar(scenario_names, imp_drop, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('改进幅度 (%)')
        ax.set_title('性能下降幅度改进（越小越好）', fontweight='bold')
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for bar, imp in zip(bars, imp_drop):
            if abs(imp) > 1:
                ax.text(bar.get_x() + bar.get_width()/2, imp,
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                       fontsize=7, fontweight='bold')

        # 9. 改进幅度（恢复时间）
        ax = fig.add_subplot(gs[2, 0])
        imp_recov = [summary[s]['improvement'].get('recovery_time', 0) for s in scenarios]
        colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in imp_recov]
        bars = ax.bar(scenario_names, imp_recov, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('改进幅度 (%)')
        ax.set_title('恢复时间改进（越小越好）', fontweight='bold')
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        for bar, imp in zip(bars, imp_recov):
            if abs(imp) > 5:
                ax.text(bar.get_x() + bar.get_width()/2, imp,
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top',
                       fontsize=7, fontweight='bold')

        # 10. 热力图
        ax = fig.add_subplot(gs[2, 1])
        metrics_subset = ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio',
                        'handover_success_rate', 'survival_rate']
        data = np.array([
            [summary[s]['enhanced'].get(m, (0,0))[0] if s in summary else 0
             for s in scenarios for m in metrics_subset if m in summary[s]['enhanced']],
            [summary[s]['traditional'].get(m, (0,0))[0] if s in summary else 0
             for s in scenarios for m in metrics_subset if m in summary[s]['traditional']]
        ])
        # 重新构建正确维度的数据
        data = np.zeros((2, len(scenarios)))
        for i, s in enumerate(scenarios):
            if 'avg_satisfaction' in summary[s]['enhanced']:
                data[0, i] = summary[s]['enhanced']['avg_satisfaction'][0]
            if 'avg_satisfaction' in summary[s]['traditional']:
                data[1, i] = summary[s]['traditional']['avg_satisfaction'][0]
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['增强算法', '传统算法'])
        ax.set_title('鲁棒性热力图（满足率）', fontweight='bold')
        for i in range(2):
            for j in range(len(scenarios)):
                val = data[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color='white' if val < 0.5 else 'black', fontsize=8)
        plt.colorbar(im, ax=ax)

        # 11. 鲁棒性评分雷达图
        ax = fig.add_subplot(gs[2, 2:], projection='polar')
        categories = ['满足率', '关键业务', '连接保持', '切换成功', '存活率']
        metrics_keys = ['avg_satisfaction', 'critical_satisfaction', 'connected_ratio',
                      'handover_success_rate', 'survival_rate']

        # 计算各场景的平均鲁棒性评分
        enh_scores = []
        trad_scores = []
        for m in metrics_keys:
            enh_vals = [summary[s]['enhanced'].get(m, (0,0))[0] for s in scenarios]
            trad_vals = [summary[s]['traditional'].get(m, (0,0))[0] for s in scenarios]
            enh_scores.append(np.mean(enh_vals))
            trad_scores.append(np.mean(trad_vals))

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        enh_scores += enh_scores[:1]
        trad_scores += trad_scores[:1]
        angles += angles[:1]

        ax.plot(angles, enh_scores, 'o-', linewidth=2, label='增强算法', color=COLORS['primary'])
        ax.fill(angles, enh_scores, alpha=0.25, color=COLORS['primary'])
        ax.plot(angles, trad_scores, 'o-', linewidth=2, label='传统算法', color=COLORS['neutral'])
        ax.fill(angles, trad_scores, alpha=0.15, color=COLORS['neutral'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('平均鲁棒性雷达图', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 12. 关键文本摘要
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')

        text = "【实验5关键发现】\n\n"

        # 基线性能
        if 'baseline' in summary:
            text += "基线性能（正常场景）：\n"
            text += f"  • 增强算法: 满足率={summary['baseline']['enhanced']['avg_satisfaction'][0]:.4f}, "
            text += f"关键业务={summary['baseline']['enhanced']['critical_satisfaction'][0]:.4f}\n"
            text += f"  • 传统算法: 满足率={summary['baseline']['traditional']['avg_satisfaction'][0]:.4f}, "
            text += f"关键业务={summary['baseline']['traditional']['critical_satisfaction'][0]:.4f}\n\n"

        # 高负载场景
        text += "高负载场景鲁棒性：\n"
        for sc in ['overload_150', 'overload_180']:
            if sc in summary:
                sat_imp = summary[sc]['improvement'].get('avg_satisfaction', 0)
                surv_enh = summary[sc]['enhanced'].get('survival_rate', (0,0))[0]
                surv_trad = summary[sc]['traditional'].get('survival_rate', (0,0))[0]
                text += f"  • {Experiment5.SCENARIOS[sc]['name']}: "
                text += f"满足率改进{sat_imp:+.1f}%, 存活率={surv_enh:.0f}% vs {surv_trad:.0f}%\n"

        # 压力场景
        text += "\n压力场景应对能力：\n"
        for sc in ['burst_50', 'bs_failure', 'mixed_stress']:
            if sc in summary:
                drop_enh = summary[sc]['enhanced'].get('performance_drop', (0,0))[0]
                drop_trad = summary[sc]['traditional'].get('performance_drop', (0,0))[0]
                recov_imp = summary[sc]['improvement'].get('recovery_time', 0)
                text += f"  • {Experiment5.SCENARIOS[sc]['name']}: "
                text += f"性能下降={drop_enh:.1f}% vs {drop_trad:.1f}%, "
                text += f"恢复时间改进{recov_imp:+.1f}%\n"

        ax.text(0.02, 0.98, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(os.path.join(RESULT_DIR, 'exp5_results.png'), dpi=200, bbox_inches='tight')
        plt.show()
