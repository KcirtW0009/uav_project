import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from scipy import stats
from .config import GLOBAL_SEED, set_global_seed, RESULT_DIR, COLORS, CMAP_PRIMARY, CMAP_SUCCESS, CMAP_WARNING
from .business import BusinessType, QoSProfile, QOS_PROFILES
from .satisfaction import HierarchicalSatisfactionMetric
from .recognition import AdaptiveRecognitionUpdater, BusinessRecognitionModel, train_or_load_recognition_model
from .environment import EnhancedNetworkEnvironment
from .algorithms import IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm
from .visualization import VisualizationHelper

# 配置字体和警告抑制
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

# -------------------- 实验1 --------------------
class Experiment1:
    """
    实验1：识别准确性的价值验证（重构版）
    
    核心问题：识别准确率如何影响系统性能？
    
    实验设计：
    - 条件A（100%准确率）：使用真实业务类型作为识别结果（基准）
    - 条件B（85%准确率）：使用高质量模型，人工注入15%噪声
    - 条件C（70%准确率）：使用中等质量模型，人工注入30%噪声
    - 条件D（随机33%）：随机分配业务类型（下界对照）
    
    所有条件使用相同的差异化QoS配置，控制其他变量一致
    """

    # 预设的识别准确率目标值（均匀梯度分布）
    ACCURACY_LEVELS = {
        'perfect': 1.00,    # 100% - 基准
        'high': 0.90,       # 90% - 高质量模型
        'medium': 0.80,      # 80% - 中等质量模型
        'low': 0.60,        # 60% - 低质量模型
        'random': 0.33,     # 33% - 随机猜测
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=15):  # 增加到15次重复以减少随机波动
        print("\n" + "="*80)
        print("实验1：识别准确性的价值验证")
        print("="*80)
        print("\n实验目的：验证业务识别准确率对系统性能的影响")
        print("\n实验条件：")
        print("  A. 100%准确率 - 使用真实类型（性能基准）")
        print("  B.  90%准确率 - 高质量识别模型")
        print("  C.  80%准确率 - 中等质量识别模型")
        print("  D.  60%准确率 - 低质量识别模型")
        print("  E.  33%准确率 - 随机分配（下界对照）")
        print("\n控制变量：差异化QoS配置、切换算法、网络环境完全相同")
        print("\n算法配置说明：使用EnhancedHandover算法，但禁用ε-greedy探索机制")
        print("  原因：ε-greedy引入随机性，会干扰识别准确率对性能影响的评估")
        print("="*80)

        # 存储各条件的结果
        results_by_accuracy = {
            'perfect': [],
            'high': [],
            'medium': [],
            'low': [],
            'random': []
        }

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")

            # 为每个准确率条件创建环境，使用不同的seed以确保独立
            for idx, (condition_name, target_accuracy) in enumerate(Experiment1.ACCURACY_LEVELS.items()):
                # 方案9：简化种子策略，使用简单的线性偏移
                # 避免与错误分配逻辑冲突
                condition_seed = GLOBAL_SEED + rep * 10000 + idx * 100
                set_global_seed(condition_seed)

                # 使用不带随机事件的环境进行对比实验
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=50,
                    recognition_model=None, scaler=None,
                    seed=condition_seed,
                    event_probability=0.0  # 关闭随机事件，专注于识别准确率的影响
                )

                # 根据目标准确率设置UAV的识别类型（使用确定性方法）
                actual_accuracy = Experiment1._setup_recognition_with_accuracy_deterministic(
                    env, target_accuracy, condition_seed
                )

                env.recognition_updater = None
                algo = EnhancedHandoverAlgorithm(env)
                # 降低epsilon以减少随机探索对实验的干扰
                algo.epsilon = 0.00  # 完全禁用探索以获得确定性的基线结果

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    algo.run_step(enable_load_balancing=True)

                # 收集结果
                stats = env.get_state_statistics()
                stats.update(algo.get_detailed_stats())
                stats['actual_recognition_accuracy'] = actual_accuracy
                results_by_accuracy[condition_name].append(stats)

                print(f" {condition_name:8s} (目标{target_accuracy*100:3.0f}%, 实际{actual_accuracy*100:5.1f}%) - "
                      f"满足率: {stats['avg_satisfaction']:.3f}")

        # 汇总结果
        summary = Experiment1._summarize_results(results_by_accuracy)
        Experiment1._print_results_table(summary)
        Experiment1._plot(summary)
        return summary

    @staticmethod
    def _setup_recognition_with_accuracy(env, target_accuracy):
        """
        根据目标准确率设置UAV的识别类型（随机版本，用于其他实验）

        Returns:
            actual_accuracy: 实际达到的准确率
        """
        correct_count = 0
        total_count = len(env.uavs)

        for uav in env.uavs.values():
            true_type = uav.true_business_type

            if np.random.random() < target_accuracy:
                # 正确识别
                recognized_type = true_type
                correct_count += 1
            else:
                # 错误识别：随机选择其他类型
                other_types = [t for t in BusinessType if t != true_type]
                recognized_type = np.random.choice(other_types)

            # 设置识别结果和QoS配置
            uav.business_type = recognized_type
            uav.qos_profile = QOS_PROFILES[recognized_type]
            uav.recognition_confidence = 0.7 + np.random.random() * 0.25  # 0.7-0.95

        return correct_count / total_count if total_count > 0 else 0.0

    @staticmethod
    def _setup_recognition_with_accuracy_deterministic(env, target_accuracy, seed):
        """
        根据目标准确率设置UAV的识别类型（确定性版本，用于实验1）

        使用确定性方法确保在相同seed下产生相同的错误模式，
        这样可以准确对比不同准确率的影响。

        方案9改进：简化错误分配逻辑，避免种子冲突

        Args:
            env: 网络环境
            target_accuracy: 目标准确率
            seed: 随机种子（用于确保确定性）

        Returns:
            actual_accuracy: 实际达到的准确率
        """
        # 使用临时随机数生成器，避免影响主随机数流
        rng = np.random.RandomState(seed)

        correct_count = 0
        total_count = len(env.uavs)

        # 方案9：直接使用简单随机数判断，避免复杂逻辑
        for uav in env.uavs.values():
            true_type = uav.true_business_type

            if rng.random() < target_accuracy:
                # 正确识别
                recognized_type = true_type
                correct_count += 1
            else:
                # 错误识别：确定性选择其他类型（按轮询方式）
                other_types = [t for t in BusinessType if t != true_type]
                # 使用UAV ID的索引来选择错误类型，确保确定性
                error_index = (uav.uav_id + seed) % len(other_types)  # 使用种子避免冲突
                recognized_type = other_types[error_index]

            # 设置识别结果和QoS配置
            uav.business_type = recognized_type
            uav.qos_profile = QOS_PROFILES[recognized_type]
            # 确定性设置置信度
            uav.recognition_confidence = 0.825  # 固定在中间值

        return correct_count / total_count if total_count > 0 else 0.0

    @staticmethod
    def _summarize_results(results_by_accuracy):
        """汇总各准确率条件的实验结果"""
        def avg_std(key, results):
            values = [r[key] for r in results]
            return np.mean(values), np.std(values)

        summary = {}
        for condition_name in Experiment1.ACCURACY_LEVELS.keys():
            results = results_by_accuracy[condition_name]
            summary[condition_name] = {
                'satisfaction': avg_std('avg_satisfaction', results),
                'true_satisfaction': avg_std('avg_true_satisfaction', results),
                'resource_match': avg_std('resource_match_ratio', results),
                'actual_accuracy': avg_std('actual_recognition_accuracy', results),
                'handover_success': avg_std('handover_success_rate', results),
                'throughput': avg_std('total_load', results),
                'critical_sat': avg_std('critical_satisfaction', results),
                'weighted_sat': avg_std('weighted_satisfaction', results),
            }
        return summary

    @staticmethod
    def _print_results_table(summary):
        """打印实验结果表格"""
        condition_names = {
            'perfect': '100%准确率',
            'high': '90%准确率',
            'medium': '80%准确率',
            'low': '60%准确率',
            'random': '33%准确率'
        }

        headers = ["指标", "100%准确率", "90%准确率", "80%准确率", "60%准确率", "33%准确率"]

        # 计算相对于100%准确率的性能损失(基于真实满意率)
        perfect_sat = summary['perfect']['true_satisfaction'][0]

        rows = [
            ["真实满足率(基于真实需求)"] + [
                f"{summary[c]['true_satisfaction'][0]:.3f}±{summary[c]['true_satisfaction'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["性能损失(基于真实需求)"] + [
                f"-"
                if c == 'perfect' else
                (f"{(summary[c]['true_satisfaction'][0] - perfect_sat)*100:+.2f}%")
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["资源匹配度"] + [
                f"{summary[c]['resource_match'][0]:.3f}±{summary[c]['resource_match'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["关键业务满足率"] + [
                f"{summary[c]['critical_sat'][0]:.3f}±{summary[c]['critical_sat'][1]:.3f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["切换成功率"] + [
                f"{summary[c]['handover_success'][0]*100:.1f}%"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["系统吞吐量(Mbps)"] + [
                f"{summary[c]['throughput'][0]:.1f}±{summary[c]['throughput'][1]:.1f}"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
            ["实际识别准确率"] + [
                f"{summary[c]['actual_accuracy'][0]*100:.1f}%"
                for c in ['perfect', 'high', 'medium', 'low', 'random']
            ],
        ]

        print("\n" + "="*100)
        print("实验1结果：识别准确率对系统性能的影响")
        print("="*100)
        VisualizationHelper.print_data_table("识别准确性价值分析", headers, rows)

        # 打印关键结论
        print("\n【关键结论】")
        high_loss = (perfect_sat - summary['high']['true_satisfaction'][0]) * 100
        medium_loss = (perfect_sat - summary['medium']['true_satisfaction'][0]) * 100
        low_loss = (perfect_sat - summary['low']['true_satisfaction'][0]) * 100
        random_loss = (perfect_sat - summary['random']['true_satisfaction'][0]) * 100

        print(f"  - 识别准确率从100%降至90%，性能损失: {high_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至80%，性能损失: {medium_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至60%，性能损失: {low_loss:+.2f}%")
        print(f"  - 识别准确率从100%降至33%，性能损失: {random_loss:+.2f}%")

        # 验证单调性
        print(f"\n【数据一致性检查】")
        sat_values = [summary[c]['true_satisfaction'][0] for c in ['perfect', 'high', 'medium', 'low', 'random']]
        print(f"  真实满足率序列: {' -> '.join([f'{v:.3f}' for v in sat_values])}")
        if sat_values == sorted(sat_values, reverse=True):
            print(f"  [OK] 真实满足率随准确率降低而下降 (符合预期)")
        else:
            print(f"  [WARN] 真实满足率未随准确率单调下降 (可能存在异常)")

        if abs(high_loss) < 5 and abs(medium_loss) < 5:
            print(f"\n  - 结论: 90%和80%识别准确率性能接近，准确率影响较小")
        elif high_loss > 10 or medium_loss > 10:
            print(f"\n  - 结论: 识别准确率对系统性能影响显著，应优先提升模型精度")
        else:
            print(f"\n  - 结论: 识别准确率对性能有影响，但90%以上已达到可接受水平")
        print("="*100)

    @staticmethod
    def _plot(summary):
        """绘制实验结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('实验1：识别准确性的价值验证', fontsize=14, fontweight='bold')

        conditions = ['perfect', 'high', 'medium', 'low', 'random']
        labels = ['100%', '90%', '80%', '60%', '33%']
        colors = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
        
        # 图1: 真实满足率 vs 识别准确率
        ax = axes[0, 0]
        accuracies = [summary[c]['actual_accuracy'][0] * 100 for c in conditions]
        true_satisfactions = [summary[c]['true_satisfaction'][0] for c in conditions]
        ax.plot(accuracies, true_satisfactions, 'o-', color=COLORS['primary'],
                linewidth=2, markersize=10)
        for i, (acc, sat) in enumerate(zip(accuracies, true_satisfactions)):
            ax.annotate(labels[i], (acc, sat), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
        ax.set_xlabel('识别准确率 (%)', fontsize=11)
        ax.set_ylabel('真实满足率(基于真实需求)', fontsize=11)
        ax.set_title('识别准确率 vs 真实系统性能', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 图2: 各指标对比柱状图
        ax = axes[0, 1]
        x = np.arange(len(labels))
        width = 0.25
        true_sat_values = [summary[c]['true_satisfaction'][0] for c in conditions]
        crit_values = [summary[c]['critical_sat'][0] for c in conditions]
        res_match_values = [summary[c]['resource_match'][0] for c in conditions]
        bars1 = ax.bar(x - width, true_sat_values, width, label='真实满足率',
                       color=COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x, crit_values, width, label='关键业务满足率',
                       color=COLORS['danger'], alpha=0.8)
        bars3 = ax.bar(x + width, res_match_values, width, label='资源匹配度',
                       color=COLORS['success'], alpha=0.8)
        ax.set_ylabel('指标值', fontsize=11)
        ax.set_title('不同准确率下的性能指标对比', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 图3: 性能损失曲线
        ax = axes[1, 0]
        perfect_sat = summary['perfect']['true_satisfaction'][0]
        losses = [(perfect_sat - summary[c]['true_satisfaction'][0]) * 100
                  for c in conditions]
        bars = ax.bar(labels, losses, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_ylabel('性能损失 (%)', fontsize=11)
        ax.set_xlabel('识别准确率', fontsize=11)
        ax.set_title('相对于100%准确率的性能损失', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, loss in zip(bars, losses):
            if loss > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{loss:.2f}%', ha='center', va='bottom', fontsize=9)

        # 图4: 切换成功率对比
        ax = axes[1, 1]
        success_rates = [summary[c]['handover_success'][0] * 100 for c in conditions]
        bars = ax.bar(labels, success_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_ylabel('切换成功率 (%)', fontsize=11)
        ax.set_xlabel('识别准确率', fontsize=11)
        ax.set_title('不同准确率下的切换成功率', fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp1_results.png'), dpi=200, bbox_inches='tight')
        plt.show()
        
        return summary


# -------------------- 统计检验工具函数 --------------------

def perform_statistical_test(group1: List[float], group2: List[float],
                            test_name: str = 'ttest',
                            alpha: float = 0.05) -> Dict[str, Any]:
    """
    执行统计显著性检验

    Args:
        group1: 第一组数据
        group2: 第二组数据
        test_name: 检验方法 ('ttest', 'mannwhitney', 'wilcoxon')
        alpha: 显著性水平

    Returns:
        Dict: 包含检验结果,包括统计量、p值、是否显著等
    """
    # 计算描述性统计
    result = {
        'group1': {
            'mean': np.mean(group1),
            'std': np.std(group1),
            'count': len(group1),
            'median': np.median(group1)
        },
        'group2': {
            'mean': np.mean(group2),
            'std': np.std(group2),
            'count': len(group2),
            'median': np.median(group2)
        },
        'effect_size': None,
        'test_method': test_name,
        'alpha': alpha
    }

    # 执行统计检验
    if test_name == 'ttest':
        # 独立样本t检验 (假设正态分布)
        statistic, p_value = stats.ttest_ind(group1, group2)
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

        # 计算Cohen's d效应量
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        if pooled_std != 0:
            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
            result['effect_size'] = cohens_d
            result['effect_size_interpretation'] = _interpret_cohens_d(cohens_d)

    elif test_name == 'mannwhitney':
        # Mann-Whitney U检验 (非参数检验)
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

        # 计算秩和效应量
        n1, n2 = len(group1), len(group2)
        u = statistic
        z = (u - n1 * n2 / 2) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        r = z / np.sqrt(n1 + n2)
        result['effect_size'] = r
        result['effect_size_interpretation'] = _interpret_rank_biserial(r)

    elif test_name == 'wilcoxon':
        # Wilcoxon符号秩检验 (配对数据)
        if len(group1) != len(group2):
            raise ValueError("Wilcoxon检验需要两组数据长度相同")
        statistic, p_value = stats.wilcoxon(group1, group2)
        result['statistic'] = statistic
        result['p_value'] = p_value
        result['significant'] = p_value < alpha

    else:
        raise ValueError(f"未知的检验方法: {test_name}")

    return result


def _interpret_cohens_d(d: float) -> str:
    """解释Cohen's d效应量"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return '微小'
    elif abs_d < 0.5:
        return '小'
    elif abs_d < 0.8:
        return '中等'
    else:
        return '大'


def _interpret_rank_biserial(r: float) -> str:
    """解释秩相关效应量"""
    abs_r = abs(r)
    if abs_r < 0.1:
        return '微小'
    elif abs_r < 0.3:
        return '小'
    elif abs_r < 0.5:
        return '中等'
    else:
        return '大'


def print_statistical_results(results: Dict[str, Any], metric_name: str = 'Metric'):
    """打印统计检验结果"""
    print(f"\n{'='*60}")
    print(f"{metric_name} 统计显著性检验")
    print(f"{'='*60}")

    print(f"\n【描述性统计】")
    print(f"  组1 (n={results['group1']['count']}):  "
          f"均值={results['group1']['mean']:.4f}±{results['group1']['std']:.4f}, "
          f"中位数={results['group1']['median']:.4f}")
    print(f"  组2 (n={results['group2']['count']}):  "
          f"均值={results['group2']['mean']:.4f}±{results['group2']['std']:.4f}, "
          f"中位数={results['group2']['median']:.4f}")

    print(f"\n【统计检验结果】")
    print(f"  检验方法: {results['test_method']}")
    print(f"  统计量: {results['statistic']:.4f}")
    print(f"  p值: {results['p_value']:.6f}")
    print(f"  显著性水平: {results['alpha']}")
    print(f"  是否显著: {'是 ✓' if results['significant'] else '否 ✗'}")

    if results['effect_size'] is not None:
        print(f"  效应量: {results['effect_size']:.4f} ({results['effect_size_interpretation']})")

    # 打印结论
    print(f"\n【结论】")
    if results['significant']:
        if results['group1']['mean'] > results['group2']['mean']:
            direction = "组1显著高于组2"
        else:
            direction = "组1显著低于组2"
        print(f"  在α={results['alpha']}水平下,{direction} (p={results['p_value']:.4f})")
    else:
        print(f"  在α={results['alpha']}水平下,两组差异无统计学意义 (p={results['p_value']:.4f})")

    print(f"{'='*60}\n")


def compare_algorithms_with_tests(enhanced_results: List[Dict],
                                  traditional_results: List[Dict],
                                  metrics: List[str]) -> Dict[str, Dict]:
    """
    对增强算法和传统算法进行多指标的统计显著性检验

    Args:
        enhanced_results: 增强算法的多次运行结果
        traditional_results: 传统算法的多次运行结果
        metrics: 需要检验的指标列表

    Returns:
        Dict: 每个指标的检验结果
    """
    all_test_results = {}

    for metric in metrics:
        if metric in enhanced_results[0] and metric in traditional_results[0]:
            group1 = [r[metric] for r in enhanced_results]
            group2 = [r[metric] for r in traditional_results]

            # 自动选择检验方法
            # Shapiro-Wilk正态性检验
            _, p1 = stats.shapiro(group1)
            _, p2 = stats.shapiro(group2)

            if p1 > 0.05 and p2 > 0.05:
                # 都符合正态分布,使用t检验
                test_method = 'ttest'
            else:
                # 不符合正态分布,使用非参数检验
                test_method = 'mannwhitney'

            test_results = perform_statistical_test(group1, group2,
                                                     test_name=test_method,
                                                     alpha=0.05)
            all_test_results[metric] = test_results

    return all_test_results


def print_comprehensive_test_summary(all_test_results: Dict[str, Dict],
                                    enhanced_name: str = '增强算法',
                                    traditional_name: str = '传统算法'):
    """打印综合检验结果摘要"""
    print(f"\n{'='*80}")
    print(f"综合统计显著性检验摘要: {enhanced_name} vs {traditional_name}")
    print(f"{'='*80}")

    significant_count = 0
    total_count = 0

    for metric, results in all_test_results.items():
        total_count += 1
        print(f"\n【{metric}】")
        print(f"  {enhanced_name}: {results['group1']['mean']:.4f}±{results['group1']['std']:.4f}")
        print(f"  {traditional_name}: {results['group2']['mean']:.4f}±{results['group2']['std']:.4f}")
        print(f"  p值: {results['p_value']:.6f}")
        if results['significant']:
            significant_count += 1
            direction = "↑" if results['group1']['mean'] > results['group2']['mean'] else "↓"
            print(f"  结论: 显著差异 {direction} (效应量={results.get('effect_size', 0):.4f})")
        else:
            print(f"  结论: 无显著差异")

    print(f"\n{'='*80}")
    print(f"总结: {significant_count}/{total_count} 个指标具有显著差异")
    print(f"{'='*80}\n")


# -------------------- 实验3：增强算法 vs 传统算法（全面对比）--------------------
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
    def run(recognition_model, scaler, num_steps=200, repeats=10):  # 增加到10次重复
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

        # 添加统计显著性检验
        print("\n" + "="*80)
        print("统计显著性检验")
        print("="*80)

        metrics_to_test = [
            'avg_satisfaction',
            'handover_success_rate',
            'critical_satisfaction',
            'avg_switching_latency_ms',
            'avg_decision_time_ms',
            'total_load',
            'load_variance'
        ]

        all_test_results = compare_algorithms_with_tests(
            enhanced_results, traditional_results, metrics_to_test
        )

        print_comprehensive_test_summary(all_test_results, "增强算法", "传统算法")

        # 将检验结果添加到summary中
        summary['statistical_tests'] = all_test_results

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


# -------------------- 实验2：机制有效性验证 --------------------
class Experiment2:
    """
    实验2：机制有效性验证

    采用逐步添加机制的方式，从传统算法开始，依次添加各个增强机制，
    验证每个机制对系统性能的贡献，为增强算法的设计提供理论依据。
    """
    MECHANISMS = {
        'traditional': '传统算法（基线）',
        'add_dynamic_threshold': '传统+动态阈值',
        'add_business_weights': '传统+动态阈值+业务权重',
        'add_epsilon_greedy': '传统+动态阈值+业务权重+ε-greedy',
        'add_load_balance': '传统+动态阈值+业务权重+ε-greedy+负载均衡',
        'add_adaptive_recognition': '传统+动态阈值+业务权重+ε-greedy+负载均衡+自适应识别',
        'full': '完整增强算法'
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=10):  # 增加到10次重复
        print("\n" + "="*80)
        print("实验2：机制有效性验证（逐步添加机制）")
        print("="*80)
        print("\n实验目的：从传统算法开始，逐步添加增强机制，验证各机制的独立贡献")
        print("\n机制添加顺序：")
        print("  1. 传统算法（基线）")
        print("  2. 添加动态阈值")
        print("  3. 添加业务特化权重")
        print("  4. 添加ε-greedy探索")
        print("  5. 添加负载均衡")
        print("  6. 添加自适应识别更新")
        print("  7. 完整增强算法（验证）")
        print("="*80)

        results = {key: [] for key in Experiment2.MECHANISMS.keys()}
        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)
            for mechanism in Experiment2.MECHANISMS.keys():
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=50,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, event_probability=0.05
                )

                # 根据机制配置创建对应的算法
                if mechanism == 'traditional':
                    algo = IntegratedHandoverAlgorithm(env)
                else:
                    algo = EnhancedHandoverAlgorithm(env)

                    # 根据机制配置启用对应的功能
                    if mechanism == 'add_dynamic_threshold':
                        # 只启用动态阈值，禁用其他增强功能
                        algo.base_threshold = 0.005
                        algo.calculate_dynamic_threshold = algo.__class__.calculate_dynamic_threshold.__get__(algo, type(algo))
                        # 禁用业务权重
                        for bt in BusinessType:
                            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}
                        # 禁用ε-greedy
                        algo.epsilon = 0.0
                        # 禁用负载均衡（在运行时控制）
                        enable_lb = False

                    elif mechanism == 'add_business_weights':
                        # 启用动态阈值和业务权重，禁用其他
                        algo.epsilon = 0.0
                        enable_lb = False

                    elif mechanism == 'add_epsilon_greedy':
                        # 启用动态阈值、业务权重和ε-greedy
                        # epsilon默认为0.05，无需修改
                        enable_lb = False

                    elif mechanism == 'add_load_balance':
                        # 启用动态阈值、业务权重、ε-greedy和负载均衡
                        enable_lb = True

                    elif mechanism == 'add_adaptive_recognition':
                        # 启用所有机制，包括自适应识别更新
                        # 自适应识别由环境控制，无需额外配置
                        enable_lb = True

                    elif mechanism == 'full':
                        # 完整增强算法
                        enable_lb = True

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    if mechanism == 'traditional':
                        algo.run_step()
                    elif mechanism in ['add_dynamic_threshold', 'add_business_weights',
                                      'add_epsilon_greedy', 'add_load_balance',
                                      'add_adaptive_recognition', 'full']:
                        algo.run_step(enable_load_balancing=enable_lb)

                stats = env.get_state_statistics()
                if hasattr(algo, 'get_detailed_stats'):
                    stats.update(algo.get_detailed_stats())
                results[mechanism].append(stats)
                print(f" {Experiment2.MECHANISMS[mechanism]}: "
                      f"满足率={stats['avg_satisfaction']:.3f}, "
                      f"切换成功率={stats.get('handover_success_rate',0)*100:.1f}%")

        summary = Experiment2._summarize(results)
        Experiment2._print_results_table(summary)
        Experiment2._plot(summary)
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
        for mechanism, name in Experiment2.MECHANISMS.items():
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
        VisualizationHelper.print_data_table("实验2结果：机制有效性验证", headers, rows)

        # 逐步添加机制的贡献分析
        print("\n【逐步添加机制贡献分析】")
        mechanism_order = ['traditional', 'add_dynamic_threshold', 'add_business_weights',
                          'add_epsilon_greedy', 'add_load_balance', 'add_adaptive_recognition', 'full']

        prev_mechanism = mechanism_order[0]
        prev_sat = summary[prev_mechanism]['avg_satisfaction'][0] if prev_mechanism in summary else 0

        print(f"\n相对于传统算法的逐步提升:")
        for mechanism in mechanism_order[1:]:
            if mechanism in summary:
                curr_sat = summary[mechanism]['avg_satisfaction'][0]
                contribution = curr_sat - prev_sat
                contribution_pct = contribution / prev_sat * 100 if prev_sat > 0 else 0

                mechanism_name = Experiment2.MECHANISMS[mechanism]

                # 提取新增的机制名称
                if mechanism == 'add_dynamic_threshold':
                    added_name = "动态阈值"
                elif mechanism == 'add_business_weights':
                    added_name = "业务权重"
                elif mechanism == 'add_epsilon_greedy':
                    added_name = "ε-greedy探索"
                elif mechanism == 'add_load_balance':
                    added_name = "负载均衡"
                elif mechanism == 'add_adaptive_recognition':
                    added_name = "自适应识别更新"
                elif mechanism == 'full':
                    added_name = "完整算法(验证)"
                else:
                    added_name = mechanism_name

                print(f"  {added_name}: 贡献 = {contribution:+.4f} ({contribution_pct:+.1f}%) "
                      f"[{prev_sat:.4f} -> {curr_sat:.4f}]")

                prev_sat = curr_sat

        # 总体提升对比
        if 'traditional' in summary and 'full' in summary:
            trad_sat = summary['traditional']['avg_satisfaction'][0]
            full_sat = summary['full']['avg_satisfaction'][0]
            total_improvement = (full_sat - trad_sat) / trad_sat * 100
            print(f"\n总体提升: 传统算法 -> 完整增强算法 = {total_improvement:+.1f}%")

    @staticmethod
    def _plot(summary):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实验2：机制有效性验证（逐步添加）', fontsize=14, fontweight='bold')
        mechanism_order = ['traditional', 'add_dynamic_threshold', 'add_business_weights',
                          'add_epsilon_greedy', 'add_load_balance', 'add_adaptive_recognition', 'full']
        mechanisms = mechanism_order
        names = [Experiment2.MECHANISMS[m] for m in mechanisms]

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

        # 逐步提升曲线图
        ax = axes[1,0]
        if all(m in summary for m in mechanism_order):
            sats = [summary[m]['avg_satisfaction'][0] for m in mechanism_order]
            stds = [summary[m]['avg_satisfaction'][1] for m in mechanism_order]
            x = range(len(mechanism_order))
            short_names = ['传统', '+动态\n阈值', '+业务\n权重', '+ε-\ngreedy', '+负载\n均衡', '+自适应\n识别', '完整']
            ax.plot(x, sats, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
            ax.fill_between(x, [s-std for s,std in zip(sats, stds)],
                           [s+std for s,std in zip(sats, stds)],
                           alpha=0.2, color=COLORS['primary'])
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=9)
            ax.set_ylabel('整体满足率')
            ax.set_title('逐步添加机制的性能提升曲线', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # 标注每个阶段的提升
            for i in range(1, len(sats)):
                improvement = sats[i] - sats[i-1]
                if abs(improvement) > 0.005:  # 只标注显著提升
                    mid_x = (i-1 + i) / 2
                    mid_y = (sats[i-1] + sats[i]) / 2
                    ax.annotate(f'{improvement:+.4f}',
                               xy=(mid_x, mid_y),
                               xytext=(mid_x, mid_y + improvement*0.5),
                               ha='center', va='center',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', lw=0.5))

        # 逐步提升柱状图
        ax = axes[1,1]
        if all(m in summary for m in mechanism_order):
            improvements = []
            contrib_names = []
            for i in range(1, len(mechanism_order)):
                if mechanism_order[i-1] in summary and mechanism_order[i] in summary:
                    prev_sat = summary[mechanism_order[i-1]]['avg_satisfaction'][0]
                    curr_sat = summary[mechanism_order[i]]['avg_satisfaction'][0]
                    improvement = curr_sat - prev_sat
                    improvements.append(improvement)

                    # 提取机制名称
                    if mechanism_order[i] == 'add_dynamic_threshold':
                        name = "动态阈值"
                    elif mechanism_order[i] == 'add_business_weights':
                        name = "业务权重"
                    elif mechanism_order[i] == 'add_epsilon_greedy':
                        name = "ε-greedy"
                    elif mechanism_order[i] == 'add_load_balance':
                        name = "负载均衡"
                    elif mechanism_order[i] == 'add_adaptive_recognition':
                        name = "自适应识别"
                    elif mechanism_order[i] == 'full':
                        name = "完整验证"
                    else:
                        name = mechanism_order[i]
                    contrib_names.append(name)

            colors = [COLORS['success'] if imp > 0 else COLORS['danger'] for imp in improvements]
            bars = ax.bar(contrib_names, improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_ylabel('满足率提升')
            ax.set_title('各机制的独立贡献', fontweight='bold')
            ax.set_xticklabels(contrib_names, rotation=30, ha='right')
            for bar, imp in zip(bars, improvements):
                ax.text(bar.get_x() + bar.get_width()/2, imp,
                       f'{imp:+.4f}',
                       ha='center', va='bottom' if imp > 0 else 'top',
                       fontsize=9, fontweight='bold')

        # 综合评分对比
        ax = axes[1,2]
        scores = []
        score_names = []
        for mechanism in mechanisms:
            if mechanism in summary:
                score = (0.4 * summary[mechanism]['avg_satisfaction'][0] +
                         0.3 * summary[mechanism]['handover_success_rate'][0] +
                         0.3 * summary[mechanism]['critical_satisfaction'][0])
                scores.append(score)
                score_names.append(mechanism)
        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        bars = ax.barh([Experiment2.MECHANISMS[m] for m in score_names], scores,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('综合评分')
        ax.set_title('各配置的综合评分', fontweight='bold')
        for bar, val in zip(bars, scores):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                   ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'exp2_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验2b：机制组合验证 --------------------
class Experiment2b:
    """
    实验2b：机制组合验证

    测试不同机制组合的性能，验证机制间的交互效应，
    找到在当前规模下的最优配置。

    设计原因：
    - 实验2发现ε-greedy在"逐步添加"时导致性能下降(-4.7%)
    - 但在"禁用"时却是正面的(+4.8%)
    - 需要通过组合验证找出机制间的交互效应
    """
    COMBINATIONS = {
        'traditional': '传统算法',
        'dyn_thresh': '+动态阈值',
        'weights': '+业务权重',
        'epsilon': '+ε-greedy',
        'load_balance': '+负载均衡',
        'dyn_thresh_weights': '动态阈值+业务权重',
        'dyn_thresh_epsilon': '动态阈值+ε-greedy',
        'weights_epsilon': '业务权重+ε-greedy',
        'dyn_thresh_weights_epsilon': '动态阈值+业务权重+ε-greedy',
        'dyn_thresh_weights_lb': '动态阈值+业务权重+负载均衡',
        'dyn_thresh_epsilon_lb': '动态阈值+ε-greedy+负载均衡',
        'weights_epsilon_lb': '业务权重+ε-greedy+负载均衡',
        'dyn_thresh_weights_epsilon_lb': '动态阈值+业务权重+ε-greedy+负载均衡',
        'full': '完整增强算法(含自适应识别)',
    }

    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=8):
        print("\n" + "="*80)
        print("实验2b：机制组合验证")
        print("="*80)
        print("\n实验目的：验证不同机制组合的性能，找出机制间的交互效应")
        print("\n测试组合：")
        print("  - 单机制：动态阈值、业务权重、ε-greedy、负载均衡")
        print("  - 双机制组合：各两两组合")
        print("  - 三机制组合：关键三机制")
        print("  - 四机制组合：所有核心机制")
        print("  - 完整算法：包含自适应识别")
        print("="*80)

        results = {key: [] for key in Experiment2b.COMBINATIONS.keys()}

        for rep in range(repeats):
            print(f"\n--- 重复 {rep+1}/{repeats} ---")
            set_global_seed(GLOBAL_SEED + rep)

            for combo_name in Experiment2b.COMBINATIONS.keys():
                env = EnhancedNetworkEnvironment(
                    num_bs=8, num_uav=50,
                    recognition_model=recognition_model, scaler=scaler,
                    seed=GLOBAL_SEED + rep, event_probability=0.05
                )

                # 根据组合配置创建算法
                if combo_name == 'traditional':
                    algo = IntegratedHandoverAlgorithm(env)
                    enable_lb = False
                else:
                    algo = EnhancedHandoverAlgorithm(env)

                    # 解析组合配置
                    has_dyn_thresh = 'dyn_thresh' in combo_name
                    has_weights = 'weights' in combo_name
                    has_epsilon = 'epsilon' in combo_name
                    has_lb = 'load_balance' in combo_name or 'lb' in combo_name

                    # 配置各机制
                    if not has_dyn_thresh:
                        # 禁用动态阈值
                        algo.base_threshold = 0.005
                        algo.calculate_dynamic_threshold = lambda uav: 0.005

                    if not has_weights:
                        # 禁用业务权重
                        for bt in BusinessType:
                            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}

                    if not has_epsilon:
                        # 禁用ε-greedy
                        algo.epsilon = 0.0

                    enable_lb = has_lb

                    # 完整算法额外启用自适应识别
                    if combo_name == 'full':
                        # 自适应识别由环境控制，无需额外配置
                        pass

                # 运行仿真
                for step in range(num_steps):
                    env.step()
                    if combo_name == 'traditional':
                        algo.run_step()
                    else:
                        algo.run_step(enable_load_balancing=enable_lb)

                stats = env.get_state_statistics()
                if hasattr(algo, 'get_detailed_stats'):
                    stats.update(algo.get_detailed_stats())
                results[combo_name].append(stats)

                if rep == 0:  # 只在第一次重复时打印，避免输出过多
                    print(f" {Experiment2b.COMBINATIONS[combo_name]:40s}: "
                          f"满足率={stats['avg_satisfaction']:.4f}")

        summary = Experiment2b._summarize(results)
        Experiment2b._print_results_table(summary)
        Experiment2b._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for combo_name, data_list in results.items():
            summary[combo_name] = {}
            for key in ['avg_satisfaction', 'handover_success_rate', 'critical_satisfaction',
                        'weighted_satisfaction', 'total_load', 'load_variance']:
                if key in data_list[0]:
                    vals = [d[key] for d in data_list]
                    summary[combo_name][key] = (np.mean(vals), np.std(vals))
        return summary

    @staticmethod
    def _print_results_table(summary):
        # 按性能排序
        sorted_combos = sorted(summary.keys(),
                           key=lambda x: summary[x]['avg_satisfaction'][0],
                           reverse=True)

        print("\n" + "="*100)
        print("【按性能排序的机制组合】")
        print("="*100)

        headers = ["排名", "组合名称", "整体满足率", "切换成功率", "关键业务满足率", "提升(相对传统)"]
        rows = []

        trad_sat = summary.get('traditional', {}).get('avg_satisfaction', (0, 0))[0]

        for rank, combo in enumerate(sorted_combos, 1):
            data = summary[combo]
            sat_mean, sat_std = data['avg_satisfaction']
            success_mean, success_std = data['handover_success_rate']
            crit_mean, crit_std = data['critical_satisfaction']
            improvement = ((sat_mean - trad_sat) / trad_sat * 100) if trad_sat > 0 else 0

            row = [
                f"{rank}",
                Experiment2b.COMBINATIONS[combo],
                f"{sat_mean:.4f}±{sat_std:.4f}",
                f"{success_mean*100:.2f}%",
                f"{crit_mean:.4f}±{crit_std:.4f}",
                f"{improvement:+.2f}%"
            ]
            rows.append(row)

        VisualizationHelper.print_data_table("实验2b结果：机制组合性能对比", headers, rows)

        # 分析机制贡献
        print("\n【机制贡献分析】")
        print("\n单机制贡献（相对于传统算法）:")
        for mechanism in ['dyn_thresh', 'weights', 'epsilon', 'load_balance']:
            if mechanism in summary and 'traditional' in summary:
                trad_sat = summary['traditional']['avg_satisfaction'][0]
                mech_sat = summary[mechanism]['avg_satisfaction'][0]
                contribution = mech_sat - trad_sat
                pct = contribution / trad_sat * 100 if trad_sat > 0 else 0
                print(f"  {Experiment2b.COMBINATIONS[mechanism]:20s}: "
                      f"{contribution:+.4f} ({pct:+.2f}%)")

        print("\n关键发现:")
        best_combo = sorted_combos[0]
        best_sat = summary[best_combo]['avg_satisfaction'][0]
        full_sat = summary['full']['avg_satisfaction'][0] if 'full' in summary else 0

        print(f"  1. 最优组合: {Experiment2b.COMBINATIONS[best_combo]} (满足率={best_sat:.4f})")
        print(f"  2. 相对传统算法提升: {((best_sat-trad_sat)/trad_sat*100):.2f}%")
        if 'full' in summary and best_combo != 'full':
            print(f"  3. 相对完整算法差异: {(best_sat-full_sat):.4f}")
            if best_sat > full_sat:
                print(f"     [发现] 存在比完整算法更优的组合!")
            else:
                print(f"     [验证] 完整算法接近最优组合")

        # 分析ε-greedy的作用
        print("\n【ε-greedy机制分析】")
        epsilon_present = ['dyn_thresh_epsilon', 'weights_epsilon', 'dyn_thresh_weights_epsilon',
                        'dyn_thresh_epsilon_lb', 'weights_epsilon_lb', 'dyn_thresh_weights_epsilon_lb', 'full']
        epsilon_absent = ['dyn_thresh', 'weights', 'dyn_thresh_weights', 'dyn_thresh_weights_lb']

        avg_with_epsilon = np.mean([summary[c]['avg_satisfaction'][0]
                                   for c in epsilon_present if c in summary])
        avg_without_epsilon = np.mean([summary[c]['avg_satisfaction'][0]
                                      for c in epsilon_absent if c in summary])

        print(f"  含ε-greedy的平均满足率: {avg_with_epsilon:.4f}")
        print(f"  不含ε-greedy的平均满足率: {avg_without_epsilon:.4f}")
        print(f"  差异: {avg_with_epsilon-avg_without_epsilon:+.4f}")

        if avg_with_epsilon < avg_without_epsilon:
            print(f"  结论: ε-greedy在当前规模下总体起负面作用")
        else:
            print(f"  结论: ε-greedy在当前规模下总体起正面作用")

        print("="*100)

    @staticmethod
    def _plot(summary):
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle('实验2b：机制组合验证', fontsize=16, fontweight='bold')
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. 按性能排名的柱状图
        ax = fig.add_subplot(gs[0, :2])
        sorted_combos = sorted(summary.keys(),
                           key=lambda x: summary[x]['avg_satisfaction'][0],
                           reverse=True)
        sats = [summary[c]['avg_satisfaction'][0] for c in sorted_combos]
        names = [Experiment2b.COMBINATIONS[c] for c in sorted_combos]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sats)))
        bars = ax.barh(names, sats, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('整体满足率')
        ax.set_title('各机制组合性能排名', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, sats):
            ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                   ha='left', va='center', fontsize=9)

        # 2. 机制组合热力图 (单机制)
        ax = fig.add_subplot(gs[0, 2])
        mechanisms = ['传统', '动态\n阈值', '业务\n权重', 'ε-\ngreedy', '负载\n均衡']
        combos = ['traditional', 'dyn_thresh', 'weights', 'epsilon', 'load_balance']
        values = [summary[c]['avg_satisfaction'][0] if c in summary else 0 for c in combos]
        im = ax.imshow([values], cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=0.9)
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms, fontsize=9)
        ax.set_yticks([])
        ax.set_title('单机制性能热力图', fontweight='bold')
        for i, val in enumerate(values):
            ax.text(i, 0, f'{val:.4f}', ha='center', va='center',
                   fontsize=10, fontweight='bold',
                   color='black' if val < 0.75 else 'white')
        plt.colorbar(im, ax=ax)

        # 3. 双机制组合矩阵
        ax = fig.add_subplot(gs[1, :2])
        mech_list = ['dyn_thresh', 'weights', 'epsilon', 'load_balance']
        mech_names = ['动态阈值', '业务权重', 'ε-greedy', '负载均衡']

        # 构建矩阵
        matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                combo_name = f"{mech_list[i]}_{mech_list[j]}"
                if combo_name in summary:
                    matrix[i, j] = summary[combo_name]['avg_satisfaction'][0]
                elif i == j:
                    # 对角线是单机制
                    matrix[i, j] = summary[mech_list[i]]['avg_satisfaction'][0]

        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.7, vmax=0.9, aspect='auto')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(mech_names)
        ax.set_yticklabels(mech_names)
        ax.set_title('双机制组合性能矩阵', fontweight='bold')
        for i in range(4):
            for j in range(i, 4):
                val = matrix[i, j]
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='black' if val < 0.78 else 'white')
        plt.colorbar(im, ax=ax)

        # 4. ε-greedy作用对比
        ax = fig.add_subplot(gs[1, 2])
        no_epsilon_combos = ['dyn_thresh', 'weights', 'dyn_thresh_weights', 'dyn_thresh_weights_lb']
        with_epsilon_combos = ['dyn_thresh_epsilon', 'weights_epsilon',
                             'dyn_thresh_weights_epsilon', 'dyn_thresh_weights_epsilon_lb']

        no_epsilon_sats = [summary[c]['avg_satisfaction'][0] for c in no_epsilon_combos if c in summary]
        with_epsilon_sats = [summary[c]['avg_satisfaction'][0] for c in with_epsilon_combos if c in summary]

        x = range(len(no_epsilon_sats))
        width = 0.35

        ax.bar([i-width/2 for i in x], no_epsilon_sats, width,
               label='不含ε-greedy', color=COLORS['primary'], alpha=0.8)
        ax.bar([i+width/2 for i in x], with_epsilon_sats, width,
               label='含ε-greedy', color=COLORS['neutral'], alpha=0.8)

        labels = ['动态阈值', '业务权重', '动态+业务',
                 '动态+业务+负载']
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.set_ylabel('整体满足率')
        ax.set_title('ε-greedy作用对比', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 5. 负载均衡作用对比
        ax = fig.add_subplot(gs[2, 0])
        no_lb_combos = ['dyn_thresh', 'weights', 'dyn_thresh_weights',
                       'dyn_thresh_epsilon', 'weights_epsilon', 'dyn_thresh_weights_epsilon']
        with_lb_combos = ['dyn_thresh_weights_lb', 'dyn_thresh_epsilon_lb',
                         'weights_epsilon_lb', 'dyn_thresh_weights_epsilon_lb']

        no_lb_sats = [summary[c]['avg_satisfaction'][0] for c in no_lb_combos if c in summary]
        with_lb_sats = [summary[c]['avg_satisfaction'][0] for c in with_lb_combos if c in summary]

        # 计算平均提升
        no_lb_avg = np.mean(no_lb_sats)
        with_lb_avg = np.mean(with_lb_sats)
        lb_improvement = (with_lb_avg - no_lb_avg) / no_lb_avg * 100 if no_lb_avg > 0 else 0

        ax.bar(['无负载均衡', '有负载均衡'], [no_lb_avg, with_lb_avg],
               color=[COLORS['neutral'], COLORS['primary']], alpha=0.8)
        ax.set_ylabel('平均满足率')
        ax.set_title(f'负载均衡作用 (提升{lb_improvement:+.2f}%)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. 关键指标散点图
        ax = fig.add_subplot(gs[2, 1])
        all_combos = list(summary.keys())
        x_vals = [summary[c]['handover_success_rate'][0]*100 for c in all_combos]
        y_vals = [summary[c]['avg_satisfaction'][0] for c in all_combos]
        colors = plt.cm.RdYlGn(y_vals)

        ax.scatter(x_vals, y_vals, s=100, c=colors, alpha=0.7, edgecolors='white', linewidth=1.5)
        ax.set_xlabel('切换成功率 (%)')
        ax.set_ylabel('整体满足率')
        ax.set_title('切换成功率 vs 整体满足率', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 标注最优组合
        best_idx = np.argmax(y_vals)
        ax.scatter([x_vals[best_idx]], [y_vals[best_idx]],
                  s=200, c='gold', edgecolors='red', linewidth=2, marker='*',
                  label=f"最优: {Experiment2b.COMBINATIONS[all_combos[best_idx]]}")
        ax.legend(fontsize=8)

        # 7. 关键文本摘要
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')

        best_combo = sorted_combos[0]
        best_sat = summary[best_combo]['avg_satisfaction'][0]
        trad_sat = summary['traditional']['avg_satisfaction'][0]
        full_sat = summary['full']['avg_satisfaction'][0]

        text = f"【实验2b关键发现】\n\n"
        text += f"最优组合: {Experiment2b.COMBINATIONS[best_combo]}\n"
        text += f"满足率: {best_sat:.4f}\n\n"
        text += f"相对传统算法: {((best_sat-trad_sat)/trad_sat*100):.2f}% 提升\n"
        text += f"相对完整算法: {((best_sat-full_sat)/full_sat*100):+.2f}% 差异\n\n"

        # 找出最优组合包含的机制
        best_mechs = []
        if 'dyn_thresh' in best_combo or 'thresh' in best_combo:
            best_mechs.append('动态阈值')
        if 'weights' in best_combo:
            best_mechs.append('业务权重')
        if 'epsilon' in best_combo:
            best_mechs.append('ε-greedy')
        if 'lb' in best_combo or 'load_balance' in best_combo:
            best_mechs.append('负载均衡')

        text += f"核心机制: {', '.join(best_mechs)}\n\n"

        # ε-greedy结论
        if 'epsilon' not in best_combo:
            text += f"[结论] 最优组合不含ε-greedy\n"
            text += f"      说明该机制在当前规模下不适用\n"
        else:
            text += f"[结论] 最优组合含ε-greedy\n"

        # 负载均衡结论
        if ('lb' not in best_combo and 'load_balance' not in best_combo):
            text += f"[结论] 最优组合不含负载均衡\n"
            text += f"      说明该机制贡献有限\n"
        else:
            text += f"[结论] 最优组合含负载均衡\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.savefig(os.path.join(RESULT_DIR, 'exp2b_results.png'), dpi=200, bbox_inches='tight')
        plt.show()


# -------------------- 实验4 --------------------
class Experiment4:
    SCENARIOS = {
        'default': {'name': '默认场景', 'desc': '标准仿真环境'},
        'urban': {'name': '城市物流', 'desc': '密集部署，障碍物多，时延敏感'},
        'emergency': {'name': '应急救援', 'desc': '高容量需求，低时延容忍，重视视频回传'},
        'agriculture': {'name': '农田监测', 'desc': '稀疏部署，大范围覆盖，周期性数据'}
    }


    @staticmethod
    def run(recognition_model, scaler, num_steps=150, repeats=10):  # 增加到10次重复
        print("\n" + "="*80)
        print("实验4：多场景对比实验")
        print("="*80)

        results = {scenario: {'enhanced': [], 'traditional': []} for scenario in Experiment4.SCENARIOS.keys()}
        for scenario, info in Experiment4.SCENARIOS.items():
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

        summary = Experiment4._summarize(results)
        Experiment4._print_results_table(summary)
        Experiment4._plot(summary)
        return summary

    @staticmethod
    def _summarize(results):
        summary = {}
        for scenario in Experiment4.SCENARIOS.keys():
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
        for scenario, info in Experiment4.SCENARIOS.items():
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
        fig.suptitle('实验4：多场景对比实验', fontsize=14, fontweight='bold')
        scenarios = list(Experiment4.SCENARIOS.keys())
        scenario_names = [Experiment4.SCENARIOS[s]['name'] for s in scenarios]
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
        plt.savefig(os.path.join(RESULT_DIR, 'exp4_results.png'), dpi=200, bbox_inches='tight')
        plt.show()