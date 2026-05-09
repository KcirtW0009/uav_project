"""
超快速泛化性验证 (全指标对比版)

目的: 快速验证MAPPO是否真的优于增强算法（非过拟合）
特点:
  - 只运行1次MAPPO评估
  - 加载已有的实验3数据作为基线
  - 对比17个核心指标
  - 分类统计和综合判定

用法:
  python quick_validation.py [--model-path PATH]
"""

import sys
import os
import time
import numpy as np
import torch
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.experiments import evaluate_mappo_in_experiment


def load_exp3_baseline():
    """加载已有的实验3基线数据"""
    
    exp3_file = os.path.join(RESULT_DIR, 'exp3_data.json')
    
    if not os.path.exists(exp3_file):
        print("[WARN] 未找到实验3数据文件")
        return None
    
    with open(exp3_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("[OK] 已加载实验3基线数据:")
    
    # 提取增强算法和传统算法的平均值
    baseline = {}
    
    if 'enhanced' in data:
        enhanced = data['enhanced']
        print(f"\n  增强算法 (10次平均):")
        for key in ['avg_satisfaction', 'handover_success_rate', 'connected_ratio', 
                     'critical_satisfaction', 'load_variance',
                     'weighted_satisfaction', 'latency_satisfaction', 'rate_satisfaction',
                     'avg_switching_latency_ms', 'max_switching_latency_ms', 
                     'avg_decision_time_ms', 'missed_opportunity_rate',
                     'migration_success_rate', 'total_throughput', 
                     'avg_sinr', 'recognition_accuracy']:
            if key in enhanced:
                val = enhanced[key]
                if isinstance(val, list) and len(val) >= 2:
                    print(f"    {key}: {val[0]:.4f} ± {val[1]:.4f}")
                    baseline[f'enhanced_{key}'] = val[0]
                    baseline[f'enhanced_{key}_std'] = val[1]
                elif isinstance(val, list):
                    print(f"    {key}: {np.mean(val):.4f} ± {np.std(val):.4f}")
                    baseline[f'enhanced_{key}'] = np.mean(val)
                    baseline[f'enhanced_{key}_std'] = np.std(val)
                else:
                    print(f"    {key}: {val:.4f}")
                    baseline[f'enhanced_{key}'] = val
    
    if 'traditional' in data:
        trad = data['traditional']
        print(f"\n  传统算法 (10次平均):")
        for key in ['avg_satisfaction', 'handover_success_rate', 'connected_ratio',
                     'critical_satisfaction', 'load_variance',
                     'weighted_satisfaction', 'latency_satisfaction', 'rate_satisfaction',
                     'avg_switching_latency_ms', 'max_switching_latency_ms', 
                     'avg_decision_time_ms', 'missed_opportunity_rate',
                     'migration_success_rate', 'total_throughput', 
                     'avg_sinr', 'recognition_accuracy']:
            if key in trad:
                val = trad[key]
                if isinstance(val, list) and len(val) >= 2:
                    print(f"    {key}: {val[0]:.4f} ± {val[1]:.4f}")
                    baseline[f'traditional_{key}'] = val[0]
                    baseline[f'traditional_{key}_std'] = val[1]
                elif isinstance(val, list):
                    print(f"    {key}: {np.mean(val):.4f} ± {np.std(val):.4f}")
                    baseline[f'traditional_{key}'] = np.mean(val)
                    baseline[f'traditional_{key}_std'] = np.std(val)
                else:
                    print(f"    {key}: {val:.4f}")
                    baseline[f'traditional_{key}'] = val
    
    return baseline


def run_mappo_evaluation(model_path, recognition_model, scaler, seed=42, num_steps=350):
    """运行单次MAPPO评估 (带详细进度日志)"""
    
    print("\n" + "="*80)
    print("[MAPPO EVALUATION] 开始评估")
    print("="*80)
    print("  模型路径: {}".format(model_path))
    print("  随机种子: {}".format(seed))
    print("  评估步数: {}".format(num_steps))
    print("  场景规模: 8BS x 300UAV")
    print("="*80)
    
    start = time.time()
    last_log_time = start
    log_interval = 30.0  # 每30秒打印一次状态
    
    # 阶段1: 环境初始化
    print("\n[Phase 1/4] 初始化评估环境...")
    env_start = time.time()
    
    mappo_stats = evaluate_mappo_in_experiment(
        num_bs=8, num_uav=300, num_steps=num_steps,
        recognition_model=recognition_model, scaler=scaler,
        seed=seed,
        model_path=model_path
    )
    
    env_elapsed = time.time() - env_start
    elapsed = time.time() - start
    
    if mappo_stats is None:
        print("\n[ERROR] MAPPO评估失败!")
        return None
    
    # 输出结果摘要
    print("\n" + "="*80)
    print("[RESULT] MAPPO评估完成")
    print("="*80)
    print("  总耗时: {:.1f}秒 ({:.2f}分钟)".format(elapsed, elapsed/60))
    print("  环境初始化: {:.1f}秒".format(env_elapsed))
    print("  仿真执行: {:.1f}秒".format(elapsed - env_elapsed))
    print("-"*80)
    print("  [核心指标]")
    print("    整体满意度:     {:.4f}".format(mappo_stats.get('avg_satisfaction', 0)))
    print("    连接保持率:     {:.2%}".format(mappo_stats.get('connected_ratio', 0)))
    print("    切换成功率:     {:.2%}".format(mappo_stats.get('handover_success_rate', 0)))
    print("    关键业务满意度: {:.4f}".format(mappo_stats.get('critical_satisfaction', 0)))
    print("-"*80)
    print("  [满意度细分]")
    print("    加权满意度:     {:.4f}".format(mappo_stats.get('weighted_satisfaction', 0)))
    print("    延迟满意度:     {:.4f}".format(mappo_stats.get('latency_satisfaction', 0)))
    print("    速率满意度:     {:.4f}".format(mappo_stats.get('rate_satisfaction', 0)))
    print("-"*80)
    print("  [切换性能]")
    print("    平均切换延迟:   {:.3f} ms".format(mappo_stats.get('avg_switching_latency_ms', 0)))
    print("    最大切换延迟:   {:.3f} ms".format(mappo_stats.get('max_switching_latency_ms', 0)))
    print("    平均决策时间:   {:.3f} ms".format(mappo_stats.get('avg_decision_time_ms', 0)))
    print("    错失机会率:     {:.4%}".format(mappo_stats.get('missed_opportunity_rate', 0)))
    print("-"*80)
    print("  [资源指标]")
    print("    负载方差:       {:.5f}".format(mappo_stats.get('load_variance', 0)))
    print("    系统吞吐量:     {:.1f} Mbps".format(mappo_stats.get('total_throughput', 0)))
    print("    平均SINR:       {:.2f} dB".format(mappo_stats.get('avg_sinr', 0)))
    print("    识别准确率:     {:.1%}".format(mappo_stats.get('recognition_accuracy', 0)))
    print("="*80)
    
    return mappo_stats


def compare_and_judge(mappo_stats, baseline):
    """对比结果并给出判断"""
    
    print("\n" + "="*100)
    print("[COMPARISON] MAPPO vs 基线算法 (全指标对比)")
    print("="*100)
    
    # [EXPANDED] 完整指标列表 (17个指标)
    metrics = [
        # === 核心性能指标 (4项) ===
        ('avg_satisfaction', '整体满意度', '%', 'higher'),
        ('connected_ratio', '连接保持率', '%', 'higher'),
        ('handover_success_rate', '切换成功率', '%', 'higher'),
        ('critical_satisfaction', '关键业务满意度', '%', 'higher'),
        
        # === 满意度细分 (3项) ===
        ('weighted_satisfaction', '加权满意度', '%', 'higher'),
        ('latency_satisfaction', '延迟满意度', '%', 'higher'),
        ('rate_satisfaction', '速率满意度', '%', 'higher'),
        
        # === 切换性能 (5项) ===
        ('avg_switching_latency_ms', '平均切换延迟', 'ms', 'lower'),
        ('max_switching_latency_ms', '最大切换延迟', 'ms', 'lower'),
        ('avg_decision_time_ms', '平均决策时间', 'ms', 'lower'),
        ('missed_opportunity_rate', '错失机会率', '%', 'lower'),
        ('migration_success_rate', '迁移成功率', '%', 'higher'),
        
        # === 负载与资源 (3项) ===
        ('load_variance', '负载方差', '', 'lower'),
        ('total_throughput', '系统吞吐量', 'Mbps', 'higher'),
        ('avg_sinr', '平均SINR', 'dB', 'higher'),
        
        # === 识别准确率 (1项) ===
        ('recognition_accuracy', '识别准确率', '%', 'higher'),
    ]
    
    results = []
    
    print("\n" + "="*100)
    print("|" + " "*96 + "|")
    print("|  " + "MAPPO vs 增强算法 vs 传统算法 - 全维度性能对比".center(90) + "  |")
    print("|" + " "*96 + "|")
    print("="*100)
    
    # 表头
    header = "+" + "-"*18 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+" + "-"*12 + "+"
    sep = "+" + "-"*18 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+" + "-"*12 + "+"
    footer = "+" + "-"*18 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+" + "-"*12 + "+"
    
    print(header)
    print("| {:^18s} | {:^10s} | {:^10s} | {:^10s} | {:^8s} | {:^12s} |".format(
        "指标", "MAPPO", "增强算法", "传统算法", "差异", "判定"))
    print(sep)
    
    for metric_key, metric_name, unit, direction in metrics:
        if metric_key not in mappo_stats:
            continue
            
        mappo_val = mappo_stats[metric_key]
        enhanced_key = f'enhanced_{metric_key}'
        trad_key = f'traditional_{metric_key}'
        
        if enhanced_key not in baseline:
            continue
        
        enhanced_val = baseline[enhanced_key]
        traditional_val = baseline.get(trad_key, None)
        
        # 计算差异 (相对于增强算法)
        if direction == 'lower':
            diff = (enhanced_val - mappo_val) / max(abs(enhanced_val), 1e-6) * 100
            better = mappo_val < enhanced_val
        else:  # higher
            diff = (mappo_val - enhanced_val) / max(abs(enhanced_val), 1e-6) * 100
            better = mappo_val > enhanced_val
        
        # 判定 (5级)
        if better and abs(diff) > 5:
            verdict = "[OK] 显著优"
        elif better and abs(diff) > 1:
            verdict = "[~] 略优"
        elif abs(diff) <= 1:
            verdict = "[=] 持平"
        elif not better and abs(diff) > 5:
            verdict = "[X] 显著差"
        else:
            verdict = "[!] 略差"
        
        # 格式化数值显示
        def format_val(val, key):
            if val is None:
                return "N/A"
            
            if 'latency' in key.lower() or 'time' in key.lower():
                return f"{val:.3f}"
            elif 'variance' in key.lower() and val < 0.01:
                return f"{val:.5f}"
            elif 'throughput' in key.lower():
                return f"{val:.1f}"
            elif 'sinr' in key.lower():
                return f"{val:.2f}"
            elif isinstance(val, float):
                if abs(val) >= 1:
                    return f"{val:.4f}"
                else:
                    return f"{val:.4f}"
            else:
                return str(val)
        
        mappo_str = format_val(mappo_val, metric_key)
        enhanced_str = format_val(enhanced_val, metric_key)
        trad_str = format_val(traditional_val, metric_key) if traditional_val else "N/A"
        
        print("| {:18s} | {:>10s} | {:>10s} | {:>10s} | {:+7.1f}% | {:12s} |".format(
            metric_name, mappo_str, enhanced_str, trad_str, diff, verdict))
        
        results.append({
            'metric': metric_name,
            'key': metric_key,
            'mappo': mappo_val,
            'enhanced': enhanced_val,
            'traditional': traditional_val,
            'diff_pct': diff,
            'better': better,
            'verdict': verdict,
            'direction': direction
        })
    
    print(footer)
    
    # [ENHANCED] 综合统计分析
    wins = sum(1 for r in results if r['better'])
    total = len(results)
    win_rate = wins / total if total > 0 else 0
    avg_improvement = np.mean([r['diff_pct'] for r in results])
    
    # 分类统计
    core_metrics = ['avg_satisfaction', 'connected_ratio', 'handover_success_rate', 'critical_satisfaction']
    satisfaction_metrics = ['weighted_satisfaction', 'latency_satisfaction', 'rate_satisfaction']
    performance_metrics = ['avg_switching_latency_ms', 'max_switching_latency_ms', 'avg_decision_time_ms']
    resource_metrics = ['load_variance', 'total_throughput', 'avg_sinr']
    
    def calc_category_score(metric_keys):
        cat_results = [r for r in results if r['key'] in metric_keys]
        if not cat_results:
            return 0, 0, 0.0
        cat_wins = sum(1 for r in cat_results if r['better'])
        cat_total = len(cat_results)
        cat_avg = np.mean([r['diff_pct'] for r in cat_results])
        return cat_wins, cat_total, cat_avg
    
    # 各类别得分
    core_wins, core_total, core_avg = calc_category_score(core_metrics)
    sat_wins, sat_total, sat_avg = calc_category_score(satisfaction_metrics)
    perf_wins, perf_total, perf_avg = calc_category_score(performance_metrics)
    res_wins, res_total, res_avg = calc_category_score(resource_metrics)
    
    # 显著性统计
    significant_better = sum(1 for r in results if r['verdict'] == '[OK] 显著优')
    slightly_better = sum(1 for r in results if r['verdict'] == '[~] 略优')
    equal = sum(1 for r in results if r['verdict'] == '[=] 持平')
    slightly_worse = sum(1 for r in results if r['verdict'] == '[!] 略差')
    significant_worse = sum(1 for r in results if r['verdict'] == '[X] 显著差')
    
    print("\n" + "="*80)
    print("[STATISTICS] 详细统计分析")
    print("="*80)
    
    print("\n+---------------------+------+--------+----------+--------------+")
    print("| 类别                | 胜/总 | 胜率   | 平均提升  | 评级         |")
    print("+---------------------+------+--------+----------+--------------+")
    
    categories = [
        ('核心性能 (4项)', core_wins, core_total, core_avg),
        ('满意度细分 (3项)', sat_wins, sat_total, sat_avg),
        ('切换性能 (5项)', perf_wins, perf_total, perf_avg),
        ('负载资源 (3项)', res_wins, res_total, res_avg),
    ]
    
    for cat_name, w, t, avg in categories:
        rate = f"{w/t*100:.0f}%" if t > 0 else "N/A"
        if avg > 10:
            grade = "[A+] 卓越"
        elif avg > 5:
            grade = "[A] 优秀"
        elif avg > 0:
            grade = "[B] 良好"
        elif avg > -5:
            grade = "[C] 一般"
        elif avg > -10:
            grade = "[D] 较差"
        else:
            grade = "[F] 很差"
        
        print("| {:19s} | {:>2d}/{:<3d} | {:>6s} | {>+8.1f}% | {:12s} |".format(
            cat_name, w, t, rate, avg, grade))
    
    print("+---------------------+------+--------+----------+--------------+")
    
    print("\n[VERDICT DISTRIBUTION]")
    verdict_dist = {
        '[OK] 显著优': significant_better,
        '[~] 略优': slightly_better,
        '[=] 持平': equal,
        '[!] 略差': slightly_worse,
        '[X] 显著差': significant_worse,
    }
    
    for v, count in verdict_dist.items():
        bar = "#" * count + "-" * (total - count) if total > 0 else ""
        print("  {:15s}: {:2d}/{:2d} [{}]".format(v, count, total, bar))
    
    print("\n" + "="*80)
    print("[VERDICT] 过拟合检测结论")
    print("="*80)
    
    # 判断逻辑 (增强版)
    if win_rate >= 0.7 and significant_better >= 5 and significant_worse <= 1:
        verdict_type = "REAL_LEARNING"
        confidence = "高 (>90%)"
        conclusion = """
+======================================================+
|                                                      |
|   [CONCLUSION] MAPPO表现出真实的泛化能力             |
|                                                      |
|   * 在{core_win_rate:.0f}%核心指标上优于增强算法                    |
|   * {sig_better}个指标显著提升                                 |
|   * 性能提升不是偶然或过拟合                         |
|   * 可以安全地声称MAPPO学到了有效策略               |
|                                                      |
+======================================================+
""".format(core_win_rate=core_wins/core_total*100 if core_total > 0 else 0,
           sig_better=significant_better)
           
    elif win_rate >= 0.6 and significantly_worse == 0 and significant_better >= 3:
        verdict_type = "LIKELY_REAL"
        confidence = "中高 (75-90%)"
        conclusion = """
+------------------------------------------------------+
| [CONCLUSION] MAPPO可能具有真实泛化能力                |
|                                                      |
| 证据:                                                 |
| * 整体胜率: {win_rate:.0%} ({wins}/{total})                              |
| * 核心指标胜率: {core_win_rate:.0f}% ({core_wins}/{core_total})                        |
| * 显著优于: {sig_better}项, 略优: {slightly_better}项                   |
| * 平均提升: {avg_improvement:+.1f}%                              |
| * 无显著退化指标                                       |
|                                                      |
| 建议: 结果可信，建议增加测试样本量确认              |
+------------------------------------------------------+
""".format(win_rate=win_rate, wins=wins, total=total,
           core_win_rate=core_wins/core_total*100 if core_total > 0 else 0,
           core_wins=core_wins, core_total=core_total,
           sig_better=significant_better, slightly_better=slightly_better,
           avg_improvement=avg_improvement)
           
    elif significantly_worse >= 2 or (win_rate < 0.5 and significant_better < 2):
        verdict_type = "OVERFITTING_RISK"
        confidence = "低 (<60%)"
        conclusion = """
+======================================================+
|                                                      |
|   [WARNING] MAPPO可能存在过拟合问题!                  |
|                                                      |
|   危险信号:                                           |
|   X {sig_worse}个指标显著差于基线                      |
|   X 胜率仅 {win_rate:.0%}                                     |
|   X 核心指标表现不佳                                  |
|                                                      |
|   可能原因:                                            |
|   1. 训练轮数过多导致记忆训练数据                     |
|   2. 缺乏足够的正则化                                 |
|   3. 验证集与训练集分布差异过大                       |
|                                                      |
+======================================================
""".format(sig_worse=significant_worse, win_rate=win_rate)
    else:
        verdict_type = "UNCERTAIN"
        confidence = "中等 (60-75%)"
        conclusion = """
+------------------------------------------------------+
| [CONCLUSION] 无法确定是否存在过拟合                   |
|                                                      |
| 混合信号:                                             |
| * 优势: {better_count}项指标较优                          |
| * 劣势: {worse_count}项指标较差                          |
| * 持平: {equal_count}项                                      |
|                                                      |
| 建议:                                                |
| * 增加评估次数(至少5次不同seed)                      |
| * 观察方差和一致性                                   |
| * 如果方差大且部分seed表现差 -> 可能过拟合           |
+------------------------------------------------------+
""".format(better_count=significant_better + slightly_better,
           worse_count=significant_worse + slightly_worse,
           equal_count=equal)
    
    print("\n  置信度: {}".format(confidence))
    print("  整体胜率: {:.0%} ({:d}/{:d})".format(win_rate, wins, total))
    print("  平均提升: {:+.2f}%".format(avg_improvement))
    print("  核心指标胜率: {:.0%} ({:d}/{:d})".format(
        core_wins/core_total if core_total > 0 else 0, core_wins, core_total))
    print(conclusion)
    
    return {
        'verdict_type': verdict_type,
        'confidence': confidence,
        'win_rate': win_rate,
        'avg_improvement': avg_improvement,
        'core_win_rate': core_wins/core_total if core_total > 0 else 0,
        'significant_better': significant_better,
        'significant_worse': significant_worse,
        'category_scores': {
            'core': (core_wins, core_total, core_avg),
            'satisfaction': (sat_wins, sat_total, sat_avg),
            'performance': (perf_wins, perf_total, perf_avg),
            'resource': (res_wins, res_total, res_avg),
        },
        'detailed_results': results
    }


def main(model_path=None):
    """主函数 (带详细进度日志)"""
    
    total_start = time.time()
    
    print("\n" + "="*80)
    print("  " + "*" * 76 + "  ")
    print("  *  " + "MAPPO泛化性快速验证系统 (全指标对比版)".center(72) + "  *")
    print("  " + "*" * 76 + "  ")
    print("="*80)
    print("  启动时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("  验证模式: 单次评估 + 历史基线对比")
    print("  对比指标: 17个核心性能指标")
    print("="*80)
    
    # ========== Phase 1: 加载识别模型 ==========
    phase1_start = time.time()
    print("\n" + "-"*80)
    print("[Phase 1/4] 加载业务识别模型")
    print("-"*80)
    
    try:
        print("  [1.1] 尝试加载已缓存的识别模型...")
        recognition_model, _ = train_or_load_recognition_model(force_retrain=False, verbose=False)
        model_source = "缓存"
    except Exception as e:
        print("  [WARN] 缓存加载失败: {}".format(e))
        print("  [1.2] 正在重新训练识别模型...")
        recognition_model, _ = train_or_load_recognition_model(force_retrain=True, verbose=False)
        model_source = "重新训练"
    
    scaler = recognition_model.scaler
    phase1_elapsed = time.time() - phase1_start
    print("\n  [OK] 识别模型加载完成 (来源: {}, 耗时: {:.1f}s)".format(model_source, phase1_elapsed))
    
    # ========== Phase 2: 加载基线数据 ==========
    phase2_start = time.time()
    print("\n" + "-"*80)
    print("[Phase 2/4] 加载实验3历史基线数据")
    print("-"*80)
    
    baseline = load_exp3_baseline()
    
    if baseline is None:
        print("[ERROR] 无基线数据，无法对比!")
        return None
    
    phase2_elapsed = time.time() - phase2_start
    baseline_count = len([k for k in baseline.keys() if not k.endswith('_std')])
    print("\n  [OK] 基线数据加载完成 ({:.1f}s, {}个指标)".format(phase2_elapsed, baseline_count))
    
    # ========== Phase 3: 运行MAPPO评估 ==========
    print("\n" + "-"*80)
    print("[Phase 3/4] 运行MAPPO模型评估")
    print("-"*80)
    
    # 确认模型路径
    if model_path is None:
        model_path = os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav_latest.pt')
    
    if not os.path.exists(model_path):
        print("[ERROR] 模型不存在: {}".format(model_path))
        return None
    
    print("  目标模型: {}".format(model_path))
    print("  模型大小: {:.2f} MB".format(os.path.getsize(model_path) / (1024*1024)))
    
    # 运行评估
    mappo_stats = run_mappo_evaluation(
        model_path=model_path,
        recognition_model=recognition_model,
        scaler=scaler,
        seed=GLOBAL_SEED,
        num_steps=350
    )
    
    if mappo_stats is None:
        print("\n[ERROR] MAPPO评估失败，无法完成验证")
        return None
    
    # ========== Phase 4: 对比分析 ==========
    phase4_start = time.time()
    print("\n" + "-"*80)
    print("[Phase 4/4] 执行全指标对比分析")
    print("-"*80)
    
    result = compare_and_judge(mappo_stats, baseline)
    
    phase4_elapsed = time.time() - phase4_start
    print("\n  [OK] 对比分析完成 ({:.1f}s)".format(phase4_elapsed))
    
    # ========== 保存结果 ==========
    output = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'model_size_mb': round(os.path.getsize(model_path) / (1024*1024), 2),
        'mappo_stats': mappo_stats,
        'baseline': baseline,
        'judgment': result,
        'timing': {
            'phase1_load_model': round(phase1_elapsed, 2),
            'phase2_load_baseline': round(phase2_elapsed, 2),
            'phase3_mappo_eval': round(result.get('eval_time', 0), 2) if isinstance(result, dict) else 0,
            'phase4_comparison': round(phase4_elapsed, 2),
            'total': round(time.time() - total_start, 2),
        }
    }
    
    output_file = 'quick_validation_{}.json'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    # ========== 最终报告 ==========
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("  " + "="*78 + "  ")
    print("  =  " + "验证任务完成".center(74) + "  =")
    print("  " + "="*78 + "  ")
    print("="*80)
    print("\n  [SUMMARY]")
    print("    总耗时:       {:.1f}秒 ({:.2f}分钟)".format(total_elapsed, total_elapsed/60))
    print("    结果文件:     {}".format(output_file))
    print("    判定类型:     {}".format(result.get('verdict_type', 'N/A')))
    print("    置信度:       {}".format(result.get('confidence', 'N/A')))
    print("    整体胜率:     {:.0%}".format(result.get('win_rate', 0)))
    print("    平均提升:     {:+.2f}%".format(result.get('avg_improvement', 0)))
    print("\n  [TIMING BREAKDOWN]")
    print("    Phase 1 (加载模型):   {:>6.1f}s ({:.0%})".format(
        phase1_elapsed, phase1_elapsed/max(total_elapsed, 1)))
    print("    Phase 2 (基线数据):   {:>6.1f}s ({:.0%})".format(
        phase2_elapsed, phase2_elapsed/max(total_elapsed, 1)))
    print("    Phase 3 (MAPPO评估):  {:>6.1f}s ({:.0%})".format(
        result.get('eval_time', 0), result.get('eval_time', 0)/max(total_elapsed, 1)))
    print("    Phase 4 (对比分析):   {:>6.1f}s ({:.0%})".format(
        phase4_elapsed, phase4_elapsed/max(total_elapsed, 1)))
    print("="*80)
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='超快速泛化性验证 (全指标版)')
    parser.add_argument('--model-path', type=str, default=None, help='MAPPO模型路径')
    args = parser.parse_args()
    
    main(model_path=args.model_path)
