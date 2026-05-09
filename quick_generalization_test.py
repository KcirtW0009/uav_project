"""
快速泛化性评估 - 验证MAPPO模型真实能力 vs 过拟合检测

目的:
  在实验三场景(8BS×300UAV)中对比:
    1. MAPPO模型 (已训练)
    2. 增强算法 (基线)
    3. 传统算法 (基线)

核心问题:
  - MAPPO的高性能是否来自真正的策略学习？
  - 还是仅仅记忆了训练数据模式(过拟合)？

过拟合判定标准:
  ✅ 真实学习: MAPPO在多个seed上都稳定优于增强算法 (>2%提升, p<0.05)
  ⚠️ 轻微过拟合: MAPPO平均更好但方差大, 部分seed不如增强算法
  ❌ 严重过拟合: MAPPO在某些seed上显著差于增强算法

用法:
  python quick_generalization_test.py [--repeats 5] [--quick]
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
from uav_system.experiments import Experiment3
from uav_system.experiments import evaluate_mappo_in_experiment


def find_best_mappo_model():
    """查找可用的MAPPO模型文件"""
    
    # 可能的模型路径列表 (按优先级排序)
    model_candidates = [
        # 当前训练的best模型
        os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav_best.pt'),
        os.path.join(RESULT_DIR, 'mappo_models', 'mappo_8bs_300uav_latest.pt'),
        # 通用模型名
        os.path.join(RESULT_DIR, 'mappo_models', 'mappo_best.pt'),
        os.path.join(RESULT_DIR, 'mappo_models', 'mappo_latest.pt'),
        # 搜索所有.pt文件
    ]
    
    # 检查候选路径
    for path in model_candidates:
        if os.path.exists(path):
            print(f"[OK] 找到模型: {path}")
            return path
    
    # 广泛搜索
    mappo_dir = os.path.join(RESULT_DIR, 'mappo_models')
    if os.path.exists(mappo_dir):
        for root, dirs, files in os.walk(mappo_dir):
            for f in files:
                if f.endswith('.pt'):
                    full_path = os.path.join(root, f)
                    print(f"[OK] 发现模型文件: {full_path}")
                    return full_path
    
    print("[WARN] 未找到任何MAPPO模型文件!")
    return None


def run_single_evaluation(seed, recognition_model, scaler, model_path, num_steps=350):
    """运行单次评估 (MAPPO + 增强算法 + 传统算法)"""
    
    results = {}
    
    # 1. MAPPO评估
    if model_path:
        try:
            mappo_stats = evaluate_mappo_in_experiment(
                num_bs=8, num_uav=300, num_steps=num_steps,
                recognition_model=recognition_model, scaler=scaler,
                seed=seed,
                model_path=model_path
            )
            results['mappo'] = mappo_stats
            if mappo_stats:
                print(f"  [MAPPO] sat={mappo_stats.get('avg_satisfaction', 0):.4f}, "
                      f"conn={mappo_stats.get('connected_ratio', 0):.1%}, "
                      f"hosr={mappo_stats.get('handover_success_rate', 0):.1%}")
        except Exception as e:
            print(f"  [ERROR] MAPPO评估失败: {e}")
            results['mappo'] = None
    
    # 2. 增强算法评估 (使用Experiment3的内部逻辑)
    from uav_system.environment import EnhancedNetworkEnvironment
    from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm
    
    # 增强算法
    env_enh = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=300,
        recognition_model=recognition_model, scaler=scaler,
        seed=seed, event_probability=0.05
    )
    algo_enh = EnhancedHandoverAlgorithm(env_enh)
    algo_enh.epsilon = 0.0
    
    # 传统算法
    env_trad = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=300,
        recognition_model=recognition_model, scaler=scaler,
        seed=seed, event_probability=0.05
    )
    algo_trad = IntegratedHandoverAlgorithm(env_trad)
    
    # 运行仿真
    for step in range(num_steps):
        env_enh.step()
        algo_enh.run_step(enable_load_balancing=True)
        env_trad.step()
        algo_trad.run_step()
    
    # 收集统计
    enh_stats = env_enh.get_state_statistics()
    enh_stats.update(algo_enh.get_detailed_stats())
    enh_stats['connected_ratio'] = enh_stats['connected_count'] / env_enh.num_uav
    results['enhanced'] = enh_stats
    
    trad_stats = env_trad.get_state_statistics()
    trad_stats.update(algo_trad.get_detailed_stats())
    trad_stats['connected_ratio'] = trad_stats['connected_count'] / env_trad.num_uav
    results['traditional'] = trad_stats
    
    print(f"  [ENHANCED] sat={enh_stats['avg_satisfaction']:.4f}, "
          f"conn={enh_stats['connected_ratio']:.1%}, "
          f"hosr={enh_stats['handover_success_rate']:.1%}")
    print(f"  [TRADITIONAL] sat={trad_stats['avg_satisfaction']:.4f}, "
          f"conn={trad_stats['connected_ratio']:.1%}, "
          f"hosr={trad_stats['handover_success_rate']:.1%}")
    
    return results


def analyze_overfitting_risk(all_results):
    """分析过拟合风险"""
    
    print("\n" + "="*80)
    print("[ANALYSIS] 过拟合风险分析")
    print("="*80)
    
    if not all_results or 'mappo' not in all_results[0] or all_results[0]['mappo'] is None:
        print("[ERROR] 无MAPPO数据，无法分析过拟合风险")
        return None
    
    # 提取关键指标
    metrics = ['avg_satisfaction', 'connected_ratio', 'handover_success_rate', 
               'critical_satisfaction', 'load_variance']
    
    analysis = {}
    
    for metric in metrics:
        mappo_vals = []
        enhanced_vals = []
        
        for r in all_results:
            if r.get('mappo') and metric in r['mappo']:
                mappo_vals.append(r['mappo'][metric])
            if r.get('enhanced') and metric in r['enhanced']:
                enhanced_vals.append(r['enhanced'][metric])
        
        if len(mappo_vals) < 2 or len(enhanced_vals) < 2:
            continue
        
        mappo_mean = np.mean(mappo_vals)
        mappo_std = np.std(mappo_vals)
        enhanced_mean = np.mean(enhanced_vals)
        enhanced_std = np.std(enhanced_vals)
        
        improvement = (mappo_mean - enhanced_mean) / max(enhanced_mean, 1e-6) * 100
        
        # 统计检验 (简化t-test)
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(mappo_vals, enhanced_vals)
        
        # 过拟合风险评估
        risk_factors = []
        risk_score = 0.0
        
        # 因子1: 方差比较 (MAPPO方差 > 增强算法方差 × 1.5 → 可疑)
        if mappo_std > enhanced_std * 1.5:
            risk_factors.append("MAPPO方差过大(不稳定)")
            risk_score += 0.25
        elif mappo_std > enhanced_std * 1.2:
            risk_factors.append("MAPPO方差略高")
            risk_score += 0.1
        
        # 因子2: 一致性 (MAPPO不是每次都赢)
        wins = sum(1 for m, e in zip(mappo_vals, enhanced_vals) if m > e)
        win_rate = wins / len(mappo_vals)
        if win_rate < 0.6:
            risk_factors.append(f"胜率低({win_rate:.0%})")
            risk_score += 0.25
        elif win_rate < 0.8:
            risk_factors.append(f"胜率中等({win_rate:.0%})")
            risk_score += 0.1
        
        # 因子3: 改善幅度
        if improvement < 1.0:
            risk_factors.append(f"改善微小({improvement:+.1f}%)")
            risk_score += 0.2
        elif improvement < 2.0:
            risk_factors.append(f"改善较小({improvement:+.1f}%)")
            risk_score += 0.1
        
        # 因子4: 统计显著性
        if p_value > 0.05:
            risk_factors.append(f"统计不显著(p={p_value:.3f})")
            risk_score += 0.3
        elif p_value > 0.1:
            risk_factors.append(f"边缘显著(p={p_value:.3f})")
            risk_score += 0.15
        
        analysis[metric] = {
            'mappo_mean': mappo_mean,
            'mappo_std': mappo_std,
            'enhanced_mean': enhanced_mean,
            'enhanced_std': enhanced_std,
            'improvement_pct': improvement,
            'p_value': p_value,
            'win_rate': win_rate,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'verdict': _get_verdict(risk_score, p_value, win_rate, improvement)
        }
    
    return analysis


def _get_verdict(risk_score, p_value, win_rate, improvement):
    """生成判定结论"""
    
    if risk_score < 0.2 and p_value < 0.05 and win_rate > 0.8:
        return "REAL_LEARNING"
    elif risk_score < 0.4 and p_value < 0.1 and win_rate > 0.6:
        return "LIKELY_REAL"
    elif risk_score < 0.6:
        return "UNCERTAIN"
    else:
        return "LIKELY_OVERFITTING"


def print_analysis_report(analysis):
    """打印分析报告"""
    
    verdicts = {
        'REAL_LEARNING': ('[OK] 真实学习', 'green'),
        'LIKELY_REAL': ('[OK] 可能是真实学习', 'yellow'),
        'UNCERTAIN': ('[?] 不确定', 'yellow'),
        'LIKELY_OVERFITTING': ('[X] 可能过拟合', 'red'),
    }
    
    metric_names = {
        'avg_satisfaction': '整体满意度',
        'connected_ratio': '连接保持率',
        'handover_success_rate': '切换成功率',
        'critical_satisfaction': '关键业务满意度',
        'load_variance': '负载方差',
    }
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│              泛化性评估结果汇总                                │")
    print("├──────────────┬──────────┬──────────┬────────┬──────────────────┤")
    print("│ 指标         │ MAPPO    │ 增强算法 │ 提升   │ 判定             │")
    print("├──────────────┼──────────┼──────────┼────────┼──────────────────┤")
    
    for metric, data in analysis.items():
        name = metric_names.get(metric, metric)
        map_val = data['mappo_mean']
        enh_val = data['enhanced_mean']
        imp = data['improvement_pct']
        verdict_text, _ = verdicts[data['verdict']]
        
        print(f"│ {name:<12} │ {map_val:>8.4f} │ {enh_val:>8.4f} │ {imp:>+5.1f}% │ {verdict_text:<16} │")
    
    print("└──────────────┴──────────┴──────────┴────────┴──────────────────┘")
    
    # 详细风险因子
    print("\n[DETAILS] 风险因子详情:")
    for metric, data in analysis.items():
        name = metric_names.get(metric, metric)
        if data['risk_factors']:
            print(f"\n  {name}:")
            for factor in data['risk_factors']:
                print(f"    - {factor}")
    
    # 总体结论
    avg_risk = np.mean([d['risk_score'] for d in analysis.values()])
    real_learning_count = sum(1 for d in analysis.values() if d['verdict'] in ['REAL_LEARNING', 'LIKELY_REAL'])
    total_metrics = len(analysis)
    
    print("\n" + "="*80)
    print("[CONCLUSION] 总体判定")
    print("="*80)
    
    if avg_risk < 0.25 and real_learning_count >= total_metrics * 0.7:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   [VERDICT] MAPPO表现出真实的泛化能力                        ║
  ║                                                              ║
  ║   证据:                                                       ║
  ║   ✓ 在多个随机种子上稳定优于增强算法                          ║
  ║   ✓ 统计检验显著 (p < 0.05)                                  ║
  ║   ✓ 方差可控 (未出现过拟合典型特征)                           ║
  ║                                                              ║
  ║   建议: 可以安全地声称MAPPO学到了有效的切换策略               ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")
    elif avg_risk < 0.5:
        print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ [VERDICT] MAPPO可能有一定泛化能力，但需谨慎解读              │
  │                                                              │
  │ 证据:                                                         │
  │ {real_learning_count}/{total_metrics} 个指标显示真实学习                              │
  │ 平均风险评分: {avg_risk:.2f}                                          │
  │                                                              │
  │ 建议:                                                         │
  │ - 可以报告正面结果，但需注明局限性                            │
  │ - 建议增加测试样本量提高置信度                               │
  └──────────────────────────────────────────────────────────────┘
""")
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   [WARNING] MAPPO可能存在过拟合问题!                          ║
  ║                                                              ║
  ║   危险信号:                                                   ║
  ║   ✗ 性能在不同种子间波动大                                    ║
  ║   ✗ 部分场景不如基线算法                                      ║
  ║   ✗ 统计检验不显著                                           ║
  ║                                                              ║
  ║   建议:                                                      ║
  ║   1. 增加正则化 (dropout, weight decay)                      ║
  ║   2. 增加训练数据多样性 (domain randomization)               ║
  ║   3. 减少模型复杂度或训练轮数                                 ║
  ║   4. 考虑early stopping机制                                  ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")


def main(repeats=5, quick=False, model_path=None):
    """主函数"""
    
    start_time = time.time()
    
    print("\n" + "="*80)
    print("[TEST] 快速泛化性评估 - MAPPO vs 增强算法 vs 传统算法")
    print("="*80)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  重复次数: {repeats}")
    print(f"  场景: 实验3 (8BS×300UAV, 350步)")
    print("="*80)
    
    # 1. 加载识别模型
    print("\n[STEP 1] 加载业务识别模型...")
    try:
        recognition_model, _ = train_or_load_recognition_model(force_retrain=False, verbose=False)
    except Exception as e:
        print(f"  [WARN] 识别模型加载失败: {e}")
        print("  [ACTION] 尝试强制重新训练...")
        recognition_model, _ = train_or_load_recognition_model(force_retrain=True, verbose=False)
    scaler = recognition_model.scaler
    print("  [OK] 识别模型加载完成")
    
    # 2. 查找/确认MAPPO模型
    print("\n[STEP 2] 查找MAPPO模型...")
    if model_path is None:
        model_path = find_best_mappo_model()
    
    if model_path is None:
        print("\n[ERROR] 未找到MAPPO模型文件!")
        print("\n可能的解决方案:")
        print("  1. 先完成或中断当前训练 (Ctrl+C)")
        print("  2. 训练会自动保存模型到 experiment_results/mappo_models/")
        print("  3. 或使用 --model-path 参数指定模型路径")
        print("\n正在尝试使用预训练权重进行测试...")
        # 尝试使用预训练缓存（如果有）
        model_path = None  # 强制为None，让evaluate函数处理
    
    # 3. 运行评估
    print(f"\n[STEP 3] 运行{repeats}次评估...")
    all_results = []
    
    actual_repeats = min(repeats, 3) if quick else repeats
    
    for i in range(actual_repeats):
        seed = GLOBAL_SEED + i * 100  # 使用间隔大的种子增加多样性
        print(f"\n--- 评估 {i+1}/{actual_repeats} (seed={seed}) ---")
        
        try:
            result = run_single_evaluation(
                seed=seed,
                recognition_model=recognition_model,
                scaler=scaler,
                model_path=model_path,
                num_steps=350 if not quick else 100  # quick模式减少步数
            )
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] 评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_results) == 0:
        print("\n[ERROR] 所有评估都失败了!")
        return None
    
    # 4. 分析结果
    print("\n[STEP 4] 分析过拟合风险...")
    analysis = analyze_overfitting_risk(all_results)
    
    if analysis:
        print_analysis_report(analysis)
    
    # 5. 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'repeats': actual_repeats,
        'model_path': model_path,
        'results': all_results,
        'analysis': analysis,
    }
    
    output_file = f'generalization_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n[SAVE] 结果已保存到: {output_file}")
    
    elapsed = time.time() - start_time
    print(f"\n[TIME] 总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    
    return analysis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='快速泛化性评估')
    parser.add_argument('--repeats', type=int, default=5, help='重复次数 (默认5)')
    parser.add_argument('--quick', action='store_true', help='快速模式 (仅3次重复, 100步)')
    parser.add_argument('--model-path', type=str, default=None, help='指定模型路径')
    args = parser.parse_args()
    
    main(repeats=args.repeats, quick=args.quick, model_path=args.model_path)
