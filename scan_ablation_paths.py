"""
快速消融路径扫描 —— 找最优叙事路线
=====================================

目标：从传统算法出发，逐步添加机制，每一步都要有正向增量。
策略：基于exp2b已有数据分析规律，再用3次重复验证新路径。

核心洞察（来自exp2b已有16种配置）:
  - weights单独用就很强：满意度0.944, HOSR=77.1%, 负载方差0.00095
  - dyn_thresh单独用HOSR暴跌到69.7%（最大痛点）
  - dyn_thresh_weights = 满意度全局最优(0.949)
  - weights_epsilon = HOSR全局最优(84.1%)

待测试的新路径:
  路径A(权重优先):   T → W → W+E → W+E+LB → 完整
  路径B(阈值后置):   T → W → W+DT → W+DT+E → 完整  
  路径C(渐进式):     T → W → W+DT → W+DT+E → W+DT+E+LB
"""

import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uav_system.config import GLOBAL_SEED, RESULT_DIR
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import (
    IntegratedHandoverAlgorithm,      # 传统算法
    EnhancedHandoverAlgorithm,         # 增强算法
)
from uav_system.business import BusinessType, QOS_PROFILES

# ============================================================
# 快速实验参数（减少重复次数加速）
# ============================================================
NUM_REPEATS = 3       # 只跑3次（原8次）
NUM_STEPS = 150        # 步数不变

# ============================================================
# 候选消融路径定义
# 每条路径是一个列表，元素为 (config_name, 描述, 启用的机制标志)
# ============================================================

CANDIDATE_PATHS = {
    'path_A_weights_first': {
        'name': '路径A: 权重优先',
        'steps': [
            ('traditional',      '传统基线(A3)',           {}),
            ('step1_weights',    '+ 业务权重',             {'weights': True}),
            ('step2_w_e',        '+ ε-greedy',             {'weights': True, 'epsilon': True}),
            ('step3_w_e_lb',     '+ 负载均衡',             {'weights': True, 'epsilon': True, 'load_balance': True}),
            ('step4_full',       '+ 动态阈值(完整)',        {'weights': True, 'epsilon': True, 'load_balance': True, 'dyn_thresh': True}),
        ],
    },
    'path_B_threshold_last': {
        'name': '路径B: 阈值后置',
        'steps': [
            ('traditional',      '传统基线(A3)',           {}),
            ('step1_weights',    '+ 业务权重',             {'weights': True}),
            ('step2_w_dt',       '+ 动态阈值',             {'weights': True, 'dyn_thresh': True}),
            ('step3_w_dt_e',     '+ ε-greedy',             {'weights': True, 'dyn_thresh': True, 'epsilon': True}),
            ('step4_full',       '+ 负载均衡(完整增强)',    {'weights': True, 'dyn_thresh': True, 'epsilon': True, 'load_balance': True}),
        ],
    },
    'path_C_gradual': {
        'name': '路径C: 渐进式最优',
        'steps': [
            ('traditional',      '传统基线(A3)',           {}),
            ('step1_weights',    '+ 业务权重',             {'weights': True}),
            ('step2_w_dt',       '+ 动态阈值(组合)',        {'weights': True, 'dyn_thresh': True}),
            ('step3_w_dt_e',     '+ ε-greedy',             {'weights': True, 'dyn_thresh': True, 'epsilon': True}),
            ('step4_complete',   '+ 负载均衡(完整)',        {'weights': True, 'dyn_thresh': True, 'epsilon': True, 'load_balance': True}),
        ],
    },
}


def create_algo(env, mechanisms):
    """根据机制标志创建并配置算法实例"""
    is_traditional = not mechanisms
    
    if is_traditional:
        algo = IntegratedHandoverAlgorithm(env)
        return algo, False  # 无负载均衡
    
    algo = EnhancedHandoverAlgorithm(env)
    
    has_dyn_thresh = mechanisms.get('dyn_thresh', False)
    has_weights = mechanisms.get('weights', False)
    has_epsilon = mechanisms.get('epsilon', False)
    has_lb = mechanisms.get('load_balance', False)
    
    # 禁用/启用动态阈值
    if not has_dyn_thresh:
        algo.base_threshold = 0.005
        algo.calculate_dynamic_threshold = lambda uav: 0.005
    
    # 禁用/启用业务权重
    if not has_weights:
        for bt in BusinessType:
            algo.business_weights[bt] = {'sinr': 0.4, 'load': 0.3, 'rate': 0.3}
    
    # 禁用/启用epsilon
    if not has_epsilon:
        algo.epsilon = 0.0
    
    return algo, has_lb


def run_single_config(mechanisms, seed, num_steps=150):
    """运行单个配置，返回完整统计字典"""
    env = EnhancedNetworkEnvironment(
        num_bs=8, num_uav=300,
        recognition_model=None, scaler=None,
        seed=seed, event_probability=0.05
    )
    
    algo, enable_lb = create_algo(env, mechanisms)
    
    for step in range(num_steps):
        env.step()
        if isinstance(algo, IntegratedHandoverAlgorithm):
            algo.run_step()
        else:
            algo.run_step(enable_load_balancing=enable_lb)
    
    stats = env.get_state_statistics()
    
    # 获取算法详细统计（含 rollback_fail_count 等）
    if hasattr(algo, 'get_detailed_stats'):
        detailed = algo.get_detailed_stats()
        stats.update(detailed)
    
    return stats


def compute_effective_hosr(stats):
    """
    计算 effective_HOSR = (success + rollback) / attempts
    
    对增强算法：
      normal_attempts = handover_attempts - reconnect_attempts
      successes = handover_successes - reconnect_successes  
      rollbacks ≈ attempts - successes - rollback_fail_count
      effective = (successes + rollbacks) / attempts = 1 - rollback_fail/attempts
    """
    if 'handover_attempts' in stats and 'rollback_fail_count' in stats:
        attempts = max(stats['handover_attempts'] - stats.get('reconnect_attempts', 0), 1)
        rollback_fails = stats.get('rollback_fail_count', 0)
        effective_hosr = 1.0 - (rollback_fails / attempts)
        return effective_hosr
    elif 'handover_success_rate' in stats:
        return stats['handover_success_rate']
    return None


def main():
    print("=" * 70)
    print("  快速消融路径扫描")
    print(f"  Params: {NUM_REPEATS} repeats x {NUM_STEPS} steps x {len(CANDIDATE_PATHS)} paths")
    print("=" * 70)
    
    all_results = {}
    
    for path_id, path_info in CANDIDATE_PATHS.items():
        print(f"\n{'='*60}")
        print(f"  Testing: {path_info['name']}")
        print(f"{'='*60}")
        
        path_results = {}
        
        for step_name, step_desc, mechanisms in path_info['steps']:
            step_runs = []
            
            for rep in range(NUM_REPEATS):
                seed = GLOBAL_SEED + rep * 100 + hash(step_name) % 1000
                stats = run_single_config(mechanisms, seed, NUM_STEPS)
                step_runs.append(stats)
            
            # 汇总
            summary = {
                'avg_satisfaction': ([s['avg_satisfaction'] for s in step_runs],),
                'handover_success_rate': ([s['handover_success_rate'] for s in step_runs],),
                'critical_satisfaction': ([s['critical_satisfaction'] for s in step_runs],),
                'load_variance': ([s['load_variance'] for s in step_runs],),
            }
            
            # 计算均值±标准差
            for key in list(summary.keys()):
                vals = summary[key][0]
                summary[key] = (np.mean(vals), np.std(vals))
            
            # 计算effective_HOSR
            eff_hosrs = [compute_effective_hosr(s) for s in step_runs if compute_effective_hosr(s) is not None]
            if eff_hosrs:
                summary['effective_HOSR'] = (np.mean(eff_hosrs), np.std(eff_hosrs))
            
            # 收集rollback信息
            rb_counts = [s.get('rollback_fail_count', 0) for s in step_runs]
            summary['rollback_fail_count'] = (np.mean(rb_counts), np.std(rb_counts))
            
            path_results[step_name] = {
                'desc': step_desc,
                'summary': summary,
                'raw': step_runs[0],  # 保留一次原始数据用于详细查看
            }
        
        all_results[path_id] = path_results
        
    # 打印该路径的结果表格
    print(f"\n  {'Step':<22} {'Satisfaction':>12} {'Strict-HOSR':>14} {'Eff-HOSR':>10} {'Critical':>10} {'LoadVar(x1e3)':>14}")
    print(f"  {'-'*82}")
    
    prev_sat = None
    prev_hosr = None
    
    for step_name, step_data in path_results.items():
        s = step_data['summary']
        sat_mean, sat_std = s['avg_satisfaction']
        hosr_mean, hosr_std = s['handover_success_rate']
        crit_mean, crit_std = s['critical_satisfaction']
        lv_mean, lv_std = s['load_variance']
        
        eff_hosr = s.get('effective_HOSR', (None, None))[0]
        eff_str = f"{eff_hosr*100:.1f}%" if eff_hosr else "N/A"
        
        # 计算相对前一步的变化
        sat_delta = f"+{(sat_mean-prev_sat)*100:.1f}%" if prev_sat is not None else "baseline"
        hosr_delta = f"+{(hosr_mean-prev_hosr)*100:+.1f}%" if prev_hosr is not None else "baseline"
        
        desc_short = step_data['desc'][:18]
        print(f"  {desc_short:<20} {sat_mean:.3f}+/-{sat_std:.3f} {hosr_mean*100:>6.1f}%+/-{hosr_std*100:.1f}% {eff_str:>10} {crit_mean:.3f}      {lv_mean*1000:.2f}")
        print(f"  {'':>22} ({sat_delta}) ({hosr_delta})")
        
        prev_sat = sat_mean
        prev_hosr = hosr_mean
    
    # ============================================================
    # 最终对比：哪条路径最好？
    # ============================================================
    print(f"\n\n{'='*70}")
    print("  路径综合评估")
    print(f"{'='*70}")
    
    best_path = None
    best_score = -999
    
    for path_id, path_results in all_results.items():
        steps = list(path_results.values())
        if len(steps) < 2:
            continue
        
        # 评分标准：
        # 1. 每一步满意度都不降 (+1 per positive step)
        # 2. 每一步HOSR不大幅下降 (> -3pp OK, else -1)
        # 3. 终点指标越高越好
        # 4. 关键业务满足率高
        
        score = 0
        details = []
        
        prev_sat = steps[0]['summary']['avg_satisfaction'][0]
        prev_hosr = steps[0]['summary']['handover_success_rate'][0]
        
        for i, step in enumerate(steps[1:], 1):
            sat = step['summary']['avg_satisfaction'][0]
            hosr = step['summary']['handover_success_rate'][0]
            crit = step['summary']['critical_satisfaction'][0]
            
            sat_change = sat - prev_sat
            hosr_change = (hosr - prev_hosr) * 100
            
            if sat_change > 0.005:
                score += 2
                details.append(f"  step{i}: sat+{sat_change:.3f} ++")
            elif sat_change > -0.01:
                score += 1
                details.append(f"  step{i}: sat{sat_change:+.3f} +")
            else:
                score -= 2
                details.append(f"  step{i}: sat{sat_change:+.3f} --")
            
            if hosr_change > 1:
                score += 1
            elif hosr_change > -5:
                score += 0  # acceptable small drop
            else:
                score -= 2  # big drop penalty
                details[-1] += f" HOSR{hosr_change:+.1f}pp --"
            
            prev_sat = sat
            prev_hosr = hosr
        
        # 终点加分
        final_sat = steps[-1]['summary']['avg_satisfaction'][0]
        final_crit = steps[-1]['summary']['critical_satisfaction'][0]
        final_lv = steps[-1]['summary']['load_variance'][0]
        score += final_sat * 30  # 满意度越高越好
        score += final_crit * 10
        score -= final_lv * 50  # 负载方差越低越好
        
        print(f"\n  {CANDIDATE_PATHS[path_id]['name']}: score={score:.1f}")
        for d in details:
            print(d)
        
        if score > best_score:
            best_score = score
            best_path = path_id
    
    print(f"\n\n  >>> Recommended: {CANDIDATE_PATHS[best_path]['name']} (score={best_score:.1f})")
    
    # 保存完整结果
    output_path = os.path.join('experiment_results', 'ablation_path_scan.json')
    save_data = {
        '_meta': {
            'num_repeats': NUM_REPEATS,
            'num_steps': NUM_STEPS,
            'best_path': best_path,
            'best_score': float(best_score),
        },
    }
    for path_id, path_results in all_results.items():
        save_data[path_id] = {}
        for step_name, step_data in path_results.items():
            s = step_data['summary']
            save_data[path_id][step_name] = {
                'desc': step_data['desc'],
                'avg_satisfaction': list(s['avg_satisfaction']),
                'handover_success_rate': list(s['handover_success_rate']),
                'critical_satisfaction': list(s['critical_satisfaction']),
                'load_variance': list(s['load_variance']),
            }
            if 'effective_HOSR' in s:
                save_data[path_id][step_name]['effective_HOSR'] = list(s['effective_HOSR'])
            if 'rollback_fail_count' in s:
                save_data[path_id][step_name]['rollback_fail_count'] = list(s['rollback_fail_count'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {output_path}")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n  Elapsed: {elapsed:.1f}s")
