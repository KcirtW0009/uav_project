# -*- coding: utf-8 -*-
"""
贝叶斯优化 — EnhancedHandoverAlgorithm权重快速调优
=====================================================
目标: 在新基站参数下(宏基站~25m, 小基站~8m, UAV 80-200m)
      搜索最优weight_config使增强算法稳定领先传统算法

搜索空间(11维):
  - 控制信令业务: sinr/load/rate (3)
  - 视频回传业务: sinr/load/rate (3)  
  - 环境监测业务: sinr/load/rate (3)
  - 切换参数: base_threshold, epsilon (2)

运行方式: 完全对齐Experiment3的执行流程

运行时间预估: ~20-40分钟 (50次迭代 × ~30秒/次)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime
from collections import defaultdict

# Windows UTF-8输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    from skopt.callbacks import CheckpointSaver
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("[WARNING] scikit-optimize not installed")

# 必须在skopt之后导入项目模块
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import (
    IntegratedHandoverAlgorithm, EnhancedHandoverAlgorithm,
    BusinessType
)
from uav_system.recognition import train_or_load_recognition_model

# ==================== 全局配置 ====================
NUM_BS = 8
NUM_UAV = 300
SIMULATION_STEPS = 100       # 调优时用较短步数加速(正式验证用350)
EVAL_ROUNDS = 3               # 每组参数评估轮数
EVAL_SEED_BASE = 50000        # 评估用种子基值
GLOBAL_SEED = 30042           # 与实验对齐的种子基值

# 优化目标权重
ALPHA_GAP = 1.0               # 增强vs传统差距的权重
BETA_HOSR = 0.5              # HOSR的权重  
GAMMA_STD = 0.3              # 标准差惩罚权重(稳定性)

# ==================== 全局变量: 识别模型(只加载一次) ====================
_recognition_model = None
_scaler = None

def get_recognition_model():
    """懒加载识别模型"""
    global _recognition_model, _scaler
    if _recognition_model is None:
        print("\n[初始化] 加载业务识别模型...")
        _recognition_model, _ = train_or_load_recognition_model(
            force_retrain=False, compare_models=False, verbose=False
        )
        _scaler = _recognition_model.scaler
        print(f"[初始化] 识别模型加载完成")
    return _recognition_model, _scaler


# ==================== 搜索空间定义 ====================
SEARCH_SPACE = [
    # ===== 控制信令业务权重 =====
    Real(0.35, 0.80, name='ctrl_sinr'),
    Real(0.05, 0.25, name='ctrl_load'),

    # ===== 视频回传业务权重 =====
    Real(0.15, 0.50, name='video_sinr'),
    Real(0.10, 0.35, name='video_load'),

    # ===== 环境监测业务权重 =====
    Real(0.10, 0.45, name='monitor_sinr'),
    Real(0.10, 0.35, name='monitor_load'),

    # ===== 切换控制参数 =====
    Real(0.001, 0.05, name='base_threshold'),   # 切换阈值基数
    Real(0.000, 0.08, name='epsilon'),           # ε-greedy探索率 (允许为0)
]

DIM_NAMES = [d.name for d in SEARCH_SPACE]


def build_weight_config(params):
    """从搜索参数构建完整的权重配置字典"""
    ctrl_sinr, ctrl_load = params[0], params[1]
    video_sinr, video_load = params[2], params[3]
    monitor_sinr, monitor_load = params[4], params[5]
    base_threshold = params[6]
    epsilon = params[7]

    def clamp(v, lo=0.01, hi=0.95):
        return max(lo, min(hi, v))

    config = {
        'business_weights': {
            'control': {
                'sinr': round(ctrl_sinr, 3),
                'load': round(ctrl_load, 3),
                'rate': round(clamp(1.0 - ctrl_sinr - ctrl_load), 3),
            },
            'video': {
                'sinr': round(video_sinr, 3),
                'load': round(video_load, 3),
                'rate': round(clamp(1.0 - video_sinr - video_load), 3),
            },
            'monitor': {
                'sinr': round(monitor_sinr, 3),
                'load': round(monitor_load, 3),
                'rate': round(clamp(1.0 - monitor_sinr - monitor_load), 3),
            },
        },
        'switch_params': {
            'base_threshold': round(base_threshold, 4),
            'epsilon': round(max(epsilon, 0.0), 4),
            'handover_cooldown': 5,
        }
    }
    return config


def evaluate_config(weight_config, rounds=EVAL_ROUNDS, verbose=False):
    """
    评估一组权重配置（完全对齐实验3的执行方式）
    
    返回: (objective_value, details_dict)  objective越小越好
    """
    recognition_model, scaler = get_recognition_model()
    
    enhanced_sats = []
    traditional_sats = []
    enhanced_hosrs = []

    for r in range(rounds):
        seed = EVAL_SEED_BASE + r * 111
        try:
            # === 传统算法 === (与实验3一致: 独立环境 + env.step + algo.run_step)
            np.random.seed(seed)
            
            env_trad = EnhancedNetworkEnvironment(
                num_bs=NUM_BS, num_uav=NUM_UAV,
                recognition_model=recognition_model, scaler=scaler,
                seed=seed, event_probability=0.05
            )
            trad_alg = IntegratedHandoverAlgorithm(env_trad)

            for step in range(SIMULATION_STEPS):
                env_trad.step()
                trad_alg.run_step()

            trad_stats = env_trad.get_state_statistics()
            trad_stats.update(trad_alg.get_detailed_stats())
            trad_sat = trad_stats['avg_satisfaction']
            trad_hosr = trad_stats['handover_success_rate']

            # === 增强算法（使用自定义权重）===
            np.random.seed(seed)
            
            env_enh = EnhancedNetworkEnvironment(
                num_bs=NUM_BS, num_uav=NUM_UAV,
                recognition_model=recognition_model, scaler=scaler,
                seed=seed, event_probability=0.05
            )
            enh_alg = EnhancedHandoverAlgorithm(
                env_enh,
                weight_config='custom',
                custom_config=weight_config
            )
            enh_alg.epsilon = 0.0  # 与实验3一致: 最终算法不含ε-greedy

            for step in range(SIMULATION_STEPS):
                env_enh.step()
                enh_alg.run_step(enable_load_balancing=True)

            enh_stats = env_enh.get_state_statistics()
            enh_stats.update(enh_alg.get_detailed_stats())
            enh_sat = enh_stats['avg_satisfaction']
            enh_hosr = enh_stats['handover_success_rate']

            enhanced_sats.append(enh_sat)
            traditional_sats.append(trad_sat)
            enhanced_hosrs.append(enh_hosr)

            if verbose:
                print(f"    Round {r+1}/{rounds}: Enh={enh_sat:.4f} Trad={trad_sat:.4f} "
                      f"gap={enh_sat-trad_sat:+.4f} HOSR={enh_hosr:.1%}")

        except Exception as e:
            import traceback
            if verbose:
                print(f"    Round {r+1} ERROR: {e}")
                traceback.print_exc()
            return 100.0, {'error': str(e)}

    # 计算统计量
    enh_mean = float(np.mean(enhanced_sats))
    trad_mean = float(np.mean(traditional_sats))
    enh_hosr_mean = float(np.mean(enhanced_hosrs))
    enh_std = float(np.std(enhanced_sats))

    gap = enh_mean - trad_mean  # 希望这个>0且尽可能大

    # 目标函数: 最小化 → 返回负gap让skopt找最小
    penalty = 0.0
    if gap < 0:
        penalty = 2.0 * abs(gap)  # 重罚反转

    objective = -(ALPHA_GAP * gap + BETA_HOSR * enh_hosr_mean) + GAMMA_STD * enh_std + penalty

    details = {
        'enhanced_mean': enh_mean,
        'traditional_mean': trad_mean,
        'gap': gap,
        'enhanced_hosr': enh_hosr_mean,
        'enhanced_std': enh_std,
        'objective': objective,
        'penalty': penalty,
        'raw_enhanced': enhanced_sats,
        'raw_traditional': traditional_sats,
    }

    return objective, details


@use_named_args(SEARCH_SPACE)
def objective(**params):
    """skopt目标函数"""
    param_values = [params[name] for name in DIM_NAMES]
    config = build_weight_config(param_values)
    obj_val, details = evaluate_config(config, rounds=EVAL_ROUNDS)
    
    objective.call_count += 1
    n = objective.call_count
    
    status = "OK" if details.get('gap', 0) > 0 else "FAIL"
    print(f"  [{n:3d}] obj={obj_val:+.4f} | "
          f"Enh={details.get('enhanced_mean',0):.4f} "
          f"Trad={details.get('traditional_mean',0):.4f} "
          f"gap={details.get('gap',0):+.4f} [{status}] "
          f"HOSR={details.get('enhanced_hosr',0):.1%}")

    return obj_val

objective.call_count = 0


def run_bayesian_optimization(n_calls=50, n_initial=15, checkpoint_path=None):
    """运行贝叶斯优化"""
    print("=" * 80)
    print("贝叶斯权重优化启动")
    print(f"  搜索空间: {len(SEARCH_SPACE)}维 ({', '.join(DIM_NAMES)})")
    print(f"  总迭代数: {n_calls} (初始随机点: {n_initial})")
    print(f"  每组评估: {EVAL_ROUNDS}轮 x {SIMULATION_STEPS}步 x {NUM_UAV}UAV")
    print(f"  预计耗时: {int(n_calls * EVAL_ROUNDS * SIMULATION_STEPS * NUM_UAV / 50000)}分钟")
    print("=" * 80)

    callbacks = []
    if checkpoint_path and SKOPT_AVAILABLE:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        callbacks.append(CheckpointSaver(checkpoint_path))

    start_time = time.time()

    if SKOPT_AVAILABLE:
        result = gp_minimize(
            func=objective,
            dimensions=SEARCH_SPACE,
            n_calls=n_calls,
            n_random_starts=n_initial,
            acq_func='EI',
            acq_optimizer='auto',
            random_state=42,
            verbose=True,
            callback=callbacks if callbacks else None
        )
    else:
        result = fallback_random_search(n_calls)

    elapsed = time.time() - start_time
    return result, elapsed


def fallback_random_search(n_calls=30):
    """简单随机搜索(当skopt不可用时)"""
    print("[FALLBACK] 使用随机搜索")
    
    best_obj = float('inf')
    best_x = None
    all_X, all_y = [], []

    for i in range(n_calls):
        x = []
        for dim in SEARCH_SPACE:
            if isinstance(dim, Real):
                val = np.random.uniform(dim.low, dim.high)
            elif isinstance(dim, Integer):
                val = np.random.randint(dim.low, dim.high + 1)
            x.append(val)

        config = build_weight_config(x)
        obj_val, details = evaluate_config(config, rounds=EVAL_ROUNDS)

        all_X.append(x)
        all_y.append(obj_val)

        if obj_val < best_obj:
            best_obj = obj_val
            best_x = x[:]

    class SimpleResult:
        def __init__(self, x, fun, X, y):
            self.x = x; self.fun = fun
            self.xs = X; self.fun_vals = y

    return SimpleResult(best_x, best_obj, all_X, all_y)


def analyze_results(result, elapsed):
    """分析并展示优化结果"""
    best_x = result.x
    best_obj = result.fun

    best_config = build_weight_config(best_x)
    _, best_details = evaluate_config(best_config, rounds=5, verbose=True)

    print("\n" + "=" * 80)
    print("优化完成!")
    print(f"  总耗时: {elapsed/60:.1f}分钟")
    print(f"  总评估次数: {objective.call_count}")
    print()
    print("★ 最佳参数配置:")
    print("-" * 60)

    print(f"\n  [业务特化权重]")
    for bname, bw in best_config['business_weights'].items():
        print(f"    {bname:>8s}: sinr={bw['sinr']:.3f} load={bw['load']:.3f} rate={bw['rate']:.3f}")

    print(f"\n  [切换控制参数]")
    sp = best_config['switch_params']
    print(f"    base_threshold = {sp['base_threshold']}")
    print(f"    epsilon        = {sp['epsilon']}")
    print(f"    cooldown       = {sp['handover_cooldown']}")

    print(f"\n  [验证性能(5轮)]")
    print(f"    增强算法均值:   {best_details['enhanced_mean']:.4f} ± {best_details['enhanced_std']:.4f}")
    print(f"    传统算法均值:   {best_details['traditional_mean']:.4f}")
    print(f"    差距(gap):      {best_details['gap']:+.4f} ", end="")
    if best_details['gap'] > 0:
        print("✅ 增强>传统")
    else:
        print("⚠️ 增强≤传统")
    print(f"    增强HOSR:      {best_details['enhanced_hosr']:.1%}")
    print(f"    目标函数值:     {best_obj:.4f}")

    # 与原始paper配置对比
    print(f"\n  [与原paper配置对比]")
    paper_config = build_weight_config([
        0.5, 0.2,     # ctrl sinr, load
        0.3, 0.25,    # video sinr, load
        0.25, 0.25,   # monitor sinr, load
        0.01, 0.01,   # threshold, epsilon
    ])
    _, paper_details = evaluate_config(paper_config, rounds=5)
    print(f"    Paper配置gap:  {paper_details['gap']:+.4f} "
          f"(Enh={paper_details['enhanced_mean']:.4f}, Trad={paper_details['traditional_mean']:.4f})")
    improvement = best_details['gap'] - paper_details['gap']
    print(f"    调优提升:      {improvement:+.4f} "
          f"({improvement/max(abs(paper_details['gap']), 0.001)*100:+.1f}%)")

    return best_config, best_details


def save_results(best_config, best_details, result, output_dir='experiment_results'):
    """保存优化结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_file = os.path.join(output_dir, f'optimized_weights_{timestamp}.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'best_config': best_config,
            'performance': {k: (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v) 
                           for k, v in best_details.items()},
            'search_space': DIM_NAMES,
            'optimization_settings': {
                'eval_rounds': EVAL_ROUNDS,
                'sim_steps': SIMULATION_STEPS,
                'num_uav': NUM_UAV,
                'num_bs': NUM_BS,
                'total_evaluations': objective.call_count,
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  配置已保存: {config_file}")
    return config_file


# ==================== 补丁: 支持custom_config ====================
_orig_init = EnhancedHandoverAlgorithm.__init__

def _patched_init(self, env, weight_config='paper', custom_config=None):
    """打补丁后的构造函数，支持custom_config参数"""
    # 先调用原初始化(用dummy weight_config避免触发默认逻辑)
    _orig_init(self, env, weight_config='paper')

    if custom_config is not None and weight_config == 'custom':
        # 应用自定义业务权重
        biz_w = custom_config.get('business_weights', {})
        self.business_weights = {
            BusinessType.CONTROL_SIGNAL: biz_w.get('control', self.business_weights[BusinessType.CONTROL_SIGNAL]),
            BusinessType.VIDEO_STREAMING: biz_w.get('video', self.business_weights[BusinessType.VIDEO_STREAMING]),
            BusinessType.ENVIRONMENT_MONITORING: biz_w.get('monitor', self.business_weights[BusinessType.ENVIRONMENT_MONITORING]),
        }

        # 应用自定义切换参数
        sp = custom_config.get('switch_params', {})
        self.base_threshold = sp.get('base_threshold', self.base_threshold)
        self.epsilon = sp.get('epsilon', self.epsilon)
        self.handover_cooldown = sp.get('handover_cooldown', self.handover_cooldown)

        self.weight_config = 'custom'

EnhancedHandoverAlgorithm.__init__ = _patched_init


# ==================== 主程序 ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='贝叶斯权重优化')
    parser.add_argument('--calls', type=int, default=50, help='总迭代次数(默认50)')
    parser.add_argument('--initial', type=int, default=15, help='初始随机点数量(默认15)')
    parser.add_argument('--rounds', type=int, default=3, help='每组参数评估轮数(默认3)')
    parser.add_argument('--steps', type=int, default=100, help='仿真步数(默认100, 加速调优)')
    parser.add_argument('--fast', action='store_true', help='快速模式(20次迭代, 2轮评估, 80步)')
    args = parser.parse_args()

    if args.fast:
        args.calls = 20
        args.rounds = 2
        args.steps = 80

    EVAL_ROUNDS = args.rounds
    SIMULATION_STEPS = args.steps

    checkpoint_path = os.path.join('experiment_results', 'tuning_checkpoint.pkl')
    result, elapsed = run_bayesian_optimization(
        n_calls=args.calls,
        n_initial=args.initial,
        checkpoint_path=checkpoint_path
    )

    best_config, best_details = analyze_results(result, elapsed)
    config_file = save_results(best_config, best_details, result)

    print("\n" + "=" * 80)
    print("下一步操作:")
    print("  1. 将最佳配置写入algorithms.py作为新的'new_env'权重选项")
    print("  2. 用新权重重跑实验3(10轮x350步)做最终验证")
    print("  3. 确认增强>传统后重跑全部实验并更新论文数据")
    print("=" * 80)
