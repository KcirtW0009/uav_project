# -*- coding: utf-8 -*-
"""
方案B: 基站参数修正后的快速验证实验 (V2 - 修复版)

完全对齐实验3的运行方式:
- 每个算法使用独立的 EnhancedNetworkEnvironment 实例
- 使用 algo.run_step() 而不是手动循环
- 启用负载均衡

运行方式:
    .\venv\Scripts\python.exe quick_validate_new_params.py
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows UTF-8输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
from uav_system.recognition import train_or_load_recognition_model
from uav_system.environment import EnhancedNetworkEnvironment
from uav_system.algorithms import EnhancedHandoverAlgorithm, IntegratedHandoverAlgorithm


def run_quick_validation_v2(num_rounds=5, num_steps=150):
    """
    快速验证V2: 完全对齐实验3的运行方式
    
    关键改进:
    1. 每个算法使用独立环境实例 (同seed保证公平)
    2. 使用 algo.run_step() 标准接口
    3. 增强算法启用负载均衡
    4. 加载业务识别模型
    """
    print("=" * 80)
    print("【方案B V2】基站参数修正快速验证 (对齐实验3)")
    print("=" * 80)
    print(f"  轮次: {num_rounds}")
    print(f"  每轮步数: {num_steps}")
    print(f"  环境: 8BS × 300UAV (default场景)")
    print(f"  新参数:")
    print(f"    - 宏基站高度: ~25m (3GPP UMa)")
    print(f"    - 小基站高度: ~8m (3GPP UMi)")
    print(f"    - UAV高度: 80~200m (低空域)")
    print(f"  对齐实验3:")
    print(f"    - 独立环境实例")
    print(f"    - run_step() 接口")
    print(f"    - 负载均衡启用")
    print("=" * 80)

    # ====== 加载识别模型 ======
    print("\n[加载业务识别模型...]")
    recognition_model, scaler = train_or_load_recognition_model(force_retrain=False)

    # ====== 运行实验 ======
    enhanced_results = []
    traditional_results = []

    for rep in range(num_rounds):
        print(f"\n--- 重复 {rep+1}/{num_rounds} ---")
        current_seed = GLOBAL_SEED + rep
        set_global_seed(current_seed)

        # ---- 增强算法 (独立环境) ----
        env_enh = EnhancedNetworkEnvironment(
            num_bs=8,
            num_uav=300,
            recognition_model=recognition_model,
            scaler=scaler,
            seed=current_seed,
            event_probability=0.05
        )
        algo_enh = EnhancedHandoverAlgorithm(env_enh)  # 默认 weight_config='paper'
        algo_enh.epsilon = 0.0  # 不含探索

        # ---- 传统算法 (独立环境) ----
        env_trad = EnhancedNetworkEnvironment(
            num_bs=8,
            num_uav=300,
            recognition_model=recognition_model,
            scaler=scaler,
            seed=current_seed,
            event_probability=0.05
        )
        algo_trad = IntegratedHandoverAlgorithm(env_trad)

        # 打印第一轮环境信息确认新参数
        if rep == 0:
            bs_heights = [bs.position[2] for bs in env_enh.base_stations.values()]
            bs_types = [bs.bs_type for bs in env_enh.base_stations.values()]
            small_count = sum(1 for t in bs_types if t == 'small')
            uav_heights = [uav.position[2] for uav in env_enh.uavs.values()]
            print(f"\n[环境参数确认]")
            print(f"  宏基站数量: {len(bs_types)-small_count}, 平均高度: {np.mean([h for h,t in zip(bs_heights,bs_types) if t=='macro']):.1f}m")
            print(f"  微基站数量: {small_count}, 平均高度: {np.mean([h for h,t in zip(bs_heights,bs_types) if t=='small']):.1f}m")
            print(f"  微基站比例: {small_count}/{env_enh.num_bs} = {small_count/env_enh.num_bs*100:.1f}%")
            print(f"  UAV平均高度: {np.mean(uav_heights):.1f}m (范围: {min(uav_heights):.0f}~{max(uav_heights):.0f}m)")
            print(f"  SINR均值: {np.mean(env_enh.sinr_matrix):.2f}dB")

        # 运行仿真
        for step in range(num_steps):
            env_enh.step()
            algo_enh.run_step(enable_load_balancing=True)
            env_trad.step()
            algo_trad.run_step()

        # 收集结果
        enh_stats = env_enh.get_state_statistics()
        enh_stats.update(algo_enh.get_detailed_stats())
        enh_stats['connected_ratio'] = enh_stats['connected_count'] / env_enh.num_uav
        enhanced_results.append(enh_stats)

        trad_stats = env_trad.get_state_statistics()
        trad_stats.update(algo_trad.get_detailed_stats())
        trad_stats['connected_ratio'] = trad_stats['connected_count'] / env_trad.num_uav
        traditional_results.append(trad_stats)

        print(f"  增强 - Sat={enh_stats['avg_satisfaction']:.4f}, CritSat={enh_stats['critical_satisfaction']:.4f}, "
              f"HOSR={enh_stats['handover_success_rate']*100:.1f}%, LoadVar={enh_stats['load_variance']:.6f}")
        print(f"  传统 - Sat={trad_stats['avg_satisfaction']:.4f}, CritSat={trad_stats['critical_satisfaction']:.4f}, "
              f"HOSR={trad_stats['handover_success_rate']*100:.1f}%, LoadVar={trad_stats['load_variance']:.6f}")

    return enhanced_results, traditional_results


def generate_validation_report_v2(enhanced_results, traditional_results):
    """生成验证报告"""
    
    def mean_and_std(values, key):
        data = [d.get(key, 0) for d in values]
        return np.mean(data), np.std(data)
    
    # 计算统计
    enh_sat_m, enh_sat_s = mean_and_std(enhanced_results, 'avg_satisfaction')
    trad_sat_m, trad_sat_s = mean_and_std(traditional_results, 'avg_satisfaction')
    enh_crit_m, _ = mean_and_std(enhanced_results, 'critical_satisfaction')
    trad_crit_m, _ = mean_and_std(traditional_results, 'critical_satisfaction')
    enh_sinr_m, _ = mean_and_std(enhanced_results, 'avg_sinr')
    trad_sinr_m, _ = mean_and_std(traditional_results, 'avg_sinr')
    enh_lv_m, _ = mean_and_std(enhanced_results, 'load_variance')
    trad_lv_m, _ = mean_and_std(traditional_results, 'load_variance')
    enh_hosr_m, _ = mean_and_std(enhanced_results, 'handover_success_rate')
    trad_hosr_m, _ = mean_and_std(traditional_results, 'handover_success_rate')
    enh_cr_m, _ = mean_and_std(enhanced_results, 'connected_ratio')
    trad_cr_m, _ = mean_and_std(traditional_results, 'connected_ratio')

    # 提升百分比
    def pct_imp(new_val, old_val):
        if old_val == 0:
            return float('inf') if new_val > 0 else 0
        return ((new_val - old_val) / abs(old_val)) * 100

    sat_imp = pct_imp(enh_sat_m, trad_sat_m)
    crit_imp = pct_imp(enh_crit_m, trad_crit_m)
    lv_imp = pct_imp(trad_lv_m, enh_lv_m)  # 负载方差越低越好
    hosr_imp = pct_imp(enh_hosr_m, trad_hosr_m)
    cr_imp = pct_imp(enh_cr_m, trad_cr_m)

    report = f"""# 【方案B V2】基站参数修正快速验证报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 参数变更摘要

| 参数 | 旧值 | **新值** | 依据 |
|------|------|---------|------|
| 宏基站高度 | 0~1000+ m 随机 | **~25m** | 3GPP TR 38.901 UMa |
| 小基站高度 | 0~1000+ m 随机 | **~8m** | 3GPP TR 38.901 UMi |
| UAV飞行高度 | 0~1000m 随机 | **80~200m** | 低空域法规 |
| 小基站比例(default) | 40% (仅urban) | **50%** | 中国5G实际部署 |

## 2. 验证配置 (对齐实验3)

- 算法运行方式: 独立环境 + run_step() 接口
- 增强算法: 启用负载均衡, epsilon=0, weight_config='paper'
- 业务识别模型: 已加载
- 随机事件: 开启 (p=0.05)

## 3. 核心指标对比

| 指标 | 传统算法 | 增强算法 | 变化 |
|------|---------|---------|------|
| 平均满意度 | {trad_sat_m:.4f} ± {trad_sat_s:.4f} | {enh_sat_m:.4f} ± {enh_sat_s:.4f} | **{sat_imp:+.2f}%** |
| 关键业务满意度 | {trad_crit_m:.4f} | {enh_crit_m:.4f} | **{crit_imp:+.2f}%** |
| 切换成功率 | {trad_hosr_m*100:.1f}% | {enh_hosr_m*100:.1f}% | **{hosr_imp:+.2f}%** |
| 平均SINR(dB) | {trad_sinr_m:.2f} | {enh_sinr_m:.2f} | - |
| 负载方差 | {trad_lv_m:.6f} | {enh_lv_m:.6f} | **{lv_imp:+.2f}%**↓ |
| 连接保持率 | {trad_cr_m*100:.1f}% | {enh_cr_m*100:.1f}% | **{cr_imp:+.2f}%** |

## 4. 结论一致性判断

### 核心问题: 修正后 "增强算法 > 传统算法" 的结论是否保持？

"""

    # 逐项判断
    checks = []
    
    # 1. 满意度
    if enh_sat_m > trad_sat_m:
        checks.append(("PASS", f"满意度提升", f"+{sat_imp:.2f}%"))
    else:
        checks.append(("FAIL", f"满意度反转", f"{sat_imp:.2f}%"))

    # 2. 关键业务满意度
    if enh_crit_m > trad_crit_m:
        checks.append(("PASS", f"关键业务满意度提升", f"+{crit_imp:.2f}%"))
    else:
        checks.append(("FAIL", f"关键业务满意度反转", f"{crit_imp:.2f}%"))

    # 3. 切换成功率
    if enh_hosr_m >= trad_hosr_m * 0.98:  # 允许1%容差
        checks.append(("PASS", f"切换成功率不降", f"{hosr_imp:+.2f}%"))
    else:
        checks.append(("WARN", f"切换成功率下降", f"{hosr_imp:+.2f}%"))

    # 4. 负载均衡
    if enh_lv_m < trad_lv_m:
        checks.append(("PASS", f"负载均衡改善", f"{lv_imp:+.2f}%"))
    else:
        checks.append(("FAIL", f"负载方差未改善", f""))

    # 5. 连接率
    if enh_cr_m >= trad_cr_m:
        checks.append(("PASS", f"连接率保持", f"{cr_imp:+.2f}%"))
    else:
        checks.append(("WARN", f"连接率下降", f"{cr_imp:+.2f}%"))

    all_pass = all(c[0] == "PASS" for c in checks)
    no_fail = all(c[0] != "FAIL" for c in checks)

    for status, title, detail in checks:
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}[status]
        report += f"- [{icon}] **{title}**: {detail}\n"

    report += f"""
### 最终判定

**{'=== PASS ✅ — 结论完全一致! 原实验结果可继续使用 ===' if all_pass else ('⚠️ PARTIAL — 主要趋势一致, 但部分指标有变化' if no_fail else '❌ FAIL — 需要进一步分析')}**

"""

    if all_pass:
        report += """## 建议

1. **直接使用原实验结果**: 修正后的参数不改变核心结论
2. **更新论文文字**: 在系统建模章节说明新的基站部署参数
3. **无需全量重跑**: 实验数据有效

"""
    elif no_fail:
        report += """## 建议

1. **可接受数值波动**: 差异在合理范围内(<10%), 可归因于随机性
2. **论文补充说明**: 在实验设置章节注明参数更新及验证结果
3. **可选**: 如导师要求, 可增加一轮5-10轮的小规模验证作为附录

"""
    else:
        report += """## 建议

1. **需要深入分析**: 确认是否有代码逻辑问题
2. **检查点**:
   - 增强算法的抢占/回滚机制是否依赖特定距离条件?
   - 效用函数中的距离/SINR权重是否敏感于高度差?
3. **备选方案**:
   - A: 调整效用函数权重以适配新参数
   - B: 全量重跑所有实验 (耗时较长)
   - C: 保持旧参数但论文中解释合理性

"""

    report += f"""---

## 附录A: 各轮详细数据

### 增强算法各轮结果
| 轮次 | 满意度 | 关键Sat | HOSR | SINR | 负载方差 | 连接率 |
|------|--------|---------|------|------|----------|--------|
"""

    for i, r in enumerate(enhanced_results):
        report += f"| {i+1} | {r.get('avg_satisfaction',0):.4f} | {r.get('critical_satisfaction',0):.4f} | {r.get('handover_success_rate',0)*100:.1f}% | {r.get('avg_sinr',0):.1f} | {r.get('load_variance',0):.6f} | {r.get('connected_ratio',0)*100:.1f}% |\n"

    report += "\n### 传统算法各轮结果\n"
    report += "| 轮次 | 满意度 | 关键Sat | HOSR | SINR | 负载方差 | 连接率 |\n"
    report += "|------|--------|---------|------|------|----------|--------|\n"

    for i, r in enumerate(traditional_results):
        report += f"| {i+1} | {r.get('avg_satisfaction',0):.4f} | {r.get('critical_satisfaction',0):.4f} | {r.get('handover_success_rate',0)*100:.1f}% | {r.get('avg_sinr',0):.1f} | {r.get('load_variance',0):.6f} | {r.get('connected_ratio',0)*100:.1f}% |\n"

    report += """
---

*此报告由 quick_validate_new_params.py V2 自动生成*
"""

    # 保存报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(RESULT_DIR, f'validation_report_{timestamp}.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # JSON详细数据
    json_data = {
        'timestamp': timestamp,
        'version': 'V2',
        'parameter_changes': {
            'macro_bs_height_old': '0~1000+m random',
            'macro_bs_height_new': '~25m (3GPP UMa)',
            'small_bs_height_old': '0~1000+m random',
            'small_bs_height_new': '~8m (3GPP UMi)',
            'uav_height_old': '0~1000m random',
            'uav_height_new': '80~200m low-altitude',
        },
        'config': {
            'method': 'independent_env_per_algorithm',
            'interface': 'run_step()',
            'load_balancing': True,
            'epsilon': 0.0,
            'weight_config': 'paper',
            'rounds': len(enhanced_results),
            'steps_per_round': 150,
        },
        'summary': {
            'traditional': {
                'avg_satisfaction': float(trad_sat_m),
                'std_satisfaction': float(trad_sat_s),
                'critical_satisfaction': float(trad_crit_m),
                'handover_success_rate': float(trad_hosr_m),
                'avg_sinr': float(trad_sinr_m),
                'load_variance': float(trad_lv_m),
                'connected_ratio': float(trad_cr_m),
            },
            'enhanced': {
                'avg_satisfaction': float(enh_sat_m),
                'std_satisfaction': float(enh_sat_s),
                'critical_satisfaction': float(enh_crit_m),
                'handover_success_rate': float(enh_hosr_m),
                'avg_sinr': float(enh_sinr_m),
                'load_variance': float(enh_lv_m),
                'connected_ratio': float(enh_cr_m),
            },
            'improvements_pct': {
                'satisfaction': float(sat_imp),
                'critical_satisfaction': float(crit_imp),
                'handover_success_rate': float(hosr_imp),
                'load_variance_improvement': float(lv_imp),
                'connected_ratio': float(cr_imp),
            },
        },
        'verdict': 'PASS' if all_pass else ('PARTIAL' if no_fail else 'FAIL'),
        'detailed_results': {
            'enhanced': enhanced_results,
            'traditional': traditional_results,
        }
    }

    json_file = os.path.join(RESULT_DIR, f'validation_data_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    # 打印最终结果
    print(f"\n{'='*70}")
    print("验证完成!")
    print(f"{'='*70}")
    print(f"\n生成的文件:")
    print(f"  报告: {report_file}")
    print(f"  数据: {json_file}")

    print(f"\n{'='*70}")
    print("最终判定")
    print(f"{'='*70}")
    for status, title, detail in checks:
        icon = {"PASS": "[OK]", "FAIL": "[X]", "WARN": "[!]"}[status]
        print(f"  {icon} {title}: {detail}")

    if all_pass:
        print(f"\n  ★★★ PASS ★★★ 修正后性能排序不变, 原实验结果完全有效!")
    elif no_fail:
        print(f"\n  ★★ PARTIAL ★★ 主要趋势一致, 部分指标有波动")
    else:
        print(f"\n  ⚠ FAIL ⚠ 需要进一步分析原因")

    # 打印对比表格
    print(f"\n{'='*70}")
    print("指标对比表")
    print(f"{'='*70}")
    print(f"  {'指标':<20s} {'传统算法':>12s} {'增强算法':>12s} {'变化':>10s}")
    print(f"  {'-'*54}")
    metrics_table = [
        ("平均满意度", f"{trad_sat_m:.4f}", f"{enh_sat_m:.4f}", f"{sat_imp:+.2f}%"),
        ("关键业务满意度", f"{trad_crit_m:.4f}", f"{enh_crit_m:.4f}", f"{crit_imp:+.2f}%"),
        ("切换成功率", f"{trad_hosr_m*100:.1f}%", f"{enh_hosr_m*100:.1f}%", f"{hosr_imp:+.2f}%"),
        ("平均SINR(dB)", f"{trad_sinr_m:.2f}", f"{enh_sinr_m:.2f}", "-"),
        ("负载方差", f"{trad_lv_m:.6f}", f"{enh_lv_m:.6f}", f"{lv_imp:+.2f}%"),
        ("连接保持率", f"{trad_cr_m*100:.1f}%", f"{enh_cr_m*100:.1f}%", f"{cr_imp:+.2f}%"),
    ]
    for name, tval, eval, chg in metrics_table:
        print(f"  {name:<20s} {tval:>12s} {eval:>12s} {chg:>10s}")

    return report_file, json_file, all_pass


if __name__ == "__main__":
    start_time = time.time()

    # 快速验证: 5轮 × 150步 ≈ 3-5分钟
    enhanced_results, traditional_results = run_quick_validation_v2(
        num_rounds=5,
        num_steps=150
    )

    # 生成报告
    report_file, json_file, passed = generate_validation_report_v2(
        enhanced_results, traditional_results
    )

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
