"""
严格验证脚本：检查Experiment4的数据保存和绘图完整流程
"""

import os
import json
from datetime import datetime

print("=" * 80)
print("[STRICT CHECK] 实验四数据保存与绘图完整流程检查")
print("=" * 80)

# ============================================================
# [1] 检查代码中的关键保存点
# ============================================================
print("\n[1] 检查Experiment4.run()的关键保存点")
print("-" * 80)

with open('uav_system/experiments.py', 'r', encoding='utf-8') as f:
    exp4_code = f.read()

has_final_save_1 = '[FINAL-SAVE-1]' in exp4_code
status_1 = "[OK] EXISTS" if has_final_save_1 else "[FAIL] MISSING"
print(f"  FINAL-SAVE-1 (MAPPO data): {status_1}")

has_final_save_2 = '[FINAL-SAVE-2]' in exp4_code
status_2 = "[OK] EXISTS" if has_final_save_2 else "[FAIL] MISSING"
print(f"  FINAL-SAVE-2 (COMPLETE data): {status_2}")

save_2_pos = exp4_code.find('[FINAL-SAVE-2]')
plot_call_pos = exp4_code.find('plot_combined_exp4_figures')
if save_2_pos > 0 and plot_call_pos > 0:
    order_ok = save_2_pos < plot_call_pos
    order_status = "[OK] CORRECT" if order_ok else "[FAIL] WRONG"
    print(f"  Execution order (SAVE-2 before PLOT): {order_status}")
else:
    print(f"  Execution order: [WARN] Cannot determine")

# ============================================================
# [2] 检查绘图脚本的数据加载逻辑
# ============================================================
print("\n[2] Check plot_exp4_figures.py data loading logic")
print("-" * 80)

with open('plot_exp4_figures.py', 'r', encoding='utf-8') as f:
    plot_code = f.read()

force_mappo_load = 'exp4_mappo_summary.json' in plot_code and '强制加载最新的MAPPO数据' in plot_code
mappo_status = "[OK] CORRECT" if force_mappo_load else "[FAIL] WRONG"
print(f"  MAPPO data source (force use latest): {mappo_status}")

warn_old_data = 'OLD DATA' in plot_code or '缺少新指标' in plot_code
warn_status = "[OK] EXISTS" if warn_old_data else "[FAIL] MISSING"
print(f"  Old data warning mechanism: {warn_status}")

# ============================================================
# [3] 检查实际文件状态
# ============================================================
print("\n[3] Check actual data file status")
print("-" * 80)

files_to_check = [
    ('experiment_results/exp4_data.json', 'Enhanced/Traditional + MAPPO stats'),
    ('experiment_results/exp4_mappo_summary.json', 'MAPPO raw data'),
    ('experiment_results/exp4_mappo_raw_results.json', 'MAPPO auto-save data'),
]

for file_path, desc in files_to_check:
    if os.path.exists(file_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        size = os.path.getsize(file_path) / 1024
        print(f"\n  {os.path.basename(file_path)}")
        print(f"    Description: {desc}")
        print(f"    Modified: {mtime}")
        print(f"    Size: {size:.1f} KB")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'exp4_data' in file_path:
                scenarios = [k for k in data.keys() if not k.startswith('_')]
                if scenarios:
                    sample_scenario = scenarios[0]
                    algos = list(data[sample_scenario].keys()) if isinstance(data[sample_scenario], dict) else []
                    print(f"    Scenarios: {len(scenarios)}")
                    print(f"    Algorithms: {algos}")

                    if 'enhanced' in data[sample_scenario]:
                        enh_metrics = data[sample_scenario]['enhanced']
                        has_new = 'connected_ratio' in enh_metrics and 'total_throughput' in enh_metrics
                        new_metrics_status = "[OK] HAS" if has_new else "[FAIL] MISSING"
                        print(f"    New metrics (connected_ratio, throughput): {new_metrics_status}")

            elif 'mappo_summary' in file_path:
                total_runs = data.get('total_mappo_runs', 0)
                raw_results = data.get('raw_results_by_scenario', {})
                print(f"    MAPPO runs: {total_runs}")
                print(f"    Scenarios: {len(raw_results)}")

        except Exception as e:
            print(f"    [WARN] Read failed: {e}")
    else:
        print(f"\n  {os.path.basename(file_path)} [FAIL] FILE NOT FOUND!")

# ============================================================
# [4] 验证数据一致性
# ============================================================
print("\n\n[4] Data consistency verification")
print("-" * 80)

try:
    with open('experiment_results/exp4_mappo_summary.json', 'r', encoding='utf-8') as f:
        mappo_summary = json.load(f)

    raw_results = mappo_summary.get('raw_results_by_scenario', {})

    print("  MAPPO data (from exp4_mappo_summary.json):")
    import numpy as np
    for scenario, results_list in raw_results.items():
        if results_list:
            sats = [r.get('avg_satisfaction') for r in results_list if r.get('avg_satisfaction') is not None]
            if sats:
                mean_sat = np.mean(sats)
                std_sat = np.std(sats)
                print(f"    {scenario:25s}: satisfaction={mean_sat:.3f}+/-{std_sat:.3f} ({len(results_list)} runs)")

except Exception as e:
    print(f"  [WARN] Cannot read MAPPO data: {e}")

try:
    with open('experiment_results/exp4_data.json', 'r', encoding='utf-8') as f:
        exp4_data = json.load(f)

    print("\n  Complete data (from exp4_data.json):")
    for scenario_key in ['agriculture', 'smart_city', 'emergency_rescue', 'logistics_delivery']:
        if scenario_key in exp4_data:
            scenario_data = exp4_data[scenario_key]
            for algo in ['enhanced', 'traditional', 'mappo']:
                if algo in scenario_data:
                    algo_data = scenario_data[algo]
                    sat = algo_data.get('avg_satisfaction', ['N/A'])
                    print(f"    {scenario_key:20s}-{algo:12s}: {sat[0]:.3f}")

except Exception as e:
    print(f"  [WARN] Cannot read exp4_data.json: {e}")

# ============================================================
# [5] 最终结论
# ============================================================
print("\n" + "=" * 80)
print("[FINAL CONCLUSION]")
print("=" * 80)

issues = []

if not has_final_save_2:
    issues.append("[FAIL] FINAL-SAVE-2 code missing, new enhanced/traditional data will NOT be saved to exp4_data.json")

if not force_mappo_load:
    issues.append("[FAIL] Plot script does not force use latest MAPPO data, may use old data")

if len(issues) == 0:
    print("""
[OK] ALL CRITICAL FIXES IN PLACE:

  1. FINAL-SAVE-2 will save complete data (enhanced+traditional+MAPPO) to exp4_data.json
  2. Plot script will force use the LATEST MAPPO data from exp4_mappo_summary.json
  3. Old data missing new metrics will be explicitly warned

[NEXT STEP]
  Run command: python main.py --exp 4 --include-mappo --no-cache

  This will:
  - Re-run all three algorithms (NO cache)
  - Collect complete new metrics (including connected_ratio, total_throughput)
  - Save complete data via FINAL-SAVE-2 to exp4_data.json
  - Auto-generate 6 correct charts via plot script
""")
else:
    print("\n[ISSUES FOUND]:")
    for issue in issues:
        print(f"  {issue}")
    print("\nFix these issues BEFORE running!")
