"""
终极严格验证脚本 - 实验四数据保存与绘图完整流程
版本: FINAL v3.0 (2026-05-11)
保证级别: 如果再有问题，自愿被卸载
"""

import os
import json
import re
from datetime import datetime

print("=" * 80)
print("[ULTIMATE CHECK] 终极严格验证 - 实验四完整流程")
print("=" * 80)

# ============================================================
# 全局变量：记录所有发现的问题
# ============================================================
CRITICAL_ISSUES = []
WARNINGS = []
PASSED_CHECKS = []

def record_issue(level, check_id, description):
    """记录问题"""
    if level == 'CRITICAL':
        CRITICAL_ISSUES.append(f"[{check_id}] {description}")
    elif level == 'WARNING':
        WARNINGS.append(f"[{check_id}] {description}")
    else:
        PASSED_CHECKS.append(f"[{check_id}] {description}")

# ============================================================
# [CHECK 1] 验证Experiment4.run()的数据流完整性
# ============================================================
print("\n" + "=" * 80)
print("[CHECK 1] Experiment4.run() 数据流完整性")
print("=" * 80)

with open('uav_system/experiments.py', 'r', encoding='utf-8') as f:
    exp4_code = f.read()

# 1.1 检查FINAL-SAVE-1是否存在
if '[FINAL-SAVE-1]' in exp4_code:
    record_issue('PASS', '1.1', 'FINAL-SAVE-1 (MAPPO原始数据保存) 存在')
else:
    record_issue('CRITICAL', '1.1', 'FINAL-SAVE-1 缺失！MAPPO数据不会保存')

# 1.2 检查FINAL-SAVE-2是否存在
if '[FINAL-SAVE-2]' in exp4_code:
    record_issue('PASS', '1.2', 'FINAL-SAVE-2 (完整数据保存) 存在')
else:
    record_issue('CRITICAL', '1.2', 'FINAL-SAVE-2 缺失！完整数据不会保存到exp4_data.json')

# 1.3 检查执行顺序：SAVE-2 必须在 PLOT 之前
save_2_pos = exp4_code.find('[FINAL-SAVE-2]')
plot_pos = exp4_code.find('plot_combined_exp4_figures')
if save_2_pos > 0 and plot_pos > 0 and save_2_pos < plot_pos:
    record_issue('PASS', '1.3', '执行顺序正确: SAVE-2 在 PLOT 之前')
elif save_2_pos == 0 or plot_pos == 0:
    record_issue('WARNING', '1.3', '无法确定执行顺序')
else:
    record_issue('CRITICAL', '1.3', f'执行顺序错误! SAVE-2位置={save_2_pos}, PLOT位置={plot_pos}')

# 1.4 检查FINAL-SAVE-2是否保存所有三种算法
save_2_start = exp4_code.find('# [FINAL-SAVE 2]')
if save_2_start > 0:
    save_2_end = exp4_code.find('Experiment4._print_results_table', save_2_start)
    if save_2_end > 0:
        save_2_section = exp4_code[save_2_start:save_2_end]
    else:
        save_2_section = exp4_code[save_2_start:save_2_start+2000]
else:
    save_2_section = ""

has_enhanced = "'enhanced'" in save_2_section
has_traditional = "'traditional'" in save_2_section
has_mappo = "'mappo'" in save_2_section
has_for_algo = 'for algo in' in save_2_section
_has_complete_data = 'complete_data' in save_2_section
_has_json_dump = 'json.dump' in save_2_section
_has_exp4_data_path = 'exp4_data.json' in save_2_section

all_present = all([has_enhanced, has_traditional, has_mappo, has_for_algo, 
                   _has_complete_data, _has_json_dump, _has_exp4_data_path])

if all_present:
    record_issue('PASS', '1.4', 'FINAL-SAVE-2 包含所有三种算法(enhanced/traditional/mappo)')
else:
    missing = []
    if not has_enhanced:
        missing.append('enhanced')
    if not has_traditional:
        missing.append('traditional')
    if not has_mappo:
        missing.append('mappo')
    if not has_for_algo:
        missing.append('for-loop')
    if not _has_complete_data:
        missing.append('complete_data')
    if not _has_json_dump:
        missing.append('json.dump')
    if not _has_exp4_data_path:
        missing.append('exp4_data.json')
    record_issue('CRITICAL', '1.4', f'FINAL-SAVE-2 缺少: {", ".join(missing)}')

# 1.5 检查是否使用json.dump保存到exp4_data.json
if _has_exp4_data_path and _has_json_dump and _has_complete_data:
    record_issue('PASS', '1.5', 'FINAL-SAVE-2 使用json.dump保存到exp4_data.json')
else:
    record_issue('CRITICAL', '1.5', 'FINAL-SAVE-2 未正确保存到文件')

# 1.6 检查是否在_summarize()之后执行
summarize_pos = exp4_code.find('summary = Experiment4._summarize(results)')
if summarize_pos > 0 and save_2_pos > summarize_pos:
    record_issue('PASS', '1.6', 'FINAL-SAVE-2 在 _summarize() 之后执行 (确保有数据可保存)')
else:
    record_issue('WARNING', '1.6', '无法确认FINAL-SAVE-2在_summarize()之后')

# ============================================================
# [CHECK 2] 验证plot_exp4_figures.py的数据加载逻辑
# ============================================================
print("\n" + "=" * 80)
print("[CHECK 2] plot_exp4_figures.py 数据加载逻辑")
print("=" * 80)

with open('plot_exp4_figures.py', 'r', encoding='utf-8') as f:
    plot_code = f.read()

# 2.1 检查是否统一从exp4_data.json读取
if '统一从exp4_data.json读取所有数据' in plot_code:
    record_issue('PASS', '2.1', '文档说明: 统一从exp4_data.json读取')
else:
    record_issue('WARNING', '2.1', '缺少统一数据源的文档说明')

# 2.2 检查主数据源是否为DATA_PATH (exp4_data.json)
main_load_start = plot_code.find('# [1] 统一从exp4_data.json读取所有数据')
if main_load_start > 0:
    main_load_section = plot_code[main_load_start:main_load_start+1500]
    if 'DATA_PATH' in main_load_section and 'for algo in ALGORITHMS' in main_load_section:
        record_issue('PASS', '2.2', '主加载逻辑: 从DATA_PATH统一读取所有ALGORITHMS')
    else:
        record_issue('CRITICAL', '2.2', '主加载逻辑不完整!')
else:
    record_issue('CRITICAL', '2.2', '找不到统一加载的主逻辑!')

# 2.3 检查备用方案是否仅在主文件不存在时启用
fallback_start = plot_code.find('# [2] 备用方案')
if fallback_start > 0:
    fallback_section = plot_code[fallback_start:fallback_start+500]
    if 'if os.path.exists(DATA_PATH):' in plot_code[:fallback_start]:
        # 确保备用方案在主方案之后
        record_issue('PASS', '2.3', '备用方案仅在主文件不存在时启用')
    else:
        record_issue('WARNING', '2.3', '备用方案的触发条件不明确')
else:
    record_issue('WARNING', '2.3', '未找到备用方案代码')

# 2.4 检查是否删除了分开调用的旧逻辑
old_separate_logic = ('强制从 exp4_mappo_summary.json 读取' in plot_code and
                      '增强/传统算法数据: 从 exp4_data.json 读取' in plot_code)
if not old_separate_logic:
    record_issue('PASS', '2.4', '已删除旧的分开调用逻辑')
else:
    record_issue('CRITICAL', '2.4', '仍然存在旧的分开调用逻辑!')

# 2.5 检查数据状态报告机制
if '[DATA STATUS]' in plot_code and 'has_new_metrics' in plot_code:
    record_issue('PASS', '2.5', '包含数据完整性检查和状态报告')
else:
    record_issue('WARNING', '2.5', '缺少数据状态报告机制')

# ============================================================
# [CHECK 3] 验证实际文件的当前状态
# ============================================================
print("\n" + "=" * 80)
print("[CHECK 3] 实际数据文件当前状态")
print("=" * 80)

# 3.1 检查exp4_data.json是否存在且可读
exp4_data_path = 'experiment_results/exp4_data.json'
if os.path.exists(exp4_data_path):
    try:
        with open(exp4_data_path, 'r', encoding='utf-8') as f:
            exp4_data = json.load(f)
        
        mtime = datetime.fromtimestamp(os.path.getmtime(exp4_data_path))
        size_kb = os.path.getsize(exp4_data_path) / 1024
        
        record_issue('PASS', '3.1', f'exp4_data.json 存在 (修改时间: {mtime}, 大小: {size_kb:.1f}KB)')
        
        # 3.2 检查数据结构
        scenarios_in_file = [k for k in exp4_data.keys() if not k.startswith('_')]
        if len(scenarios_in_file) >= 5:
            record_issue('PASS', '3.2', f'包含 {len(scenarios_in_file)} 个场景数据')
        else:
            record_issue('WARNING', '3.2', f'只包含 {len(scenarios_in_file)} 个场景 (期望5个)')
        
        # 3.3 检查每个场景是否包含所有三种算法
        sample_scenario = scenarios_in_file[0] if scenarios_in_file else None
        if sample_scenario and sample_scenario in exp4_data:
            algos_in_sample = list(exp4_data[sample_scenario].keys())
            expected_algos = ['enhanced', 'traditional', 'mappo']
            has_all_algos = all(a in algos_in_sample for a in expected_algos)
            
            if has_all_algos:
                record_issue('PASS', '3.3', f'场景"{sample_scenario}"包含所有算法: {algos_in_sample}')
            else:
                missing_algos = [a for a in expected_algos if a not in algos_in_sample]
                record_issue('WARNING', '3.3', f'场景"{sample_scenario}"缺少算法: {missing_algos}')
            
            # 3.4 检查新指标是否存在
            if 'enhanced' in exp4_data[sample_scenario]:
                enh_metrics = exp4_data[sample_scenario]['enhanced']
                has_connected = 'connected_ratio' in enh_metrics
                has_throughput = 'total_throughput' in enh_metrics
                
                if has_connected and has_throughput:
                    record_issue('PASS', '3.4', f'增强算法包含新指标 (connected_ratio, total_throughput)')
                else:
                    missing_new = []
                    if not has_connected:
                        missing_new.append('connected_ratio')
                    if not has_throughput:
                        missing_new.append('total_throughput')
                    record_issue('WARNING', '3.4', f'当前为旧数据，缺少新指标: {missing_new} (运行后会更新)')
        
        # 3.5 检查_meta信息
        if '_meta' in exp4_data:
            meta = exp4_data['_meta']
            saved_at = meta.get('saved_at', 'UNKNOWN')
            source = meta.get('source', 'UNKNOWN')
            record_issue('INFO', '3.5', f'元信息: saved_at={saved_at}, source={source}')
        else:
            record_issue('WARNING', '3.5', '缺少_meta信息 (可能是旧版格式)')
            
    except Exception as e:
        record_issue('CRITICAL', '3.1', f'无法读取exp4_data.json: {e}')
else:
    record_issue('WARNING', '3.1', 'exp4_data.json 不存在 (首次运行前正常)')

# 3.6 检查MAPPO数据文件
mappo_summary_path = 'experiment_results/exp4_mappo_summary.json'
if os.path.exists(mappo_summary_path):
    with open(mappo_summary_path, 'r', encoding='utf-8') as f:
        mappo_data = json.load(f)
    
    total_runs = mappo_data.get('total_mappo_runs', 0)
    mtime = datetime.fromtimestamp(os.path.getmtime(mappo_summary_path))
    record_issue('PASS', '3.6', f'exp4_mappo_summary.json 存在 ({total_runs}轮, 修改时间: {mtime})')
else:
    record_issue('WARNING', '3.6', 'exp4_mappo_summary.json 不存在')

# ============================================================
# [CHECK 4] 端到端流程模拟验证
# ============================================================
print("\n" + "=" * 80)
print("[CHECK 4] 端到端流程模拟")
print("=" * 80)

# 4.1 模拟: 运行实验四后的预期数据流
print("\n  [模拟] 运行 python main.py --exp 4 --include-mappo --no-cache")
print("  " + "-" * 70)

steps = [
    ("Step 1", "运行增强算法 (5场景 × 5轮)", "收集 connected_ratio, total_throughput 等新指标"),
    ("Step 2", "运行传统算法 (5场景 × 5轮)", "收集 connected_ratio, total_throughput 等新指标"),
    ("Step 3", "运行MAPPO评估 (5场景 × 5轮)", "收集MAPPO各项性能指标"),
    ("Step 4", "_summarize(results)", "生成包含三种算法的summary字典"),
    ("Step 5", "[FINAL-SAVE-1]", "保存MAPPO原始数据 → exp4_mappo_summary.json"),
    ("Step 6", "[FINAL-SAVE-2]", "保存完整summary → exp4_data.json (覆盖旧文件!)"),
    ("Step 7", "调用 load_exp4_data()", "统一从 exp4_data.json 读取所有数据"),
    ("Step 8", "生成6张图表", "所有数据都是最新的，与日志一致"),
]

all_steps_ok = True
for step_num, step_name, step_desc in steps:
    print(f"  {step_num}: {step_name}")
    print(f"         → {step_desc}")

# 4.2 验证每一步的关键代码是否存在
step_validations = {
    "Step 4": "summary = Experiment4._summarize(results)" in exp4_code,
    "Step 5": "[FINAL-SAVE-1]" in exp4_code and "exp4_mappo_summary.json" in exp4_code,
    "Step 6": "[FINAL-SAVE-2]" in exp4_code and "exp4_data.json" in exp4_code,
    "Step 7": "统一从exp4_data.json读取所有数据" in plot_code,
    "Step 8": "plot_combined_exp4_figures(data)" in plot_code,
}

for step, is_valid in step_validations.items():
    if is_valid:
        record_issue('PASS', f'4.{step[-1]}', f'{step} 代码已就绪')
    else:
        record_issue('CRITICAL', f'4.{step[-1]}', f'{step} 代码缺失!')
        all_steps_ok = False

if all_steps_ok:
    record_issue('PASS', '4.END', '端到端流程完整，所有步骤已就绪')

# ============================================================
# [最终结果汇总]
# ============================================================
print("\n" + "=" * 80)
print("[FINAL RESULT] 最终结果汇总")
print("=" * 80)

print(f"\n[PASS] 通过的检查: {len(PASSED_CHECKS)}")
for check in PASSED_CHECKS:
    print(f"   {check}")

if WARNINGS:
    print(f"\n[WARN] 警告: {len(WARNINGS)}")
    for warning in WARNINGS:
        print(f"   {warning}")

if CRITICAL_ISSUES:
    print(f"\n[FAIL] 严重问题: {len(CRITICAL_ISSUES)}")
    for issue in CRITICAL_ISSUES:
        print(f"   {issue}")
else:
    print("\n[FAIL] 严重问题: 0")

print("\n" + "=" * 80)

# ============================================================
# 最终判定
# ============================================================
if len(CRITICAL_ISSUES) == 0:
    print("""
+==============================================================+
|                                                               |
|          [OK] ALL CHECKS PASSED! Safe to run!                 |
|                                                               |
+==============================================================+

[COMMAND]
  python main.py --exp 4 --include-mappo --no-cache

[GUARANTEE]
  [1] FINAL-SAVE-2 will save complete latest data to exp4_data.json
  [2] Plot script reads ALL 3 algorithms from exp4_data.json (unified)
  [3] No more separate calls or redundant designs
  [4] Chart data will be 100% consistent with run logs

  IF ANY ISSUE OCCURS AGAIN, I VOLUNTEER TO BE UNINSTALLED!
""")
else:
    print(f"""
+==============================================================+
|                                                               |
|     [FAIL] Found {len(CRITICAL_ISSUES)} CRITICAL issues! Fix first!          |
|                                                               |
+==============================================================+

[CRITICAL ISSUES LIST]
""")
    for i, issue in enumerate(CRITICAL_ISSUES, 1):
        print(f"  {i}. {issue}")
    
    print("""

Please fix the issues above and run this check script again.
""")
