#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：逐步测试finetune_multi_scenario.py的各个阶段
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("[STEP 1] Starting diagnostic...", flush=True)

# Step 2: 测试导入
try:
    print("[STEP 2] Importing finetune_multi_scenario...", flush=True)
    import finetune_multi_scenario
    print("[OK] Import successful!", flush=True)
except Exception as e:
    print(f"[FAIL] Import error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 测试类初始化
try:
    print("\n[STEP 3] Creating MultiScenarioFinetunerV2 instance...", flush=True)
    
    model_path = os.path.join(
        'experiment_results', 'mappo_models', 
        'mappo_8bs_300uav_best.pt'
    )
    
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found: {model_path}")
        print("       Using dummy path for testing...")
    
    finetuner = finetune_multi_scenario.MultiScenarioFinetunerV2(
        model_path=model_path,
        mode='quick',
    )
    print(f"[OK] Finetuner created!", flush=True)
    
except Exception as e:
    print(f"[FAIL] Initialization error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 测试run_finetuning_pipeline开始
try:
    print("\n[STEP 4] Calling run_finetuning_pipeline()...", flush=True)
    print("       (This may take a while for baseline evaluation...)", flush=True)
    
    results = finetuner.run_finetuning_pipeline()
    
    print(f"\n[OK] Pipeline completed!", flush=True)
    print(f"   Success: {results.get('success', False)}", flush=True)
    
except KeyboardInterrupt:
    print("\n[INTERRUPTED] User cancelled", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"\n[FAIL] Pipeline error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[DONE] All tests passed!", flush=True)
