# -*- coding: utf-8 -*-
"""Quick test for Phase 2 visualization"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test import
from phase2_evaluation import VisualizationGenerator, UnifiedAlgorithmEvaluator
import numpy as np

# Create mock data
mock_results = {
    'scenario_0': {
        'traditional': {'avg_satisfaction': 0.5, 'throughput': 3.0, 'connected_ratio': 1.0,
                       'handover_success_rate': 0.8, 'bs_load_balance': 0.6},
        'enhanced': {'avg_satisfaction': 0.6, 'throughput': 3.5, 'connected_ratio': 0.95,
                   'handover_success_rate': 0.85, 'bs_load_balance': 0.7},
        'mappo': {'avg_satisfaction': 0.7, 'throughput': 4.0, 'connected_ratio': 0.98,
                 'handover_success_rate': 0.9, 'bs_load_balance': 0.8}
    },
    'scenario_1': {
        'traditional': {'avg_satisfaction': 0.4, 'throughput': 2.8, 'connected_ratio': 0.9,
                       'handover_success_rate': 0.7, 'bs_load_balance': 0.5},
        'enhanced': {'avg_satisfaction': 0.55, 'throughput': 3.2, 'connected_ratio': 0.92,
                   'handover_success_rate': 0.8, 'bs_load_balance': 0.65},
        'mappo': {'avg_satisfaction': 0.65, 'throughput': 3.8, 'connected_ratio': 0.96,
                 'handover_success_rate': 0.88, 'bs_load_balance': 0.75}
    }
}

mock_scenarios = [
    {'num_bs': 4, 'num_uav': 10, 'max_steps': 50, 'name': 'Small'},
    {'num_bs': 4, 'num_uav': 20, 'max_steps': 60, 'name': 'Medium'}
]

print("Testing visualization generator...")
viz = VisualizationGenerator(output_dir='test_output')

try:
    result = viz.generate_comprehensive_report(mock_results, mock_scenarios)
    if result:
        print(f"SUCCESS! Report saved to: {result}")
    else:
        print("FAILED - returned None")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
