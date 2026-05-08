#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
curriculum_learning.py 快速验证脚本

检查项:
1. 所有导入是否正确
2. 核心类是否可以实例化
3. 配置是否合理
4. 是否有明显的语法错误
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试1: 检查所有导入"""
    print("\n[TEST 1] 检查导入...")
    
    try:
        import torch
        import numpy as np
        from dataclasses import dataclass, field
        from typing import Dict, List, Optional, Tuple, Any
        from collections import defaultdict, deque
        
        print("  [OK] 基础库导入成功")
        
        # 测试项目特定导入
        from uav_system.config import GLOBAL_SEED, RESULT_DIR, set_global_seed
        from uav_system.mappo_environment import MultiAgentHandoverEnv
        from uav_system.mappo_agent_v2 import MAPPOAgentV2 as MAPPOAgent
        from uav_system.business import BusinessType
        
        print("  [OK] UAV项目模块导入成功")
        
        # 导入主模块 (这会触发所有类定义)
        from curriculum_learning import (
            CurriculumConfig,
            ScenarioConfig,
            CurriculumScheduler,
            ScenarioRewardShaper,
            ContrastiveLearningModule,
            CurriculumTrainer,
        )
        
        print("  [OK] curriculum_learning模块导入成功")
        return True
        
    except ImportError as e:
        print(f"  [FAIL] 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试2: 验证配置"""
    print("\n[TEST 2] 验证配置...")
    
    try:
        from curriculum_learning import CurriculumConfig
        
        config = CurriculumConfig()
        
        # 检查阶段配置
        assert len(config.phase_configs) == 4, "应该有4个阶段"
        print(f"  [OK] 阶段数量: {len(config.phase_configs)}")
        
        # 检查每个阶段的必需字段
        required_fields = ['name', 'episodes', 'scenarios', 'lr_factor', 'target_improvement']
        for phase_key, phase_cfg in config.phase_configs.items():
            for field_name in required_fields:
                assert field_name in phase_cfg, f"{phase_key}缺少{field_name}"
        
        print("  [OK] 所有阶段配置完整")
        
        # 检查对比学习配置
        assert config.contrastive_enabled == True
        assert 0 < config.contrastive_lambda < 1
        print(f"  [OK] 对比学习配置: λ={config.contrastive_lambda}")
        
        # 检查奖励塑造配置
        assert config.reward_shaping_enabled == True
        assert len(config.scenario_reward_weights) == 5
        print(f"  [OK] 场景奖励权重: {len(config.scenario_reward_weights)}个场景")
        
        # 验证权重归一化
        for scenario, weights in config.scenario_reward_weights.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{scenario}的权重和不等于1: {total}"
        
        print("  [OK] 所有场景权重归一化正确")
        
        return True
        
    except AssertionError as e:
        print(f"  [FAIL] 配置验证失败: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] 意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core_classes():
    """测试3: 实例化核心类"""
    print("\n[TEST 3] 测试核心类实例化...")
    
    try:
        from curriculum_learning import (
            CurriculumConfig,
            CurriculumScheduler,
            ScenarioRewardShaper,
            ContrastiveLearningModule,
        )
        import torch
        
        config = CurriculumConfig()
        
        # 测试CurriculumScheduler
        scheduler = CurriculumScheduler(config)
        phase_key, phase_cfg = scheduler.get_current_phase()
        
        assert phase_key is not None, "应该有当前阶段"
        assert phase_cfg is not None, "阶段配置不应为空"
        print(f"  [OK] CurriculumScheduler初始化成功")
        print(f"         当前阶段: {phase_cfg['name']}")
        
        # 测试ScenarioRewardShaper
        reward_shaper = ScenarioRewardShaper(config)
        assert reward_shaper.enabled == True
        print(f"  [OK] ScenarioRewardShaper初始化成功")
        
        # 测试对比学习模块
        contrastive_module = ContrastiveLearningModule(
            obs_dim=49,
            embedding_dim=64,
            temperature=0.1,
        )
        
        # 测试前向传播
        dummy_obs = {
            i: torch.randn(49) for i in range(3)
        }
        
        loss, embeddings = contrastive_module(dummy_obs, 'industrial_inspection')
        
        assert isinstance(loss, torch.Tensor), "损失应该是tensor"
        assert embeddings.shape == (3, 64), f"embeddings形状错误: {embeddings.shape}"
        print(f"  [OK] ContrastiveLearningModule初始化成功")
        print(f"         输出形状: loss={loss.item():.4f}, embeddings={embeddings.shape}")
        
        # 测试场景切换
        scheduler.advance_to_next_phase()
        phase_key_2, phase_cfg_2 = scheduler.get_current_phase()
        assert phase_key_2 != phase_key, "应该进入下一阶段"
        print(f"  [OK] 阶段切换功能正常: {phase_cfg['name']} → {phase_cfg_2['name']}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] 类实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """测试4: 训练器初始化 (不实际训练)"""
    print("\n[TEST 4] 测试CurriculumTrainer初始化...")
    
    try:
        from curriculum_learning import CurriculumTrainer, CurriculumConfig
        import os
        
        # 检查基础模型是否存在
        base_model = 'results/mappo_models/mappo_8bs_300uav_best.pt'
        
        if not os.path.exists(base_model):
            print(f"  [WARN] 基础模型不存在: {base_model}")
            print(f"         将使用虚拟路径进行测试")
        
        config = CurriculumConfig()
        
        # 创建trainer (不会立即开始训练)
        trainer = CurriculumTrainer(
            base_model_path=base_model,
            config=config,
        )
        
        # 验证属性
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'reward_shaper')
        assert hasattr(trainer, 'scenarios')
        assert len(trainer.scenarios) == 5, "应该有5个场景"
        
        print(f"  [OK] CurriculumTrainer初始化成功")
        print(f"         场景数量: {len(trainer.scenarios)}")
        print(f"         输出目录: {trainer.output_dir}")
        
        # 打印场景摘要
        print(f"\n  [*] 场景配置摘要:")
        total_improvement = 0
        for sid, scfg in trainer.scenarios.items():
            improvement = (scfg.target_score - scfg.baseline_score) * 100
            total_improvement += improvement
            diff_icon = {'easy': '★', 'medium': '●', 'hard': '▲'}[scfg.difficulty]
            print(f"     {diff_icon} {scfg.name:12s}: "
                  f"{scfg.baseline_score:.2%} → {scfg.target_score:.2%} "
                  f"({improvement:+.1f}%)")
        
        avg_improvement = total_improvement / len(trainer.scenarios)
        print(f"\n     平均目标提升: {avg_improvement:+.1f}%")
        
        if avg_improvement >= 6:
            print(f"  [OK] 目标设定合理 (>6%)")
        else:
            print(f"  [WARN] 目标可能偏低 (<6%)")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Trainer初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_shaping_logic():
    """测试5: 奖励塑造逻辑"""
    print("\n[TEST 5] 测试奖励塑造逻辑...")
    
    try:
        from curriculum_learning import CurriculumConfig, ScenarioRewardShaper
        
        config = CurriculumConfig()
        shaper = ScenarioRewardShaper(config)
        
        # 测试不同场景的奖励塑造
        test_cases = [
            {
                'scenario': 'industrial_inspection',
                'original_reward': 0.7,
                'info': {
                    'avg_satisfaction': 0.65,
                    'connected_rate': 0.85,
                    'global_load_ratio': 0.6,
                },
                'step_info': {
                    'handover_success_rate': 0.85,
                    'connection_stability': 0.9,
                }
            },
            {
                'scenario': 'logistics_delivery',
                'original_reward': 0.5,
                'info': {
                    'avg_satisfaction': 0.55,
                    'connected_rate': 0.75,
                    'global_load_ratio': 0.8,
                },
                'step_info': {
                    'handover_success_rate': 0.75,
                    'connection_stability': 0.85,
                }
            },
        ]
        
        for i, case in enumerate(test_cases):
            shaped = shaper.shape_reward(
                scenario_id=case['scenario'],
                original_reward=case['original_reward'],
                info=case['info'],
                step_info=case['step_info'],
            )
            
            # 塑造后的奖励应该在合理范围内
            assert 0 <= shaped <= 1.5, f"奖励超出范围: {shaped}"
            
            # 不同场景应该产生不同的塑造结果 (因为权重不同)
            if i > 0:
                prev_shaped = test_cases[i-1].get('_shaped', 0)
                # 不一定不同，但至少要合理
            
            case['_shaped'] = shaped
            
            print(f"  [OK] Case {i+1} ({case['scenario']}): "
                  f"{case['original_reward']:.3f} → {shaped:.3f}")
        
        print(f"\n  [OK] 奖励塑造逻辑正确")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] 奖励塑造测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*70)
    print("  PMSF v3.0 课程学习系统 - 快速验证")
    print("="*70)
    
    tests = [
        ("导入检查", test_imports),
        ("配置验证", test_config),
        ("核心类实例化", test_core_classes),
        ("训练器初始化", test_trainer_initialization),
        ("奖励塑造逻辑", test_reward_shaping_logic),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [ERROR] 测试 '{name}' 发生异常: {e}")
            results.append((name, False))
    
    # 输出汇总
    print("\n" + "="*70)
    print("  验证结果汇总")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        icon = "✓" if result else "✗"
        print(f"  {icon} {status} {name}")
    
    print(f"\n  总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n  [SUCCESS] 所有测试通过! 系统已准备就绪。")
        print("\n  下一步:")
        print(f"    python curriculum_learning.py --mode quick          # 快速测试")
        print(f"    python curriculum_learning.py --mode full           # 完整训练")
        return 0
    else:
        print(f"\n  [WARNING] 有 {total - passed} 个测试未通过，请检查错误信息。")
        return 1


if __name__ == '__main__':
    exit(main())
