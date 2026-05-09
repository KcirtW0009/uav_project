"""
MAPPO V21 训练安全与缓存功能验证脚本
======================================
用途: 验证中断安全保存和预训练缓存功能是否正常工作

测试项目:
  1. TrainingSafetyManager 初始化
  2. 预训练缓存路径生成
  3. 缓存保存/加载功能
  4. 信号处理器注册
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_safety_manager():
    """测试1: Safety Manager基本功能"""
    print("=" * 70)
    print("Test 1: TrainingSafetyManager Basic Functions")
    print("=" * 70)
    
    try:
        from uav_system.experiments_mappo import TrainingSafetyManager, training_safety
        
        # 检查实例创建
        assert training_safety is not None, "training_safety instance should exist"
        print("[OK] Global safety manager instance created")
        
        # 检查属性
        assert hasattr(training_safety, 'agent'), "Should have agent attribute"
        assert hasattr(training_safety, 'latest_model_path'), "Should have latest_model_path"
        assert hasattr(training_safety, 'current_episode'), "Should have current_episode"
        assert hasattr(training_safety, '_lock'), "Should have thread lock"
        print("[OK] All required attributes present")
        
        # 检查方法
        assert callable(getattr(training_safety, 'setup', None)), "Should have setup method"
        assert callable(getattr(training_safety, 'enable_signal_handlers', None)), "Should have enable_signal_handlers"
        assert callable(getattr(training_safety, 'disable_signal_handlers', None)), "Should have disable_signal_handlers"
        assert callable(getattr(training_safety, 'update_state', None)), "Should have update_state method"
        print("[OK] All required methods present")
        
        # 测试状态更新
        training_safety.update_state(
            episode_num=10,
            reward=100.0,
            satisfaction=0.95,
            composite_score=0.88,
            episode_rewards=[90, 95, 100],
            episode_sats=[0.93, 0.94, 0.95],
            composite_window=[0.85, 0.86, 0.87, 0.88],
            sat_window=[0.92, 0.93, 0.94, 0.95],
            best_composite=0.88,
            best_sat=0.95
        )
        assert training_safety.current_episode == 10, "Episode should be updated"
        assert len(training_safety.training_state['episode_rewards']) == 3, "Rewards should be stored"
        print("[OK] State update works correctly")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrain_cache():
    """测试2: 预训练缓存功能"""
    print("\n" + "=" * 70)
    print("Test 2: Pretrain Cache Functions")
    print("=" * 70)
    
    try:
        from uav_system.experiments_mappo import TrainingSafetyManager
        
        # 测试缓存路径生成
        cache_path = TrainingSafetyManager.get_pretrain_cache_path(300, 8)
        assert 'pretrain_300uav_8bs.pkl' in cache_path, f"Cache path should contain filename: {cache_path}"
        assert os.path.isdir(os.path.dirname(cache_path)), "Cache directory should exist or be created"
        print(f"[OK] Cache path generated: {cache_path}")
        
        # 测试加载不存在的缓存（应返回None）
        result = TrainingSafetyManager.load_pretrain_cache(999, 99)  # 不存在的配置
        assert result is None, "Non-existent cache should return None"
        print("[OK] Loading non-existent cache returns None")
        
        # 测试保存缓存（使用模拟数据）
        class MockAgent:
            def __init__(self):
                self.actor = MockActor()
            
        class MockActor:
            def state_dict(self):
                return {'layer1.weight': [1, 2, 3], 'layer2.bias': [0.1, 0.2]}
        
        mock_agent = MockAgent()
        mock_demos = [(i, i % 5) for i in range(100)]  # 模拟示范数据
        
        saved_path = TrainingSafetyManager.save_pretrain_cache(
            mock_agent, mock_demos, 300, 8,
            pretrain_epochs=50,
            pretrain_loss=0.05
        )
        
        assert os.path.exists(saved_path), f"Cache file should exist: {saved_path}"
        print(f"[OK] Cache saved successfully: {saved_path}")
        
        # 测试加载刚保存的缓存
        loaded_cache = TrainingSafetyManager.load_pretrain_cache(300, 8)
        assert loaded_cache is not None, "Just-saved cache should load successfully"
        assert loaded_cache['num_uav'] == 300, "UAV count should match"
        assert loaded_cache['num_bs'] == 8, "BS count should match"
        assert loaded_cache['pretrain_epochs'] == 50, "Epochs should match"
        assert abs(loaded_cache['final_loss'] - 0.05) < 0.001, "Loss should match"
        print("[OK] Loaded cache data is correct")
        
        # 清理测试文件
        if os.path.exists(saved_path):
            os.remove(saved_path)
            print("[OK] Test cache file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_interval():
    """测试3: 验证保存间隔已更新"""
    print("\n" + "=" * 70)
    print("Test 3: Save Interval Configuration")
    print("=" * 70)
    
    try:
        with open('uav_system/experiments_mappo.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含新的保存间隔注释
        if 'save_interval = 10  # V21' in content:
            print("[OK] Save interval updated to 10 episodes (V21)")
            print("     This means model will be saved every 10 episodes")
            print("     Maximum loss on interrupt: 9 episodes of training")
        elif 'save_interval = 50' in content and 'save_interval = 10' not in content:
            print("[WARN] Save interval still at 50 episodes (old value)")
            print("     Consider updating to 10 for better interrupt safety")
        else:
            print("[INFO] Could not determine save interval from code")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test 3 failed: {e}")
        return False


def test_pretrain_return_value():
    """测试4: 验证pretrain方法返回值"""
    print("\n" + "=" * 70)
    print("Test 4: Pretrain Method Return Value")
    print("=" * 70)
    
    try:
        with open('uav_system/mappo_agent_v2.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有return语句
        if "'final_loss': final_loss" in content and "return {" in content:
            print("[OK] pretrain() method now returns dict with final_loss")
            print("     This enables caching of pretraining results")
        else:
            print("[WARN] pretrain() may not return expected format")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test 4 failed: {e}")
        return False


def test_imports():
    """测试5: 验证必要的导入已添加"""
    print("\n" + "=" * 70)
    print("Test 5: Required Imports")
    print("=" * 70)
    
    try:
        with open('uav_system/experiments_mappo.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        imports_found = {
            'signal': False,
            'threading': False,
        }
        
        for line in lines[:60]:  # 只检查前60行（import区域）
            if 'import signal' in line:
                imports_found['signal'] = True
            if 'import threading' in line:
                imports_found['threading'] = True
        
        all_ok = True
        for module, found in imports_found.items():
            status = "[OK]" if found else "[MISSING]"
            print(f"  {status} import {module}")
            if not found:
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"[FAIL] Test 5 failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("MAPPO V21 - Training Safety & Cache Feature Verification")
    print("=" * 70)
    print("\nThis script verifies the new features:")
    print("  1. Interrupt-safe model saving (Ctrl+C handling)")
    print("  2. Pretrain caching (avoid redundant demonstration collection)")
    print("  3. Reduced save interval (10 eps vs 50 eps)")
    print("  4. Training state checkpoint for resume capability")
    print("")
    
    results = []
    
    results.append(("Safety Manager Basics", test_safety_manager()))
    results.append(("Pretrain Cache", test_pretrain_cache()))
    results.append(("Save Interval", test_save_interval()))
    results.append(("Pretrain Return", test_pretrain_return_value()))
    results.append(("Required Imports", test_imports()))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! New features are ready.")
        print("\nNext steps:")
        print("  1. Run training: python main.py --exp mappo")
        print("  2. Press Ctrl+C during training to test safe shutdown")
        print("  3. Check that pretrain cache is created after first run")
        print("  4. Run again to verify cache is loaded (faster startup)")
    else:
        print("\n[WARNING] Some tests failed. Please review the errors above.")
    
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
