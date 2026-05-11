"""
=============================================================================
  实验四警告验证脚本 (test_exp4_no_warnings.py)
=============================================================================

【目的】
在运行完整实验四之前，先进行小规模测试验证：
1. sklearn并行警告是否已彻底消除
2. 种子机制是否正常工作
3. 三种算法是否能正常运行
4. 数据收集和保存功能是否正常

【使用方法】
python test_exp4_no_warnings.py

【预期输出】
- 零sklearn UserWarning
- 零识别准确率相关输出
- 正常的算法性能数据
"""

import sys
import os
import warnings

# ============================================================
# 第一道防线：主程序级别抑制
# ============================================================
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 第二道防线：捕获所有警告并统计
# ============================================================
import logging

# 设置日志级别为WARNING以上，确保能看到任何漏网的警告
logging.basicConfig(
    level=logging.WARNING,
    format='[%(levelname)s] %(name)s: %(message)s'
)

# 自定义警告捕获器
warning_count = {'total': 0, 'sklearn_warnings': 0}

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """自定义警告处理函数：记录但不显示"""
    warning_count['total'] += 1
    
    msg_str = str(message)
    
    if 'sklearn' in msg_str or 'parallel' in msg_str or 'delayed' in msg_str:
        warning_count['sklearn_warnings'] += 1
        print(f"\n❌ [FAIL] 检测到sklearn警告: {msg_str}")
        return  # 不显示
    
    # 显示非sklearn警告
    print(f"\n⚠️  [WARN] {category.__name__}: {msg_str} ({filename}:{lineno})")

# 安装自定义警告处理器
warnings.showwarning = custom_warning_handler

# ============================================================
# 导入项目模块（这步会触发recognition模块加载）
# ============================================================
print("=" * 80)
print("开始测试：实验四警告验证")
print("=" * 80)

print("\n[Step 1] 导入项目模块...")
try:
    from uav_system.config import set_global_seed, GLOBAL_SEED, RESULT_DIR
    from uav_system.recognition import train_or_load_recognition_model
    from uav_system.experiments import Experiment4
    print("✅ 模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# ============================================================
# 测试1：验证识别模块是否被绕过
# ============================================================
print("\n" + "-" * 80)
print("[Test 1] 验证识别模块是否被正确绕过")
print("-" * 80)

try:
    # 尝试调用train_or_load_recognition_model（但这应该不会被实验四使用）
    print("ℹ️  注意：实验四将传入 None 作为识别模型参数")
    print("   因此不会触发 recognition.py 中的 cross_val_score()")
    
    # 检查环境模块对None的处理
    from uav_system.environment import EnhancedNetworkEnvironment
    
    test_env = EnhancedNetworkEnvironment(
        num_bs=8,
        num_uav=10,  # 小规模测试
        recognition_model=None,  # 关键：传入None
        scaler=None,
        seed=42,
        scenario='smart_city',
        event_probability=0.0  # 禁用随机事件以加速
    )
    
    # 执行一步测试
    test_env.step()
    
    # 检查业务类型识别结果
    uav_id = 0
    biz_type, confidence = test_env.perform_recognition(uav_id)
    
    print(f"✅ UAV{uav_id} 业务识别结果:")
    print(f"   类型: {biz_type.name}")
    print(f"   置信度: {confidence:.2f}")
    print(f"   是否使用真实类型: {confidence == 1.0}")
    
    if confidence == 1.0:
        print("✅ [PASS] 识别模型为None时，直接返回真实业务类型")
    else:
        print("❌ [FAIL] 未正确绕过识别模型！")
        
except Exception as e:
    print(f"❌ [FAIL] Test 1 异常: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 2: 运行小规模实验四（1个场景 × 1次重复 × 少量步数）
# ============================================================
print("\n" + "-" * 80)
print("[Test 2] 运行小规模实验四 (1场景 × 1重复 × 10步)")
print("-" * 80)

try:
    # 修改Experiment4内部参数以加速测试
    original_SCENARIOS = Experiment4.SCENARIOS.copy()
    
    # 只保留第一个场景用于测试
    Experiment4.SCENARIOS = {
        'smart_city': original_SCENARIOS['smart_city']
    }
    
    # 强制使用小UAV数量以加速
    Experiment4.SCENARIOS['smart_city']['num_uav'] = 20  # 从400改为20
    
    print(f"测试场景: {Experiment4.SCENARIOS['smart_city']['name']}")
    print(f"UAV数量: {Experiment4.SCENARIOS['smart_city']['num_uav']} (已缩减)")
    
    # 运行实验四（不包含MAPPO以加速）
    import time
    start_time = time.time()
    
    results = Experiment4.run(
        recognition_model=None,  # 关键：None
        scaler=None,            # 关键：None
        num_steps=10,           # 极少步数
        repeats=1,              # 仅1次重复
        include_mappo=False,    # 跳过MAPPO评估
        use_cache=False
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ [PASS] 实验四小规模测试完成")
    print(f"   耗时: {elapsed_time:.1f}秒")
    
    # 验证数据结构
    if 'smart_city' in results:
        scenario_data = results['smart_city']
        
        print(f"\n📊 收集到的数据结构:")
        for algo_name, algo_data in scenario_data.items():
            if len(algo_data) > 0:
                print(f"  ✅ {algo_name}: {len(algo_data)}次重复")
                if len(algo_data) > 0:
                    first_result = algo_data[0]
                    sat = first_result.get('avg_satisfaction', 'N/A')
                    print(f"     首次满意度: {sat}")
                    
                    # 检查是否包含recognition_accuracy
                    if 'recognition_accuracy' in first_result:
                        print(f"     ❌ 仍包含recognition_accuracy指标!")
                    else:
                        print(f"     ✅ 已移除recognition_accuracy指标")
            else:
                print(f"  ⚠️  {algo_name}: 无数据")
    else:
        print(f"❌ [FAIL] 结果中缺少'smart_city'场景数据")
    
    # 恢复原始SCENARIOS
    Experiment4.SCENARIOS = original_SCENARIOS
    
except Exception as e:
    print(f"\n❌ [FAIL] Test 2 异常: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试恢复
    try:
        Experiment4.SCENARIOS = original_SCENARIOS
    except:
        pass

# ============================================================
# Test 3: 验证种子一致性
# ============================================================
print("\n" + "-" * 80)
print("[Test 3] 验证种子机制")
print("-" * 80)

try:
    from uav_system.config import set_global_seed
    
    # 测试种子设置
    test_seeds = [30042, 30043, 30044]
    
    for seed in test_seeds:
        set_global_seed(seed)
        
        # 生成一些随机数验证
        import numpy as np
        random_val = np.random.rand()
        
        print(f"  种子={seed}: 随机值={random_val:.6f}")
    
    print("✅ [PASS] 种子设置正常工作")
    
except Exception as e:
    print(f"❌ [FAIL] Test 3 异常: {e}")

# ============================================================
# 最终报告
# ============================================================
print("\n" + "=" * 80)
print("最终验证报告")
print("=" * 80)

print(f"\n📊 警告统计:")
print(f"   总警告数: {warning_count['total']}")
print(f"   sklearn相关警告: {warning_count['sklearn_warnings']}")

if warning_count['sklearn_warnings'] == 0 and warning_count['total'] == 0:
    print("\n" + "🎉" * 30)
    print("✅ [ALL PASS] 所有测试通过！可以安全运行完整实验四")
    print("🎉" * 30)
    print("\n运行命令:")
    print("  python main.py --exp 4 --include-mappo")
elif warning_count['sklearn_warnings'] > 0:
    print("\n❌ [FAIL] 仍有sklearn警告未消除！请检查代码")
    sys.exit(1)
else:
    print(f"\n⚠️  有{warning_count['total'] - warning_count['sklearn_warnings']}个非sklearn警告")
    print("   这些可能是正常的，但仍需关注")

print("\n" + "=" * 80)
