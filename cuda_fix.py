"""
CUDA环境修复脚本

功能：
1. 自动检测GPU可用性
2. 设置CUDA环境变量
3. GPU不可用时自动回退到CPU模式
4. 验证PyTorch环境

使用方法：
    from cuda_fix import setup_environment
    device = setup_environment()
"""

import os
import sys

def setup_environment():
    """
    设置CUDA环境并返回设备类型
    
    Returns:
        str: 'cuda' 或 'cpu'
    """
    print("=" * 60)
    print("环境检测")
    print("=" * 60)
    
    # 1. 设置CUDA环境变量
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp",
    ]
    
    current_path = os.environ.get("PATH", "")
    cuda_added = False
    
    for path in cuda_paths:
        if path not in current_path and os.path.exists(path):
            os.environ["PATH"] = path + os.pathsep + current_path
            cuda_added = True
            print(f"[OK] 添加CUDA路径: {path}")
    
    # 2. 设置CUDA优化选项
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 3. 检测PyTorch和CUDA
    print("\n[检测] PyTorch和CUDA状态...")
    
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"  GPU设备数: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    [{i}] {gpu_name} ({gpu_mem:.2f} GB)")
            
            # 测试GPU计算
            print("\n[测试] GPU计算测试...")
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = test_tensor @ test_tensor
                print("  GPU计算测试: 通过")
                device = "cuda"
            except Exception as e:
                print(f"  GPU计算测试: 失败 ({e})")
                print("  回退到CPU模式")
                device = "cpu"
        else:
            print("  GPU不可用，使用CPU模式")
            device = "cpu"
            
    except ImportError as e:
        print(f"  PyTorch未安装或导入失败: {e}")
        print("  警告: 训练可能无法正常进行")
        device = "cpu"
    except Exception as e:
        print(f"  检测过程出错: {e}")
        device = "cpu"
    
    print(f"\n最终设备: {device.upper()}")
    print("=" * 60)
    
    return device


def verify_environment():
    """验证完整的环境配置"""
    device = setup_environment()
    
    print("\n[验证] 依赖包状态...")
    
    required_packages = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
    }
    
    all_ok = True
    for name, import_name in required_packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {name}: OK ({version})")
        except ImportError:
            print(f"  {name}: 未安装")
            all_ok = False
    
    if not all_ok:
        print("\n[警告] 部分依赖缺失，请运行: pip install numpy matplotlib scipy scikit-learn pandas")
    
    return device, all_ok


if __name__ == "__main__":
    device, all_ok = verify_environment()
    
    if device == "cuda":
        print("\n可以启动GPU训练")
    else:
        print("\n将使用CPU训练（速度较慢）")
    
    print("\n启动MAPPO训练:")
    print("  python run_mappo_standard.py")
