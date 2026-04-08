"""
环境诊断脚本

诊断PyTorch和依赖库的问题，提供解决方案。
"""

import sys
import os

print("=" * 70)
print("环境诊断报告")
print("=" * 70)

# 1. Python版本
print("\n[1] Python环境:")
print(f"  版本: {sys.version}")
print(f"  路径: {sys.executable}")

# 2. 检查虚拟环境
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
if os.path.exists(venv_path):
    print(f"\n[2] 虚拟环境:")
    print(f"  存在: {venv_path}")
    venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        print(f"  Python: {venv_python}")
else:
    print("\n[2] 虚拟环境:")
    print(f"  未找到: {venv_path}")

# 3. 检查PyTorch安装
print("\n[3] PyTorch安装检查:")
torch_paths = [
    os.path.join(venv_path, 'Lib', 'site-packages', 'torch'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch'),
    r"E:\Anaconda3\Lib\site-packages\torch",
]

torch_found = None
for p in torch_paths:
    if os.path.exists(p):
        torch_found = p
        print(f"  找到: {p}")
        # 检查DLL
        dll_path = os.path.join(p, 'lib')
        if os.path.exists(dll_path):
            dlls = [f for f in os.listdir(dll_path) if f.endswith('.dll')]
            print(f"  DLL文件数: {len(dlls)}")
            if 'c10.dll' in dlls:
                print(f"  [OK] c10.dll存在")
            else:
                print(f"  [警告] c10.dll不存在")
        break

if not torch_found:
    print("  [错误] 未找到PyTorch安装")

# 4. 尝试导入PyTorch
print("\n[4] PyTorch导入测试:")
try:
    import torch
    print(f"  [OK] PyTorch {torch.__version__} 导入成功")
    print(f"  [OK] CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  [错误] 导入失败: {e}")
except OSError as e:
    print(f"  [错误] DLL加载失败: {e}")

# 5. 检查Visual C++ Runtime
print("\n[5] Visual C++ Runtime检查:")
import ctypes
try:
    msvcr = ctypes.CDLL("msvcr120.dll")
    print("  [OK] msvcr120.dll (VS 2013) 已安装")
except OSError:
    print("  [警告] msvcr120.dll 未找到")
    print("         可能需要安装 Visual C++ Redistributable")

try:
    vcruntime = ctypes.CDLL("vcruntime140.dll")
    print("  [OK] vcruntime140.dll (VS 2015+) 已安装")
except OSError:
    print("  [警告] vcruntime140.dll 未找到")
    print("         请安装最新的 Visual C++ Redistributable")

# 6. 系统信息
print("\n[6] 系统信息:")
import platform
print(f"  系统: {platform.system()}")
print(f"  版本: {platform.version()}")
print(f"  架构: {platform.machine()}")

# 7. 解决方案
print("\n" + "=" * 70)
print("解决方案")
print("=" * 70)

print("""
方案1: 安装Visual C++ Redistributable (最可能解决问题)
------------------------------------------------------
下载并安装:
- VC_redist.x64.exe (VS 2015-2022):
  https://aka.ms/vs/17/release/vc_redist.x64.exe

方案2: 重新安装PyTorch CPU版本 (绕过CUDA问题)
------------------------------------------------------
在venv环境中执行:
  .\\venv\\Scripts\\activate
  pip uninstall torch -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

方案3: 使用系统Python (如果有)
------------------------------------------------------
  python run_mappo_standard.py

方案4: 重新创建虚拟环境
------------------------------------------------------
  rd /s /q venv
  python -m venv venv
  .\\venv\\Scripts\\activate
  pip install numpy matplotlib scipy scikit-learn pandas
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

方案5: 清理Anaconda环境变量 (如果同时安装了Anaconda)
------------------------------------------------------
在PowerShell中:
  $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notmatch 'Anaconda' }) -join ';'
  $env:PYTHONPATH = $null
  python run_mappo_standard.py
""")

print("=" * 70)
print("诊断完成")
print("=" * 70)
