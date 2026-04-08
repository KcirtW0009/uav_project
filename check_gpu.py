#!/usr/bin/env python3
"""
GPU可用性检测脚本
功能：
1. 检测当前系统GPU状态、显存大小及驱动版本
2. 检测PyTorch/TensorFlow的GPU支持
3. 提供问题诊断和解决方案
4. 自动识别操作系统并给出适配的安装命令
"""

import platform
import subprocess
import sys
import os
from pathlib import Path


class GPUDetector:
    """GPU检测器类"""
    
    def __init__(self):
        self.results = {
            "os_type": platform.system(),
            "os_version": platform.release(),
            "os_details": platform.platform(),
            "python_version": sys.version,
            "nvidia_smi_available": False,
            "nvidia_driver_version": None,
            "gpu_count": 0,
            "gpu_info": [],
            "pytorch_installed": False,
            "pytorch_version": None,
            "pytorch_cuda_available": False,
            "pytorch_gpu_devices": [],
            "tensorflow_installed": False,
            "tensorflow_version": None,
            "tensorflow_gpu_available": False,
            "cuda_toolkit_version": None,
            "issues": [],
            "solutions": []
        }
    
    def print_header(self, text):
        """打印标题"""
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)
    
    def print_success(self, text):
        print(f"  [OK] {text}")
    
    def print_error(self, text):
        print(f"  [X] {text}")
    
    def print_warning(self, text):
        print(f"  [!] {text}")
    
    def print_info(self, text):
        print(f"  [i] {text}")
    
    def run_command(self, cmd, capture=True):
        """运行命令并返回输出"""
        try:
            if capture:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, 
                    text=True, timeout=10, encoding='utf-8', errors='ignore'
                )
                return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
            else:
                result = subprocess.run(cmd, shell=True, timeout=10)
                return result.returncode == 0, "", ""
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)
    
    def detect_os(self):
        """检测操作系统"""
        self.print_header("操作系统检测")
        
        os_type = self.results["os_type"]
        os_version = self.results["os_version"]
        os_details = self.results["os_details"]
        
        print(f"  系统类型: {os_type}")
        print(f"  系统版本: {os_version}")
        print(f"  详细信息: {os_details}")
        
        # 检测是否为Windows
        if os_type == "Windows":
            self.print_success("Windows 系统")
            # 检测是否为Windows 11
            if "10" in os_version or "11" in os_version:
                self.print_success("Windows 10/11 兼容CUDA")
            # 检测WSL
            if "WSL" in os_details:
                self.print_warning("检测到WSL环境，可能需要额外配置")
        elif os_type == "Linux":
            self.print_success("Linux 系统")
            # 检测是否为WSL2
            if "WSL2" in os_details or "microsoft-standard" in os_details:
                self.print_warning("检测到WSL2环境")
                if not os.path.exists("/usr/local/cuda"):
                    self.results["issues"].append("WSL2环境下可能需要安装CUDA Toolkit")
        elif os_type == "Darwin":
            self.print_warning("macOS系统，NVIDIA GPU支持有限")
            self.results["issues"].append("macOS通常不支持NVIDIA CUDA")
        else:
            self.print_warning(f"未知操作系统: {os_type}")
        
        return os_type
    
    def detect_nvidia_driver(self):
        """检测NVIDIA驱动"""
        self.print_header("NVIDIA驱动检测")
        
        # 尝试多种方式检测nvidia-smi
        possible_paths = ["nvidia-smi"]
        if self.results["os_type"] == "Windows":
            # Windows上可能的路径
            possible_paths.extend([
                "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
                "${env:ProgramFiles}\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
            ])
        
        success = False
        for path in possible_paths:
            ok, stdout, stderr = self.run_command(f'"{path}" --query-gpu=driver_version --format=csv,noheader' if self.results["os_type"] == "Windows" else f"{path} --query-gpu=driver_version --format=csv,noheader 2>/dev/null")
            if ok and stdout:
                success = True
                break
        
        if not success:
            ok, stdout, stderr = self.run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        
        if ok and stdout:
            self.results["nvidia_smi_available"] = True
            self.print_success("nvidia-smi 可用")
            
            # 获取驱动版本
            _, driver_version, _ = self.run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
            self.results["nvidia_driver_version"] = driver_version
            print(f"  驱动版本: {driver_version}")
            
            # 获取GPU数量
            ok, gpu_count, _ = self.run_command("nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l")
            if not ok:
                ok, gpu_count, _ = self.run_command("nvidia-smi -L | findstr /C:\"GPU\"")
            
            if ok:
                try:
                    self.results["gpu_count"] = len(stdout.split('\n')) if '\n' in stdout else 1
                except:
                    self.results["gpu_count"] = 1
            
            self.print_success(f"检测到 {self.results['gpu_count']} 个GPU")
            
            # 获取每个GPU的详细信息
            ok, stdout, _ = self.run_command("nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv")
            if ok:
                lines = stdout.strip().split('\n')
                for line in lines[1:]:  # 跳过标题行
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpu_info = {
                            "index": parts[0],
                            "name": parts[1],
                            "memory_total": parts[2],
                            "memory_used": parts[3],
                            "memory_free": parts[4],
                            "utilization": parts[5],
                            "temperature": parts[6]
                        }
                        self.results["gpu_info"].append(gpu_info)
                        print(f"  GPU {parts[0]}: {parts[1]}")
                        print(f"    显存: {parts[2]} (已用: {parts[3]}, 可用: {parts[4]})")
                        print(f"    利用率: {parts[5]}, 温度: {parts[6]}")
        else:
            self.print_error("nvidia-smi 不可用")
            self.results["issues"].append("NVIDIA驱动未安装或不可访问")
            self.results["solutions"].append({
                "problem": "NVIDIA驱动未安装",
                "solutions": self._get_driver_install_instructions()
            })
    
    def _get_driver_install_instructions(self):
        """获取驱动安装指南"""
        os_type = self.results["os_type"]
        instructions = []
        
        if os_type == "Windows":
            instructions = [
                "方案1 - 自动安装:",
                "  1. 打开 NVIDIA官网: https://www.nvidia.com/Download/index.aspx",
                "  2. 选择你的GPU型号和Windows版本",
                "  3. 下载并运行驱动安装程序",
                "",
                "方案2 - 使用GeForce Experience:",
                "  1. 下载并安装 GeForce Experience: https://www.nvidia.com/geforce/geforce-experience/",
                "  2. 自动检测并安装最新驱动",
                "",
                "方案3 - 命令行安装(需要管理员权限):",
                '  winget install Nvidia.GeForceExperience'
            ]
        elif os_type == "Linux":
            instructions = [
                "Ubuntu/Debian:",
                "  sudo apt update",
                "  sudo apt install nvidia-driver-535  # 选择适合你的版本",
                "  sudo reboot",
                "",
                "Arch Linux:",
                "  sudo pacman -S nvidia-utils",
                "  sudo reboot",
                "",
                "验证安装:",
                "  nvidia-smi"
            ]
        else:
            instructions = ["当前系统不支持NVIDIA驱动安装"]
        
        return instructions
    
    def detect_cuda_toolkit(self):
        """检测CUDA Toolkit"""
        self.print_header("CUDA Toolkit检测")
        
        # 尝试多种方式检测CUDA
        cuda_paths = []
        
        if self.results["os_type"] == "Windows":
            cuda_paths = [
                os.environ.get("CUDA_PATH", ""),
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*"
            ]
        else:
            cuda_paths = [
                os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "")),
                "/usr/local/cuda",
                "/usr/local/cuda-*"
            ]
        
        found_cuda = False
        for path in cuda_paths:
            if path and os.path.exists(path):
                found_cuda = True
                # 尝试读取版本文件
                version_file = os.path.join(path, "version.txt")
                if os.path.exists(version_file):
                    with open(version_file, 'r') as f:
                        version = f.read().strip()
                    self.results["cuda_toolkit_version"] = version
                    print(f"  CUDA路径: {path}")
                    print(f"  CUDA版本: {version}")
                else:
                    # 尝试从路径提取版本
                    import re
                    match = re.search(r'v?(\d+\.\d+)', path)
                    if match:
                        version = match.group(1)
                        self.results["cuda_toolkit_version"] = version
                        print(f"  CUDA路径: {path}")
                        print(f"  CUDA版本: {version}")
                break
        
        if not found_cuda:
            self.print_warning("CUDA Toolkit 未检测到")
            self.results["issues"].append("CUDA Toolkit未安装")
            self.results["solutions"].append({
                "problem": "CUDA Toolkit未安装",
                "solutions": self._get_cuda_install_instructions()
            })
        else:
            self.print_success("CUDA Toolkit 已安装")
    
    def _get_cuda_install_instructions(self):
        """获取CUDA安装指南"""
        os_type = self.results["os_type"]
        driver_version = self.results.get("nvidia_driver_version", "")
        
        # 从驱动版本推断支持的CUDA版本
        cuda_versions = []
        if driver_version:
            try:
                major = int(driver_version.split('.')[0])
                if major >= 560:
                    cuda_versions = ["12.6", "12.4", "12.2", "12.1"]
                elif major >= 550:
                    cuda_versions = ["12.4", "12.2", "12.1", "11.8"]
                elif major >= 545:
                    cuda_versions = ["12.2", "12.1", "11.8", "11.7"]
                elif major >= 535:
                    cuda_versions = ["12.2", "12.1", "11.8", "11.7"]
                elif major >= 525:
                    cuda_versions = ["12.1", "11.8", "11.7", "11.6"]
                elif major >= 515:
                    cuda_versions = ["11.8", "11.7", "11.6", "11.5"]
                else:
                    cuda_versions = ["11.8", "11.7", "11.6"]
            except:
                cuda_versions = ["12.4", "12.2", "11.8"]
        
        instructions = []
        if os_type == "Windows":
            instructions = [
                "CUDA Toolkit 下载地址: https://developer.nvidia.com/cuda-downloads",
                "",
                "推荐安装步骤:",
                "  1. 访问 https://developer.nvidia.com/cuda-downloads",
                "  2. 选择: Windows > x86_64 > 10 > exe(local)",
                f"  3. 推荐版本: CUDA {cuda_versions[0] if cuda_versions else '12.4'}",
                "  4. 下载并运行安装程序",
                "  5. 安装完成后，重新打开终端验证: nvidia-smi"
            ]
        elif os_type == "Linux":
            instructions = [
                "Ubuntu/Debian 安装CUDA:",
                f"  wget https://developer.download.nvidia.com/compute/cuda/repos/{'ubuntu' if 'Ubuntu' in platform.platform() else 'debian'}/x86_64/cuda-keyring_1.0-1_all.deb",
                "  sudo dpkg -i cuda-keyring_1.0-1_all.deb",
                "  sudo apt update",
                f"  sudo apt install cuda-cudart-12-4  # 以CUDA 12.4为例",
                "  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc",
                "  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc",
                "  source ~/.bashrc",
                "",
                f"推荐版本: CUDA {cuda_versions[0] if cuda_versions else '12.4'}"
            ]
        else:
            instructions = ["当前系统CUDA安装方法未知"]
        
        return instructions
    
    def detect_pytorch(self):
        """检测PyTorch及GPU支持"""
        self.print_header("PyTorch GPU支持检测")
        
        try:
            import torch
            self.results["pytorch_installed"] = True
            self.results["pytorch_version"] = torch.__version__
            print(f"  PyTorch版本: {torch.__version__}")
            self.print_success("PyTorch 已安装")
            
            # 检测CUDA支持
            cuda_available = torch.cuda.is_available()
            self.results["pytorch_cuda_available"] = cuda_available
            
            if cuda_available:
                self.print_success("PyTorch CUDA支持 已启用")
                print(f"  PyTorch编译时的CUDA版本: {torch.version.cuda}")
                print(f"  可用的GPU数量: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    cap = torch.cuda.get_device_capability(i)
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {name}")
                    print(f"    计算能力: {cap[0]}.{cap[1]}")
                    print(f"    总显存: {props.total_memory / 1024**3:.2f} GB")
                    
                    gpu_info = {
                        "index": i,
                        "name": name,
                        "compute_capability": f"{cap[0]}.{cap[1]}",
                        "total_memory_gb": props.total_memory / 1024**3
                    }
                    self.results["pytorch_gpu_devices"].append(gpu_info)
            else:
                self.print_error("PyTorch CUDA支持 不可用")
                
                # 诊断原因
                issues = []
                if not self.results["nvidia_smi_available"]:
                    issues.append("NVIDIA驱动未安装")
                if not self.results["cuda_toolkit_version"]:
                    issues.append("CUDA Toolkit未安装")
                
                # 检查PyTorch是否为CPU版本
                if "cpu" in torch.__version__ or "+cpu" in torch.__version__:
                    issues.append("安装的是CPU版本PyTorch")
                else:
                    issues.append("PyTorch与当前CUDA版本不兼容")
                
                for issue in issues:
                    self.print_error(f"原因: {issue}")
                
                self.results["issues"].append(f"PyTorch无法使用GPU: {'; '.join(issues)}")
                self.results["solutions"].append({
                    "problem": "PyTorch无法使用GPU",
                    "details": issues,
                    "solutions": self._get_pytorch_install_instructions()
                })
                
        except ImportError:
            self.print_error("PyTorch 未安装")
            self.results["issues"].append("PyTorch未安装")
            self.results["solutions"].append({
                "problem": "PyTorch未安装",
                "solutions": self._get_pytorch_install_instructions()
            })
    
    def _get_pytorch_install_instructions(self):
        """获取PyTorch安装指南"""
        os_type = self.results["os_type"]
        cuda_version = self.results.get("cuda_toolkit_version", "")
        
        # 确定PyTorch版本
        instructions = [
            "=" * 50,
            "PyTorch GPU版本安装指南",
            "=" * 50,
            "",
            "重要提示: 安装GPU版PyTorch前，请确保:",
            "  1. NVIDIA驱动已安装 (运行 nvidia-smi 验证)",
            "  2. CUDA Toolkit已安装",
            "",
            "步骤1: 卸载CPU版本(如果有)",
            "  pip uninstall torch torchvision torchaudio -y",
            "",
            "步骤2: 安装GPU版本",
        ]
        
        # 根据CUDA版本给出推荐安装命令
        if cuda_version:
            major_minor = '.'.join(cuda_version.split('.')[:2])
        else:
            major_minor = "12.4"  # 默认推荐
        
        instructions.extend([
            f"  # CUDA {major_minor} 版本",
            f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{major_minor.replace('.', '')}",
            "",
            "  或使用稳定版 (自动匹配CUDA):",
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
            "",
            "  或使用最新预览版:",
            "  pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124",
            "",
            "验证安装:",
            "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"",
            "",
            "Windows额外步骤:",
            "  1. 安装 Visual Studio Build Tools",
            "  2. 确保CUDA_PATH环境变量已设置",
            "  3. 重启终端/PowerShell"
        ])
        
        return instructions
    
    def detect_tensorflow(self):
        """检测TensorFlow及GPU支持"""
        self.print_header("TensorFlow GPU支持检测")
        
        try:
            import tensorflow as tf
            self.results["tensorflow_installed"] = True
            self.results["tensorflow_version"] = tf.__version__
            print(f"  TensorFlow版本: {tf.__version__}")
            self.print_success("TensorFlow 已安装")
            
            # 检测GPU支持
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.results["tensorflow_gpu_available"] = True
                self.print_success("TensorFlow GPU支持 已启用")
                print(f"  检测到的GPU数量: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
            else:
                self.print_error("TensorFlow GPU支持 不可用")
                self.results["issues"].append("TensorFlow无法使用GPU")
                self.results["solutions"].append({
                    "problem": "TensorFlow GPU不可用",
                    "solutions": [
                        "TensorFlow GPU版本安装:",
                        "  pip uninstall tensorflow -y",
                        "  pip install tensorflow[and-cuda]  # 或",
                        "  pip install nvidia-tensorflow  # NVIDIA优化版本",
                        "",
                        "或使用tensorflow-gpu包(已弃用):",
                        "  pip uninstall tensorflow-gpu -y",
                        "  pip install tensorflow-gpu==2.15.0  # 指定版本"
                    ]
                })
                
        except ImportError:
            self.print_warning("TensorFlow 未安装 (可选)")
    
    def detect_anaconda_env(self):
        """检测Anaconda环境"""
        self.print_header("Python环境检测")
        
        # 检测虚拟环境
        venv = os.environ.get("VIRTUAL_ENV", "")
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
        
        if venv:
            print(f"  虚拟环境: {venv}")
        if conda_env:
            print(f"  Conda环境: {conda_env}")
            print("  建议: 在Conda环境中安装GPU版本以避免依赖冲突")
        
        # 检查pip版本
        ok, stdout, _ = self.run_command("pip --version")
        if ok:
            print(f"  {stdout}")
        
        # 检查包位置
        ok, stdout, _ = self.run_command("pip show torch 2>nul" if self.results["os_type"] == "Windows" else "pip show torch 2>/dev/null")
        if ok and stdout:
            for line in stdout.split('\n'):
                if line.startswith("Location:"):
                    print(f"  PyTorch安装位置: {line.split(':', 1)[1].strip()}")
                    break
    
    def check_project_compatibility(self):
        """检查项目依赖兼容性"""
        self.print_header("项目兼容性检查")
        
        project_file = Path(__file__).parent / "requirements.txt"
        if project_file.exists():
            self.print_info("检测到requirements.txt文件")
            with open(project_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'torch' in content.lower() or 'tensorflow' in content.lower():
                    self.print_info("项目依赖包含深度学习框架")
                    
                    # 检查是否有CPU版本指定
                    if '+cpu' in content or '-cpu' in content:
                        self.print_warning("requirements.txt可能指定了CPU版本")
                        self.print_info("建议: 修改为GPU版本或移除版本后缀")
        else:
            self.print_info("未找到requirements.txt，跳过兼容性检查")
    
    def generate_summary(self):
        """生成检测报告"""
        self.print_header("检测结果汇总")
        
        print(f"\n  操作系统: {self.results['os_type']} {self.results['os_version']}")
        print(f"  Python版本: {sys.version.split()[0]}")
        
        if self.results["nvidia_smi_available"]:
            print(f"  NVIDIA驱动: {self.results['nvidia_driver_version']}")
            print(f"  GPU数量: {self.results['gpu_count']}")
            for gpu in self.results["gpu_info"]:
                print(f"    - {gpu['name']}: {gpu['memory_total']}")
        else:
            print("  NVIDIA驱动: 未安装")
        
        if self.results["cuda_toolkit_version"]:
            print(f"  CUDA Toolkit: {self.results['cuda_toolkit_version']}")
        else:
            print("  CUDA Toolkit: 未安装")
        
        if self.results["pytorch_installed"]:
            print(f"  PyTorch: {self.results['pytorch_version']}")
            if self.results["pytorch_cuda_available"]:
                self.print_success("PyTorch GPU: 可用")
            else:
                self.print_error("PyTorch GPU: 不可用")
        else:
            print("  PyTorch: 未安装")
        
        # 总体状态
        print("\n" + "-" * 50)
        print("  GPU可用性状态:")
        
        if (self.results["nvidia_smi_available"] and 
            self.results["cuda_toolkit_version"] and
            self.results["pytorch_installed"] and
            self.results["pytorch_cuda_available"]):
            self.print_success("所有组件已就绪，可以正常使用GPU!")
        else:
            self.print_error("GPU不可用，请按照以下解决方案进行修复")
        
        return len(self.results["issues"]) == 0
    
    def generate_solutions_report(self):
        """生成解决方案报告"""
        if not self.results["solutions"]:
            return
        
        self.print_header("解决方案")
        
        for i, solution in enumerate(self.results["solutions"], 1):
            print(f"\n问题 {i}: {solution['problem']}")
            if 'details' in solution:
                print("  原因分析:")
                for detail in solution['details']:
                    print(f"    - {detail}")
            print("\n  解决步骤:")
            for step in solution["solutions"]:
                print(f"    {step}")
    
    def export_report(self):
        """导出检测报告到文件"""
        report_file = Path("gpu_detection_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("GPU检测报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"操作系统: {self.results['os_type']} {self.results['os_version']}\n")
            f.write(f"Python版本: {sys.version.split()[0]}\n\n")
            
            f.write("NVIDIA驱动:\n")
            f.write(f"  nvidia-smi可用: {self.results['nvidia_smi_available']}\n")
            if self.results['nvidia_driver_version']:
                f.write(f"  驱动版本: {self.results['nvidia_driver_version']}\n")
            f.write(f"  GPU数量: {self.results['gpu_count']}\n")
            for gpu in self.results['gpu_info']:
                f.write(f"  - {gpu['name']}: {gpu['memory_total']}\n")
            
            f.write("\nCUDA Toolkit:\n")
            f.write(f"  版本: {self.results['cuda_toolkit_version'] or '未安装'}\n")
            
            f.write("\nPyTorch:\n")
            f.write(f"  已安装: {self.results['pytorch_installed']}\n")
            if self.results['pytorch_version']:
                f.write(f"  版本: {self.results['pytorch_version']}\n")
            f.write(f"  CUDA支持: {self.results['pytorch_cuda_available']}\n")
            
            f.write("\n检测到的问题:\n")
            for issue in self.results['issues']:
                f.write(f"  - {issue}\n")
            
            if self.results['solutions']:
                f.write("\n解决方案:\n")
                for i, sol in enumerate(self.results['solutions'], 1):
                    f.write(f"\n{i}. {sol['problem']}\n")
                    for step in sol['solutions']:
                        f.write(f"   {step}\n")
        
        print(f"\n报告已保存到: {report_file.absolute()}")
    
    def run_quick_fix(self):
        """运行快速修复建议"""
        self.print_header("快速修复建议")
        
        print("\n根据检测结果，建议执行以下操作:\n")
        
        if not self.results["nvidia_smi_available"]:
            print("【紧急】安装NVIDIA驱动程序:")
            for step in self._get_driver_install_instructions():
                print(f"  {step}")
            print()
        
        if not self.results["cuda_toolkit_version"]:
            print("【重要】安装CUDA Toolkit:")
            for step in self._get_cuda_install_instructions():
                print(f"  {step}")
            print()
        
        if not self.results["pytorch_cuda_available"]:
            print("【关键】安装PyTorch GPU版本:")
            for step in self._get_pytorch_install_instructions():
                print(f"  {step}")
            print()
        
        if (self.results["nvidia_smi_available"] and 
            self.results["pytorch_installed"] and 
            not self.results["pytorch_cuda_available"]):
            print("\n常见问题排查:")
            print("  1. 驱动与CUDA版本不匹配?")
            print("     - 检查 https://docs.nvidia.com/deploy/cuda-compatibility/ ")
            print("  2. PyTorch编译CUDA版本与安装的CUDA不匹配?")
            print(f"     - 当前CUDA: {self.results.get('cuda_toolkit_version', '未知')}")
            print("     - PyTorch需要重新安装匹配版本")
            print("  3. Windows环境变量未设置?")
            print("     - 设置 CUDA_PATH 指向CUDA安装目录")
    
    def run(self):
        """执行完整检测流程"""
        print("\n" + "=" * 60)
        print("  GPU可用性自动检测工具")
        print("  " + "=" * 58)
        print("  版本: 1.0")
        print("  时间: 2026-04-08")
        print("=" * 60)
        
        # 执行各项检测
        self.detect_os()
        self.detect_nvidia_driver()
        self.detect_cuda_toolkit()
        self.detect_pytorch()
        self.detect_tensorflow()
        self.detect_anaconda_env()
        self.check_project_compatibility()
        
        # 生成报告
        is_ready = self.generate_summary()
        self.generate_solutions_report()
        
        # 如果有问题，提供快速修复建议
        if not is_ready:
            self.run_quick_fix()
        
        # 导出报告
        self.export_report()
        
        # 最终建议
        print("\n" + "=" * 60)
        if is_ready:
            print("  恭喜! 环境已就绪，可以开始GPU加速训练!")
            print("  建议运行以下命令验证:")
            print("    python -c \"import torch; print('GPU:', torch.cuda.get_device_name(0))\n\"")
        else:
            print("  环境检测完成，请根据上述解决方案进行配置")
            print("  修复后，重新运行本脚本验证")
        print("=" * 60 + "\n")
        
        return is_ready


def main():
    """主函数"""
    detector = GPUDetector()
    success = detector.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
