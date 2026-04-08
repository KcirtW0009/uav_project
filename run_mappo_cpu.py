"""
MAPPO实验启动脚本 - 强制使用CPU版本
"""
import os
import sys

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用所有GPU

# 确保使用venv中的Python和包
venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)
else:
    # venv不存在，直接导入CPU版本的torch
    import torch
    if torch.cuda.is_available():
        print("警告: 检测到CUDA，将强制使用CPU")
        torch.cuda.is_available = lambda: False
    
    # 启动主程序
    from run_mappo_standard import main
    main()
