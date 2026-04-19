@echo off
chcp 65001 >nul
echo ============================================================
echo  主实验环境公平对比实验 - 一键运行脚本 (V17优化版)
echo ============================================================
echo.

cd /d "%~dp0"

:: 检查虚拟环境是否存在
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 venv
    echo 请先创建虚拟环境: python -m venv venv
    pause
    exit /b 1
)

:: 激活虚拟环境
echo [1/4] 激活虚拟环境...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 无法激活虚拟环境
    pause
    exit /b 1
)
echo ✓ 虚拟环境已激活

:: 清理旧模型（可选）
echo.
echo [2/4] 清理旧模型...
for /d %%i in (experiment_logs\mappo_main_env_*) do (
    echo   删除旧模型目录: %%i
    rmdir /s /q "%%i" >nul 2>&1
)
echo ✓ 旧模型已清理

:: 运行实验
echo.
echo [3/4] 开始运行实验...
echo ============================================================
echo.
echo 实验配置:
echo   - 环境类: EnhancedNetworkEnvironment
echo   - UAV数量: 300
echo   - BS数量: 8
echo   - 负载率: ~77%
echo   - 评估轮数: 30 episodes
echo   - 训练轮数: 300 episodes (最大)
echo   - 学习率: actor=2e-5, critic=3e-4
echo   - Clip范围: 0.3
echo.
echo 预期运行时间:
echo   - 传统算法评估: ~15-20 分钟
echo   - 增强算法评估: ~20-25 分钟  
echo   - MAPPO训练+评估: ~60-90 分钟 (可能触发早停)
echo   - 总计: 约 2-3 小时
echo.
echo ============================================================
echo.
echo 开始时间: %date% %time%
echo.

python run_main_experiment_comparison.py

set EXIT_CODE=%errorlevel%

:: 显示结果
echo.
echo ============================================================
echo.
echo 结束时间: %date% %time%
echo.
if %EXIT_CODE% equ 0 (
    echo ✓ 实验成功完成！
) else (
    echo ✗ 实验失败，退出代码: %EXIT_CODE%
)

echo.
echo [4/4] 生成的文件:
echo   📊 JSON报告: main_experiment_comparison_*.json
echo   📝 Markdown报告: main_experiment_comparison_report_*.md
echo   📈 性能可视化报告: performance_reports\
echo   🤖 MAPPO模型: experiment_logs\mappo_main_env_*\

echo.
echo ============================================================

pause