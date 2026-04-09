@echo off
chcp 65001
setlocal enabledelayedexpansion

echo ========================================
echo MAPPO 实验运行脚本
echo ========================================
echo.

:: 设置项目目录
set PROJECT_DIR=f:\桌面\本科毕业论文\结题\uav_project
cd /d "%PROJECT_DIR%"

echo 当前目录: %CD%
echo.

:: 显示菜单
echo 请选择要运行的实验:
echo.
echo [1] 原MAPPO实验（对比基准）--small模式
echo [2] 优化MAPPO实验 - 高负载场景（128 UAVs）
echo [3] 优化MAPPO实验 - 中负载场景（64 UAVs）
echo [4] 优化MAPPO实验 - 低负载场景（32 UAVs）
echo [5] 参数搜索实验（小规模测试）
echo [6] 自定义命令
echo [0] 退出
echo.

set /p choice="请输入选项 (0-6): "

if "%choice%"=="1" (
    echo.
    echo 正在运行原MAPPO实验（对比基准）...
    venv\Scripts\python.exe main.py --exp mappo --rl-phase both --small
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 正在运行优化MAPPO实验 - 高负载场景...
    venv\Scripts\python.exe run_optimized_mappo.py --scenario high --train --eval
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 正在运行优化MAPPO实验 - 中负载场景...
    venv\Scripts\python.exe run_optimized_mappo.py --scenario medium --train --eval
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 正在运行优化MAPPO实验 - 低负载场景...
    venv\Scripts\python.exe run_optimized_mappo.py --scenario low --train --eval
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 正在运行参数搜索实验...
    venv\Scripts\python.exe mappo_parameter_search.py
    goto end
)

if "%choice%"=="6" (
    echo.
    echo 请输入自定义命令（不需要输入 venv\Scripts\python.exe）:
    set /p custom_cmd="> "
    echo 执行: venv\Scripts\python.exe %custom_cmd%
    venv\Scripts\python.exe %custom_cmd%
    goto end
)

if "%choice%"=="0" (
    echo 退出脚本
    goto end
)

echo 无效选项，请重新运行脚本

:end
echo.
echo ========================================
echo 实验运行完成
echo ========================================
pause
