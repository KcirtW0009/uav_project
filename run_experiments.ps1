# MAPPO 实验运行脚本 (PowerShell)
# 使用方法: 右键选择"使用 PowerShell 运行"或在PowerShell中执行 .\run_experiments.ps1

$PROJECT_DIR = "f:\桌面\本科毕业论文\结题\uav_project"
Set-Location $PROJECT_DIR

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MAPPO 实验运行脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "当前目录: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

function Show-Menu {
    Write-Host "请选择要运行的实验:" -ForegroundColor Green
    Write-Host ""
    Write-Host "[1] 原MAPPO实验（对比基准）--small模式" -ForegroundColor White
    Write-Host "[2] 优化MAPPO实验 - 高负载场景（128 UAVs）" -ForegroundColor White
    Write-Host "[3] 优化MAPPO实验 - 中负载场景（64 UAVs）" -ForegroundColor White
    Write-Host "[4] 优化MAPPO实验 - 低负载场景（32 UAVs）" -ForegroundColor White
    Write-Host "[5] 参数搜索实验（小规模测试）" -ForegroundColor White
    Write-Host "[6] 自定义命令" -ForegroundColor White
    Write-Host "[0] 退出" -ForegroundColor Red
    Write-Host ""
}

function Run-OriginalMAPPO {
    Write-Host ""
    Write-Host "正在运行原MAPPO实验（对比基准）..." -ForegroundColor Yellow
    Write-Host "命令: venv\Scripts\python.exe main.py --exp mappo --rl-phase both --small" -ForegroundColor Gray
    Write-Host ""
    & venv\Scripts\python.exe main.py --exp mappo --rl-phase both --small
}

function Run-OptimizedMAPPO {
    param([string]$Scenario)
    
    $scenarioNames = @{
        "low" = "低负载（32 UAVs）"
        "medium" = "中负载（64 UAVs）"
        "high" = "高负载（128 UAVs）"
        "extreme" = "极高负载（150 UAVs）"
    }
    
    Write-Host ""
    Write-Host "正在运行优化MAPPO实验 - $($scenarioNames[$Scenario])..." -ForegroundColor Yellow
    Write-Host "命令: venv\Scripts\python.exe run_optimized_mappo.py --scenario $Scenario --train --eval" -ForegroundColor Gray
    Write-Host ""
    & venv\Scripts\python.exe run_optimized_mappo.py --scenario $Scenario --train --eval
}

function Run-ParameterSearch {
    Write-Host ""
    Write-Host "正在运行参数搜索实验..." -ForegroundColor Yellow
    Write-Host "命令: venv\Scripts\python.exe mappo_parameter_search.py" -ForegroundColor Gray
    Write-Host ""
    & venv\Scripts\python.exe mappo_parameter_search.py
}

function Run-CustomCommand {
    Write-Host ""
    Write-Host "请输入自定义命令（不需要输入 venv\Scripts\python.exe）:" -ForegroundColor Green
    $customCmd = Read-Host "> "
    Write-Host ""
    Write-Host "执行: venv\Scripts\python.exe $customCmd" -ForegroundColor Yellow
    Write-Host ""
    & venv\Scripts\python.exe $customCmd
}

# 主循环
while ($true) {
    Show-Menu
    $choice = Read-Host "请输入选项 (0-6)"
    
    switch ($choice) {
        "1" { Run-OriginalMAPPO; break }
        "2" { Run-OptimizedMAPPO -Scenario "high"; break }
        "3" { Run-OptimizedMAPPO -Scenario "medium"; break }
        "4" { Run-OptimizedMAPPO -Scenario "low"; break }
        "5" { Run-ParameterSearch; break }
        "6" { Run-CustomCommand; break }
        "0" { 
            Write-Host ""
            Write-Host "退出脚本" -ForegroundColor Red
            exit 
        }
        default { 
            Write-Host ""
            Write-Host "无效选项，请重新选择" -ForegroundColor Red
            Write-Host ""
        }
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "实验运行完成" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $continue = Read-Host "是否继续运行其他实验? (y/n)"
    if ($continue -ne "y") {
        Write-Host "退出脚本" -ForegroundColor Red
        break
    }
    Write-Host ""
}
