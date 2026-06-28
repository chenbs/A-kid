# 一键注册 A股选股策略的 Windows 计划任务
# 用法：右键本文件 -> "使用 PowerShell 运行"，或在【管理员】PowerShell 里执行：
#   powershell -ExecutionPolicy Bypass -File E:\private_work\A-kid\setup_schedule.ps1
#
# 注册两个任务：
#   StockDaily  —— 周一到周五 17:30，跑 run_strategy.py --mode daily（出当日信号）
#   StockWeekly —— 周六 09:30，跑 run_strategy.py --mode weekly（重训+回测）

$ErrorActionPreference = "Stop"

$py      = "E:\private_work\A-kid\.venv\Scripts\python.exe"
$script  = "E:\private_work\A-kid\run_strategy.py"
$workdir = "E:\private_work\A-kid"

# 必须管理员
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "请以【管理员身份】运行本脚本（右键 PowerShell -> 以管理员身份运行）。" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $py))     { Write-Host "找不到 python: $py" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $script)) { Write-Host "找不到脚本: $script" -ForegroundColor Red; exit 1 }

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
# 以当前用户运行，且仅在登录时运行（无需保存密码）
$prin = New-ScheduledTaskPrincipal -UserId $id.Name -LogonType Interactive -RunLevel Limited

# 每日
$dailyAction  = New-ScheduledTaskAction -Execute $py -Argument "`"$script`" --mode daily" -WorkingDirectory $workdir
$dailyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 17:30
Register-ScheduledTask -TaskName "StockDaily" -Action $dailyAction -Trigger $dailyTrigger -Settings $settings -Principal $prin -Description "A股选股每日预测+出信号" -Force | Out-Null
Write-Host "已注册 StockDaily：周一到周五 17:30" -ForegroundColor Green

# 每周
$weeklyAction  = New-ScheduledTaskAction -Execute $py -Argument "`"$script`" --mode weekly" -WorkingDirectory $workdir
$weeklyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 09:30
Register-ScheduledTask -TaskName "StockWeekly" -Action $weeklyAction -Trigger $weeklyTrigger -Settings $settings -Principal $prin -Description "A股选股每周重训+回测" -Force | Out-Null
Write-Host "已注册 StockWeekly：周六 09:30" -ForegroundColor Green

Write-Host ""
Write-Host "完成。可在【任务计划程序】里看到 StockDaily / StockWeekly。" -ForegroundColor Cyan
Write-Host "手动测试：schtasks /run /tn StockDaily"
