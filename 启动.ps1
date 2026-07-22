# 蔗循智策 - 一键启动（PowerShell 版本）
# 用法: .\启动.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "       蔗循智策 - 甘蔗副产物循环经济决策系统               " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  1. Streamlit 可视化界面  -> http://localhost:8501" -ForegroundColor Cyan
Write-Host "  2. FastAPI 数据接口     -> http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "[启动] 正在启动 Streamlit 可视化界面..." -ForegroundColor Yellow
$streamlit = Start-Process -FilePath "cmd" -ArgumentList "/c", "D:\python\python.exe -m streamlit run app.py --server.port 8501" -PassThru

Write-Host "[启动] 正在启动 FastAPI 数据接口..." -ForegroundColor Yellow
$api = Start-Process -FilePath "cmd" -ArgumentList "/c", "D:\python\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload" -PassThru

Write-Host ""
Write-Host "[完成] 两个服务已启动！" -ForegroundColor Green
Write-Host "   Streamlit: http://localhost:8501" -ForegroundColor Cyan
Write-Host "   API 文档:  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "按 Enter 停止所有服务..." -ForegroundColor Yellow

$null = Read-Host

# 停止进程
if ($streamlit) { Stop-Process -Id $streamlit.Id -Force -ErrorAction SilentlyContinue }
if ($api) { Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue }

Write-Host "[已停止] 所有服务已关闭" -ForegroundColor Green