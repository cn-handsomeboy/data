@echo off
chcp 65001 >nul
title 蔗循智策 - 一键启动

echo.
echo ===========================================
echo  蔗循智策 - 甘蔗副产物循环经济决策系统
echo ===========================================
echo  1. Streamlit: http://localhost:8501
echo  2. API Docs:  http://localhost:8000/docs
echo ===========================================
echo.

echo [启动] Streamlit ...
start "Streamlit" cmd /c "D:\python\python.exe -m streamlit run app.py --server.port 8501"

echo [启动] FastAPI ...
start "API" cmd /c "D:\python\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo [完成] 服务已启动！按 Enter 停止...
pause >nul