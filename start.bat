@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  QuantPilot 本地部署
echo ============================================
echo.
echo [1/4] 检查 Python...
python --version
if errorlevel 1 (
  echo   [失败] 找不到 python
  pause
  exit /b 1
)
echo.
echo [2/4] 检查依赖...
python -c "import numpy,pandas,scipy,fastapi,uvicorn,akshare" >nul 2>&1
if errorlevel 1 (
  echo   缺少依赖，正在安装...
  python -m pip install -r requirements.txt
)
echo   依赖齐全
echo.
echo [3/4] 运行自检...
python backend\preflight.py
if errorlevel 1 (
  echo   [失败] 自检未通过，已中止
  pause
  exit /b 1
)
echo.
echo [4/4] 启动服务 http://127.0.0.1:8421
echo   关闭本窗口即停止
start "" http://127.0.0.1:8421
python backend\api_v2.py
pause
