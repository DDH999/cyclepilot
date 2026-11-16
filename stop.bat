@echo off
chcp 65001 >nul
echo 停止 QuantPilot（端口 8421）
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8421 ^| findstr LISTENING') do (
  taskkill /PID %%a /F >nul 2>&1
  echo   已结束进程 %%a
  set FOUND=1
)
if %FOUND%==0 echo   没有在运行的实例
pause
