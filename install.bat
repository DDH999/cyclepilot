@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 安装 QuantPilot 依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [失败] 安装失败，可换国内源重试：
  echo   python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
  exit /b 1
)
echo 依赖完成，运行自检...
python backend\preflight.py
pause
