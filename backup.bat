@echo off
chcp 65001 >nul
cd /d "%~dp0"
set STAMP=%date:~0,4%%date:~5,2%%date:~8,2%
set DEST=backups\%STAMP%
echo 备份到 %DEST%
if not exist "%DEST%" mkdir "%DEST%"
xcopy reports "%DEST%\reports" /E /I /Q /Y >nul 2>&1
if exist .state xcopy .state "%DEST%\.state" /E /I /Q /Y >nul 2>&1
for %%f in (holdings.csv dca_ledger.csv watchlist.txt) do (
  if exist %%f copy /Y %%f "%DEST%\" >nul
)
echo 完成：持仓、定投台账、报告、风控记录已备份
pause
