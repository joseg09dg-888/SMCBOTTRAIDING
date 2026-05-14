@echo off
setlocal

set BOT_DIR=C:\Users\jose-\projects\trading_agent
set PYTHON=%BOT_DIR%\.venv\Scripts\python.exe
set LOGFILE=%BOT_DIR%\logs\bot.log
set CAPITAL=1000

cd /d %BOT_DIR%

echo [%DATE% %TIME%] Iniciando SMC Bot... >> "%LOGFILE%"

%PYTHON% startup.py --auto --capital %CAPITAL% --reason auto_restart >> "%LOGFILE%" 2>&1

echo [%DATE% %TIME%] Proceso terminado con codigo %ERRORLEVEL% >> "%LOGFILE%"
endlocal
