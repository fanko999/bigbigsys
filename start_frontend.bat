@echo off
cd /d %~dp0

echo Starting AI Web Chat Frontend...
echo.

:: 使用Python内置HTTP服务器
python -m http.server 5180

pause
