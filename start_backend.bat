@echo off
cd /d %~dp0\backend

echo Starting AI Web Chat Backend...
echo.

pip install -r requirements.txt

python main.py

pause
