@echo off
REM ChatAudit full chatbot app for Windows - double-click this file.
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)
%PY% -m pip install --quiet flask scikit-learn pandas
echo Starting the ChatAudit app at http://localhost:5051 - keep this window open.
start "" http://localhost:5051
%PY% chat_app.py
pause
