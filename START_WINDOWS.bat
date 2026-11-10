@echo off
REM ChatAudit launcher for Windows - double-click this file.
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)

%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Download it from https://www.python.org/downloads/
  echo During installation, tick "Add Python to PATH". Then double-click this file again.
  pause
  exit /b
)

echo Installing required libraries (only slow the first time)...
%PY% -m pip install --quiet flask scikit-learn pandas

echo.
echo Starting ChatAudit. Your browser will open at http://localhost:5050
echo KEEP THIS WINDOW OPEN while using the app. Close it to stop.
echo.
start "" http://localhost:5050
%PY% app.py
pause
