@echo off
title Audio Signal Processing Agent
cd /d "%~dp0"

echo.
echo  =====================================================
echo   Audio Signal Processing Agent  -  Starting...
echo  =====================================================
echo.

:: Check Python is available
where python >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python from https://python.org
    pause
    exit /b 1
)

:: Install/upgrade dependencies silently
echo  Checking dependencies...
python -m pip install flask librosa numpy soundfile scipy --quiet --exists-action i 2>nul

:: Open browser after 3 seconds (in background)
start "" /B cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:5000"

echo  Dashboard opening at http://127.0.0.1:5000
echo  Close this window to stop the server.
echo.

:: Start Flask app
python app.py

pause
