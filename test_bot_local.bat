@echo off
cd /d "%~dp0"
echo ========================================
echo Slack Bot Local Testing
echo ========================================
echo.

REM Use system Python directly (most reliable)
set PYTHON_CMD=python

REM Verify Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please make sure Python is installed and in your PATH.
    echo.
    pause
    exit /b 1
)

echo Found Python, starting test script...
echo.

REM Run the test script
python test_bot_local.py

REM Keep window open
echo.
echo ========================================
echo Test script finished.
echo ========================================
pause
