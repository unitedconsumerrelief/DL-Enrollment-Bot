@echo off
echo ========================================
echo Slack Document Bot - Production Run
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    echo ✅ Virtual environment found, using venv...
    set PYTHON_CMD=venv\Scripts\python.exe
) else (
    echo ⚠️  Virtual environment not found, using system Python...
    set PYTHON_CMD=python
)

REM Install/update dependencies
echo Installing dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo Please create a .env file with your API keys
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo ✅ Environment file found
echo.
echo 🚀 Starting Slack Document Bot...
echo (Press Ctrl+C to stop the bot)
echo.

REM Run the bot
%PYTHON_CMD% slack_doc_bot.py

echo.
echo Bot stopped.
pause

