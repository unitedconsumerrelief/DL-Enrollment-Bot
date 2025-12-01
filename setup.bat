@echo off
echo ========================================
echo Slack Document Bot Setup
echo ========================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully
echo.

echo Step 2: Checking for .env file...
if not exist ".env" (
    echo ⚠️  .env file not found!
    echo Please create a .env file with the following content:
    echo.
    echo SLACK_BOT_TOKEN=xoxb-your-bot-token-here
    echo SLACK_APP_TOKEN=xapp-your-app-token-here
    echo OPENAI_API_KEY=your_openai_api_key_here
    echo.
    echo Replace 'your_openai_api_key_here' with your actual OpenAI API key
    echo.
    pause
    exit /b 1
) else (
    echo ✅ .env file found
)
echo.

echo Step 3: Checking documents folder...
if not exist "documents" (
    echo ❌ documents folder not found!
    echo Please ensure all document files are in the documents/ folder
    pause
    exit /b 1
) else (
    echo ✅ documents folder found
)
echo.

echo Step 4: Testing bot startup...
echo Starting bot for testing...
python slack_doc_bot.py
echo.
echo Setup complete! The bot should now be ready to run.
echo Use 'run_bot.bat' to start the bot in production.
pause 