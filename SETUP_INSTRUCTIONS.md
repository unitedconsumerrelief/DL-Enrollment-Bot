# Slack Document Bot Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (to clone the repository)

## Step 1: Clone/Download the Project
Download all files to a folder on the target computer.

## Step 2: Install Python Dependencies
Open a terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

## Step 3: Install System Dependencies (if needed)
The bot uses OCR for PDF processing. You may need to install Tesseract:

**Windows:**
- Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your system PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

## Step 4: Create Environment File
Create a file named `.env` in the project root with the following content:
```
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Step 5: Verify Document Files
Ensure the `documents/` folder contains all the PDF and TXT files:
- 10. ELEVATE_ FORTH Enrollment Criteria_v07.08.24 (3) (3).pdf
- 11. State List_v06.01.25 (3).pdf
- Clarity.pdf
- ComparisonTable.txt
- Debt Program Comparison Table.pdf
- Disqualified.txt
- Elevate.pdf
- Elevate.txt
- State List.pdf
- StateList.txt
- Unacceptable Credit Union.pdf
- UnacceptableCreditUnion.txt

## Step 6: Test the Bot
Run the bot to test if everything is working:
```bash
python slack_doc_bot.py
```

You should see output like:
```
🚀 Starting final patched Slack DocGPT bot with codex and document fallback...
📄 Loading and chunking documents...
✅ Bot is ready.
```

## Step 7: Run the Bot (Production)
Use the provided batch file for Windows:
```bash
run_bot.bat
```

Or run directly:
```bash
python slack_doc_bot.py
```

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Run `pip install -r requirements.txt` again
2. **Tesseract not found**: Install Tesseract OCR (see Step 3)
3. **OpenAI API errors**: Check your API key in the `.env` file
4. **Slack connection errors**: Verify your tokens are correct

### Logs to Watch For:
- ✅ "Bot is ready" - Everything is working
- ❌ Error messages - Check the specific error and resolve accordingly

## Security Notes:
- Keep your `.env` file secure and never commit it to version control
- The bot tokens provided are already configured for your Slack workspace
- Only share the OpenAI API key with trusted team members

## Bot Features:
- Answers questions about Elevate and Clarity debt relief programs
- Supports both English and Spanish
- Uses document search and hardcoded rules
- Responds to @mentions in Slack channels 