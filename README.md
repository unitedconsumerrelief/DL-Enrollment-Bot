# DL Enrollment Bot - Slack Document Assistant

A Slack bot that automatically answers agent questions about Elevate and Clarity debt relief programs using document search and AI.

## Features

- 🤖 **Automatic Question Answering** - Responds to @mentions in Slack
- 🔍 **Hybrid Document Search** - Combines vector and keyword search for better accuracy
- 🌐 **Bilingual Support** - Answers in both English and Spanish
- 🛡️ **Reliable** - Retry logic, error handling, and comprehensive logging
- 📚 **Document-Based** - Uses policy documents and training materials
- ✨ **Smart Preprocessing** - Fixes typos and clarifies incomplete questions

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Slack Bot Token and App Token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/unitedconsumerrelief/DL-Enrollment-Bot.git
cd DL-Enrollment-Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```env
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
OPENAI_API_KEY=sk-your-openai-key
```

4. Run the bot:
```bash
python slack_doc_bot.py
```

## Testing

Test locally without Slack:
```bash
python test_bot_local.py
```

## Deployment

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for deployment instructions to Render.

## Documentation

- [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) - Complete deployment guide
- [SEARCH_IMPROVEMENTS.md](SEARCH_IMPROVEMENTS.md) - Document search improvements
- [QUESTION_PREPROCESSING.md](QUESTION_PREPROCESSING.md) - Question preprocessing system
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Local testing guide

## Project Structure

```
DL-Enrollment-Bot/
├── slack_doc_bot.py          # Main bot application
├── policy_codex_full_ready.py # Policy codex definitions
├── documents/                 # Policy documents (PDF/TXT)
├── requirements.txt           # Python dependencies
├── render.yaml               # Render deployment config
├── Procfile                  # Alternative deployment config
└── test_bot_local.py         # Local testing script
```

## License

Private repository - United Consumer Relief

