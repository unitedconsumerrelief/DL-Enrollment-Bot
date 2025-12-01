# Quick Test Instructions

## The Problem
The test script was closing immediately after loading. This has been fixed!

## How to Run the Test

### Method 1: Use the Batch File (Easiest)
Double-click `test_bot_local.bat` in Windows Explorer, or run:
```bash
test_bot_local.bat
```

This will:
- Find Python automatically
- Run the test script
- Keep the window open so you can see results

### Method 2: Run from Command Prompt
1. Open Command Prompt
2. Navigate to the project folder:
   ```bash
   cd X:\Deudas_slack_doc_gpt_bot
   ```
3. Run:
   ```bash
   python test_bot_local.py
   ```

## What You Should See

1. **Initialization** (takes 1-2 minutes):
   ```
   🚀 Initializing bot for local testing...
   ✅ Loaded X chunks from documents
   🔢 Creating embeddings (this may take a minute)...
   ✅ Bot initialized and ready for testing!
   ```

2. **Mode Selection**:
   ```
   Select testing mode:
   1. Interactive mode (enter questions one by one)
   2. Batch test mode (test predefined questions)
   3. Quick test (test one question and exit)
   ```

3. **Then you can test questions!**

## If It Still Closes Immediately

1. **Check for errors**: Look at the output before it closes
2. **Run from Command Prompt**: This shows errors better
3. **Check your .env file**: Make sure `OPENAI_API_KEY` is set
4. **Check documents folder**: Make sure PDF/TXT files exist

## Common Issues

### "OPENAI_API_KEY not found"
- Create a `.env` file with your API key
- Format: `OPENAI_API_KEY=sk-your-key-here`

### "No document chunks loaded"
- Check that `documents/` folder exists
- Make sure it contains PDF or TXT files

### "Python not found"
- Make sure Python is installed
- Add Python to your PATH
- Or use: `py test_bot_local.py` instead

---

**Try running `test_bot_local.bat` now - it should work!**

