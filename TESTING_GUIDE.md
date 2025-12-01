# Local Testing Guide

## Test Without Slack

You can now test the bot's document search improvements **without connecting to Slack**. This prevents disrupting your active Slack channel.

## Quick Start

### 1. Run the Test Script

```bash
python test_bot_local.py
```

### 2. Choose Testing Mode

**Option 1: Interactive Mode** (Recommended)
- Enter questions one by one
- See results immediately
- Type `quit` to exit

**Option 2: Batch Test Mode**
- Tests 8 predefined questions automatically
- Good for quick validation

## What Gets Tested

✅ Document loading and chunking  
✅ Hybrid search (vector + keyword)  
✅ Chunk validation  
✅ GPT answer generation  
✅ Full question-answering pipeline  

## Example Test Questions

Try these to verify improvements:

1. **"Does Elevate accept mortgage loans?"**
   - Should find: Mortgage rejection info

2. **"Is Oportun accepted in California?"**
   - Should find: State-specific Oportun rules

3. **"What is the minimum payment for Clarity?"**
   - Should find: Payment information

4. **"Does Clarity accept Regional Finance?"**
   - Should find: Regional Finance acceptance rules

5. **"What creditors are disqualified?"**
   - Should find: Disqualified creditor lists

## Understanding the Output

### During Testing, You'll See:

```
🔍 Running hybrid search...
   Found X chunks from hybrid search
   After validation: Y valid chunks

🤖 Generating answer...

💬 ANSWER:
[Full answer with English and Spanish]
```

### What to Look For:

✅ **Good Signs:**
- "Found 20-35 chunks" - Good coverage
- "After validation: 5-15 chunks" - Reasonable filtering
- Answer contains specific information from documents
- No "information not found" when info exists

⚠️ **Warning Signs:**
- "Found 0 chunks" - Search failed
- "After validation: 0 chunks" - Too strict filtering
- "Information not found" when you know it exists

## Advanced: Show Retrieved Chunks

In interactive mode, type `chunks` before your question to see what chunks were retrieved:

```
❓ Your question: chunks
✅ Chunk display enabled. Enter your question:
❓ Your question: Does Elevate accept mortgage loans?
```

This helps debug why information might be missed.

## Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` file exists with your API key

### "No document chunks loaded"
- Check that `documents/` folder exists
- Verify PDF/TXT files are in the folder

### "Vector search failed"
- Check OpenAI API key is valid
- Check internet connection
- May be rate limited (wait a minute)

### Answers seem wrong
- Check the chunks shown (use `chunks` mode)
- Verify the information is actually in your documents
- Try rephrasing the question

## Comparing Before/After

### Before Improvements:
- Retrieved only 5 chunks
- Often said "not found" when info existed
- Missed keyword matches

### After Improvements:
- Retrieves 20-35 chunks
- Hybrid search catches more
- More lenient validation
- Better GPT instructions

## Next Steps

1. **Test locally** with `test_bot_local.py`
2. **Verify answers** are accurate
3. **Check logs** for any issues
4. **Deploy to Render** when satisfied

---

**Ready to test? Run:** `python test_bot_local.py`

