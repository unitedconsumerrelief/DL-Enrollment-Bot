# Question Preprocessing System

## Problem
Agents sometimes send poorly formatted questions like:
- "Oportun is eligble?" (typo + incomplete)
- "mortgage accepted?" (missing context)
- "regional finance" (not even a question)

## Solution
Added `preprocess_question()` function that:

### 1. **Fixes Common Typos** ✅
Automatically corrects:
- "eligble" → "eligible"
- "eligibile" → "eligible"
- "accepts" → "accept"
- "elevat" → "elevate"
- "clariy" → "clarity"
- "mortagage" → "mortgage"

### 2. **Expands Incomplete Questions** ✅
Uses GPT-3.5-turbo to intelligently reformat:
- "Oportun is eligble?" → "Is Oportun eligible for Elevate or Clarity programs?"
- "mortgage accepted?" → "Are mortgage loans accepted in Elevate or Clarity programs?"
- "regional finance" → "Is Regional Finance accepted in Elevate or Clarity programs?"

### 3. **Adds Missing Context** ✅
If question mentions a creditor but not the programs:
- Adds "for Elevate or Clarity programs" context
- Adds proper question starters ("Is", "Does", "What", etc.)
- Adds question marks if missing

### 4. **Smart Detection** ✅
Only processes questions that need help:
- Very short questions (≤3 words)
- Questions missing question words ("is", "does", "what", etc.)
- Questions mentioning creditors/programs

## Examples

| Input | Output |
|-------|--------|
| "Oportun is eligble?" | "Is Oportun eligible for Elevate or Clarity programs?" |
| "mortgage accepted?" | "Are mortgage loans accepted in Elevate or Clarity programs?" |
| "regional finance" | "Is Regional Finance accepted in Elevate or Clarity programs?" |
| "What is minimum payment?" | "What is the minimum payment for Elevate or Clarity programs?" |

## Performance
- **Fast**: Uses GPT-3.5-turbo (cheaper, faster than GPT-4)
- **Efficient**: Only processes questions that need help
- **Fallback**: If clarification fails, uses original question
- **Logged**: All clarifications are logged for debugging

## Configuration
The preprocessing is automatic and happens before document search. No configuration needed!

---

**This ensures the bot understands even poorly formatted questions from agents!** 🎯

