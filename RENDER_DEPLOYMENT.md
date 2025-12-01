# Render Deployment Guide

This guide will walk you through deploying the Slack Document Bot to Render for 24/7 operation.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com) (free tier available)
2. **GitHub/GitLab Account**: Render deploys from Git repositories
3. **Environment Variables**: Have your tokens ready:
   - `SLACK_BOT_TOKEN`
   - `SLACK_APP_TOKEN`
   - `OPENAI_API_KEY`

---

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Initial commit - Ready for Render deployment"
```

### 1.2 Push to GitHub/GitLab

1. Create a new repository on GitHub or GitLab
2. Push your code:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

**Important**: Make sure `.env` is in `.gitignore` (already added) - never commit secrets!

---

## Step 2: Create Render Service

### 2.1 Create New Service

1. Log into [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Background Worker"**
3. Connect your GitHub/GitLab account if not already connected
4. Select your repository

### 2.2 Configure Service

**Basic Settings:**
- **Name**: `slack-doc-bot` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python slack_doc_bot.py`

**OR** if you're using `render.yaml`:
- Render will auto-detect and use it

### 2.3 Set Environment Variables

In the Render dashboard, go to **Environment** tab and add:

```
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
OPENAI_API_KEY=sk-your-openai-key-here
```

**Important**: 
- Click **"Save Changes"** after adding each variable
- Never commit these to Git!

---

## Step 3: Deploy

### 3.1 Manual Deploy

1. Click **"Manual Deploy"** → **"Deploy latest commit"**
2. Watch the build logs
3. Wait for deployment to complete

### 3.2 Auto-Deploy (Recommended)

- Render will automatically deploy when you push to your repository
- Enable in **Settings** → **Auto-Deploy**: `Yes`

---

## Step 4: Verify Deployment

### 4.1 Check Logs

1. Go to **Logs** tab in Render dashboard
2. Look for:
   - ✅ `🚀 Starting Slack DocGPT bot...`
   - ✅ `📚 Loaded X chunks from documents.`
   - ✅ `✅ Bot is ready and connected to Slack!`

### 4.2 Test in Slack

1. Go to your Slack workspace
2. Mention your bot: `@YourBotName test question`
3. You should receive a response

---

## Step 5: Monitor & Maintain

### 5.1 Monitoring

- **Logs**: Check Render dashboard logs regularly
- **Metrics**: Monitor CPU/Memory usage
- **Uptime**: Render free tier has some limitations

### 5.2 Updates

To update the bot:
1. Make changes locally
2. Commit and push to Git
3. Render auto-deploys (or manually deploy)

---

## Troubleshooting

### Bot Not Responding

1. **Check Logs**: Look for errors in Render dashboard
2. **Verify Tokens**: Ensure environment variables are set correctly
3. **Check Slack**: Verify bot is added to channel and has permissions

### Build Failures

1. **Dependencies**: Check `requirements.txt` is correct
2. **Python Version**: Render uses Python 3.9+ by default
3. **Build Logs**: Review error messages in build output

### Runtime Errors

1. **Memory Issues**: Free tier has 512MB RAM limit
   - Solution: Upgrade plan or optimize code
2. **Timeout**: Long-running operations may timeout
   - Solution: Already handled with retries in code
3. **API Rate Limits**: OpenAI rate limits
   - Solution: Already handled with exponential backoff

### OCR Issues (Tesseract)

If you see OCR errors:
- **Not Critical**: Bot will continue without OCR, using PDF text extraction only
- **To Fix**: Would require custom Docker image with Tesseract installed

---

## Cost Considerations

### Free Tier Limitations

- **512MB RAM**: Should be sufficient for this bot
- **750 hours/month**: Enough for 24/7 operation
- **Auto-sleep**: Free tier services sleep after 15 min inactivity
  - **Solution**: Upgrade to paid plan ($7/month) for always-on

### Paid Tier ($7/month)

- Always-on (no sleep)
- More resources
- Better performance

---

## Security Best Practices

1. ✅ **Never commit `.env` file** (already in `.gitignore`)
2. ✅ **Use Render's environment variables** (not hardcoded)
3. ✅ **Rotate tokens periodically**
4. ✅ **Monitor logs for suspicious activity**
5. ✅ **Keep dependencies updated**

---

## Next Steps After Deployment

1. ✅ Test bot responses in Slack
2. ✅ Monitor logs for first 24 hours
3. ✅ Set up alerts (if using paid tier)
4. ✅ Document any custom configurations
5. ✅ Share bot with your team

---

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Render Support**: Available in dashboard
- **Slack API Docs**: [api.slack.com](https://api.slack.com)

---

## Quick Reference

**Render Dashboard**: https://dashboard.render.com  
**Service Type**: Background Worker  
**Build Command**: `pip install -r requirements.txt`  
**Start Command**: `python slack_doc_bot.py`  
**Required Env Vars**: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `OPENAI_API_KEY`

---

**Ready to deploy? Start with Step 1!** 🚀

