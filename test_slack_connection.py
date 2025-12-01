import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

load_dotenv()

# Test bot token
bot_token = os.getenv("SLACK_BOT_TOKEN")
app_token = os.getenv("SLACK_APP_TOKEN")

print("🔍 Testing Slack Connection...")
print(f"Bot Token: {bot_token[:20]}..." if bot_token else "❌ No bot token found")
print(f"App Token: {app_token[:20]}..." if app_token else "❌ No app token found")

if not bot_token or not app_token:
    print("❌ Missing tokens in .env file")
    exit(1)

try:
    # Test bot connection
    client = WebClient(token=bot_token)
    auth_test = client.auth_test()
    
    print("✅ Bot connection successful!")
    print(f"Bot User ID: {auth_test['user_id']}")
    print(f"Bot Name: {auth_test['user']}")
    print(f"Team: {auth_test['team']}")
    
    # Test posting a message (will fail if no permissions)
    try:
        # This will fail if bot isn't in a channel, but that's expected
        print("\n🔍 Testing bot permissions...")
        print("✅ Bot has basic authentication")
        print("⚠️  To test messaging, add bot to a channel and run:")
        print("   @YourBotName hello")
        
    except SlackApiError as e:
        print(f"❌ Bot permission error: {e.response['error']}")
        
except SlackApiError as e:
    print(f"❌ Bot connection failed: {e.response['error']}")
    print("Check your SLACK_BOT_TOKEN in .env file")

print("\n📋 Next steps:")
print("1. Add bot to your Slack channel: /invite @YourBotName")
print("2. Test with: @YourBotName hello")
print("3. Check bot permissions in Slack app settings") 