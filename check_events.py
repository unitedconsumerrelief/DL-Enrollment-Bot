import os
import logging
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = App(token=os.getenv("SLACK_BOT_TOKEN"))

@app.event("app_mention")
def handle_app_mention_events(body, event, say):
    logger.info(f"🔔 Received app_mention event: {event}")
    
    # Get the text and user info
    text = event.get("text", "")
    user = event.get("user", "")
    channel = event.get("channel", "")
    
    logger.info(f"📝 Text: {text}")
    logger.info(f"👤 User: {user}")
    logger.info(f"📢 Channel: {channel}")
    
    # Respond immediately to test
    try:
        say(f"✅ Bot received your message: {text}")
        logger.info("✅ Response sent successfully")
    except Exception as e:
        logger.error(f"❌ Error sending response: {e}")

@app.event("message")
def handle_message_events(body, event, say):
    logger.info(f"📨 Received message event: {event.get('text', '')}")

if __name__ == "__main__":
    print("🔍 Starting bot with debug logging...")
    print("📋 This will show all events the bot receives")
    print("💡 Try mentioning the bot in Slack now")
    
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start() 