"""
BODYBALANCE.AI - WhatsApp Bot Template
Integration with Twilio WhatsApp Business API.

SETUP INSTRUCTIONS:
1. Create a Twilio account at https://www.twilio.com
2. Enable WhatsApp Sandbox in Twilio Console
3. Get your Account SID, Auth Token, and WhatsApp number
4. Set environment variables (see below)
5. Deploy this script to a server with a public URL
6. Configure Twilio webhook to point to your /webhook endpoint

ENVIRONMENT VARIABLES:
- TWILIO_ACCOUNT_SID: Your Twilio Account SID
- TWILIO_AUTH_TOKEN: Your Twilio Auth Token
- TWILIO_WHATSAPP_NUMBER: Your Twilio WhatsApp number (e.g., whatsapp:+14155238886)

DEPLOYMENT OPTIONS:
- Heroku: Add Procfile with "web: python whatsapp_bot.py"
- Railway: Deploy directly from GitHub
- Render: Add as a Web Service
- AWS Lambda: Use Zappa or Serverless Framework
"""

import os
import sys
from pathlib import Path
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import ChatbotEngine, Session, load_qa_pairs
from src.utils import get_config, setup_logging, log_query, generate_session_id

# Initialize
config = get_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# Initialize Twilio client (optional, for proactive messaging)
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize chatbot
qa_pairs = load_qa_pairs(config)
chatbot = ChatbotEngine(
    qa_pairs=qa_pairs,
    similarity_threshold=config.get("similarity_threshold", 0.3)
)

# Session storage (in production, use Redis or database)
sessions = {}


def get_or_create_session(phone_number: str) -> Session:
    """Get existing session or create new one for a phone number."""
    if phone_number not in sessions:
        session_id = generate_session_id()
        sessions[phone_number] = Session(session_id)
        logger.info(f"New session created for {phone_number[:6]}***")
    return sessions[phone_number]


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Handle incoming WhatsApp messages from Twilio.
    
    Twilio sends POST requests with:
    - From: Sender's WhatsApp number (e.g., whatsapp:+1234567890)
    - Body: Message text
    - To: Your Twilio WhatsApp number
    """
    # Get message details
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")
    
    logger.info(f"Received message from {from_number[:15]}***: {incoming_msg[:50]}...")
    
    # Get or create session
    session = get_or_create_session(from_number)
    
    # Process message
    response = MessagingResponse()
    
    if not incoming_msg:
        response.message("👋 Hello! How can I help you today?")
        return str(response)
    
    # Check for special commands
    if incoming_msg.lower() in ["hi", "hello", "start", "menu"]:
        welcome_msg = (
            f"👋 Welcome to {config.get('app_title', 'BODYBALANCE.AI')}!\n\n"
            "I can help you with:\n"
            "📦 Product information\n"
            "🚚 Shipping & delivery\n"
            "↩️ Returns & refunds\n"
            "🛠️ Technical support\n\n"
            "Just type your question!"
        )
        response.message(welcome_msg)
        return str(response)
    
    if incoming_msg.lower() in ["bye", "goodbye", "exit", "quit"]:
        goodbye_msg = (
            "Thank you for chatting with us! 🙏\n\n"
            "Was this conversation helpful?\n"
            "Reply: YES / NO / SOMEWHAT"
        )
        response.message(goodbye_msg)
        return str(response)
    
    # Handle feedback responses
    if incoming_msg.upper() in ["YES", "NO", "SOMEWHAT"]:
        from src.utils import log_feedback
        log_feedback(
            session_id=session.session_id,
            helpful=incoming_msg.capitalize(),
            exchange_count=session.exchange_count
        )
        response.message("Thank you for your feedback! 🙏 Have a great day!")
        return str(response)
    
    # Get chatbot response
    answer, confidence, matched_question = chatbot.find_answer(incoming_msg)
    
    if answer:
        bot_response = answer
        
        # Add confidence note for low-confidence matches
        if 0.3 <= confidence < 0.5:
            bot_response += f"\n\n_ℹ️ I'm {int(confidence*100)}% confident about this answer._"
    else:
        # No match found
        suggestions = chatbot.get_suggestions(incoming_msg, n=3)
        bot_response = "I'm sorry, I couldn't find a relevant answer. 😔\n\n"
        
        if suggestions:
            bot_response += "Did you mean:\n"
            for i, suggestion in enumerate(suggestions, 1):
                bot_response += f"{i}. {suggestion}\n"
            bot_response += "\n"
        
        bot_response += f"For further help, contact: {config.get('support_email', 'support@bodybalance.com')}"
    
    # Update session
    session.add_exchange(incoming_msg, bot_response, confidence, matched_question)
    
    # Log query
    log_query(
        session_id=session.session_id,
        user_input=incoming_msg,
        bot_response=bot_response,
        confidence=confidence,
        matched_question=matched_question,
        metadata={"channel": "whatsapp", "from": from_number[:10] + "***"}
    )
    
    # Send response
    response.message(bot_response)
    
    # Ask for feedback after 3 exchanges
    if session.should_ask_feedback():
        response.message(
            "💬 Quick question: Was this conversation helpful?\n"
            "Reply: YES / NO / SOMEWHAT"
        )
        session.mark_feedback_collected()
    
    return str(response)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return {"status": "healthy", "qa_pairs": len(qa_pairs)}


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with setup instructions."""
    return """
    <h1>BODYBALANCE.AI WhatsApp Bot</h1>
    <p>This is the WhatsApp webhook endpoint.</p>
    <h2>Setup Instructions:</h2>
    <ol>
        <li>Configure your Twilio WhatsApp Sandbox</li>
        <li>Set the webhook URL to: <code>https://your-domain.com/webhook</code></li>
        <li>Send a message to your Twilio WhatsApp number</li>
    </ol>
    <p>Health check: <a href="/health">/health</a></p>
    """


def send_proactive_message(to_number: str, message: str):
    """
    Send a proactive message to a WhatsApp number.
    
    Note: This requires an approved WhatsApp template for business-initiated messages.
    
    Args:
        to_number: Recipient's WhatsApp number (e.g., whatsapp:+1234567890)
        message: Message to send
    """
    if not twilio_client:
        logger.error("Twilio client not initialized")
        return False
    
    try:
        msg = twilio_client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number
        )
        logger.info(f"Proactive message sent: {msg.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send proactive message: {e}")
        return False


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           BODYBALANCE.AI WhatsApp Bot                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Server running on port {port}                                  ║
    ║  Webhook URL: http://localhost:{port}/webhook                   ║
    ║  Health check: http://localhost:{port}/health                   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  For production, deploy to Heroku/Railway/Render             ║
    ║  and configure Twilio webhook to your public URL             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host="0.0.0.0", port=port, debug=True)
