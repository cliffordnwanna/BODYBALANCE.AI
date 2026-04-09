"""
BODYBALANCE.AI - Web Widget Backend API
FastAPI server for embeddable chat widget.

SETUP INSTRUCTIONS:
1. Install dependencies: pip install -r requirements.txt
2. Run server: uvicorn backend_api:app --reload --port 8000
3. Embed widget on your website using the provided HTML snippet

DEPLOYMENT OPTIONS:
- Fly.io: flyctl launch
- Render: Add as Web Service
- Railway: Deploy from GitHub
- Vercel: Use vercel-python runtime
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import ChatbotEngine, Session, load_qa_pairs
from src.utils import get_config, setup_logging, log_query, log_feedback, generate_session_id

# Initialize
config = get_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="BODYBALANCE.AI API",
    description="REST API for the BODYBALANCE.AI chatbot",
    version="2.0.0"
)

# CORS - Allow all origins for widget embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
qa_pairs = load_qa_pairs(config)
chatbot = ChatbotEngine(
    qa_pairs=qa_pairs,
    similarity_threshold=config.get("similarity_threshold", 0.3)
)

# Session storage (in production, use Redis)
sessions = {}


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    session_id: str
    suggestions: Optional[List[str]] = None
    ask_feedback: bool = False


class FeedbackRequest(BaseModel):
    session_id: str
    helpful: str  # Yes, No, Somewhat
    suggestion: Optional[str] = None


class WidgetConfig(BaseModel):
    title: str
    subtitle: str
    primary_color: str
    position: str = "bottom-right"
    greeting: str


def get_or_create_session(session_id: Optional[str]) -> tuple:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return sessions[session_id], session_id
    
    new_id = generate_session_id()
    sessions[new_id] = Session(new_id)
    return sessions[new_id], new_id


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "BODYBALANCE.AI API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "POST /api/chat",
            "feedback": "POST /api/feedback",
            "widget_config": "GET /api/widget/config",
            "widget_demo": "GET /widget/demo",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "qa_pairs_loaded": len(qa_pairs),
        "active_sessions": len(sessions)
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response.
    
    Args:
        request: ChatRequest with message and optional session_id
        
    Returns:
        ChatResponse with bot response, confidence, and session info
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Get or create session
    session, session_id = get_or_create_session(request.session_id)
    
    # Get chatbot response
    answer, confidence, matched_question = chatbot.find_answer(request.message)
    
    suggestions = None
    if answer:
        bot_response = answer
    else:
        # No match - get suggestions
        suggestions = chatbot.get_suggestions(request.message, n=3)
        bot_response = (
            "I'm sorry, I couldn't find a relevant answer to your question. "
            f"For further assistance, please contact {config.get('support_email', 'support@bodybalance.com')}"
        )
    
    # Update session
    session.add_exchange(request.message, bot_response, confidence, matched_question)
    
    # Log query
    log_query(
        session_id=session_id,
        user_input=request.message,
        bot_response=bot_response,
        confidence=confidence,
        matched_question=matched_question,
        metadata={"channel": "web_widget"}
    )
    
    return ChatResponse(
        response=bot_response,
        confidence=confidence,
        session_id=session_id,
        suggestions=suggestions,
        ask_feedback=session.should_ask_feedback()
    )


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    """
    Submit user feedback.
    
    Args:
        request: FeedbackRequest with session_id and feedback
        
    Returns:
        Success message
    """
    session = sessions.get(request.session_id)
    exchange_count = session.exchange_count if session else 0
    
    log_feedback(
        session_id=request.session_id,
        helpful=request.helpful,
        suggestion=request.suggestion,
        exchange_count=exchange_count
    )
    
    if session:
        session.mark_feedback_collected()
    
    return {"status": "success", "message": "Thank you for your feedback!"}


@app.get("/api/widget/config", response_model=WidgetConfig)
async def widget_config():
    """Get widget configuration for embedding."""
    return WidgetConfig(
        title=config.get("app_title", "BODYBALANCE.AI"),
        subtitle=config.get("app_subtitle", "Your AI-Powered Wellness Assistant"),
        primary_color=config.get("primary_color", "#2E7D32"),
        position="bottom-right",
        greeting="👋 Hi! How can I help you today?"
    )


@app.get("/widget/demo", response_class=HTMLResponse)
async def widget_demo():
    """Demo page showing the embedded widget."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BODYBALANCE.AI Widget Demo</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                background: #f5f5f5;
            }
            h1 { color: #2E7D32; }
            .demo-content {
                background: white;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            code {
                background: #f0f0f0;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.9rem;
            }
            pre {
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 1rem;
                border-radius: 8px;
                overflow-x: auto;
            }
        </style>
    </head>
    <body>
        <h1>💪 BODYBALANCE.AI Widget Demo</h1>
        <div class="demo-content">
            <p>This page demonstrates the embeddable chat widget. Look at the bottom-right corner!</p>
            <h2>Integration Code</h2>
            <p>Add this snippet to your website:</p>
            <pre>&lt;script src="https://your-domain.com/widget.js"&gt;&lt;/script&gt;
&lt;script&gt;
  BodyBalanceWidget.init({
    apiUrl: 'https://your-api-domain.com',
    position: 'bottom-right',
    primaryColor: '#2E7D32'
  });
&lt;/script&gt;</pre>
        </div>
        
        <!-- Embedded Widget -->
        <div id="bodybalance-widget"></div>
        <script>
            // Inline widget for demo
            (function() {
                const API_URL = window.location.origin;
                let sessionId = null;
                let isOpen = false;
                
                // Create widget HTML
                const widgetHTML = `
                    <div id="bb-widget-container" style="
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        z-index: 9999;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    ">
                        <div id="bb-chat-window" style="
                            display: none;
                            width: 350px;
                            height: 500px;
                            background: white;
                            border-radius: 12px;
                            box-shadow: 0 5px 40px rgba(0,0,0,0.16);
                            flex-direction: column;
                            overflow: hidden;
                        ">
                            <div style="
                                background: #2E7D32;
                                color: white;
                                padding: 16px;
                                font-weight: 600;
                            ">
                                💪 BODYBALANCE.AI
                                <span onclick="toggleWidget()" style="float: right; cursor: pointer;">✕</span>
                            </div>
                            <div id="bb-messages" style="
                                flex: 1;
                                overflow-y: auto;
                                padding: 16px;
                                background: #f9f9f9;
                            ">
                                <div style="
                                    background: #E8F5E9;
                                    padding: 12px;
                                    border-radius: 12px;
                                    margin-bottom: 8px;
                                ">
                                    👋 Hi! How can I help you today?
                                </div>
                            </div>
                            <div style="padding: 12px; border-top: 1px solid #eee;">
                                <input type="text" id="bb-input" placeholder="Type your message..." style="
                                    width: calc(100% - 60px);
                                    padding: 10px;
                                    border: 1px solid #ddd;
                                    border-radius: 20px;
                                    outline: none;
                                " onkeypress="if(event.key==='Enter')sendMessage()">
                                <button onclick="sendMessage()" style="
                                    width: 50px;
                                    padding: 10px;
                                    background: #2E7D32;
                                    color: white;
                                    border: none;
                                    border-radius: 20px;
                                    cursor: pointer;
                                ">➤</button>
                            </div>
                        </div>
                        <button id="bb-toggle" onclick="toggleWidget()" style="
                            width: 60px;
                            height: 60px;
                            border-radius: 50%;
                            background: #2E7D32;
                            color: white;
                            border: none;
                            cursor: pointer;
                            font-size: 24px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        ">💬</button>
                    </div>
                `;
                
                document.getElementById('bodybalance-widget').innerHTML = widgetHTML;
                
                window.toggleWidget = function() {
                    const chatWindow = document.getElementById('bb-chat-window');
                    const toggleBtn = document.getElementById('bb-toggle');
                    isOpen = !isOpen;
                    chatWindow.style.display = isOpen ? 'flex' : 'none';
                    toggleBtn.style.display = isOpen ? 'none' : 'block';
                };
                
                window.sendMessage = async function() {
                    const input = document.getElementById('bb-input');
                    const message = input.value.trim();
                    if (!message) return;
                    
                    const messagesDiv = document.getElementById('bb-messages');
                    
                    // Add user message
                    messagesDiv.innerHTML += `
                        <div style="
                            background: #E3F2FD;
                            padding: 12px;
                            border-radius: 12px;
                            margin-bottom: 8px;
                            text-align: right;
                        ">${message}</div>
                    `;
                    
                    input.value = '';
                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    
                    // Call API
                    try {
                        const response = await fetch(API_URL + '/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({message, session_id: sessionId})
                        });
                        const data = await response.json();
                        sessionId = data.session_id;
                        
                        // Add bot response
                        messagesDiv.innerHTML += `
                            <div style="
                                background: #E8F5E9;
                                padding: 12px;
                                border-radius: 12px;
                                margin-bottom: 8px;
                            ">${data.response}</div>
                        `;
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    } catch (error) {
                        messagesDiv.innerHTML += `
                            <div style="
                                background: #FFEBEE;
                                padding: 12px;
                                border-radius: 12px;
                                margin-bottom: 8px;
                            ">Sorry, something went wrong. Please try again.</div>
                        `;
                    }
                };
            })();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           BODYBALANCE.AI Web API                             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Server running on port {port}                                  ║
    ║  API Docs: http://localhost:{port}/docs                         ║
    ║  Widget Demo: http://localhost:{port}/widget/demo               ║
    ║  Health: http://localhost:{port}/health                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=port)
