"""
BODYBALANCE.AI - Main Streamlit Entry Point
Initialize vector store + chain in session state on first load, chat interface.
"""
import streamlit as st
import sys
import os
import time

# Handle module imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.vector_store import BodyBalanceVectorStore
from src.core.chains import BodyBalanceChain
from src.core.guardrails import check_red_flags
from src.ui.styles import inject_custom_styles
from src.ui.components import (
    render_exercise_card, render_cta_button,
    render_emergency_alert,
    render_whatsapp_link, render_clinic_cta_card
)

# Page Configuration
st.set_page_config(
    page_title="BodyBalance AI Concierge",
    layout="wide"
)

# Vector store with caching (lazy init to avoid refresh issues)
@st.cache_resource
def get_vector_store():
    """Initialize vector store once per server instance (cached)."""
    try:
        return BodyBalanceVectorStore()
    except Exception as e:
        st.error(f"⚠️ Clinical knowledge base failed to load. Please check your configuration or contact support.")
        st.error(f"Technical details: {str(e)}")
        st.stop()

# Lazy init chain on first use
def get_chain():
    if "chain" not in st.session_state:
        st.session_state.chain = BodyBalanceChain(get_vector_store())
    return st.session_state.chain

def extract_name_from_greeting(text):
    """Extract name from greeting like 'Hello, my name is John' or 'I am Jane'."""
    import re
    patterns = [
        r"my name is (\w+)",
        r"i am (\w+)",
        r"i'm (\w+)",
        r"call me (\w+)",
        r"this is (\w+)"
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1).capitalize()
    return None

def detect_intent(text):
    """Detect user intent from message."""
    text_lower = text.lower()
    
    pain_keywords = ['pain', 'hurt', 'ache', 'sore', 'injury', 'injured', 'back', 'neck', 'knee', 'shoulder', 'hip', 'ankle', 'wrist', 'headache', 'migraine', 'sprain', 'strain', 'swelling', 'stiff', 'stiffness', 'difficulty moving', 'cant move', "can't move", 'uncomfortable', 'suffering']
    booking_keywords = ['book', 'appointment', 'schedule', 'consultation', 'session', 'visit', 'see cherry', 'meet', 'register', 'sign up']
    curious_keywords = ['curious', 'wondering', 'what is', 'how does', 'tell me about', 'information', 'learn', 'know more', 'services', 'pricing', 'cost', 'price']
    testing_keywords = ['test', 'testing', 'try', 'trying out', 'demo', 'how does this work', 'what can you do']
    
    for keyword in pain_keywords:
        if keyword in text_lower:
            return 'pain'
    for keyword in booking_keywords:
        if keyword in text_lower:
            return 'booking'
    for keyword in curious_keywords:
        if keyword in text_lower:
            return 'curious'
    for keyword in testing_keywords:
        if keyword in text_lower:
            return 'testing'
    
    return 'general'

def get_intent_response(intent, name=None):
    """Get appropriate response based on intent."""
    greeting = f"Hello{name and ' ' + name or ''}! Welcome to BodyBalance Physiotherapy Clinic." if not st.session_state.chat_history else ""
    
    responses = {
        'pain': {
            'message': f"{greeting}\n\nI'm sorry to hear you're dealing with pain. That can be really uncomfortable and frustrating.\n\nTo help you best, may I ask:\n\n1. **Where exactly** is the pain located? (e.g., lower back, neck, left knee, right shoulder)\n2. **How long** have you had it? (days, weeks, months)\n\nOr if you prefer, I can connect you directly with Cherry for a proper assessment.",
            'next_step': 'pain_details',
            'show_quick_replies': True,
            'quick_replies': ['My back hurts', 'Neck pain for 2 weeks', 'Knee injury', 'Connect me with Cherry']
        },
        'booking': {
            'message': f"{greeting}\n\nI'd be happy to help you book an appointment with Cherry Nwanna, our lead physiotherapist.\n\nWe offer:\n- **In-Person Session**: ₦150,000\n- **Virtual Consultation**: ₦50,000\n\nClick below to book via WhatsApp:",
            'next_step': 'ready',
            'show_quick_replies': False,
            'quick_replies': []
        },
        'curious': {
            'message': f"{greeting}\n\nGreat question! I'd love to tell you more about physiotherapy and how we can help.\n\nPhysiotherapy helps with:\n- Pain relief and management\n- Recovery from injuries\n- Improving mobility and strength\n- Preventing future problems\n\nWhat would you like to know? Feel free to ask about our services, pricing, or how physiotherapy works.",
            'next_step': 'ready',
            'show_quick_replies': True,
            'quick_replies': ['What services do you offer?', 'How much does it cost?', 'Do you treat sports injuries?', 'What is physiotherapy?']
        },
        'testing': {
            'message': f"{greeting}\n\nNo problem at all! Feel free to test me out. I'm here to help answer questions about physiotherapy, our clinic services, or general wellness advice.\n\nWhat would you like to know?",
            'next_step': 'ready',
            'show_quick_replies': True,
            'quick_replies': ['What can you do?', 'Tell me about BodyBalance', 'Book an appointment', 'I have back pain']
        },
        'general': {
            'message': f"{greeting}\n\nHello! I'm the BodyBalance virtual assistant. I can help you with:\n\n- Understanding physiotherapy and how it can help\n- Information about our clinic and services\n- Booking an appointment with Cherry\n- General wellness guidance\n\nWhat brings you here today?",
            'next_step': 'intent',
            'show_quick_replies': True,
            'quick_replies': ['I am in pain', 'I want to book an appointment', 'Just curious about physiotherapy', 'Testing the app']
        }
    }
    
    return responses.get(intent, responses['general'])

def main():
    # Validate secrets first - fail fast with clear error
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("⚠️ OpenAI API key not configured.")
        st.info("Please add OPENAI_API_KEY to your Streamlit Cloud Secrets (Manage App → Secrets).")
        st.stop()
    
    # Initialize Session State - MUST be first thing in main()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "patient_name" not in st.session_state:
        st.session_state.patient_name = None
    if "intent" not in st.session_state:
        st.session_state.intent = None
    if "pain_location" not in st.session_state:
        st.session_state.pain_location = None
    if "pain_duration" not in st.session_state:
        st.session_state.pain_duration = None
    if "bmi" not in st.session_state:
        st.session_state.bmi = None
    if "bmi_group" not in st.session_state:
        st.session_state.bmi_group = None
    if "intake_step" not in st.session_state:
        st.session_state.intake_step = "greeting"
    if "last_message" not in st.session_state:
        st.session_state.last_message = None
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "total_queries": 0,
            "types": {"medical_advice": 0, "appointment": 0, "pricing": 0, "general": 0, "emergency": 0}
        }
    if "questions_after_ready" not in st.session_state:
        st.session_state.questions_after_ready = 0
    if "last_message_time" not in st.session_state:
        st.session_state.last_message_time = 0
    
    # Inject Styling
    inject_custom_styles()
    
    # Sidebar: Clinic Info & Metrics
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2em;">
            <div style="color: #1B5E20; font-weight: 700; font-size: 1.4em;">BodyBalance Clinic</div>
            <div style="color: #666; font-size: 0.9em;">360 Degree Wellness</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Clinic Hours")
        st.info("Mon-Fri: 8:00 AM - 6:00 PM\n\nSat: 9:00 AM - 2:00 PM")
        
        st.markdown("---")
        st.markdown("### Location")
        st.info("Lagos, Nigeria")
        
        st.markdown("---")
        st.markdown("### Book Treatment")
        render_clinic_cta_card(is_sidebar=True)
        
        st.markdown("---")
        st.markdown("### Metrics")
        
        # Defensive check for analytics
        if "analytics" in st.session_state and isinstance(st.session_state.analytics, dict):
            st.metric("Total Queries", st.session_state.analytics.get("total_queries", 0))
            
            with st.expander("Response Types"):
                types_data = st.session_state.analytics.get("types", {})
                for t, count in types_data.items():
                    st.write(f"**{t.replace('_', ' ').capitalize()}:** {count}")
        else:
            st.metric("Total Queries", 0)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.75em; color: #666; margin-top: 1em;">
            <strong>Clinical AI Disclaimer</strong><br/>
            BodyBalance AI provides educational information, not medical diagnosis.<br/>
            Always consult a human physiotherapist for persistent pain.
        </div>
        """, unsafe_allow_html=True)

    # Main Chat Area

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, dict):
                    # Render structured clinical response from dict
                    st.write(content.get("message", ""))
                    for exercise in content.get("exercises", []):
                        render_exercise_card(exercise)
                    if content.get("cta"):
                        render_cta_button(content["cta"])
                    
                    # Simple WhatsApp link below every response
                    render_whatsapp_link()
                else:
                    # Fallback for legacy object format
                    try:
                        st.write(content.message)
                        for exercise in content.exercises:
                            render_exercise_card(exercise.model_dump())
                        if content.cta:
                            render_cta_button(content.cta)
                        render_whatsapp_link()
                    except Exception:
                        st.write(str(content))
            else:
                st.write(message["content"])

    # INTAKE FLOW
    # Show greeting on first visit (no chat history)
    if len(st.session_state.chat_history) == 0 and st.session_state.intake_step == "greeting":
        with st.chat_message("assistant"):
            greeting_msg = """Hello! I'm BodyBalance.

I'm here to help you with pain relief, booking appointments, or answering questions about physiotherapy.

What brings you here today?"""
            
            st.write(greeting_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": greeting_msg})
            
            # Show hint suggestions (visual only, user types response)
            st.markdown("""
            <div style="margin-top:15px;">
                <div style="font-size:0.85em;color:#666;margin-bottom:8px;"><strong>Follow up:</strong></div>
                <ul style="font-size:0.9em;color:#333;margin:0;padding-left:20px;">
                    <li>"I have back pain"</li>
                    <li>"Book an appointment"</li>
                    <li>"What is physiotherapy?"</li>
                    <li>"Just testing"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.intake_step = "intent"

    # Chat Input - always render so user can type
    query = st.chat_input("How can BodyBalance Physiotherapy help you today?")
    
    # Input sanitization: strip, guard empty, and truncate at word boundary
    if query:
        query = query.strip()
        if not query:
            st.warning("Please type a message before sending.")
            query = None
        else:
            # Truncate at word boundary (max 500 chars)
            if len(query) > 500:
                query = query[:500].rsplit(' ', 1)[0] + "..."
        
        # Rate limiting: min 1 second between messages
        if query:
            current_time = time.time()
            if current_time - st.session_state.last_message_time < 1.0:
                st.warning("Please wait a moment before sending another message.")
                query = None  # Drop the message
            else:
                st.session_state.last_message_time = current_time
    
    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Process with Guardrails and Chain
        with st.chat_message("assistant"):
            # Ensure analytics exists
            if "analytics" not in st.session_state or not isinstance(st.session_state.analytics, dict):
                st.session_state.analytics = {
                    "total_queries": 0,
                    "types": {"medical_advice": 0, "appointment": 0, "pricing": 0, "general": 0, "emergency": 0}
                }
            
            st.session_state.analytics["total_queries"] += 1
            
            # 1. Check Guardrails first
            red_flag_msg = check_red_flags(query)
            if red_flag_msg:
                # Ensure types dict exists
                if "types" not in st.session_state.analytics:
                    st.session_state.analytics["types"] = {"medical_advice": 0, "appointment": 0, "pricing": 0, "general": 0, "emergency": 0}
                st.session_state.analytics["types"]["emergency"] = st.session_state.analytics["types"].get("emergency", 0) + 1
                render_emergency_alert(red_flag_msg)
                render_whatsapp_link()
                st.session_state.chat_history.append({"role": "assistant", "content": red_flag_msg})
            else:
                # INTAKE LOGIC - Don't count questions until intake is complete
                if st.session_state.intake_step == "intent":
                    # Try to extract name first
                    name = extract_name_from_greeting(query)
                    if name and not st.session_state.patient_name:
                        st.session_state.patient_name = name
                    
                    # Detect intent
                    intent = detect_intent(query)
                    st.session_state.intent = intent
                    
                    # Show WhatsApp CTA for booking intent (no follow-up hints - push to action)
                    if intent == 'booking':
                        response_data = get_intent_response(intent, st.session_state.patient_name)
                        st.write(response_data['message'])
                        st.session_state.chat_history.append({"role": "assistant", "content": response_data['message']})
                        st.session_state.intake_step = response_data['next_step']
                        render_whatsapp_link()
                        st.stop()  # Stop here to prevent further chat, force WhatsApp action
                    
                    # For pain intent: extract what we can, ask for rest only if needed
                    elif intent == 'pain':
                        text_lower = query.lower()
                        locations = ['back', 'neck', 'knee', 'shoulder', 'hip', 'ankle', 'wrist', 'head', 'arm', 'leg', 'foot']
                        for loc in locations:
                            if loc in text_lower:
                                st.session_state.pain_location = loc
                                break
                        # Extract duration if present
                        if any(word in text_lower for word in ['day', 'days']):
                            st.session_state.pain_duration = 'days'
                        elif any(word in text_lower for word in ['week', 'weeks']):
                            st.session_state.pain_duration = 'weeks'
                        elif any(word in text_lower for word in ['month', 'months']):
                            st.session_state.pain_duration = 'months'
                        elif any(word in text_lower for word in ['year', 'years']):
                            st.session_state.pain_duration = 'chronic'
                        
                        # Require BOTH location AND duration for accurate, tailored answers
                        missing_info = []
                        if not st.session_state.pain_location:
                            missing_info.append("where exactly the pain is (e.g., lower back, left knee, right shoulder)")
                        if not st.session_state.pain_duration:
                            missing_info.append("how long you've had it")
                        
                        if missing_info:
                            # Ask for missing info specifically
                            if len(missing_info) == 2:
                                clarify_msg = "I'm sorry to hear you're in pain. To give you the safest guidance, could you tell me:\n\n1. **Where exactly** is the pain? (e.g., lower back, left knee, right shoulder)\n2. **How long** have you had it?\n\nThis helps me suggest appropriate exercises."
                            else:
                                clarify_msg = f"Thanks for that. Just one more thing to help me tailor my advice. Could you share **{missing_info[0]}**?"
                            st.write(clarify_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": clarify_msg})
                            st.session_state.intake_step = "pain_details"
                        else:
                            # Have both location AND duration - proceed to RAG
                            st.session_state.intake_step = "ready"
                            # Fall through to RAG chain below
                    
                    # For other intents: show response and hints, don't go to RAG yet
                    else:
                        response_data = get_intent_response(intent, st.session_state.patient_name)
                        st.write(response_data['message'])
                        st.session_state.chat_history.append({"role": "assistant", "content": response_data['message']})
                        st.session_state.intake_step = response_data['next_step']
                        
                        # Show hint suggestions for follow-up (non-booking/pain intents only)
                        if response_data['show_quick_replies'] and response_data['quick_replies']:
                            hints_list = "".join([f'<li>"{h}"</li>' for h in response_data['quick_replies']])
                            st.markdown(f"""
                            <div style="margin-top:12px;">
                                <div style="font-size:0.85em;color:#666;margin-bottom:6px;"><strong>Follow up:</strong></div>
                                <ul style="font-size:0.9em;color:#333;margin:0;padding-left:20px;">
                                    {hints_list}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                
                elif st.session_state.intake_step == "pain_details":
                    # Extract pain location and duration from response
                    text_lower = query.lower()
                    
                    # Try to extract pain location (only if not already set)
                    if not st.session_state.pain_location:
                        locations = ['back', 'neck', 'knee', 'shoulder', 'hip', 'ankle', 'wrist', 'head', 'arm', 'leg', 'foot']
                        for loc in locations:
                            if loc in text_lower:
                                st.session_state.pain_location = loc
                                break
                    
                    # Try to extract duration (only if not already set)
                    if not st.session_state.pain_duration:
                        if any(word in text_lower for word in ['day', 'days']):
                            st.session_state.pain_duration = 'days'
                        elif any(word in text_lower for word in ['week', 'weeks']):
                            st.session_state.pain_duration = 'weeks'
                        elif any(word in text_lower for word in ['month', 'months']):
                            st.session_state.pain_duration = 'months'
                        elif any(word in text_lower for word in ['year', 'years']):
                            st.session_state.pain_duration = 'chronic'
                    
                    # Check if we now have BOTH location AND duration
                    if not st.session_state.pain_location or not st.session_state.pain_duration:
                        # Still missing something - ask specifically for what's missing
                        missing = []
                        if not st.session_state.pain_location:
                            missing.append("where exactly the pain is")
                        if not st.session_state.pain_duration:
                            missing.append("how long you've had it")
                        clarify_msg = f"Thanks for that. Just one more thing. Could you share **{missing[0]}**? This helps me give you tailored advice."
                        st.write(clarify_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": clarify_msg})
                        # Stay in pain_details step - don't fall through yet
                    else:
                        # Have BOTH location AND duration - proceed to RAG
                        st.session_state.intake_step = "ready"
                        # Fall through to RAG chain below - no st.stop()
                
                # 2. Run RAG Chain (only if ready)
                if st.session_state.intake_step == "ready":
                    # Check question limit BEFORE incrementing (prevent mid-sentence cutoffs)
                    if st.session_state.questions_after_ready >= 3:
                        limit_msg = "You've reached the 3-question limit for this session. Please refresh the page to start a new conversation with Cherry."
                        st.info(limit_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": limit_msg})
                        st.stop()
                    
                    # Count questions only after intake is complete
                    st.session_state.questions_after_ready += 1
                    
                    with st.spinner("Analyzing"):
                        try:
                            # Build context with patient info (privacy-first: BMI group only)
                            bmi_display = st.session_state.bmi_group.replace('_', ' ').title() if st.session_state.bmi_group else 'Not provided'
                            patient_context = f"""
Patient: {st.session_state.patient_name or 'Anonymous'}
Intent: {st.session_state.intent}
Pain Location: {st.session_state.pain_location or 'Not specified'}
Pain Duration: {st.session_state.pain_duration or 'Not specified'}
BMI Category: {bmi_display}
                            """.strip()
                            
                            # Enhance query with patient context
                            enhanced_query = f"{query}\n\n[Patient Context: {patient_context}]"
                            
                            response = get_chain().run(enhanced_query)
                            # Ensure analytics types dict exists
                            if "types" not in st.session_state.analytics:
                                st.session_state.analytics["types"] = {"medical_advice": 0, "appointment": 0, "pricing": 0, "general": 0, "emergency": 0}
                            # Handle response.type safely (string or enum)
                            response_type = str(response.type) if response.type else "general"
                            st.session_state.analytics["types"][response_type] = st.session_state.analytics["types"].get(response_type, 0) + 1
                            
                            # Deduplication: If message contains exercise descriptions AND we have exercise cards,
                            # show only the intro text (before exercises) and let cards handle the details
                            message_text = response.message
                            if response.exercises and len(response.exercises) > 0:
                                # Check if message has exercise descriptions (Steps, Reps, How to do it, etc.)
                                exercise_keywords = ['steps:', 'reps:', 'how to do it:', 'reps/sets:', 'caution:', 'sets:']
                                has_exercise_desc = any(keyword in message_text.lower() for keyword in exercise_keywords)
                                
                                if has_exercise_desc:
                                    # Try to find where exercise descriptions start and truncate
                                    # Common patterns: exercise names with markdown, section headers, etc.
                                    split_patterns = [
                                        '\n\n**Pelvic',
                                        '\n**Pelvic',
                                        '\n\n*Pelvic',
                                        '\n*Pelvic',
                                        '\n\nPelvic',
                                        '\nPelvic',
                                        'Here are some exercises',
                                        'Here are the exercises',
                                        'Try these exercises',
                                        'Steps:',
                                        '\n\n1. **',
                                        '\n1. **'
                                    ]
                                    truncated = False
                                    for pattern in split_patterns:
                                        if pattern in message_text:
                                            message_text = message_text.split(pattern)[0].strip()
                                            truncated = True
                                            break
                                    
                                    # Clean up any trailing artifacts
                                    message_text = re.sub(r'\*+$', '', message_text).strip()  # Remove trailing asterisks
                                    message_text = re.sub(r'\n+$', '', message_text).strip()  # Remove trailing newlines
                                    
                                    # Add proper ending
                                    if truncated:
                                        if not message_text.endswith('.'):
                                            message_text += '.'
                                        message_text += ' See the exercise cards below:'
                            
                            # Render message (possibly deduplicated)
                            st.write(message_text)
                            
                            # Render exercises as cards
                            for exercise in response.exercises:
                                render_exercise_card(exercise.model_dump())
                            
                            # Render CTA
                            if response.cta:
                                render_cta_button(response.cta)
                            
                            # Simple WhatsApp link below every response
                            render_whatsapp_link()
                                
                            # Save to history as dict for serialization
                            st.session_state.chat_history.append({"role": "assistant", "content": response.model_dump()})
                            
                            # Show follow-up hints
                            st.markdown("""
                            <div style="margin-top:12px;">
                                <div style="font-size:0.85em;color:#666;margin-bottom:6px;"><strong>Follow up:</strong></div>
                                <ul style="font-size:0.9em;color:#333;margin:0;padding-left:20px;">
                                    <li>"Book an appointment"</li>
                                    <li>"Tell me more"</li>
                                    <li>"What else helps?"</li>
                                    <li>"Thanks"</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            # User-friendly error with fallback to human contact
                            st.warning("I'm having trouble accessing my clinical knowledge base right now.")
                            st.info("For immediate assistance with your question, please contact Cherry Nwanna directly. She'll be happy to help!")
                            render_whatsapp_link()
                            
                            # Log error for debugging (not shown to user)
                            error_msg = f"AI service error: {str(e)}"
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg + "\n\n[Fallback: WhatsApp CTA shown to user]"})

if __name__ == "__main__":
    main()
