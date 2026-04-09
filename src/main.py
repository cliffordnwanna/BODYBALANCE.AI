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

# Initialize Session State
if "vector_store" not in st.session_state:
    with st.spinner("Initializing Clinical Knowledge Base..."):
        st.session_state.vector_store = BodyBalanceVectorStore()

if "chain" not in st.session_state:
    st.session_state.chain = BodyBalanceChain(st.session_state.vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analytics" not in st.session_state or not isinstance(st.session_state.get("analytics"), dict):
    st.session_state.analytics = {
        "total_queries": 0,
        "types": {"medical_advice": 0, "appointment": 0, "pricing": 0, "general": 0, "emergency": 0}
    }

# Inject Styling
inject_custom_styles()

def main():
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
        st.metric("Total Patient Queries", st.session_state.analytics["total_queries"])
        
        with st.expander("Response Types"):
            for t, count in st.session_state.analytics["types"].items():
                st.write(f"**{t.replace('_', ' ').capitalize()}:** {count}")

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
                if isinstance(message["content"], str):
                    st.write(message["content"])
                else:
                    # Render structured clinical response
                    response = message["content"]
                    st.write(response.message)
                    for exercise in response.exercises:
                        render_exercise_card(exercise.model_dump())
                    if response.cta:
                        render_cta_button(response.cta)
                    
                    # Simple WhatsApp link below every response
                    render_whatsapp_link()
            else:
                st.write(message["content"])

    # Chat Input
    if query := st.chat_input("How can BodyBalance Physiotherapy help you today?"):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Process with Guardrails and Chain
        with st.chat_message("assistant"):
            st.session_state.analytics["total_queries"] += 1
            
            # 1. Check Guardrails first
            red_flag_msg = check_red_flags(query)
            if red_flag_msg:
                st.session_state.analytics["types"]["emergency"] += 1
                render_emergency_alert(red_flag_msg)
                render_whatsapp_link() # Show booking link even in emergency
                st.session_state.chat_history.append({"role": "assistant", "content": red_flag_msg})
            else:
                # Check question limit
                user_questions = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
                if user_questions >= 3:
                    st.warning("You've reached the 3-question limit. Please refresh the page to start a new session.")
                    st.stop()
                # 2. Run RAG Chain
                with st.spinner("Analyzing"):
                    try:
                        response = st.session_state.chain.run(query)
                        st.session_state.analytics["types"][response.type] += 1
                        
                        # Render message
                        st.write(response.message)
                        
                        # Render exercises if any
                        for exercise in response.exercises:
                            render_exercise_card(exercise.model_dump())
                        
                        # Render CTA
                        if response.cta:
                            render_cta_button(response.cta)
                        
                        # Simple WhatsApp link below every response
                        render_whatsapp_link()
                            
                        # Save to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"I'm sorry, I encountered an error. Please try again later. (Error: {str(e)})"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
