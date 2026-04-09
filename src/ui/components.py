"""
BODYBALANCE.AI - Components
Streamlit render functions for: exercise cards, CTA button, disclaimer badge, escalation alert.
"""
import streamlit as st

def render_exercise_card(exercise: dict):
    """Render an exercise card with steps and metadata."""
    st.markdown(f"""
    <div class="exercise-card">
        <div class="exercise-name">{exercise.get('name', 'N/A')}</div>
        <div class="exercise-steps">
            <strong>How to do it:</strong><br/>
            {" ".join([f"<li>{step}</li>" for step in exercise.get('steps', [])])}
        </div>
        <div class="exercise-meta">
            <strong>Reps/Sets:</strong> {exercise.get('reps', 'N/A')}<br/>
            <strong>Caution:</strong> {exercise.get('caution', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_cta_button(text: str = "Book Appointment"):
    """Render a WhatsApp-styled CTA button."""
    whatsapp_url = "https://wa.me/2348136293596?text=Hello%20BodyBalance,%20I%20would%20like%20to%20book%20a%20physiotherapy%20session."
    st.markdown(f"""
    <div style="text-align: center;">
        <a href="{whatsapp_url}" target="_blank" class="cta-btn">
            {text}
        </a>
    </div>
    """, unsafe_allow_html=True)

def render_clinic_cta_card(is_sidebar: bool = False):
    """Render a persistent clinic CTA card with professional therapist info."""
    whatsapp_url = "https://wa.me/2348136293596?text=Hello%20BodyBalance,%20I%20would%20like%20to%20book%20a%20physiotherapy%20session."
    virtual_url = "https://wa.me/2348136293596?text=Hello%20BodyBalance,%20I%20would%20like%20to%20book%20a%20virtual%20consultation."
    
    card_style = "background-color: #F1F8E9; border: 2px solid #1B5E20; border-radius: 12px; padding: 1.2em; margin: 1em 0;"
    if is_sidebar:
        card_style = "background-color: #FFFFFF; border: 1px solid #1B5E20; border-radius: 8px; padding: 1em; margin: 0.5em 0;"
    
    st.markdown(f"""
    <div style="{card_style}">
        <div style="color: #1B5E20; font-weight: 700; font-size: 1.1em; margin-bottom: 5px;">BodyBalance Clinic</div>
        <div style="font-size: 0.9em; font-weight: 600; color: #333;">Lead Therapist: Cherry Nwanna (BMR.PT)</div>
        <div style="font-size: 0.85em; color: #555; margin-top: 8px;">
            <b>Services:</b><br/>
            • In-Person Session: ₦150,000<br/>
            • Virtual Consultation: ₦50,000
        </div>
        <div style="margin-top: 12px; display: flex; flex-direction: column; gap: 8px;">
            <a href="{whatsapp_url}" target="_blank" style="background-color: #1B5E20; color: white; text-decoration: none; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.85em; font-weight: 600;">
                Book In-Person
            </a>
            <a href="{virtual_url}" target="_blank" style="background-color: #FFFFFF; color: #1B5E20; border: 1px solid #1B5E20; text-decoration: none; padding: 8px; border-radius: 6px; text-align: center; font-size: 0.85em; font-weight: 600;">
                Book Virtual (₦50k)
            </a>
        </div>
        <div style="font-size: 0.7em; color: #666; margin-top: 10px; font-style: italic;">
            "AI guidance is not a substitute for professional care. Book a real session today."
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_disclaimer_badge():
    """Render a clinical disclaimer badge."""
    st.markdown("""
    <div style="text-align: center; margin-top: 2em;">
        <span class="disclaimer-badge">Clinical AI Disclaimer</span><br/>
        <p style="font-size: 0.75em; color: #777; margin-top: 8px;">
            BodyBalance AI provides educational information, not medical diagnosis. 
            Always consult a human physiotherapist for persistent pain.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_emergency_alert(message: str):
    """Render a hardcoded emergency escalation alert."""
    st.markdown(f"""
    <div class="emergency-alert">
        {message}
    </div>
    """, unsafe_allow_html=True)

def render_whatsapp_link():
    """Render a simple WhatsApp booking link below responses."""
    whatsapp_url = "https://wa.me/2348136293596?text=Hello%20BodyBalance,%20I%20would%20like%20to%20book%20a%20physiotherapy%20session."
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1em;">
        <a href="{whatsapp_url}" target="_blank" style="color: #1B5E20; font-size: 0.9em; text-decoration: none; border-bottom: 1px dashed #1B5E20;">
            Book a session via WhatsApp
        </a>
    </div>
    """, unsafe_allow_html=True)

def render_clinic_cta_card(is_sidebar=False):
    """Renders the clinic booking CTA card."""
    container = st.sidebar if is_sidebar else st
    container.markdown("""
    <div style="background:#1B5E20;padding:16px;border-radius:10px;color:white;margin-bottom:12px;">
        <h4 style="margin:0">BodyBalance Clinic</h4>
        <p style="margin:4px 0">Lead Therapist: <b>Cherry Nwanna (BMR.PT)</b></p>
        <p style="margin:4px 0">Services:</p>
        <ul style="margin:4px 0">
            <li>In-Person Session: 150,000</li>
            <li>Virtual Consultation: 50,000</li>
        </ul>
        <p style="font-size:0.8em;font-style:italic">"AI guidance is not a substitute for professional care."</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.link_button("Book In-Person", "https://wa.me/2348136293596", use_container_width=True)
    st.sidebar.link_button("Book Virtual", "https://wa.me/2348136293596", use_container_width=True)
