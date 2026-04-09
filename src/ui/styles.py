"""
BODYBALANCE.AI - Styles
Inject custom CSS: white background, deep green accents, clean medical font.
"""
import streamlit as st

def inject_custom_styles():
    st.markdown("""
    <style>
        /* Modern Medical Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #1B1B1B;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F1F8E9;
            border-right: 1px solid #C8E6C9;
        }

        /* Clinic Header Styling */
        .clinic-header {
            color: #1B5E20;
            font-weight: 700;
            font-size: 2.2em;
            margin-bottom: 0.2em;
        }
        .clinic-subtitle {
            color: #666;
            font-size: 1.1em;
            margin-bottom: 1.5em;
        }

        /* Exercise Card Styling */
        .exercise-card {
            background-color: #FFFFFF;
            border: 2px solid #1B5E20;
            border-radius: 8px;
            padding: 1.2em;
            margin: 1em 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .exercise-name {
            color: #1B5E20;
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 0.5em;
        }
        .exercise-steps {
            font-size: 0.95em;
            color: #444;
            line-height: 1.5;
        }
        .exercise-meta {
            font-size: 0.85em;
            color: #666;
            margin-top: 0.8em;
            font-style: italic;
        }

        /* CTA Button Styling */
        .cta-btn {
            display: inline-block;
            background-color: #1B5E20;
            color: white !important;
            padding: 10px 24px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            text-align: center;
            margin-top: 1em;
            transition: background-color 0.2s;
        }
        .cta-btn:hover {
            background-color: #2E7D32;
        }

        /* Alert Styling */
        .emergency-alert {
            background-color: #FFEBEE;
            border: 2px solid #D32F2F;
            border-radius: 12px;
            padding: 1.5em;
            margin: 1em 0;
            color: #B71C1C;
        }

        /* Disclaimer Styling */
        .disclaimer-badge {
            display: inline-block;
            background-color: #FFF9C4;
            color: #F57F17;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
    </style>
    """, unsafe_allow_html=True)
