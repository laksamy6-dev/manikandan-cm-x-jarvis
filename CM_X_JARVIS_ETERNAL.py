%%writefile app.py
# -*- coding: utf-8 -*-
# ==============================================================================
#   PROJECT: CM-X JARVIS: ETERNAL EDITION (STREAMLIT WEB APP)
#   OWNER: Boss Manikandan
#   AI CORE: JARVIS (Shadow Emperor)
#   PURPOSE: Permanent Cloud Deployment with Security Login
# ==============================================================================

!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import google.generativeai as genai
import hmac

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="CM-X JARVIS: WAR ROOM",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SECURITY LOCK (பாதுகாப்பு பூட்டு) ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        # பாஸ்வேர்ட்: "boss" (இதை நீங்க மாத்திக்கலாம்)
        if st.session_state["password"] == "boss":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "🔑 ENTER ACCESS CODE:", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "🔑 ENTER ACCESS CODE:", type="password", on_change=password_entered, key="password"
        )
        st.error("⛔ ACCESS DENIED: தவறான கடவுச்சொல்!")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Stop here if not logged in

# --- 3. UI THEME (HACKER STYLE) ---
st.markdown("""
    <style>
    .main { background-color: #000000; color: #00f3ff; font-family: 'Courier New', monospace; }
    .stApp { background-color: #000000; }
    h1, h2, h3 { color: #ffffff; text-shadow: 0 0 10px #00f3ff; font-weight: 900; }
    .metric-card { background: rgba(20, 20, 30, 0.9); border: 1px solid #00f3ff; padding: 20px; border-radius: 10px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.2); text-align: center; }
    .stButton>button { background: linear-gradient(45deg, #f59e0b, #d97706); color: black; font-weight: bold; border: none; width: 100%; padding: 15px; border-radius: 5px; }
    .stButton>button:hover { box-shadow: 0 0 20px #f59e0b; color: white; }
    .fire-text { background: linear-gradient(to top, #ff0000, #ff8800, #ffff00); -webkit-background-clip: text; color: transparent; font-weight: 900; animation: flicker 0.1s infinite; }
    @keyframes flicker { 0% { opacity: 1; } 50% { opacity: 0.8; } } 
    </style>
""", unsafe_allow_html=True)

# --- 4. SECRETS LOADING (ரகசிய சாவி) ---
# Streamlit Cloud-ல் Secrets-ஐ பாதுகாப்பாக வைக்கலாம்.
# இப்போதைக்கு இன்புட் பாக்ஸ் வைக்கிறேன், அப்புறம் Cloud Secrets-க்கு மாறலாம்.

with st.sidebar:
    st.header("🔐 SECURITY VAULT")
    upstox_key = st.text_input("Upstox Key", type="password")
    gemini_key = st.text_input("Gemini Key", type="password")
    telegram_token = st.text_input("Telegram Token", type="password")
    chat_id = st.text_input("Chat ID", value="8580047711")

    if st.button("💾 SAVE KEYS"):
        st.session_state['keys_saved'] = True
        st.success("KEYS LOCKED!")

# --- 5. HEADER SECTION ---
col_logo, col_title, col_status = st.columns([1, 4, 1])

with col_logo:
    st.markdown("## 🦁")

with col_title:
    st.markdown(f"<h1>CM-X <span style='color:#00f3ff'>JARVIS</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 class='fire-text'>CREATOR: BOSS MANIKANDAN</h3>", unsafe_allow_html=True)

with col_status:
    st.markdown("### 🟢 LIVE")

# --- 6. DASHBOARD METRICS ---
col1, col2, col3 = st.columns(3)

# Simulated Live Data (Upstox API Connection will go here)
price = 19500 + np.random.randint(-10, 15)
vwap = 19500 + np.random.randint(-5, 5)
rsi = 50 + np.random.randint(-5, 5)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#ffffff">NIFTY 50</h4>
            <h1 style="color:#00f3ff">{price}</h1>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#ffffff">VWAP</h4>
            <h1 style="color:#f59e0b">{vwap}</h1>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="color:#ffffff">RSI</h4>
            <h1 style="color:#d946ef">{rsi}</h1>
        </div>
    """, unsafe_allow_html=True)

# --- 7. JARVIS BRAIN & CHART ---
st.markdown("---")
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 NEURAL SCANNER")
    # Fake Chart for Demo
    chart_data = pd.DataFrame(
        np.random.randn(20, 2) + [price, vwap],
        columns=['Price', 'VWAP']
    )
    st.line_chart(chart_data)

with c2:
    st.subheader("🤖 JARVIS INTELLIGENCE")

    # AI Analysis Logic
    decision = "WAIT"
    reason = "Scanning..."

    if price > vwap + 5:
        decision = "BUY CE 🚀"
        reason = "Momentum Breakout!"
    elif price < vwap - 5:
        decision = "BUY PE 🩸"
        reason = "Market Crash!"

    st.info(f"**DECISION:** {decision}")
    st.caption(f"REASON: {reason}")

    # Gemini Chat
    user_query = st.text_input("Ask Jarvis:", placeholder="e.g., Analyze Trend")
    if user_query and gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(f"Act as JARVIS for Boss Manikandan. Question: {user_query}")
            st.success(f"🦁 **JARVIS:** {response.text}")
        except Exception as e:
            st.error("AI Error")

# --- 8. APPROVAL SYSTEM ---
if decision != "WAIT":
    st.markdown("### ⚠️ TRADE DETECTED!")
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button(f"✅ APPROVE {decision}"):
            st.toast(f"Order Executed: {decision}", icon="🚀")
            # Upstox Order Code Here
    with ac2:
        if st.button("❌ REJECT"):
            st.toast("Trade Cancelled", icon="🛑")

# --- 9. AUTO-REFRESH ---
time.sleep(1)
st.rerun()
            
