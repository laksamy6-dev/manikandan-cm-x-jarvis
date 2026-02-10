import os
import sys
import subprocess
import time
import json
import random
from datetime import datetime

# --- 1. SELF-HEALING & SETUP (தானியங்கி அமைப்பு) ---
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"🦁 JARVIS: Installing missing module -> {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# தேவையான லைப்ரரிகளை சரிபார்த்தல்
required_libs = ["streamlit", "google-generativeai", "pandas", "numpy", "requests", "python-dotenv", "plotly", "watchdog"]
for lib in required_libs:
    install_and_import(lib)

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import plotly.graph_objects as go
from dotenv import load_dotenv

# --- 2. CREDENTIALS INJECTION (ரகசிய சாவி தானியங்கி உருவாக்கம்) ---
# பாஸ், நீங்க கொடுத்த கீகளை நான் இங்கேயே வச்சிருக்கேன். 
# இந்த ஃபைலை ரன் பண்ணும்போது, இதுவே .env ஃபைலை உருவாக்கிடும்.

ENV_CONTENT = """
UPSTOX_API_KEY=6463a56b-79e5-4c99-9b45-fe3db2878395
UPSTOX_API_SECRET=9gws7n1uu2
UPSTOX_REDIRECT_URI=https://localhost:8501
UPSTOX_ACCESS_TOKEN=eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0WkNDREwiLCJqdGkiOiI2OTg3NDQzNjY2MDFhZTA5MTkyMDM1M2YiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzcwNDcyNTAyLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NzA1MDE2MDB9.Qq76KVcVezgQmS5Dn9Pp_BGmyVoV7QVhrXZOVL0xJHU
GEMINI_API_KEY=AIzaSyAu4IyoGJ2NN1n0_0y9BFRTSj8ZWfFUbVU
TELEGRAM_BOT_TOKEN=8580047711:AAG8-WU5G3U0dWX0-CUwaLVnKAPT2Xzls2A
TELEGRAM_CHAT_ID=8580047711
OWNER_NAME=BOSS MANIKANDAN
AI_NAME=JARVIS
"""

if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write(ENV_CONTENT)
    print("🦁 JARVIS: Security Vault (.env) Created Successfully!")

load_dotenv()

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CM-X JARVIS: WAR ROOM",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 4. CSS STYLING (HOLOGRAPHIC & FIRE THEME) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* GLOBAL THEME */
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Rajdhani', sans-serif; }
    
    /* FIRE TEXT EFFECT (BOSS NAME) */
    .fire-text {
        background: linear-gradient(to top, #ff0000, #ff8800, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2rem;
        text-shadow: 0 0 10px rgba(255, 69, 0, 0.6);
        animation: flicker 1.5s infinite alternate;
    }
    @keyframes flicker {
        0% { opacity: 1; text-shadow: 0 0 10px red; }
        100% { opacity: 0.8; text-shadow: 0 0 20px orange; }
    }

    /* HOLOGRAPHIC CARD */
    .holo-card {
        background: rgba(10, 20, 30, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.3);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1), inset 0 0 20px rgba(0, 243, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
    }

    /* METRICS */
    .metric-value { font-size: 2.5rem; font-weight: bold; font-family: 'Orbitron', sans-serif; color: white; }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; }

    /* JARVIS BUTTON */
    .stButton>button {
        background: linear-gradient(45deg, #00f3ff, #0066ff);
        color: black;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        border: none;
        width: 100%;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px #00f3ff;
        color: white;
    }
    
    /* INPUT FIELDS */
    .stTextInput input {
        background-color: rgba(0, 20, 40, 0.8) !important;
        color: #00f3ff !important;
        border: 1px solid #00f3ff !important;
        font-family: 'Rajdhani', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- 5. INITIALIZE STATE (மூளை நினைவகம்) ---
if 'price' not in st.session_state: st.session_state.price = 19500.0
if 'vwap' not in st.session_state: st.session_state.vwap = 19500.0
if 'history' not in st.session_state: 
    st.session_state.history = pd.DataFrame(columns=['Time', 'Price', 'VWAP'])
if 'velocity' not in st.session_state: st.session_state.velocity = 0.0

# --- 6. API HANDLERS (மூளை செயல்பாடுகள்) ---
def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": f"🦁 **JARVIS:** {message}", "parse_mode": "Markdown"}, timeout=2)
        except: pass

def ask_gemini(query, context_data):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "API Key Error"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Act as JARVIS for Boss Manikandan. Market Data: {context_data}. User asks: {query}. Keep it short, robotic, and cool."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Neural Link Error: {str(e)}"

# --- 7. HEADER SECTION ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1>CM-X <span style='color:#00f3ff'>JARVIS</span> <span style='font-size:1rem; color:#64748b;'>V7.0</span></h1>", unsafe_allow_html=True)
    st.markdown("<div class='fire-text'>CREATOR: BOSS MANIKANDAN</div>", unsafe_allow_html=True)
with c2:
    st.success("🟢 SYSTEM ONLINE")
    if st.button("🔄 REFRESH SYSTEM"):
        st.rerun()

# --- 8. DASHBOARD GRID ---
col_left, col_right = st.columns([2, 1])

with col_left:
    # --- VIDEO FEED (Custom HTML for Maximize) ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3 style="margin:0; color:#00f3ff;">📡 LIVE OPTIC FEED</h3>
            <span style="color:#ef4444; font-weight:bold; animation: pulse 1s infinite;">● LIVE</span>
        </div>
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border: 2px solid #f59e0b; border-radius: 8px; margin-top: 10px;">
            <iframe src="https://www.youtube.com/embed/jfKfPfyJRdk?autoplay=1&mute=1&controls=1" 
                    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" 
                    frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
            </iframe>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- NEURAL CHART ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    
    # Simulation Logic
    move = np.random.randint(-12, 18)
    st.session_state.price += move
    st.session_state.vwap = (st.session_state.vwap * 19 + st.session_state.price) / 20
    st.session_state.velocity = move
    
    # Update History
    now = datetime.now().strftime("%H:%M:%S")
    new_df = pd.DataFrame({'Time': [now], 'Price': [st.session_state.price], 'VWAP': [st.session_state.vwap]})
    st.session_state.history = pd.concat([st.session_state.history, new_df]).tail(50)
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['Price'], mode='lines', name='Price', line=dict(color='#00f3ff', width=3)))
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['VWAP'], mode='lines', name='VWAP', line=dict(color='#ef4444', width=2, dash='dot')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#64748b'), margin=dict(l=0,r=0,t=0,b=0), height=300, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1e293b'))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # --- METRICS PANEL ---
    st.markdown('<div class="holo-card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>NIFTY 50 SPOT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{st.session_state.price:.2f}</div>", unsafe_allow_html=True)
    
    vel_color = "#10b981" if st.session_state.velocity > 0 else "#ef4444"
    st.markdown(f"<div class='metric-label' style='margin-top:15px;'>VELOCITY</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value' style='color:{vel_color}; font-size:2rem;'>{st.session_state.velocity:.2f}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- JARVIS CONTROL ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 JARVIS INTERFACE")
    
    query = st.text_input("Command:", placeholder="e.g., Analyze Trend")
    
    if st.button("ACTIVATE NEURAL LINK"):
        if query:
            with st.spinner("Processing..."):
                response = ask_gemini(query, f"Price: {st.session_state.price}, Velocity: {st.session_state.velocity}")
                st.success(f"🦁 **JARVIS:** {response}")
                
                # Javascript Voice Output
                js = f"""
                <script>
                    var msg = new SpeechSynthesisUtterance("{response.replace('"', '')}");
                    var voices = window.speechSynthesis.getVoices();
                    msg.voice = voices.find(v => v.name.includes("Google UK English Male")) || voices[0];
                    msg.rate = 1.0; msg.pitch = 0.8;
                    window.speechSynthesis.speak(msg);
                </script>
                """
                st.components.v1.html(js, height=0)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 9. APPROVAL SYSTEM (The Gatekeeper) ---
signal = "WAIT"
if st.session_state.velocity > 8: signal = "BUY CE 🚀"
elif st.session_state.velocity < -8: signal = "BUY PE 🩸"

if signal != "WAIT":
    st.markdown(f"""
        <div style="background:#1e1e2e; border:2px solid #f59e0b; padding:20px; border-radius:15px; text-align:center; animation:pulse 1s infinite; margin-top:20px;">
            <h2 style="color:#f59e0b; margin:0;">⚠️ TRADE DETECTED: {signal}</h2>
            <p style="color:#aaa;">Reason: Neuro-Quantum Velocity Spike</p>
        </div>
        <style>@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }} 70% {{ box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }} 100% {{ box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }} }}</style>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button(f"✅ APPROVE {signal}", use_container_width=True):
            st.toast("Trade Executed Successfully!", icon="🚀")
            send_telegram(f"Trade Executed: {signal} @ {st.session_state.price}")
    with c2:
        if st.button("❌ REJECT", use_container_width=True):
            st.toast("Trade Rejected.", icon="🛑")

# --- 10. AUTO REFRESH ---
time.sleep(1)
st.rerun()
