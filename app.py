import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import google.generativeai as genai
import plotly.graph_objects as go
import hmac
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="CM-X JARVIS: WAR ROOM",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SECURITY LOCK (பாதுகாப்பு) ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        # டீஃபால்ட் பாஸ்வேர்ட்: "boss"
        correct_password = st.secrets.get("system", {}).get("admin_password", "boss")
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Login UI
    st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #00f3ff; }
        .login-box { border: 2px solid #00f3ff; padding: 40px; border-radius: 15px; text-align: center; box-shadow: 0 0 20px rgba(0, 243, 255, 0.3); max-width: 400px; margin: 100px auto; }
        .stTextInput input { color: #00f3ff !important; background-color: #111 !important; border: 1px solid #00f3ff !important; }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='login-box'><h1>🦁 ACCESS DENIED</h1><p>IDENTIFY YOURSELF</p></div>", unsafe_allow_html=True)
    st.text_input("ENTER ACCESS CODE:", type="password", key="password", on_change=password_entered)
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("⛔ ACCESS DENIED: தவறான கடவுச்சொல்!")
        
    return False

if not check_password():
    st.stop()

# --- 3. UI THEME (SHADOW EMPEROR) ---
st.markdown("""
    <style>
    /* Global Theme */
    .stApp { background-color: #02040a; color: #e2e8f0; }
    
    /* Fire Text */
    .fire-text {
        background: linear-gradient(to top, #ff0000, #ff8800, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 1.8rem;
        text-shadow: 0 0 10px rgba(255, 69, 0, 0.6);
        animation: flicker 1.5s infinite alternate;
        font-family: 'Orbitron', sans-serif;
    }
    @keyframes flicker {
        0% { opacity: 1; text-shadow: 0 0 10px red; }
        100% { opacity: 0.8; text-shadow: 0 0 20px orange; }
    }

    /* Holographic Cards */
    .holo-card {
        background: rgba(10, 20, 35, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.3);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        color: #00f3ff;
    }

    /* Metrics */
    .metric-val { font-size: 2.5rem; font-weight: bold; font-family: monospace; color: white; }
    .metric-lbl { font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #00f3ff, #0066ff);
        color: black;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px #00f3ff;
        color: white;
    }
    
    /* Chat Input */
    .stTextInput input {
        background-color: rgba(0,0,0,0.5) !important;
        color: #00f3ff !important;
        border: 1px solid #0044ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. SECRETS LOADING (Cloud Safe) ---
# Streamlit Cloud uses st.secrets to store keys securely
try:
    UPSTOX_KEY = st.secrets["api_keys"]["Upstox_api_key"]
    GEMINI_KEY = st.secrets["api_keys"]["Gemini_api_key"]
    TELEGRAM_TOKEN = st.secrets["api_keys"]["Telegram_bot_token"]
    CHAT_ID = st.secrets["api_keys"]["Telegram_chat_id"]
except:
    # Fallback for local testing if secrets.toml is missing
    GEMINI_KEY = "" # Will prompt user
    TELEGRAM_TOKEN = ""
    CHAT_ID = ""

# --- 5. TELEGRAM ALERT SYSTEM ---
def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": f"🦁 **JARVIS:** {message}", "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=2)
        except:
            pass

# --- 6. HEADER & IDENTITY ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1>CM-X <span style='color:#00f3ff'>JARVIS</span></h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='fire-text'>CREATOR: BOSS MANIKANDAN</h3>", unsafe_allow_html=True)
with c2:
    st.success("🟢 SYSTEM ONLINE")
    if st.button("🔄 REFRESH SYSTEM"):
        st.rerun()

# --- 7. MAIN DASHBOARD ---
col_left, col_right = st.columns([2, 1])

# Initialize State
if 'price' not in st.session_state: st.session_state.price = 19500.0
if 'vwap' not in st.session_state: st.session_state.vwap = 19500.0
if 'history' not in st.session_state: 
    st.session_state.history = pd.DataFrame(columns=['Time', 'Price', 'VWAP'])

with col_left:
    # --- VIDEO FEED (Maximize Button Included in Streamlit Video) ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("### 📡 LIVE MARKET FEED")
    # This URL is a placeholder. Replace with your trading view stream link or keep for demo.
    st.video("https://www.w3schools.com/html/mov_bbb.mp4", format="video/mp4", start_time=0)
    st.caption("Secure Feed: Upstox Data Stream")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LIVE NEURAL CHART ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("### 📈 NEURAL SCANNER")
    
    # Simulate Data Logic (Brain Simulation)
    move = np.random.randint(-12, 18)
    st.session_state.price += move
    st.session_state.vwap = (st.session_state.vwap * 19 + st.session_state.price) / 20
    
    # History Update
    new_row = pd.DataFrame({
        'Time': [datetime.now().strftime("%H:%M:%S")], 
        'Price': [st.session_state.price], 
        'VWAP': [st.session_state.vwap]
    })
    st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(60)

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['Price'], 
                             mode='lines', name='Price', line=dict(color='#00f3ff', width=3)))
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['VWAP'], 
                             mode='lines', name='VWAP', line=dict(color='#ef4444', width=2, dash='dot')))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#64748b'),
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1e293b')
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # --- METRICS PANEL ---
    st.markdown('<div class="holo-card" style="text-align:center;">', unsafe_allow_html=True)
    st.markdown("<span class='metric-lbl'>NIFTY 50 SPOT</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-val'>{st.session_state.price:.2f}</div>", unsafe_allow_html=True)
    
    velocity = move
    vel_color = "#10b981" if velocity > 0 else "#ef4444"
    st.markdown(f"<br><span class='metric-lbl'>VELOCITY</span><br><span style='color:{vel_color}; font-size:1.5rem; font-weight:bold;'>{velocity:.2f}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- JARVIS AI CHAT ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("🤖 ASK JARVIS")
    
    # If no secrets, ask for key
    if not GEMINI_KEY:
        GEMINI_KEY = st.text_input("🔑 API Key Required:", type="password")
    
    user_query = st.text_input("Command:", placeholder="e.g. Trend Status")
    
    if st.button("ACTIVATE NEURAL LINK"):
        if not GEMINI_KEY:
            st.error("⚠️ Gemini Key Missing!")
        elif not user_query:
            st.warning("⚠️ Enter command.")
        else:
            try:
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Act as JARVIS for Boss Manikandan. Market Price: {st.session_state.price}. Query: {user_query}. Keep it robotic, short, and authoritative."
                
                with st.spinner("Processing..."):
                    response = model.generate_content(prompt)
                
                st.success(f"🦁 **JARVIS:** {response.text}")
                
                # Voice Output (Browser Native TTS)
                js_code = f"""
                    <script>
                        var msg = new SpeechSynthesisUtterance("{response.text.replace('"', '')}");
                        var voices = window.speechSynthesis.getVoices();
                        // Try to find a male English voice
                        msg.voice = voices.find(v => v.name.includes("Google UK English Male")) || voices[0];
                        window.speechSynthesis.speak(msg);
                    </script>
                """
                st.components.v1.html(js_code, height=0)
                
            except Exception as e:
                st.error(f"AI Error: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

# --- 8. APPROVAL SYSTEM (The Final Gate) ---
# Logic: High Velocity triggers Approval Request
signal = "WAIT"
if velocity > 8: signal = "BUY CE"
elif velocity < -8: signal = "BUY PE"

if signal != "WAIT":
    st.markdown(f"""
        <div style="background:#1e1e2e; border:2px solid #f59e0b; padding:20px; border-radius:15px; text-align:center; animation:pulse 1s infinite; margin-top:20px;">
            <h2 style="color:#f59e0b; margin:0;">⚠️ SIGNAL DETECTED: {signal}</h2>
            <p style="color:#aaa;">Reason: High Velocity Momentum</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Approval Buttons
    ac1, ac2 = st.columns(2)
    with ac1:
        if st.button(f"✅ APPROVE {signal}", use_container_width=True):
            st.toast(f"Order Executed: {signal}", icon="🚀")
            send_telegram(f"Trade Executed: {signal} @ {st.session_state.price}")
            # Add Upstox order placement code here
    with ac2:
        if st.button("❌ REJECT", use_container_width=True):
            st.toast("Trade Cancelled", icon="🛑")
            send_telegram(f"Trade Rejected: {signal}")

# --- 9. AUTO-REFRESH (Live Simulation) ---
time.sleep(1)
st.rerun()
