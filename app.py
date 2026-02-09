import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import requests
import json
import os
import math
import base64
from datetime import datetime
import pytz
from gtts import gTTS
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="CM-X JARVIS: OMEGA 2.0", layout="wide", page_icon="🦁")

# CSS Styling (Hacker Theme)
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', monospace; }
    .holo-card { background: rgba(10, 20, 30, 0.8); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px; color: #00f3ff; }
    .stButton>button { background: linear-gradient(45deg, #00f3ff, #0066ff); color: black; font-weight: bold; border: none; width: 100%; }
    h1, h2, h3 { color: #fff; text-shadow: 0 0 10px #00f3ff; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. UTILITIES & CONFIG
# ==========================================
def get_indian_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def send_telegram_msg(message):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            url = f"https://api.telegram.org/bot{st.secrets['TELEGRAM_BOT_TOKEN']}/sendMessage"
            params = {"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": message}
            requests.get(url, params=params)
    except: pass

def play_voice(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("voice_out.mp3")
        with open("voice_out.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        os.remove("voice_out.mp3")
    except: pass

# ==========================================
# 3. MATH & PHYSICS CORE
# ==========================================
def calculate_physics(price_history):
    if len(price_history) < 5: return 0, 0
    # List-ஆ அல்லது DataFrame-ஆ என்று பார்த்து கையாளுதல்
    hist = list(price_history)
    velocity = hist[-1] - hist[-2]
    acceleration = velocity - (hist[-2] - hist[-3])
    return velocity, acceleration

def predict_future_price(history):
    if len(history) < 20: return history[-1]
    y = np.array(history[-20:])
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p(len(y) + 10) # Next 10 mins

class KalmanFilter:
    def __init__(self):
        self.last_estimate = 0
        self.err_estimate = 1
        self.q = 0.01
    def update(self, measurement):
        kalman_gain = self.err_estimate / (self.err_estimate + 1)
        current_estimate = self.last_estimate + kalman_gain * (measurement - self.last_estimate)
        self.err_estimate = (1.0 - kalman_gain) * self.err_estimate + abs(self.last_estimate - current_estimate) * self.q
        self.last_estimate = current_estimate
        return current_estimate

# ==========================================
# 4. BRAIN CORE (UNIFIED GEMINI 2.0)
# ==========================================
class ChellakiliBrain:
    def __init__(self):
        self.model = None
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            # ✅ CORRECTED: Using Gemini 2.0 Flash
            self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze(self, price, vel, trend):
        if not self.model: return '{"decision": "WAIT", "reason": "API Key Missing"}'
        
        prompt = f"""
        Act as 'Chellakili', an elite scalper.
        Data: Price={price}, Velocity={vel}, Trend={trend}.
        Logic:
        - Vel > 2 & Trend UP -> BUY CE
        - Vel < -2 & Trend DOWN -> BUY PE
        
        Output JSON: {{"decision": "BUY_CE/BUY_PE/WAIT", "reason": "Tanglish explanation"}}
        """
        try:
            res = self.model.generate_content(prompt)
            return res.text.replace('```json', '').replace('```', '').strip()
        except Exception as e:
            return f'{{"decision": "WAIT", "reason": "Error: {str(e)}"}}'

# ==========================================
# 5. MAIN APP LOGIC
# ==========================================

# Initialize Session State
if 'price' not in st.session_state: st.session_state.price = 19500.0
if 'vwap' not in st.session_state: st.session_state.vwap = 19500.0
if 'history' not in st.session_state: st.session_state.history = []
if 'brain' not in st.session_state: st.session_state.brain = ChellakiliBrain()
if 'kf' not in st.session_state: st.session_state.kf = KalmanFilter()

# Data Simulation
move = np.random.randint(-10, 15)
st.session_state.price += move
st.session_state.vwap = (st.session_state.vwap * 19 + st.session_state.price) / 20
kf_price = st.session_state.kf.update(st.session_state.price)

# Update History
st.session_state.history.append(st.session_state.price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# Calculations
vel, acc = calculate_physics(st.session_state.history)
future_target = predict_future_price(st.session_state.history)
trend = "UP" if vel > 0 else "DOWN"

# Brain Analysis
ai_output = st.session_state.brain.analyze(st.session_state.price, vel, trend)
try:
    ai_data = json.loads(ai_output)
    decision = ai_data.get("decision", "WAIT")
    reason = ai_data.get("reason", "")
except:
    decision = "WAIT"
    reason = "Parsing Error"

# ==========================================
# 6. DASHBOARD UI
# ==========================================
c1, c2 = st.columns([3, 1])
with c1: st.markdown("<h1>🦁 CM-X <span style='color:#00f3ff'>OMEGA 2.0</span></h1>", unsafe_allow_html=True)
with c2: st.markdown(f"**IST:** {get_indian_time().strftime('%H:%M:%S')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ CONTROLS")
    auto_ref = st.toggle("🔄 Auto-Refresh", value=True)
    voice_on = st.toggle("🔊 Voice", value=False)
    if st.button("Test Telegram"): send_telegram_msg("CM-X Online!")

# Main Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("PRICE", f"{st.session_state.price:.2f}", f"{move:.2f}")
m2.metric("VELOCITY", f"{vel:.2f}")
m3.metric("FUTURE (10m)", f"{future_target:.2f}")
m4.metric("AI DECISION", decision, delta_color="normal")

# Charts
c_left, c_right = st.columns([2, 1])
with c_left:
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.history, mode='lines', name='Price', line=dict(color='#00f3ff')))
    fig.update_layout(title="Quantum Price Flow", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("🧠 CHELLAKILI (Gemini 2.0)")
    st.info(f"**Signal:** {decision}")
    st.caption(f"Reason: {reason}")
    
    # User Chat
    user_q = st.text_input("Ask Jarvis:", placeholder="Trend analysis...")
    if st.button("ASK"):
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            # ✅ CORRECTED: Gemini 2.0 Here too
            model = genai.GenerativeModel('gemini-2.0-flash')
            res = model.generate_content(f"Act as Jarvis. Price: {st.session_state.price}. Query: {user_q}")
            st.success(f"🦁 {res.text}")
            if voice_on: play_voice(res.text)
    st.markdown('</div>', unsafe_allow_html=True)

# Signal Alert & Telegram
if decision != "WAIT":
    st.toast(f"{decision} DETECTED!", icon="🔥")
    if st.button(f"EXECUTE {decision}"):
        send_telegram_msg(f"🚀 TRADE: {decision} @ {st.session_state.price}\nReason: {reason}")

# --- HTML EMBED (OPTIONAL) ---
# If you have the HTML file, it will show below
try:
    with open("CM_X_Ultimate_Brahmastra_AI_FIXED_V3.html", "r") as f:
        st.components.v1.html(f.read(), height=600, scrolling=True)
except: pass

if auto_ref:
    time.sleep(1)
    st.rerun()
