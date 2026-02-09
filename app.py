import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import os
import math
import base64
from datetime import datetime
import pytz
import plotly.graph_objects as go
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIG & IRON MAN HUD UI
# ==========================================
st.set_page_config(
    page_title="CM-X: BRAHMASTRA LIVE",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ULTRA-MODERN CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #000000 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit Header/Footer */
    header, footer {visibility: hidden;}
    
    /* Neon Cards */
    .neon-card {
        background: rgba(10, 15, 20, 0.95);
        border: 1px solid #00f3ff;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
        transition: all 0.3s ease-in-out;
    }
    .neon-card:hover {
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.5);
        transform: scale(1.02);
    }
    
    /* Text Colors */
    .big-text { font-size: 32px; font-weight: 800; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.5); }
    .neon-green { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    .neon-red { color: #ff0000; text-shadow: 0 0 10px #ff0000; }
    .neon-cyan { color: #00f3ff; text-shadow: 0 0 10px #00f3ff; }
    
    /* Terminal */
    .terminal {
        background: #050505;
        border-left: 4px solid #00ff00;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        font-size: 12px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SYSTEM UTILITIES
# ==========================================
def get_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def send_telegram(msg):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            requests.get(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_BOT_TOKEN']}/sendMessage",
                         params={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg})
    except: pass

# ==========================================
# 3. UPSTOX LIVE ENGINE (REAL TRADING)
# ==========================================
def get_live_price():
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {st.secrets["UPSTOX_ACCESS_TOKEN"]}'
            }
            # NIFTY BANK or NIFTY 50 (Change as needed)
            params = {'instrument_key': 'NSE_INDEX|Nifty Bank'} 
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if data['status'] == 'success':
                price = data['data']['NSE_INDEX:Nifty Bank']['last_price']
                return float(price), "LIVE 🟢"
    except Exception as e:
        return None, f"ERR: {str(e)}"
    return None, "NO TOKEN 🔴"

def place_order(signal, price):
    """
    உண்மையான ஆர்டர் போடும் இடம்.
    (Safety: தற்போதைக்கு Telegram Alert மட்டும் அனுப்புகிறேன்.
    API Call உள்ளே கமெண்ட் செய்யப்பட்டுள்ளது)
    """
    msg = f"🚀 AUTOMATED ORDER TRIGGERED!\nSignal: {signal}\nIndex Price: {price}\nTime: {get_time()}"
    send_telegram(msg)
    # உண்மையான ஆர்டர் போட கீழே உள்ளதை Uncomment செய்யவும்:
    # requests.post("https://api.upstox.com/v2/order/place", headers=..., data=...)
    return True

# ==========================================
# 4. AGGRESSIVE SCALPING BRAIN (Physics + Math)
# ==========================================
def calculate_physics(history):
    if len(history) < 3: return 0, 0
    vel = history[-1] - history[-2]
    acc = vel - (history[-2] - history[-3])
    return vel, acc

def run_monte_carlo(price, vol, sims=500):
    wins = 0
    vol = 0.01 if vol == 0 else vol
    for _ in range(sims):
        sim_p = price * math.exp((0 - 0.5 * vol**2) + vol * np.random.normal())
        if sim_p > price: wins += 1
    return (wins / sims) * 100

class ChellakiliBrain:
    def __init__(self):
        self.model = None
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel('gemini-2.0-flash') # 2.0 Flash

    def ask(self, prompt):
        if not self.model: return "BRAIN OFFLINE"
        try:
            res = self.model.generate_content(f"Act as Elite Trader. {prompt}. Short Tanglish answer.")
            return res.text
        except: return "THINKING..."

# ==========================================
# 5. MAIN DASHBOARD LOGIC
# ==========================================

# A. SECURITY CHECK
if 'auth' not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br><h1 style='text-align:center; color:#00f3ff;'>🔐 IDENTITY VERIFICATION</h1>", unsafe_allow_html=True)
        pwd = st.text_input("ACCESS CODE", type="password")
        if st.button("INITIATE SYSTEM", use_container_width=True):
            if pwd == "boss": st.session_state.auth = True; st.rerun()
    st.stop()

# B. DATA INIT
if 'brain' not in st.session_state: st.session_state.brain = ChellakiliBrain()

# Get Price First to Initialize History correctly (Fixing the Jump Issue)
live_price, status = get_live_price()
if live_price is None: 
    # Fallback to simulation
    live_price = 25000.0 if 'history' not in st.session_state else st.session_state.history[-1] + np.random.randint(-10, 15)
    status = "SIMULATION 🟠"

if 'history' not in st.session_state: 
    # Fill history with current price to avoid graph jump
    st.session_state.history = [live_price] * 50 

# Update History
st.session_state.history.append(live_price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# C. SIDEBAR CONTROLS
with st.sidebar:
    st.header("⚙️ SYSTEM CONTROL")
    st.markdown(f"**STATUS:** {status}")
    auto_trade = st.toggle("🚀 AUTO-EXECUTE ORDERS", value=False)
    aggro_mode = st.toggle("🔥 AGGRESSIVE SCALPING", value=True)
    if st.button("📡 PING TELEGRAM"): send_telegram("System Live.")

# D. CALCULATIONS (THE BRAIN)
vel, acc = calculate_physics(st.session_state.history)
volatility = np.std(st.session_state.history[-10:]) / np.mean(st.session_state.history[-10:])
win_prob = run_monte_carlo(live_price, volatility)

# --- DECISION LOGIC (AGGRESSIVE) ---
score = 0
reasons = []

# 1. Physics (Threshold lowered for Scalping)
limit = 0.5 if aggro_mode else 1.5
if vel > limit: 
    score += 2
    reasons.append(f"Vel +{vel:.2f}")
elif vel < -limit: 
    score -= 2
    reasons.append(f"Vel {vel:.2f}")

# 2. Acceleration (Momentum)
if acc > 0.1: score += 1
elif acc < -0.1: score -= 1

# 3. Probability
if win_prob > 60: score += 1
elif win_prob < 40: score -= 1

# Final Verdict
decision = "WAIT"
color_cls = "neon-cyan"

if score >= 3:
    decision = "BUY CE 🚀"
    color_cls = "neon-green"
elif score <= -3:
    decision = "BUY PE 🩸"
    color_cls = "neon-red"

# E. UI DISPLAY (The Brahmastra Look)
st.markdown(f"<h1 style='text-align:center; color:white;'>🦁 CM-X <span style='color:#00f3ff'>BRAHMASTRA</span></h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:#aaa;'>{get_time().strftime('%H:%M:%S')} | {status}</p>", unsafe_allow_html=True)

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f"<div class='neon-card'><small>INDEX PRICE</small><br><div class='big-text neon-cyan'>{live_price}</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='neon-card'><small>VELOCITY</small><br><div class='big-text {('neon-green' if vel>0 else 'neon-red')}'>{vel:.2f}</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='neon-card'><small>WIN PROB</small><br><div class='big-text neon-cyan'>{win_prob:.0f}%</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='neon-card' style='border:2px solid {('#00ff00' if 'CE' in decision else ('#ff0000' if 'PE' in decision else '#00f3ff'))}'><small>SIGNAL</small><br><div class='big-text {color_cls}'>{decision}</div></div>", unsafe_allow_html=True)

# Chart & Logs
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 📈 QUANTUM FLOW")
    fig = go.Figure()
    # Gradient Line Chart
    fig.add_trace(go.Scatter(
        y=st.session_state.history[-40:], 
        mode='lines', 
        name='Price',
        line=dict(color='#00f3ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 243, 255, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='white'), 
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### 🖥️ NEURAL LOGS")
    log_text = f"""
    > [SYSTEM] Scanning Market...
    > [MODE] {'AGGRESSIVE 🔥' if aggro_mode else 'NORMAL 🛡️'}
    > [DATA] Price: {live_price}
    > [PHYSICS] Vel: {vel:.2f} | Acc: {acc:.2f}
    > [LOGIC] Score: {score} | Reasons: {reasons}
    > [DECISION] {decision}
    """
    st.markdown(f"<div class='terminal'>{log_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

# F. AUTOMATION & AI
if decision != "WAIT":
    # 1. Trigger Order (If Auto-Trade is ON)
    if auto_trade:
        place_order(decision, live_price)
        st.toast(f"ORDER SENT: {decision}", icon="🚀")
    
    # 2. Ask Gemini
    if "GEMINI_API_KEY" in st.secrets:
        with st.expander("🤖 CHELLAKILI ANALYSIS (Click to Open)", expanded=True):
            prompt = f"Market is {live_price}. Physics Velocity is {vel}. Signal is {decision}. Why? Quick answer."
            ai_reply = st.session_state.brain.ask(prompt)
            st.info(f"🦁 {ai_reply}")

# Auto-Refresh Speed
time.sleep(1) # Fast refresh for scalping
st.rerun()
    
