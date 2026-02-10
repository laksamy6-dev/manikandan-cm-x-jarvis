import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import google.generativeai as genai
import time
from datetime import datetime
import pytz
import json
import os
from gtts import gTTS
import base64
from collections import deque

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="CM-X GOD MODE: BOSS EDITION",
    layout="wide",
    page_icon="👑",
    initial_sidebar_state="collapsed"
)

# SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 

# --- 2. CSS STYLING (BOSS STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp { background-color: #000000; color: #00ff41; font-family: 'Orbitron', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ff41; text-shadow: 0 0 5px #00ff41; }
    div[data-testid="stMetricLabel"] { color: #888; }
    
    /* Approval Box */
    .approval-box {
        border: 2px solid #ffff00;
        background-color: #333300;
        color: #ffff00;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
        animation: flash 1s infinite;
    }
    @keyframes flash { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 5px;
        height: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIG & SECRETS ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
except: 
    st.error("⚠️ SECRETS ERROR: Check .streamlit/secrets.toml")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. MEMORY ---
def init_brain():
    if not os.path.exists(MEMORY_FILE):
        return {"position": None, "pnl": 0.0, "orders": []}
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return {"position": None, "pnl": 0.0, "orders": []}

def save_brain(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

brain = init_brain()

# Session State
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'position' not in st.session_state: st.session_state.position = brain.get('position', None)
if 'pnl' not in st.session_state: st.session_state.pnl = brain.get('pnl', 0.0)

# --- 5. AUDIO (Voice Fix Attempt) ---
def speak_aether(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "alert.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        
        # Hidden audio player with autoplay enabled
        md = f"""
            <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except: pass

# --- 6. PHYSICS CORE ---
def calculate_metrics(prices):
    p = np.array(list(prices))
    if len(p) < 10: return 0,0,0
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    entropy = np.std(p[-10:])
    return v, a, entropy

# --- 7. DATA ---
def get_live_data():
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            key = next((k for k in [REQ_INSTRUMENT_KEY, REQ_INSTRUMENT_KEY.replace('|', ':')] if k in data), list(data.keys())[0])
            return float(data[key]['last_price'])
    except: pass
    return None

# --- 8. UI LAYOUT ---
st.markdown(f"<h1 style='text-align:center;'>👑 CM-X GOD MODE: {OWNER_NAME}</h1>", unsafe_allow_html=True)

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
price_ph = c1.empty()
v_ph = c2.empty()
a_ph = c3.empty()
pnl_ph = c4.empty()

# Chart
chart_ph = st.empty()

# Controls
st.write("---")
col_btn1, col_btn2 = st.columns(2)
if col_btn1.button("🟢 START ENGINE"): 
    st.session_state.bot_active = True
    speak_aether("Engine Started Boss")
if col_btn2.button("🔴 STOP ENGINE"): 
    st.session_state.bot_active = False
    st.session_state.pending_signal = None

# --- 9. APPROVAL SECTION (The Boss Feature) ---
approval_container = st.empty()

# --- 10. MAIN LOOP ---
if st.session_state.bot_active:
    
    # 1. Fetch Data
    price = get_live_data()
    
    if price:
        st.session_state.prices.append(price)
        v, a, entropy = calculate_metrics(st.session_state.prices)
        
        # 2. Logic (Physics Brain)
        signal = None
        if not st.session_state.position:
            if v > 1.5 and a > 0.5: signal = "BUY"
            elif v < -1.5 and a < -0.5: signal = "SELL"
        
        # 3. Handle Pending Approval (BOSS PERMISSION)
        if signal and not st.session_state.pending_signal:
            st.session_state.pending_signal = {"type": signal, "price": price}
            speak_aether(f"Boss! {signal} Signal Detected. Waiting for approval.")
        
        # 4. Display Approval Box if Signal exists
        if st.session_state.pending_signal:
            sig = st.session_state.pending_signal
            with approval_container.container():
                st.markdown(f"""
                <div class="approval-box">
                    <h2>⚠️ PERMISSION REQUIRED ⚠️</h2>
                    <h3>SIGNAL: {sig['type']} @ {sig['price']}</h3>
                    <p>Physics Velocity: {v:.2f} | Acceleration: {a:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                ac1, ac2 = st.columns(2)
                if ac1.button("✅ APPROVE TRADE", key="approve"):
                    # EXECUTE TRADE
                    st.session_state.position = {"type": sig['type'], "entry": sig['price'], "qty": 50}
                    brain['position'] = st.session_state.position
                    save_brain(brain)
                    st.session_state.pending_signal = None # Clear signal
                    speak_aether("Trade Executed Successfully.")
                    st.rerun()
                    
                if ac2.button("❌ REJECT", key="reject"):
                    st.session_state.pending_signal = None # Clear signal
                    speak_aether("Trade Rejected.")
                    st.rerun()

        # 5. Exit Logic (Automatic)
        if st.session_state.position:
            pos = st.session_state.position
            pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
            
            # Simple Target/SL
            if pnl > 500 or pnl < -300:
                st.session_state.pnl += pnl
                brain['pnl'] = st.session_state.pnl
                st.session_state.position = None
                brain['position'] = None
                save_brain(brain)
                speak_aether("Position Closed.")
                st.rerun()

        # 6. Update UI
        price_ph.metric("NIFTY 50", f"{price:,.2f}")
        v_ph.metric("VELOCITY", f"{v:.2f}")
        a_ph.metric("ACCEL", f"{a:.2f}")
        
        pnl_color = "green" if st.session_state.pnl >= 0 else "red"
        pnl_ph.markdown(f"<h3 style='color:{pnl_color}'>P&L: ₹{st.session_state.pnl:.2f}</h3>", unsafe_allow_html=True)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41')))
        fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)

    # Loop Delay
    time.sleep(1)
    # Only rerun if NOT waiting for approval (to prevent button flicker)
    if not st.session_state.pending_signal:
        st.rerun()
