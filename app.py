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
    page_title="PROJECT AETHER: GOD MODE",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# TIMEZONE FIX (IST) - முக்கியம்!
IST = pytz.timezone('Asia/Kolkata')

# AETHER SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 
KILL_SWITCH_LOSS = -2000 

# --- 2. ADVANCED CYBERPUNK STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
    
    .stApp { background-color: #050505; color: #00ff41; font-family: 'Courier New', monospace; }
    
    /* Neon Text */
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; text-shadow: 0 0 10px #00ff41; color: #fff; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        color: #00ff41;
        text-shadow: 0 0 5px #00ff41;
    }
    div[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #333; box-shadow: 0 0 10px rgba(0, 255, 65, 0.1); }
    
    /* Live Terminal Log */
    .terminal-box {
        font-family: 'Fira Code', monospace; background-color: #000; border: 1px solid #333;
        color: #00ff41; padding: 10px; height: 200px; overflow-y: auto; font-size: 14px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #000; color: #00ff41; border: 1px solid #00ff41;
        font-family: 'Orbitron', sans-serif; height: 50px; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #00ff41; color: #000; box-shadow: 0 0 20px #00ff41; }
    
    /* Active Trade Warning */
    .active-trade-box {
        border: 2px solid #ff0000; background-color: #220000; color: #ff4444;
        padding: 15px; text-align: center; font-family: 'Orbitron'; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    
    /* Input Box */
    .stTextInput>div>div>input { background-color: #111; color: #00ff41; border: 1px solid #00ff41; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SECRETS LOADING ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    # Optional Telegram
    if "telegram" in st.secrets:
        TG_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_ID = st.secrets["telegram"]["chat_id"]
    else: TG_TOKEN = None; TG_ID = None
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
except Exception as e:
    st.error(f"⚠️ SYSTEM FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. BLACK BOX MEMORY ---
def init_aether_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"position": None, "orders": [], "pnl": 0.0, "daily_stats": {"wins": 0, "losses": 0}, "last_thought": "System Initialized."}
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return {"position": None, "orders": [], "pnl": 0.0, "daily_stats": {"wins":0, "losses":0}}

def save_aether_memory(pos, orders, pnl, stats, thought):
    data = {"position": pos, "orders": orders, "pnl": pnl, "daily_stats": stats, "last_thought": thought}
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

brain = init_aether_memory()

# Session State
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = brain['position']
if 'orders' not in st.session_state: st.session_state.orders = brain['orders']
if 'pnl' not in st.session_state: st.session_state.pnl = brain['pnl']
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = 0.0

# --- 5. LOGGING (IST FIXED) ---
def add_log(msg, type="info"):
    timestamp = datetime.now(IST).strftime("%H:%M:%S") # IST Time Fix
    color = "#00ff41"
    if type == "warn": color = "#ffff00"
    if type == "danger": color = "#ff0000"
    st.session_state.live_logs.appendleft(f"<span style='color:#888'>[{timestamp}]</span> <span style='color:{color}'>{msg}</span>")

def send_telegram(msg):
    if TG_TOKEN and TG_ID:
        try: requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", params={"chat_id": TG_ID, "text": f"🧬 CM-X: {msg}"})
        except: pass

# --- 6. AUDIO ENGINE (GHOST VOICE) ---
def speak_aether(text):
    try:
        brain['last_thought'] = text
        add_log(f"JARVIS: {text}", "warn")
        
        # Tamil Voice Priority
        tts = gTTS(text=text, lang='ta', tld='co.in')
        filename = "ghost_voice.mp3"
        tts.save(filename)
        
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

# --- 7. QUANTUM PHYSICS CORE (THE SHARP LOGIC) ---
def calculate_quantum_state(price_deque):
    if len(price_deque) < 10: return 0, 0, 0, 0
    p = np.array(price_deque)
    velocity = np.diff(p)[-1]
    acceleration = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    energy = abs(velocity) * 100 
    entropy = np.std(p[-10:])
    return velocity, acceleration, energy, entropy

# --- 8. GEMINI AI (CHAT) ---
def consult_ghost_brain(query, price):
    try:
        prompt = f"You are AETHER (Tamil Trading Bot). Price: {price}. User: {query}. Reply in Tamil."
        return model.generate_content(prompt).text
    except: return "Connection Lost."

# --- 9. DATA FETCHING ---
def get_live_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None, "NO TOKEN"
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            key = list(data.keys())[0]
            return float(data[key]['last_price']), "CONNECTED"
    except: pass
    return None, "ERROR"

# --- 10. UI LAYOUT ---
st.markdown(f"""
<div style="text-align: center;">
    <h1>PROJECT AETHER: GOD MODE</h1>
    <p style="color: #00ff41;">OPERATOR: {OWNER_NAME} | SYSTEM: <b>ONLINE</b></p>
</div>
""", unsafe_allow_html=True)
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Active Trade Alert
if st.session_state.position:
    pos = st.session_state.position
    st.markdown(f"""<div class="active-trade-box"><h3>⚠ TRADE ACTIVE: {pos['type']} @ {pos['entry']}</h3></div>""", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 Schrödinger Wave Function")
    chart_ph = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    p_metric = m1.empty(); v_metric = m2.empty(); a_metric = m3.empty(); e_metric = m4.empty()
    
    st.write("---")
    st.subheader("🖥️ LIVE TERMINAL")
    log_ph = st.empty()

with c2:
    st.subheader("👻 JARVIS LINK")
    
    # JARVIS INPUT (The Feature You Asked For)
    user_q = st.text_input("Talk to Jarvis:", placeholder="Type here...")
    if st.button("SEND COMMAND"):
        if user_q:
            p = st.session_state.prices[-1] if st.session_state.prices else 0
            add_log(f"BOSS: {user_q}", "info")
            rep = consult_ghost_brain(user_q, p)
            speak_aether(rep)
            
    st.write("---")
    pnl_display = st.empty()
    
    # Controls
    b1, b2 = st.columns(2)
    start_btn = b1.button("🔥 INITIATE")
    stop_btn = b2.button("🛑 KILL SWITCH")
    
    if st.button("❌ EMERGENCY EXIT"):
        st.session_state.position = None
        speak_aether("Emergency Exit.")
        st.rerun()

# Control Logic
if start_btn: st.session_state.bot_active = True
if stop_btn: st.session_state.bot_active = False

# --- 11. MAIN LOOP ---
if st.session_state.bot_active:
    
    price, status = get_live_market_data()
    if status == "CONNECTED":
        st.session_state.prices.append(price)
        
        # 1. PHYSICS
        v, a, energy, entropy = calculate_quantum_state(st.session_state.prices)
        
        # 2. DECISION (The Sharp Logic)
        # Entry
        if st.session_state.position is None:
            # STRICT ENTRY RULES (Chaos < 10, V > 1.5)
            if v > 1.5 and a > 0.3 and entropy < 10:
                st.session_state.position = {"type": "BUY", "entry": price, "qty": 50}
                st.session_state.trailing_high = 0.0
                speak_aether(f"Momentum High. Buying at {price}")
                add_log("EXECUTING BUY ORDER", "warn")
                send_telegram(f"BUY: {price}")
                st.rerun()
                
            elif v < -1.5 and a < -0.3 and entropy < 10:
                st.session_state.position = {"type": "SELL", "entry": price, "qty": 50}
                st.session_state.trailing_high = 0.0
                speak_aether(f"Gravity High. Selling at {price}")
                add_log("EXECUTING SELL ORDER", "warn")
                send_telegram(f"SELL: {price}")
                st.rerun()
                
        # Exit (Zero Loss Logic)
        else:
            pos = st.session_state.position
            pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
            
            if pnl > st.session_state.trailing_high: st.session_state.trailing_high = pnl
            high = st.session_state.trailing_high
            
            exit = False; reason = ""
            
            # THE RULES
            if high > 500 and pnl < 200: exit = True; reason = "ZERO LOSS HIT"
            elif high > 1000 and pnl < high*0.7: exit = True; reason = "TRAILING HIT"
            elif pnl < -300: exit = True; reason = "STOP LOSS"
            elif (pos['type']=="BUY" and v < -1.5): exit = True; reason = "REVERSAL"
            elif (pos['type']=="SELL" and v > 1.5): exit = True; reason = "REVERSAL"
            
            if exit:
                st.session_state.pnl += pnl
                brain['daily_stats']['wins' if pnl>0 else 'losses'] += 1
                save_aether_memory(None, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], f"Closed: {reason}")
                
                st.session_state.position = None
                speak_aether(f"Trade Closed. {reason}. PNL: {pnl}")
                add_log(f"CLOSED: {reason}", "danger")
                send_telegram(f"CLOSED: {pnl}")
                st.rerun()

        # 3. UI UPDATE
        p_metric.metric("NIFTY 50", f"{price:,.2f}", f"{v:.2f}")
        v_metric.metric("VELOCITY", f"{v:.2f}")
        a_metric.metric("ACCEL", f"{a:.2f}")
        e_metric.metric("ENTROPY", f"{entropy:.2f}")
        
        # PNL Display
        curr_pnl = 0.0
        if st.session_state.position:
            pos = st.session_state.position
            curr_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
        tot = st.session_state.pnl + curr_pnl
        pnl_display.markdown(f"<h1 style='color:{'#0f0' if tot>=0 else '#f00'}; text-align:center;'>PNL: ₹{tot:.2f}</h1>", unsafe_allow_html=True)
        
        # Logs
        l_html = "".join([l for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{l_html}</div>', unsafe_allow_html=True)
        
        # Chart (With CRASH FIX)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
        if st.session_state.position:
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        
        # *** THIS IS THE CRITICAL FIX FOR YOUR ERROR ***
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
        
        time.sleep(0.2)
        st.refresh()
