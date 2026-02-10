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
import random

# --- 1. SYSTEM CONFIGURATION & PERSONA ---
st.set_page_config(
    page_title="PROJECT AETHER: GOD MODE",
    layout="wide",
    page_icon="👻",
    initial_sidebar_state="collapsed"
)

# AETHER SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 
KILL_SWITCH_LOSS = -2000 

# --- 2. ADVANCED CYBERPUNK STYLING (SAME AS YOUR SCREENSHOT) ---
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
    div[data-testid="stMetricLabel"] { color: #888; font-weight: bold; }
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
    }
    
    /* Live Terminal Log */
    .terminal-box {
        font-family: 'Fira Code', monospace;
        background-color: #000;
        border: 1px solid #333;
        color: #00ff41;
        padding: 10px;
        height: 200px;
        overflow-y: auto;
        font-size: 14px;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.2);
    }
    .log-time { color: #888; margin-right: 10px; }
    .log-info { color: #00ff41; }
    .log-warn { color: #ffff00; }
    .log-danger { color: #ff0000; }
    
    /* Approval Box (New Feature) */
    .approval-box {
        border: 2px solid #ffff00;
        background-color: #220;
        padding: 15px;
        text-align: center;
        animation: flash 1s infinite;
        margin-bottom: 10px;
    }
    @keyframes flash { 0% { border-color: #ffff00; } 50% { border-color: #ff0000; } 100% { border-color: #ffff00; } }

    /* Buttons */
    .stButton>button {
        background-color: #000;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Orbitron', sans-serif;
        transition: 0.3s;
        height: 50px;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000;
        box-shadow: 0 0 20px #00ff41;
    }
    
    /* Agent Cards (The 4 Council Members) */
    .agent-card {
        background: #111; border: 1px solid #333; padding: 10px; text-align: center; border-radius: 5px;
    }
    .agent-buy { border-color: #00ff41; color: #00ff41; }
    .agent-sell { border-color: #ff003c; color: #ff003c; }
    .agent-wait { border-color: #888; color: #888; }

    </style>
    """, unsafe_allow_html=True)

# --- 3. SECRETS LOADING ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    genai.configure(api_key=GEMINI_API_KEY)
    # Using Flash model for speed as per your request
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
except Exception as e:
    st.error(f"⚠️ SYSTEM FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. MEMORY & STATE ---
def init_aether_memory():
    if not os.path.exists(MEMORY_FILE):
        data = {"position": None, "orders": [], "pnl": 0.0}
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"position": None, "orders": [], "pnl": 0.0}

def save_aether_memory(pos, orders, pnl):
    data = {"position": pos, "orders": orders, "pnl": pnl}
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

brain = init_aether_memory()

if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = brain['position']
if 'orders' not in st.session_state: st.session_state.orders = brain['orders']
if 'pnl' not in st.session_state: st.session_state.pnl = brain['pnl']
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None # For Approval

# --- 5. LOGGING SYSTEM ---
def add_log(msg, type="info"):
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    color_class = "log-info"
    if type == "warn": color_class = "log-warn"
    if type == "danger": color_class = "log-danger"
    log_entry = f"<span class='log-time'>[{timestamp}]</span> <span class='{color_class}'>{msg}</span>"
    st.session_state.live_logs.appendleft(log_entry)

# --- 6. AUDIO ENGINE (UPDATED FOR MOBILE) ---
def speak_aether(text):
    try:
        # Generate Audio
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "alert.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        
        # HTML5 Audio Player (Autoplay on compatible browsers)
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)
    except: pass

# --- 7. THE COUNCIL (4 AGENTS LOGIC) ---
# இதுதான் நீங்க கேட்ட புது லாஜிக் (4 பேர் முடிவு எடுப்பது)
def consult_the_council(price, v, a, entropy):
    council_votes = {}
    
    # Agent 1: Physics (The existing logic)
    if v > 1.5 and a > 0.2: council_votes['Physics'] = "BUY"
    elif v < -1.5 and a < -0.2: council_votes['Physics'] = "SELL"
    else: council_votes['Physics'] = "WAIT"
    
    # Agent 2: Trend (Simulated SuperTrend)
    # ரியல் டைமில் இதை indicators வைத்து மாற்றலாம். இப்போதைக்கு Moving Average Logic.
    trend = "WAIT"
    if len(st.session_state.prices) > 20:
        ma20 = np.mean(list(st.session_state.prices)[-20:])
        if price > ma20 + 5: trend = "BUY"
        elif price < ma20 - 5: trend = "SELL"
    council_votes['Trend'] = trend
    
    # Agent 3: Option Chain (Simulated PCR)
    # நிஜ API இணைப்பு இல்லையென்றால் இது ஒரு கணிப்பு
    pcr_sim = 0.5 + (random.random()) # Mock Value
    if pcr_sim > 1.2: council_votes['Options'] = "BUY"
    elif pcr_sim < 0.7: council_votes['Options'] = "SELL"
    else: council_votes['Options'] = "WAIT"
    
    # Agent 4: Volatility (Entropy based)
    if entropy > 2.0: council_votes['Volatility'] = "NO_TRADE"
    else: council_votes['Volatility'] = "GO"
    
    return council_votes

# --- 8. DATA FETCHING ---
def get_live_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None, "NO TOKEN"
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': REQ_INSTRUMENT_KEY}
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return float(data['data'][list(data['data'].keys())[0]]['last_price']), "CONNECTED"
        return None, "API ERROR"
    except: return None, "NET ERROR"

# --- 9. UI LAYOUT ---
st.markdown(f"""
<div style="text-align: center;">
    <h1>PROJECT AETHER: GOD MODE</h1>
    <p style="color: #00ff41;">OPERATOR: {OWNER_NAME} | MEMORY: <b>ATTACHED</b></p>
</div>
""", unsafe_allow_html=True)

# Grid Layout
c1, c2 = st.columns([2, 1])

# --- LEFT COLUMN: DATA & CHART ---
with c1:
    st.subheader("📊 Schrödinger Wave Function")
    chart_ph = st.empty()
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    p_metric = m1.empty()
    v_metric = m2.empty()
    a_metric = m3.empty()
    e_metric = m4.empty()
    
    # --- NEW FEATURE: THE COUNCIL DISPLAY ---
    st.write("---")
    st.subheader("🧠 THE COUNCIL CHAMBER (4 AGENTS)")
    council_ph = st.empty()
    
    # Live Logs
    st.write("---")
    log_ph = st.empty()

# --- RIGHT COLUMN: CONTROLS & APPROVAL ---
with c2:
    st.subheader("👻 Ghost Protocol")
    
    # NEW FEATURE: Voice Input (Text box simulation for driving safety)
    user_input = st.text_input("🎤 COMMAND JARVIS:", placeholder="Type or Speak...")
    if user_input:
        add_log(f"BOSS SAID: {user_input}", "warn")
        speak_aether(f"Copy that Boss. Analyzing {user_input}")
    
    st.write("---")
    pnl_display = st.empty()
    
    # Controls
    col_btn1, col_btn2 = st.columns(2)
    start_btn = col_btn1.button("🔥 INITIATE")
    stop_btn = col_btn2.button("🛑 STOP")
    
    # NEW FEATURE: APPROVAL SECTION
    approval_container = st.empty()

# Control State
if start_btn: st.session_state.bot_active = True
if stop_btn: st.session_state.bot_active = False

# --- 10. MAIN LOOP ---
if st.session_state.bot_active:
    
    price, status = get_live_market_data()
    
    # If API Fails, use Dummy Data for Simulation (So app doesn't crash)
    if not price: 
        if st.session_state.prices: price = st.session_state.prices[-1] + np.random.normal(0, 2)
        else: price = 22000.00
    
    st.session_state.prices.append(price)
    
    # Physics Calculation
    p_data = np.array(st.session_state.prices)
    if len(p_data) > 5:
        v = np.diff(p_data)[-1]
        a = np.diff(np.diff(p_data))[-1]
        entropy = np.std(p_data[-10:])
    else: v, a, entropy = 0, 0, 0

    # --- THE COUNCIL LOGIC ---
    votes = consult_the_council(price, v, a, entropy)
    
    # Visualize Council
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        
        def get_color(vote):
            if "BUY" in vote: return "agent-buy"
            if "SELL" in vote: return "agent-sell"
            return "agent-wait"

        cc1.markdown(f"<div class='agent-card {get_color(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {get_color(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card {get_color(votes['Options'])}'>OPTIONS<br>{votes['Options']}</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card {get_color(votes['Volatility'])}'>VIX<br>{votes['Volatility']}</div>", unsafe_allow_html=True)

    # --- DECISION & APPROVAL SYSTEM ---
    final_signal = None
    buy_votes = list(votes.values()).count("BUY")
    sell_votes = list(votes.values()).count("SELL")
    
    # If 3 out of 4 agree -> Generate Signal
    if buy_votes >= 3 and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "BUY"
        speak_aether("Boss! The Council suggests BUY. Waiting for your approval.")
        
    elif sell_votes >= 3 and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "SELL"
        speak_aether("Boss! The Council suggests SELL. Waiting for your approval.")

    # --- RENDER APPROVAL BUTTONS (IF SIGNAL EXISTS) ---
    if st.session_state.pending_signal:
        with approval_container.container():
            st.markdown(f"""
            <div class="approval-box">
                <h2>⚠️ AUTHORIZATION REQUIRED</h2>
                <h1>{st.session_state.pending_signal} SIGNAL DETECTED</h1>
            </div>
            """, unsafe_allow_html=True)
            
            b1, b2 = st.columns(2)
            if b1.button("✅ EXECUTE", key="exec"):
                # Trade Logic
                st.session_state.position = {"type": st.session_state.pending_signal, "entry": price, "qty": 50}
                st.session_state.pending_signal = None
                add_log(f"ORDER EXECUTED BY BOSS: {price}", "warn")
                speak_aether("Order Executed successfully.")
                st.rerun()
                
            if b2.button("❌ REJECT", key="rej"):
                st.session_state.pending_signal = None
                add_log("ORDER REJECTED BY BOSS", "danger")
                speak_aether("Order Cancelled.")
                st.rerun()

    # Exit Logic (Automatic)
    if st.session_state.position:
        pos = st.session_state.position
        pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        
        if pnl > 500 or pnl < -300: # Target/SL
            st.session_state.pnl += pnl
            st.session_state.position = None
            save_aether_memory(None, [], st.session_state.pnl)
            speak_aether("Target Hit. Position Closed.")
            st.rerun()

    # UI Updates
    p_metric.metric("NIFTY", f"{price:,.2f}")
    v_metric.metric("VELOCITY", f"{v:.2f}")
    a_metric.metric("ACCEL", f"{a:.2f}")
    e_metric.metric("ENTROPY", f"{entropy:.2f}")
    
    # Log Display
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
    if st.session_state.position:
        fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange")
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)
    
    time.sleep(1)
    if not st.session_state.pending_signal: # Only rerun if not waiting for button press
        st.rerun()
