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
import math
import random

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="AETHER: FUSION GOD MODE",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 # 126-Period Momentum (Research Based)
TELEGRAM_INTERVAL = 120 # 2 Minutes
KILL_SWITCH_LOSS = -2000 

# --- 2. PROFESSIONAL LIGHT CSS (CLEAN & POWERFUL) ---
st.markdown("""
    <style>
    /* Global Font & Colors */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    .stApp { background-color: #f0f4f8; color: #102a43; font-family: 'Rajdhani', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #003e6b; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 800; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1); }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #0077cc; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #627d98; font-weight: 600; }
    
    /* Terminal Box */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #102a43;
        color: #00ff41;
        padding: 15px;
        height: 300px;
        overflow-y: auto;
        font-size: 13px;
        border-radius: 8px;
        border: 2px solid #334e68;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .log-time { color: #829ab1; margin-right: 8px; }
    .log-info { color: #40c3ff; }
    .log-warn { color: #f0b429; font-weight: bold; }
    .log-danger { color: #ef4e4e; font-weight: bold; }
    .log-brain { color: #d645bb; font-weight: bold; } /* AI Thoughts */
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0077cc, #005299);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        height: 50px;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(0, 119, 204, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #005299, #003e6b);
        box-shadow: 0 6px 12px rgba(0, 119, 204, 0.3);
        transform: scale(1.02);
    }
    
    /* Active Trade Pulse */
    .active-trade-box {
        background-color: #fff1f0;
        border: 2px solid #e12d39;
        color: #cf1124;
        padding: 15px;
        text-align: center;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(225, 45, 57, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(225, 45, 57, 0); } 100% { box-shadow: 0 0 0 0 rgba(225, 45, 57, 0); } }
    
    /* Status Indicators */
    .status-online { background-color: #e3f9e5; color: #137333; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid #137333; }
    .status-offline { background-color: #fce8e6; color: #c5221f; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid #c5221f; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SECRETS LOADING ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
except Exception as e:
    st.error(f"⚠️ CORE FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. BLACK BOX MEMORY (PERSISTENT STORAGE) ---
def init_brain_memory():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "position": None,
            "orders": [],
            "pnl": 0.0,
            "learning_rate": 1.0, 
            "win_streak": 0,
            "loss_streak": 0,
            "last_thought": "System Initialized."
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"position": None, "orders": [], "pnl": 0.0, "learning_rate": 1.0}

def save_brain(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

# LOAD BRAIN IMMEDIATELY
brain = init_brain_memory()

# SESSION STATE SYNC (CRITICAL FOR RELOADS)
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = time.time()
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=50)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""

# Restore State from File (MEMORY RECALL)
# This ensures if you refresh, the bot remembers the trade!
if 'position' not in st.session_state: st.session_state.position = brain.get('position', None)
if 'orders' not in st.session_state: st.session_state.orders = brain.get('orders', [])
if 'pnl' not in st.session_state: st.session_state.pnl = brain.get('pnl', 0.0)

# --- 5. LOGGING SYSTEM ---
def add_log(msg, type="info"):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    color_class = "log-info"
    if type == "warn": color_class = "log-warn"
    if type == "danger": color_class = "log-danger"
    if type == "brain": color_class = "log-brain"
    
    st.session_state.live_logs.appendleft(f"<span class='log-time'>[{ts}]</span> <span class='{color_class}'>{msg}</span>")

def send_telegram_report(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧬 AETHER REPORT:\n{msg}"}
        requests.get(url, params=params)
    except: pass

# --- 6. AUDIO ENGINE (GHOST SPEAKER) ---
def speak_aether(text):
    try:
        add_log(f"SPEAKING: {text}", "brain")
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "ghost.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

# --- 7. ADVANCED MATH CORE (FUSION: ROCKET + PHYSICS) ---
def rocket_formula(v, vol_current, vol_avg):
    if vol_avg == 0: vol_avg = 1
    mass_ratio = abs(vol_current / vol_avg)
    if mass_ratio <= 0: mass_ratio = 0.1
    thrust = v * math.log(mass_ratio + 1)
    return thrust

def monte_carlo_simulation(prices, num_sims=100, steps=10):
    # Convert to list first to avoid deque issues
    p_array = np.array(list(prices))
    if len(p_array) < 20: return 0.5
    
    last_price = p_array[-1]
    returns = np.diff(p_array) / p_array[:-1]
    mu = np.mean(returns)
    sigma = np.std(returns)
    bullish_paths = 0
    for _ in range(num_sims):
        sim_price = last_price
        for _ in range(steps):
            shock = np.random.normal(mu, sigma)
            sim_price = sim_price * (1 + shock)
        if sim_price > last_price: bullish_paths += 1
    return bullish_paths / num_sims

def calculate_singularity_metrics(prices):
    p = np.array(list(prices))
    if len(p) < 10: return 0,0,0,0,0
    
    # 1. Physics (Newton)
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    
    # 2. Chaos (Entropy)
    entropy = np.std(p[-10:])
    
    # 3. Rocket Thrust
    volatility = entropy if entropy > 0 else 1
    thrust = rocket_formula(v, volatility*1.5, volatility) 
    
    # 4. Prediction (Monte Carlo)
    prob = monte_carlo_simulation(prices)
    
    return v, a, entropy, thrust, prob

# --- 8. AI CONSULTANT (THE GHOST) ---
def consult_ghost(price, v, a, t, p):
    try:
        prompt = f"Market: {price}, Thrust: {t:.2f}, WinProb: {p:.2f}. One line sci-fi trading advice?"
        res = model.generate_content(prompt)
        return res.text
    except: return "AI Recalibrating..."

# --- 9. REAL DATA FETCH (SMART KEY FIX) ---
def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            # Handles both ':' and '|' automatically
            key = next((k for k in [REQ_INSTRUMENT_KEY, REQ_INSTRUMENT_KEY.replace('|', ':')] if k in data), list(data.keys())[0])
            return float(data[key]['last_price'])
    except: pass
    return None

# --- 10. SELF-CORRECTION (LEARNING) ---
def update_learning_rate(result):
    lr = brain.get("learning_rate", 1.0)
    if result == "WIN":
        lr = min(1.5, lr + 0.1) # Boost confidence
        brain["win_streak"] = brain.get("win_streak", 0) + 1
        brain["loss_streak"] = 0
    else:
        lr = max(0.5, lr - 0.2) # Reduce risk
        brain["loss_streak"] = brain.get("loss_streak", 0) + 1
        brain["win_streak"] = 0
    brain["learning_rate"] = lr
    save_brain(brain)
    return lr

# --- 11. DASHBOARD LAYOUT (UI) ---
st.markdown(f"""
<div style="text-align:center; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1;">
    <h1 style="margin-bottom: 5px;">AETHER: FUSION GOD MODE</h1>
    <p style="color:#486581; font-weight:bold;">OPERATOR: {OWNER_NAME} | BRAIN: ACTIVE | LEARNING RATE: {brain.get('learning_rate', 1.0):.2f}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Connection Status
price_check = get_live_data()
if price_check:
    st.markdown('<div class="status-online">🟢 UPSTOX CONNECTED | DATA FLOWING</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-offline">🔴 CONNECTION LOST | CHECK TOKEN</div>', unsafe_allow_html=True)

# Active Trade Warning (Persistent)
if st.session_state.position:
    pos = st.session_state.position
    st.markdown(f"""
    <div class="active-trade-box">
        🔥 ACTIVE POSITION RESTORED: {pos['type']} @ {pos['entry']} | QTY: {pos['qty']}
    </div>
    """, unsafe_allow_html=True)

# METRICS ROW
c1, c2, c3, c4, c5 = st.columns(5)
p_ph = c1.empty() # Price
v_ph = c2.empty() # Velocity
t_ph = c3.empty() # Thrust
m_ph = c4.empty() # Win Prob
e_ph = c5.empty() # Chaos

# MAIN GRID
g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Quantum Trajectory")
    chart_ph = st.empty()
    st.subheader("🖥️ Neural Core Logs")
    log_ph = st.empty()

with g2:
    st.subheader("🧠 Control Deck")
    
    # GHOST BUTTON
    if st.button("🔊 CONSULT AETHER"):
        if st.session_state.prices:
            p_curr = list(st.session_state.prices)[-1]
            speak_aether(f"Market at {p_curr}. Analyzing fields.")
            st.toast("Ghost Protocol Initiated")

    st.write("---")
    pnl_ph = st.empty() # P&L
    
    st.write("---")
    c_start, c_stop = st.columns(2)
    if c_start.button("▶️ ACTIVATE"):
        st.session_state.bot_active = True
        add_log("SYSTEM ACTIVATED", "brain")
    if c_stop.button("⏹️ DEACTIVATE"):
        st.session_state.bot_active = False
        add_log("SYSTEM STOPPED", "danger")
        
    st.caption("Status:")
    if brain['position']:
        st.info(f"HOLDING: {brain['position']['type']}")
    else:
        st.success("SCANNING...")

# --- 12. THE MAIN LOOP ---
if st.session_state.bot_active:
    
    if not price_check:
        st.error("NO DATA CONNECTION")
        st.stop()
        
    while st.session_state.bot_active:
        
        # 1. KILL SWITCH CHECK
        if st.session_state.pnl < KILL_SWITCH_LOSS:
            speak_aether("Critical Failure. Kill Switch Activated.")
            st.session_state.bot_active = False
            add_log("MAX LOSS REACHED. SHUTTING DOWN.", "danger")
            break

        # 2. LIVE DATA FETCH
        price = get_live_data()
        if not price:
            time.sleep(1)
            continue
        
        st.session_state.prices.append(price)
        
        # 3. HYPER-CALCULATIONS (Fusion of ALL Theories)
        v, a, entropy, thrust, prob = calculate_singularity_metrics(st.session_state.prices)
        lr = brain.get("learning_rate", 1.0)
        
        # 4. DECISION LOGIC (FUSION)
        
        # BUY LOGIC: (Rocket + Monte Carlo + Physics + Entropy)
        if (prob > 0.6) and (thrust > 0.5) and (v > 1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "BUY", "entry": price, "qty": 50}
                save_brain(brain) # Auto-Save
                
                # Update Session State Immediately
                st.session_state.position = brain['position']
                
                speak_aether("Thrust Detected. Buying Call.")
                add_log(f"BUY ORDER | P={prob:.2f}", "warn")
                st.rerun()
        
        # SELL LOGIC
        elif (prob < 0.4) and (thrust < -0.5) and (v < -1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "SELL", "entry": price, "qty": 50}
                save_brain(brain) # Auto-Save
                
                # Update Session State Immediately
                st.session_state.position = brain['position']
                
                speak_aether("Negative Thrust. Selling Put.")
                add_log(f"SELL ORDER | P={prob:.2f}", "warn")
                st.rerun()
                
        # EXIT LOGIC (Adaptive)
        if brain['position']:
            pos = brain['position']
            pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
            
            target = 500 * lr
            stoploss = -250 * (1/lr)
            
            if pnl > target or pnl < stoploss:
                res = "WIN" if pnl > 0 else "LOSS"
                new_lr = update_learning_rate(res)
                
                brain['pnl'] += pnl
                brain['position'] = None
                brain['orders'].insert(0, f"{res} | P&L: {pnl:.0f}")
                save_brain(brain) # Auto-Save
                
                # Update Session State Immediately
                st.session_state.position = None
                st.session_state.pnl = brain['pnl']
                
                speak_aether(f"Position Closed. {res}.")
                add_log(f"EXIT | P&L: {pnl}", "brain")
                st.rerun()

        # 5. TELEGRAM AUTO-REPORT (2 Mins)
        if time.time() - st.session_state.last_tg_time > TELEGRAM_INTERVAL:
            report = f"⏰ {datetime.now().strftime('%H:%M')}\n💰 NIFTY: {price}\n🚀 THRUST: {thrust:.2f}\n💵 P&L: {brain['pnl']:.2f}"
            send_telegram_report(report)
            st.session_state.last_tg_time = time.time()
            add_log("TELEGRAM REPORT SENT", "info")

        # 6. UI UPDATE
        p_ph.metric("NIFTY 50", f"{price:,.2f}")
        v_ph.metric("VELOCITY", f"{v:.2f}")
        t_ph.metric("THRUST", f"{thrust:.2f}")
        m_ph.metric("WIN PROB", f"{prob*100:.0f}%")
        e_ph.metric("CHAOS", f"{entropy:.2f}")
        
        # P&L Display
        total = brain['pnl'] + (pnl if brain['position'] else 0)
        col = "#16a34a" if total >= 0 else "#dc2626" 
        pnl_ph.markdown(f"<h1 style='color:{col}; text-align:center;'>₹{total:.2f}</h1>", unsafe_allow_html=True)
        
        # Live Terminal
        log_html = "".join([f"<div>{l}</div>" for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
        
        # Chart (Professional Light)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#0077cc', width=2), fill='tozeroy', fillcolor='rgba(0, 119, 204, 0.1)'))
        if brain['position']:
            fig.add_hline(y=brain['position']['entry'], line_dash="dash", line_color="#ff9900")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
