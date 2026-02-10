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
    page_title="AETHER: LIGHT MODE",
    layout="wide",
    page_icon="☀️",
    initial_sidebar_state="collapsed"
)

# CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 
TELEGRAM_INTERVAL = 120 # 2 Minutes
KILL_SWITCH_LOSS = -2000 

# --- 2. PROFESSIONAL LIGHT CSS (CLEAN LOOK) ---
st.markdown("""
    <style>
    /* Main Background - White */
    .stApp { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #0f172a; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Metrics Box */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-2px); }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #2563eb; /* Royal Blue */
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b; /* Slate Grey */
        font-weight: 600;
    }
    
    /* Terminal / Logs (Light Version) */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        font-size: 13px;
        border-left: 5px solid #2563eb;
        border-radius: 5px;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.05);
    }
    .log-time { color: #94a3b8; font-weight: bold; }
    .log-info { color: #0f172a; }
    .log-warn { color: #d97706; font-weight: bold; } /* Amber */
    .log-danger { color: #dc2626; font-weight: bold; } /* Red */
    .log-brain { color: #7c3aed; font-weight: bold; } /* Purple */
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        height: 50px;
        transition: 0.3s;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
        box-shadow: 0 6px 10px rgba(37, 99, 235, 0.3);
    }
    
    /* Active Trade Box */
    .active-trade-box {
        background-color: #fff7ed;
        border: 2px solid #f97316;
        color: #c2410c;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(249, 115, 22, 0.1);
    }
    
    /* Status Bar */
    .status-connected { background-color: #dcfce7; color: #166534; padding: 8px; text-align: center; border-radius: 5px; font-weight: bold; border: 1px solid #166534; margin-bottom: 10px; }
    .status-disconnected { background-color: #fee2e2; color: #991b1b; padding: 8px; text-align: center; border-radius: 5px; font-weight: bold; border: 1px solid #991b1b; margin-bottom: 10px; }
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

# --- 4. MEMORY & SESSION ---
def init_brain():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "position": None, "orders": [], "pnl": 0.0,
            "learning_rate": 1.0, "win_streak": 0, "loss_streak": 0
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"position": None, "orders": [], "pnl": 0.0, "learning_rate": 1.0}

def save_brain(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

brain = init_brain()

# Session Sync
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = time.time()
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=30)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""

# --- 5. LOGGING ---
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
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"☀️ AETHER LIGHT:\n{msg}"}
        requests.get(url, params=params)
    except: pass

# --- 6. AUDIO ENGINE ---
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

# --- 7. MATH CORE (FIXED ERROR HERE) ---
def rocket_formula(v, vol_current, vol_avg):
    if vol_avg == 0: vol_avg = 1
    mass_ratio = abs(vol_current / vol_avg) # Fix: Ensure positive
    if mass_ratio <= 0: mass_ratio = 0.1
    thrust = v * math.log(mass_ratio + 1) # Fix: log domain error
    return thrust

def monte_carlo_simulation(prices, num_sims=100, steps=10):
    # FIX: Convert deque to numpy array immediately to prevent TypeError
    p_array = np.array(list(prices))
    
    if len(p_array) < 20: return 0.5
    
    last_price = p_array[-1]
    # Calculate returns safely
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
    # Ensure prices is a list/array before processing
    p = np.array(list(prices))
    
    if len(p) < 10: return 0,0,0,0,0
    
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    entropy = np.std(p[-10:])
    volatility = entropy if entropy > 0 else 1
    thrust = rocket_formula(v, volatility*1.5, volatility) 
    prob = monte_carlo_simulation(prices) # Pass original deque, function handles it
    
    return v, a, entropy, thrust, prob

# --- 8. REAL DATA FETCH ---
def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            key = next((k for k in [REQ_INSTRUMENT_KEY, REQ_INSTRUMENT_KEY.replace('|', ':')] if k in data), list(data.keys())[0])
            return float(data[key]['last_price'])
    except: pass
    return None

# --- 9. LEARNING ---
def update_learning_rate(result):
    lr = brain.get("learning_rate", 1.0)
    if result == "WIN": lr = min(1.5, lr + 0.1)
    else: lr = max(0.5, lr - 0.2)
    brain["learning_rate"] = lr
    save_brain(brain)
    return lr

# --- 10. UI LAYOUT (LIGHT THEME) ---
st.markdown(f"""
<div style="text-align:center; padding-bottom: 20px; border-bottom: 2px solid #e2e8f0;">
    <h1 style="color:#1e3a8a; margin-bottom: 5px;">AETHER: SINGULARITY (LIGHT)</h1>
    <p style="color:#64748b;">OPERATOR: <b>{OWNER_NAME}</b> | BRAIN: <b>ACTIVE</b> | LR: <b>{brain.get('learning_rate', 1.0):.2f}</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Connection Status
price_check = get_live_data()
if price_check:
    st.markdown('<div class="status-connected">🟢 SYSTEM ONLINE | DATA FLOWING</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-disconnected">🔴 CONNECTION LOST | CHECK TOKEN</div>', unsafe_allow_html=True)

# Active Trade Warning
if st.session_state.position:
    pos = st.session_state.position
    st.markdown(f"""
    <div class="active-trade-box">
        🔥 ACTIVE TRADE RUNNING: {pos['type']} @ {pos['entry']}
    </div>
    """, unsafe_allow_html=True)

# METRICS ROW
c1, c2, c3, c4, c5 = st.columns(5)
p_ph = c1.empty()
v_ph = c2.empty()
t_ph = c3.empty()
m_ph = c4.empty()
e_ph = c5.empty()

# MAIN AREA
g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Market Analysis")
    chart_ph = st.empty()
    st.subheader("🖥️ System Logs")
    log_ph = st.empty()

with g2:
    st.subheader("🤖 Command Center")
    
    if st.button("🔊 ASK GHOST"):
        if st.session_state.prices:
            p_curr = list(st.session_state.prices)[-1]
            speak_aether(f"Market at {p_curr}. Analyzing structure.")
            st.toast("Ghost Analysis Requested")

    st.write("---")
    pnl_ph = st.empty()
    
    st.write("---")
    c_start, c_stop = st.columns(2)
    if c_start.button("▶️ START AUTO"):
        st.session_state.bot_active = True
        add_log("AUTO-PILOT ENGAGED", "brain")
    if c_stop.button("⏹️ STOP SYSTEM"):
        st.session_state.bot_active = False
        add_log("MANUAL STOP TRIGGERED", "danger")
        
    st.caption("Current Status:")
    if brain['position']:
        st.info(f"HOLDING: {brain['position']['type']}")
    else:
        st.success("SCANNING MARKET...")

# --- 11. MAIN LOOP ---
if st.session_state.bot_active:
    
    if not price_check:
        st.error("NO DATA CONNECTION")
        st.stop()
        
    while st.session_state.bot_active:
        
        # 1. LIVE DATA
        price = get_live_data()
        if not price:
            time.sleep(1)
            continue
        
        st.session_state.prices.append(price)
        
        # 2. CALCULATIONS (Fixed Error Here)
        v, a, entropy, thrust, prob = calculate_singularity_metrics(st.session_state.prices)
        lr = brain.get("learning_rate", 1.0)
        
        # 3. DECISION LOGIC
        # BUY
        if (prob > 0.6) and (thrust > 0.5) and (v > 1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "BUY", "entry": price, "qty": 50}
                save_brain(brain)
                speak_aether("Thrust Confirmed. Buying Call.")
                add_log(f"BUY ORDER | P={prob:.2f}", "warn")
                st.rerun()
        
        # SELL
        elif (prob < 0.4) and (thrust < -0.5) and (v < -1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "SELL", "entry": price, "qty": 50}
                save_brain(brain)
                speak_aether("Negative Thrust. Selling Put.")
                add_log(f"SELL ORDER | P={prob:.2f}", "warn")
                st.rerun()
                
        # EXIT
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
                save_brain(brain)
                speak_aether(f"Position Closed. {res}.")
                add_log(f"EXIT | P&L: {pnl}", "brain")
                st.rerun()

        # 4. TELEGRAM (2 Min)
        if time.time() - st.session_state.last_tg_time > TELEGRAM_INTERVAL:
            report = f"⏰ {datetime.now().strftime('%H:%M')}\n💰 NIFTY: {price}\n🚀 THRUST: {thrust:.2f}\n💵 P&L: {brain['pnl']:.2f}"
            send_telegram_report(report)
            st.session_state.last_tg_time = time.time()
            add_log("REPORT SENT", "info")

        # 5. UI UPDATE
        p_ph.metric("NIFTY 50", f"{price:,.2f}")
        v_ph.metric("VELOCITY", f"{v:.2f}")
        t_ph.metric("THRUST", f"{thrust:.2f}")
        m_ph.metric("WIN PROB", f"{prob*100:.0f}%")
        e_ph.metric("CHAOS", f"{entropy:.2f}")
        
        # P&L
        total = brain['pnl'] + (pnl if brain['position'] else 0)
        col = "#16a34a" if total >= 0 else "#dc2626" # Green/Red
        pnl_ph.markdown(f"<h1 style='color:{col}; text-align:center;'>₹{total:.2f}</h1>", unsafe_allow_html=True)
        
        # Logs
        log_html = "".join([f"<div>{l}</div>" for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
        
        # Chart (Light Theme)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#2563eb', width=2), fill='tozeroy', fillcolor='rgba(37, 99, 235, 0.1)'))
        if brain['position']:
            fig.add_hline(y=brain['position']['entry'], line_dash="dash", line_color="#f97316")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
