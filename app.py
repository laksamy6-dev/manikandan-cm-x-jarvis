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
    page_title="AETHER: ULTIMATE SINGULARITY",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 
TELEGRAM_INTERVAL = 120 # 2 Minutes Report
KILL_SWITCH_LOSS = -2000 

# --- 2. ADVANCED CSS (HOLOGRAPHIC THEME) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    .stApp { background-color: #000000; color: #00e5ff; font-family: 'Rajdhani', sans-serif; }
    
    /* Holographic Text */
    h1, h2, h3 { text-shadow: 0 0 10px #00e5ff; color: #fff; text-transform: uppercase; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00e5ff;
        text-shadow: 0 0 8px #00e5ff;
    }
    div[data-testid="stMetricLabel"] { color: #888; font-weight: bold; }
    div[data-testid="stMetric"] {
        background: rgba(0, 20, 40, 0.8);
        border: 1px solid #004466;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* Terminal Box */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #050505;
        border: 1px solid #333;
        color: #00ff41;
        padding: 10px;
        height: 250px;
        overflow-y: auto;
        font-size: 13px;
        border-left: 3px solid #00ff41;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #000, #111);
        color: #00e5ff;
        border: 1px solid #00e5ff;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #00e5ff;
        color: #000;
        box-shadow: 0 0 20px #00e5ff;
    }
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

# --- 4. SELF-CORRECTING MEMORY ---
def init_brain():
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

brain = init_brain()

# Session Sync
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = time.time()
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=30)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""

# --- 5. LOGGING & TELEGRAM ---
def add_log(msg, type="info"):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    color = "#00ff41" # Green
    if type == "warn": color = "#ffeb3b" # Yellow
    if type == "danger": color = "#ff0000" # Red
    if type == "brain": color = "#00e5ff" # Cyan
    st.session_state.live_logs.appendleft(f"<span style='color:#888'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")

def send_telegram_report(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧬 AETHER REPORT:\n{msg}"}
        requests.get(url, params=params)
    except: pass

# --- 6. AUDIO ENGINE (GHOST SPEAKER) ---
# பாஸ்! இதுதான் நீங்க கேட்ட ஸ்பீக்கர் வசதி. இது இங்க பத்திரமா இருக்கு!
def speak_aether(text):
    try:
        add_log(f"VOCALIZING: {text}", "brain")
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "ghost.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # Hidden Player
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

# --- 7. ADVANCED MATH CORE ---
def rocket_formula(v, vol_current, vol_avg):
    if vol_avg == 0: vol_avg = 1
    mass_ratio = vol_current / vol_avg
    if mass_ratio <= 0: mass_ratio = 0.1
    thrust = v * math.log(mass_ratio)
    return thrust

def monte_carlo_simulation(prices, num_sims=100, steps=10):
    if len(prices) < 20: return 0.5
    last_price = prices[-1]
    returns = np.diff(prices) / prices[:-1]
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
    if len(prices) < 10: return 0,0,0,0,0
    p = np.array(prices)
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    entropy = np.std(p[-10:])
    volatility = entropy # Using entropy as volatility proxy
    thrust = rocket_formula(v, volatility*1.5, volatility) 
    prob = monte_carlo_simulation(prices)
    return v, a, entropy, thrust, prob

# --- 8. AI CONSULTANT ---
def consult_ghost(price, v, a, t, p):
    try:
        prompt = f"Market: {price}, Thrust: {t:.2f}, WinProb: {p:.2f}. One line sci-fi advice?"
        res = model.generate_content(prompt)
        return res.text
    except: return "Calculating..."

# --- 9. DATA FETCH ---
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

# --- 10. LEARNING LOGIC ---
def update_learning_rate(result):
    lr = brain.get("learning_rate", 1.0)
    if result == "WIN":
        lr = min(1.5, lr + 0.1)
        brain["win_streak"] = brain.get("win_streak", 0) + 1
        brain["loss_streak"] = 0
    else:
        lr = max(0.5, lr - 0.2)
        brain["loss_streak"] = brain.get("loss_streak", 0) + 1
        brain["win_streak"] = 0
    brain["learning_rate"] = lr
    save_brain(brain)
    return lr

# --- 11. UI LAYOUT ---
st.markdown(f"""
<div style="text-align:center; margin-bottom: 20px;">
    <h1>AETHER: SINGULARITY MODE</h1>
    <p style="color:#00e5ff; letter-spacing:2px;">OPERATOR: {OWNER_NAME} | BRAIN: <b>EVOLVING</b> | LR: <b>{brain.get('learning_rate', 1.0):.2f}</b></p>
</div>
""", unsafe_allow_html=True)

# Audio Injection
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Metrics
c1, c2, c3, c4, c5 = st.columns(5)
p_ph = c1.empty()
v_ph = c2.empty()
t_ph = c3.empty() # Rocket Thrust
m_ph = c4.empty() # Monte Carlo
e_ph = c5.empty() # Entropy

# Main Grid
g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Quantum Probability Field")
    chart_ph = st.empty()
    st.subheader("🖥️ Neural Core Terminal")
    log_ph = st.empty()

with g2:
    st.subheader("👻 Ghost Protocol")
    
    # MANUAL VOICE TRIGGER
    if st.button("🔊 CONSULT AETHER"):
        if st.session_state.prices:
            p_c = st.session_state.prices[-1]
            v, a, e, t, p = calculate_singularity_metrics(st.session_state.prices)
            msg = consult_ghost(p_c, v, a, t, p)
            speak_aether(msg)
            st.toast(f"AETHER: {msg}", icon="👻")

    st.write("---")
    pnl_ph = st.empty()
    
    st.write("---")
    c_start, c_stop = st.columns(2)
    if c_start.button("🔥 ACTIVATE"):
        st.session_state.bot_active = True
        add_log("PROTOCOL STARTED.", "brain")
    if c_stop.button("🛑 SHUTDOWN"):
        st.session_state.bot_active = False
        add_log("SYSTEM HALTED.", "danger")
        
    st.caption("Active Memory")
    if brain['position']:
        st.info(f"OPEN: {brain['position']['type']} @ {brain['position']['entry']}")
    else:
        st.success("SCANNING...")

# --- 12. SINGULARITY LOOP ---
if st.session_state.bot_active:
    
    price = get_live_data()
    if not price:
        st.error("DATALINK ERROR")
        st.stop()
        
    while st.session_state.bot_active:
        
        # 1. LIVE DATA
        price = get_live_data()
        if not price:
            time.sleep(1)
            continue
        
        st.session_state.prices.append(price)
        
        # 2. METRICS (Rocket + Physics + Monte Carlo)
        v, a, entropy, thrust, prob = calculate_singularity_metrics(st.session_state.prices)
        lr = brain.get("learning_rate", 1.0)
        
        # 3. DECISION LOGIC (Combined)
        # BUY
        if (prob > 0.6) and (thrust > 0.5) and (v > 1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "BUY", "entry": price, "qty": 50}
                save_brain(brain)
                speak_aether("Rocket Thrust Detected. Probability High. Buying.")
                add_log(f"BUY | P={prob:.2f} T={thrust:.2f}", "warn")
                st.rerun()
        
        # SELL
        elif (prob < 0.4) and (thrust < -0.5) and (v < -1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "SELL", "entry": price, "qty": 50}
                save_brain(brain)
                speak_aether("Downward Thrust Detected. Selling.")
                add_log(f"SELL | P={prob:.2f} T={thrust:.2f}", "warn")
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
                speak_aether(f"Trade Closed. Result: {res}.")
                add_log(f"EXIT. P&L: {pnl}. New LR: {new_lr:.2f}", "brain")
                st.rerun()

        # 4. TELEGRAM AUTO-REPORT (2 Mins)
        if time.time() - st.session_state.last_tg_time > TELEGRAM_INTERVAL:
            report = f"⏰ {datetime.now().strftime('%H:%M')}\n💰 NIFTY: {price}\n🚀 THRUST: {thrust:.2f}\n🎲 PROB: {prob*100:.0f}%\n💵 P&L: {brain['pnl']:.2f}"
            send_telegram_report(report)
            st.session_state.last_tg_time = time.time()
            add_log("TELEGRAM REPORT SENT", "info")

        # 5. UI UPDATE
        p_ph.metric("NIFTY 50", f"{price:,.2f}")
        v_ph.metric("VELOCITY", f"{v:.2f}")
        t_ph.metric("THRUST", f"{thrust:.2f}")
        m_ph.metric("PROBABILITY", f"{prob*100:.0f}%")
        e_ph.metric("CHAOS", f"{entropy:.2f}")
        
        # P&L
        total = brain['pnl'] + (pnl if brain['position'] else 0)
        col = "#00ff41" if total >= 0 else "#ff0000"
        pnl_ph.markdown(f"<h1 style='color:{col}; text-align:center;'>₹{total:.2f}</h1>", unsafe_allow_html=True)
        
        # Logs
        log_html = "".join([f"<div>{l}</div>" for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00e5ff', width=2), fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)'))
        if brain['position']:
            fig.add_hline(y=brain['position']['entry'], line_dash="dash", line_color="orange")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
