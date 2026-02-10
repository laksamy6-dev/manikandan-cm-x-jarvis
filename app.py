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
MAX_HISTORY_LEN = 126 
TELEGRAM_INTERVAL = 120 # 2 Minutes
KILL_SWITCH_LOSS = -2000 

# --- 2. PROFESSIONAL SYSTEM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #f0f4f8; color: #102a43; font-family: 'Rajdhani', sans-serif; }
    h1, h2, h3 { color: #003e6b; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 800; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #0077cc; font-weight: bold; }
    
    /* Terminal Box (CSV Style Log) */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #102a43;
        color: #00ff41;
        padding: 15px;
        height: 350px;
        overflow-y: auto;
        font-size: 14px;
        border-radius: 8px;
        border: 2px solid #334e68;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    .log-row { border-bottom: 1px solid #334e68; padding: 2px 0; }
    .log-time { color: #829ab1; margin-right: 10px; font-weight:bold; }
    .log-msg { color: #40c3ff; }
    .log-pnl-win { color: #00e676; font-weight: bold; }
    .log-pnl-loss { color: #ff1744; font-weight: bold; }
    
    /* Buttons & Active Trade */
    .stButton>button {
        background: linear-gradient(135deg, #0077cc, #005299);
        color: white; border: none; border-radius: 8px; font-weight: bold; height: 50px;
    }
    .active-trade-box {
        background-color: #fff1f0; border: 2px solid #e12d39; color: #cf1124;
        padding: 15px; text-align: center; border-radius: 8px; font-weight: bold; font-size: 18px;
        margin-bottom: 20px; animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(225, 45, 57, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(225, 45, 57, 0); } 100% { box-shadow: 0 0 0 0 rgba(225, 45, 57, 0); } }
    
    .status-online { background-color: #e3f9e5; color: #137333; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid #137333; }
    .status-offline { background-color: #fce8e6; color: #c5221f; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid #c5221f; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIG ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
except: st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. MEMORY ---
def init_brain():
    if not os.path.exists(MEMORY_FILE):
        data = {"position": None, "orders": [], "pnl": 0.0, "learning_rate": 1.0, "win_streak": 0, "loss_streak": 0}
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"position": None, "orders": [], "pnl": 0.0, "learning_rate": 1.0}

def save_brain(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

brain = init_brain()

if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = time.time()
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=50) # The CSV style logs
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""

# Restore
if 'position' not in st.session_state: st.session_state.position = brain.get('position', None)
if 'pnl' not in st.session_state: st.session_state.pnl = brain.get('pnl', 0.0)

# --- 5. LOGGING (CSV STYLE) ---
def add_log(msg):
    ts = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    # Formatting based on msg content for color
    css_class = "log-msg"
    if "WIN" in msg or "PROFIT" in msg or "BUY DETECTED" in msg: css_class = "log-pnl-win"
    elif "LOSS" in msg or "SELL DETECTED" in msg: css_class = "log-pnl-loss"
    
    entry = f"<div class='log-row'><span class='log-time'>{ts}</span> <span class='{css_class}'>{msg}</span></div>"
    st.session_state.live_logs.appendleft(entry)

def send_telegram_report(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧬 AETHER REPORT:\n{msg}"}
        requests.get(url, params=params)
    except: pass

# --- 6. AUDIO ---
def speak_aether(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "ghost.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

# --- 7. MATH CORE ---
def rocket_formula(v, vol_current, vol_avg):
    if vol_avg == 0: vol_avg = 1
    mass_ratio = abs(vol_current / vol_avg)
    if mass_ratio <= 0: mass_ratio = 0.1
    thrust = v * math.log(mass_ratio + 1)
    return thrust

def monte_carlo_simulation(prices, num_sims=100, steps=10):
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

def calculate_metrics(prices):
    p = np.array(list(prices))
    if len(p) < 10: return 0,0,0,0,0
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    entropy = np.std(p[-10:])
    thrust = rocket_formula(v, entropy*1.5, entropy if entropy > 0 else 1) 
    prob = monte_carlo_simulation(prices)
    return v, a, entropy, thrust, prob

# --- 8. AI ---
def consult_ghost(price, v, a, t, p):
    try:
        prompt = f"Market: {price}, Thrust: {t:.2f}, WinProb: {p:.2f}. One line sci-fi trading advice?"
        res = model.generate_content(prompt)
        return res.text
    except: return "AI Recalibrating..."

# --- 9. DATA ---
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

# --- 10. LEARNING ---
def update_learning_rate(result):
    lr = brain.get("learning_rate", 1.0)
    if result == "WIN": lr = min(1.5, lr + 0.1)
    else: lr = max(0.5, lr - 0.2)
    brain["learning_rate"] = lr
    save_brain(brain)
    return lr

# --- 11. UI ---
st.markdown(f"""
<div style="text-align:center; padding-bottom: 10px; border-bottom: 2px solid #cbd5e1;">
    <h1 style="margin-bottom: 5px;">AETHER: FUSION GOD MODE</h1>
    <p style="color:#486581; font-weight:bold;">OPERATOR: {OWNER_NAME} | BRAIN: ACTIVE | LR: {brain.get('learning_rate', 1.0):.2f}</p>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

price_check = get_live_data()
if price_check: st.markdown('<div class="status-online">🟢 UPSTOX CONNECTED | DATA FLOWING</div>', unsafe_allow_html=True)
else: st.markdown('<div class="status-offline">🔴 CONNECTION LOST | CHECK TOKEN</div>', unsafe_allow_html=True)

if st.session_state.position:
    pos = st.session_state.position
    st.markdown(f"""<div class="active-trade-box">🔥 ACTIVE: {pos['type']} @ {pos['entry']} | QTY: {pos['qty']}</div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
p_ph = c1.empty()
v_ph = c2.empty()
t_ph = c3.empty()
m_ph = c4.empty()
e_ph = c5.empty()

g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Quantum Trajectory")
    chart_ph = st.empty()
    st.subheader("🖥️ Neural Core Logs")
    log_ph = st.empty()

with g2:
    st.subheader("🧠 Control Deck")
    if st.button("🔊 CONSULT AETHER"):
        if st.session_state.prices:
            p_curr = list(st.session_state.prices)[-1]
            speak_aether(f"Market at {p_curr}. Analyzing fields.")
    st.write("---")
    pnl_ph = st.empty() 
    st.write("---")
    c_start, c_stop = st.columns(2)
    if c_start.button("▶️ ACTIVATE"):
        st.session_state.bot_active = True
        add_log("SYSTEM ACTIVATED")
    if c_stop.button("⏹️ DEACTIVATE"):
        st.session_state.bot_active = False
        add_log("SYSTEM STOPPED")
        
    st.caption("Status:")
    if brain['position']: st.info(f"HOLDING: {brain['position']['type']}")
    else: st.success("SCANNING...")

# --- 12. LOOP ---
if st.session_state.bot_active:
    if not price_check: st.stop()
    while st.session_state.bot_active:
        price = get_live_data()
        if not price:
            time.sleep(1)
            continue
        
        st.session_state.prices.append(price)
        v, a, entropy, thrust, prob = calculate_metrics(st.session_state.prices)
        lr = brain.get("learning_rate", 1.0)
        
        # BUY
        if (prob > 0.6) and (thrust > 0.5) and (v > 1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "BUY", "entry": price, "qty": 50}
                save_brain(brain)
                st.session_state.position = brain['position']
                speak_aether("Thrust Detected. Buying Call.")
                add_log(f"BUY DETECTED @ {price}")
                st.rerun()
        
        # SELL
        elif (prob < 0.4) and (thrust < -0.5) and (v < -1.0) and (entropy < 10):
            if not brain['position']:
                brain['position'] = {"type": "SELL", "entry": price, "qty": 50}
                save_brain(brain)
                st.session_state.position = brain['position']
                speak_aether("Negative Thrust. Selling Put.")
                add_log(f"SELL DETECTED @ {price}")
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
                
                # Update Session
                st.session_state.position = None
                st.session_state.pnl = brain['pnl']
                
                save_brain(brain)
                speak_aether(f"Position Closed. {res}.")
                add_log(f"CLOSED {pos['type']} | P&L: {pnl:.0f}")
                st.rerun()

        # TELEGRAM
        if time.time() - st.session_state.last_tg_time > TELEGRAM_INTERVAL:
            report = f"⏰ {datetime.now().strftime('%H:%M')}\n💰 NIFTY: {price}\n🚀 THRUST: {thrust:.2f}\n💵 P&L: {brain['pnl']:.2f}"
            send_telegram_report(report)
            st.session_state.last_tg_time = time.time()
            add_log("TELEGRAM REPORT SENT")

        # UI UPDATE
        p_ph.metric("NIFTY 50", f"{price:,.2f}")
        v_ph.metric("VELOCITY", f"{v:.2f}")
        t_ph.metric("THRUST", f"{thrust:.2f}")
        m_ph.metric("WIN PROB", f"{prob*100:.0f}%")
        e_ph.metric("CHAOS", f"{entropy:.2f}")
        
        total = brain['pnl'] + (pnl if brain['position'] else 0)
        col = "#16a34a" if total >= 0 else "#dc2626" 
        pnl_ph.markdown(f"<h1 style='color:{col}; text-align:center;'>₹{total:.2f}</h1>", unsafe_allow_html=True)
        
        # LOGS
        log_html = "".join([l for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
        
        # CHART
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#0077cc', width=2), fill='tozeroy', fillcolor='rgba(0, 119, 204, 0.1)'))
        if brain['position']:
            fig.add_hline(y=brain['position']['entry'], line_dash="dash", line_color="#ff9900")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
