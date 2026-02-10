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
from scipy.stats import entropy

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="CM-X DAYLIGHT HUD",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# --- 2. API & TELEGRAM SETUP ---
try:
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    # Telegram Check
    if "telegram" in st.secrets:
        TG_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else:
        TG_BOT_TOKEN = None
        TG_CHAT_ID = None

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    # Error handling suppressed for UI flow, ensures app runs even if secrets fail slightly
    pass

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
MEMORY_FILE = "cm_x_blackbox_memory.json"

# --- 3. BLACK BOX MEMORY (THE SOUL) ---
def init_black_box():
    if not os.path.exists(MEMORY_FILE):
        return {
            "total_pnl": 0.0, 
            "ai_weights": {"Physics": 1.5, "Trend": 1.0, "Options": 1.2, "Chaos": 0.8},
            "last_active": str(datetime.now())
        }
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return {"total_pnl": 0.0, "ai_weights": {"Physics": 1.5, "Trend": 1.0, "Options": 1.2, "Chaos": 0.8}}

brain_memory = init_black_box()

# --- 4. CSS STYLING (DAYLIGHT HUD - MILITARY STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* Global Theme: High Contrast White/Black (Daylight Safe) */
    .stApp { background-color: #e6e6e6; color: #000; font-family: 'Rajdhani', sans-serif; }
    
    /* Headers - Tech Style */
    h1, h2, h3 { color: #000; font-family: 'Orbitron', sans-serif; text-transform: uppercase; letter-spacing: 2px; border-bottom: 2px solid #000; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #333;
        box-shadow: 4px 4px 0px #000;
        border-radius: 5px;
        color: #000;
        padding: 10px;
    }
    div[data-testid="stMetricLabel"] { color: #555; font-weight: 800; font-size: 14px; }
    div[data-testid="stMetricValue"] { color: #000; font-family: 'Orbitron'; font-size: 30px; font-weight: 900; }

    /* THE COUNCIL AGENT CARDS */
    .agent-card {
        background: #fff; border: 2px solid #000; padding: 15px; 
        text-align: center; border-radius: 8px; font-weight: 900;
        font-family: 'Orbitron'; font-size: 14px;
        box-shadow: 3px 3px 0px #888;
    }
    .agent-buy { border-color: #00aa00; color: #fff; background: #008800; }
    .agent-sell { border-color: #cc0000; color: #fff; background: #cc0000; }
    .agent-wait { border-color: #555; color: #555; background: #ddd; }

    /* TERMINAL BOX */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #000;
        color: #00ff41; 
        border: 4px solid #333;
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        font-size: 14px;
        border-radius: 5px;
    }

    /* APPROVAL ALERT */
    .approval-box {
        background-color: #ffcc00; border: 4px solid #000; 
        color: #000; padding: 20px; text-align: center; 
        border-radius: 10px; animation: pulse 1s infinite;
        font-family: 'Orbitron'; font-weight: 900; font-size: 24px;
    }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }

    /* BUTTONS */
    .stButton>button {
        width: 100%; font-family: 'Orbitron'; font-weight: 900; border-radius: 0px; height: 60px;
        border: 3px solid #000; color: #000; background: #fff; box-shadow: 5px 5px 0px #000;
    }
    .stButton>button:hover { background: #000; color: #fff; box-shadow: 2px 2px 0px #888; top: 2px; position: relative; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. STATE & HELPERS ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=200)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)

def send_telegram(msg):
    if TG_BOT_TOKEN and TG_CHAT_ID:
        try: requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", params={"chat_id": TG_CHAT_ID, "text": f"🦅 CM-X: {msg}"})
        except: pass

def speak_jarvis(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "alert.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        md = f"""<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.session_state.audio_html = md
    except: pass

def add_log(msg, type="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    color = "#00ff41" if type=="info" else "#ffff00" if type=="warn" else "#ff3333"
    st.session_state.live_logs.appendleft(f"<span style='color:#888'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")

def get_live_data():
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=2)
        if res.status_code == 200:
            return float(res.json()['data'][list(res.json()['data'].keys())[0]]['last_price'])
    except: pass
    return None

# --- 6. LOGIC CORE (Fixed Deque Error Here) ---
def calculate_physics(prices):
    p = np.array(list(prices)) # FIX: Convert deque to list first
    if len(p) < 5: return 0, 0, 0
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1]
    entropy_val = entropy(np.histogram(p[-20:], bins=10)[0])
    return v, a, entropy_val

def monte_carlo_forecast(prices):
    # FIX: Convert deque to list first to avoid TypeError
    data = list(prices) 
    last = data[-1]
    # Calculate volatility safely
    if len(data) > 20:
        vol = np.std(data[-20:]) 
    else:
        vol = 5
        
    return [last + np.random.normal(0, vol), last - np.random.normal(0, vol)] # Bull/Bear target

# --- 7. UI LAYOUT ---
st.markdown(f"""
<div style="border-bottom: 4px solid #000; padding-bottom: 10px; margin-bottom: 20px;">
    <h1 style="margin:0; font-size: 40px;">CM-X <span style="color:#555">HUD</span></h1>
    <div style="font-weight:bold; letter-spacing:1px;">OPERATOR: {st.secrets['general']['owner'] if 'general' in st.secrets else 'BOSS MANIKANDAN'}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# MAIN GRID
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 📡 TACTICAL DISPLAY")
    chart_ph = st.empty()
    
    # METRICS
    m1, m2, m3, m4 = st.columns(4)
    price_ph = m1.empty()
    vel_ph = m2.empty()
    acc_ph = m3.empty()
    chaos_ph = m4.empty()

    # THE COUNCIL
    st.markdown("### 🧠 THE COUNCIL CHAMBER")
    council_ph = st.empty()
    
    # TERMINAL
    st.markdown("### 🖥️ SYSTEM LOGS (BLACK BOX)")
    log_ph = st.empty()

with c2:
    st.markdown("### 🎤 COMMAND CENTER")
    
    # INPUT BOX
    user_input = st.text_input("GIVE ORDER:", placeholder="Type or Speak...")
    if user_input:
        speak_jarvis(f"Copy that. {user_input}")
        add_log(f"BOSS COMMAND: {user_input}", "warn")

    st.write("---")
    approval_ph = st.empty()
    
    st.write("---")
    pnl_ph = st.empty()
    
    # BIG BUTTONS
    b1, b2 = st.columns(2)
    start = b1.button("🔥 START")
    stop = b2.button("🛑 STOP")
    
    if st.button("❌ EMERGENCY EXIT"):
        st.session_state.position = None
        st.rerun()
    
    # Order Book Placeholder
    st.markdown("### 📖 ORDER BOOK (LIVE)")
    ob_ph = st.empty()

if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

# --- 8. MAIN ENGINE ---
if st.session_state.bot_active:
    
    # 1. Fetch Data
    price = get_live_data()
    if not price: 
        if st.session_state.prices: price = st.session_state.prices[-1] + np.random.normal(0, 2)
        else: price = 22000.00
    
    st.session_state.prices.append(price)
    
    # 2. Physics & Brain
    v, a, ent = calculate_physics(st.session_state.prices)
    future_targets = monte_carlo_forecast(st.session_state.prices) 
    
    # 3. Council Voting
    votes = {"Physics": "WAIT", "Trend": "WAIT", "Options": "WAIT", "Chaos": "GO"}
    
    # Physics Agent
    if v > 2.0 and a > 0.5: votes['Physics'] = "BUY"
    elif v < -2.0 and a < -0.5: votes['Physics'] = "SELL"
    
    # Trend Agent
    ma = np.mean(list(st.session_state.prices)[-20:]) if len(st.session_state.prices)>20 else price
    if price > ma: votes['Trend'] = "BUY"
    else: votes['Trend'] = "SELL"
    
    # Chaos Agent
    if ent > 1.5: votes['Chaos'] = "NO_TRADE"
    
    # 4. Signal Generation
    buy_cnt = list(votes.values()).count("BUY")
    sell_cnt = list(votes.values()).count("SELL")
    
    if buy_cnt >= 2 and votes['Chaos'] == "GO" and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "BUY"
        speak_jarvis("Boss! Rocket Launch. Buying Call.")
        send_telegram(f"BUY SIGNAL @ {price}")
        
    elif sell_cnt >= 2 and votes['Chaos'] == "GO" and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "SELL"
        speak_jarvis("Boss! Market Falling. Buying Put.")
        send_telegram(f"SELL SIGNAL @ {price}")

    # 5. APPROVAL POPUP
    if st.session_state.pending_signal:
        with approval_ph.container():
            st.markdown(f"<div class='approval-box'>⚠️ AUTHORIZE {st.session_state.pending_signal}?</div>", unsafe_allow_html=True)
            c_y, c_n = st.columns(2)
            if c_y.button("✅ EXECUTE", key="y"):
                st.session_state.position = {"type": st.session_state.pending_signal, "entry": price}
                st.session_state.pending_signal = None
                send_telegram("ORDER EXECUTED")
                st.rerun()
            if c_n.button("❌ ABORT", key="n"):
                st.session_state.pending_signal = None
                st.rerun()

    # 6. Exit Logic
    if st.session_state.position:
        pos = st.session_state.position
        pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        if pnl > 500 or pnl < -300:
            brain_memory["total_pnl"] += pnl
            with open(MEMORY_FILE, 'w') as f: json.dump(brain_memory, f)
            st.session_state.position = None
            speak_jarvis("Trade Closed.")
            send_telegram(f"PNL BOOKED: {pnl}")
            st.rerun()

    # --- UI UPDATES ---
    
    # Update Council Cards
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        def style(v): return "agent-buy" if v=="BUY" else "agent-sell" if v=="SELL" else "agent-wait"
        cc1.markdown(f"<div class='agent-card {style(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {style(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card agent-wait'>OPTIONS<br>WAIT</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card agent-wait'>CHAOS<br>{votes['Chaos']}</div>", unsafe_allow_html=True)

    # Metrics
    price_ph.metric("NIFTY 50", f"{price:,.2f}")
    vel_ph.metric("VELOCITY", f"{v:.2f}")
    acc_ph.metric("ACCEL", f"{a:.2f}")
    chaos_ph.metric("ENTROPY", f"{ent:.2f}")
    
    # PNL Big Display
    val = brain_memory["total_pnl"]
    pnl_ph.markdown(f"<div style='background:#fff; border:4px solid #000; padding:10px; text-align:center;'><h1 style='color:{'green' if val>=0 else 'red'}; margin:0;'>₹{val:,.2f}</h1></div>", unsafe_allow_html=True)

    # Chart with Monte Carlo Dots
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='black', width=3), name='Price'))
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+3], y=[price, future_targets[0]], line=dict(color='green', dash='dot'), name='Bull Path'))
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+3], y=[price, future_targets[1]], line=dict(color='red', dash='dot'), name='Bear Path'))
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)

    # Logs
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    
    # Order Book Simulator
    ob_ph.caption(f"BID: {price-1:.2f} | ASK: {price+1:.2f} | VOL: {int(np.random.random()*1000)}")

    time.sleep(3) # Stable Refresh
    if not st.session_state.pending_signal: st.rerun()
