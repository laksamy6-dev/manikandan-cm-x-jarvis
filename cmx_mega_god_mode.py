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

# ==========================================
# 1. SYSTEM CONFIGURATION (DAYLIGHT HUD)
# ==========================================
st.set_page_config(
    page_title="CM-X MEGA GOD MODE",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. SECRETS & API SETUP
# ==========================================
try:
    # Secrets Loading (Make sure these are in .streamlit/secrets.toml)
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    # Telegram Check (Optional)
    if "telegram" in st.secrets:
        TG_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else:
        TG_BOT_TOKEN = None
        TG_CHAT_ID = None

    # AI Config
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # Fast Model
except:
    # Error handling suppressed for UI flow (Will use simulation if keys fail)
    pass

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
MEMORY_FILE = "cm_x_mega_memory.json"

# ==========================================
# 3. BLACK BOX MEMORY (THE SOUL)
# ==========================================
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "total_pnl": 0.0,
            "trade_log": [],
            # சுய கற்றல் எடைகள் (Self-Learning Weights)
            "weights": {"Physics": 1.5, "Trend": 1.0, "Global": 1.0, "Chaos": 0.8},
            "global_sentiment": "NEUTRAL",
            "market_knowledge": "System Initialized."
        }
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return init_memory()

def save_memory(mem):
    with open(MEMORY_FILE, 'w') as f: json.dump(mem, f, indent=4)

brain_memory = init_memory()

# ==========================================
# 4. STATE MANAGEMENT
# ==========================================
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=300)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = 0.0

# ==========================================
# 5. AUDIO & LOGS SYSTEM
# ==========================================
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
    color = "#000" if type=="info" else "#cc0000" if type=="danger" else "#ff9900"
    st.session_state.live_logs.appendleft(f"<span style='color:#555'>[{ts}]</span> <span style='color:{color}; font-weight:bold'>{msg}</span>")

def send_telegram(msg):
    if TG_BOT_TOKEN and TG_CHAT_ID:
        try: requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", params={"chat_id": TG_CHAT_ID, "text": f"🦅 CM-X: {msg}"})
        except: pass

# ==========================================
# 6. UI STYLING (DAYLIGHT MILITARY THEME)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* Global Theme */
    .stApp { background-color: #e6e6e6; color: #000; font-family: 'Rajdhani', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #000; font-family: 'Orbitron', sans-serif; text-transform: uppercase; border-bottom: 2px solid #000; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #000;
        box-shadow: 4px 4px 0px #000;
        border-radius: 5px;
        color: #000;
        padding: 10px;
    }
    div[data-testid="stMetricValue"] { color: #000; font-family: 'Orbitron'; font-size: 30px; font-weight: 900; }

    /* The Council Agent Cards */
    .agent-card {
        background: #fff; border: 2px solid #000; padding: 10px; 
        text-align: center; border-radius: 8px; font-weight: 900;
        font-family: 'Orbitron'; font-size: 14px;
        box-shadow: 3px 3px 0px #888; margin-bottom: 5px;
    }
    .agent-buy { border-color: #00aa00; color: #fff; background: #008800; }
    .agent-sell { border-color: #cc0000; color: #fff; background: #cc0000; }
    .agent-wait { border-color: #555; color: #555; background: #ddd; }

    /* Approval Box (Flashing) */
    .approval-box {
        background-color: #ffcc00; border: 4px solid #000; 
        color: #000; padding: 20px; text-align: center; 
        border-radius: 10px; animation: pulse 1s infinite;
        font-family: 'Orbitron'; font-weight: 900; font-size: 24px;
    }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }

    /* Buttons */
    .stButton>button {
        width: 100%; font-family: 'Orbitron'; font-weight: 900; border-radius: 0px; height: 60px;
        border: 3px solid #000; color: #000; background: #fff; box-shadow: 5px 5px 0px #000;
    }
    .stButton>button:hover { background: #000; color: #fff; top: 2px; position: relative; box-shadow: 2px 2px 0px #000; }
    
    /* Terminal Logs */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #000;
        color: #00ff41; 
        border: 4px solid #333;
        padding: 10px;
        height: 200px;
        overflow-y: auto;
        font-size: 14px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 7. THE BRAIN LOGIC (PHYSICS + ZERO LOSS)
# ==========================================
class MegaBrain:
    
    def calculate_physics(self, prices):
        # Convert Deque to List to avoid Error
        p = np.array(list(prices))
        if len(p) < 10: return 0, 0, 0
        v = np.diff(p)[-1]
        a = np.diff(np.diff(p))[-1]
        
        # Entropy Calculation (Safe Method)
        try:
            hist, _ = np.histogram(p[-20:], bins=10, density=True)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy_val = -np.sum(probs * np.log(probs))
        except: entropy_val = 0
        
        return v, a, entropy_val

    def monte_carlo_forecast(self, prices):
        data = list(prices)
        last = data[-1]
        vol = np.std(data[-20:]) if len(data)>20 else 5
        # 3 Candle Prediction
        return [last + np.random.normal(0, vol), last - np.random.normal(0, vol)]

    def get_auto_strike(self, spot_price, direction):
        # ATM Strike Selection
        strike = round(spot_price / 50) * 50
        if direction == "BUY": return f"{strike} CE"
        else: return f"{strike} PE"

    def update_learning(self, result):
        # Brain Growth Logic
        w = brain_memory["weights"]
        factor = 0.05
        if result == "WIN":
            w["Physics"] += factor
            w["Trend"] += factor
            brain_memory["market_knowledge"] += " | Win recorded."
        else:
            w["Physics"] -= factor
            w["Trend"] -= factor
            brain_memory["market_knowledge"] += " | Loss recorded. Adjusting."
        save_memory(brain_memory)

brain = MegaBrain()

def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: 
        # Simulation Fallback
        if st.session_state.prices: return st.session_state.prices[-1] + np.random.normal(0, 3)
        return 22100.00
    try:
        headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=2)
        if res.status_code == 200:
            return float(res.json()['data'][list(res.json()['data'].keys())[0]]['last_price'])
    except: pass
    if st.session_state.prices: return st.session_state.prices[-1]
    return 22100.00

# ==========================================
# 8. MAIN UI LAYOUT
# ==========================================
st.markdown(f"""
<div style="border-bottom: 4px solid #000; padding-bottom: 10px; margin-bottom: 20px;">
    <h1 style="margin:0; font-size: 40px;">CM-X <span style="color:#555">MEGA AUTOMATON</span></h1>
    <div style="font-weight:bold; letter-spacing:1px;">OPERATOR: {st.secrets['general']['owner'] if 'general' in st.secrets else 'BOSS MANIKANDAN'}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# GRID SYSTEM
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

    # THE COUNCIL CHAMBER
    st.markdown("### 🏛️ THE COUNCIL (DECISION CORE)")
    council_ph = st.empty()
    
    # TERMINAL LOGS
    st.markdown("### 🖥️ BLACK BOX LOGS")
    log_ph = st.empty()

with c2:
    st.markdown("### 🌍 GLOBAL & CONTROLS")
    
    # GLOBAL SETTING
    g_sent = st.select_slider("Global Sentiment", ["BEARISH", "NEUTRAL", "BULLISH"], value=brain_memory["global_sentiment"])
    if g_sent != brain_memory["global_sentiment"]:
        brain_memory["global_sentiment"] = g_sent
        save_memory(brain_memory)

    # GEMINI CHAT INPUT
    st.markdown("### 💬 TALK TO BRAIN")
    user_input = st.text_input("Command:", placeholder="Type 'Status' or Ask Question...")
    if st.button("SEND COMMAND"):
        if user_input:
            speak_jarvis("Processing command.")
            add_log(f"BOSS: {user_input}", "warn")
            # Gemini Reply
            try:
                ctx = f"You are CM-X Trading Bot. PnL: {brain_memory['total_pnl']}. User says: {user_input}"
                reply = model.generate_content(ctx).text
                speak_jarvis(reply)
                add_log(f"AI: {reply}", "info")
            except: pass

    st.write("---")
    # APPROVAL AREA (Important)
    approval_ph = st.empty()
    st.write("---")
    
    # BUTTONS
    b1, b2 = st.columns(2)
    start = b1.button("🔥 START ENGINE")
    stop = b2.button("🛑 STOP ENGINE")
    
    if st.button("❌ EMERGENCY EXIT"):
        st.session_state.position = None
        speak_jarvis("Emergency Exit Triggered.")
        st.rerun()
        
    pnl_ph = st.empty()

if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

# ==========================================
# 9. MAIN EXECUTION LOOP
# ==========================================
if st.session_state.bot_active:
    
    # 1. Fetch Data
    price = get_live_data()
    st.session_state.prices.append(price)
    
    # 2. Physics & Brain Logic
    v, a, ent = brain.calculate_physics(st.session_state.prices)
    future_targets = brain.monte_carlo_forecast(st.session_state.prices)
    
    # 3. Council Voting
    votes = {}
    weights = brain_memory["weights"]
    
    # Physics Agent
    if v > 1.5 and a > 0.3: votes['Physics'] = "BUY"
    elif v < -1.5 and a < -0.3: votes['Physics'] = "SELL"
    else: votes['Physics'] = "WAIT"
    
    # Trend Agent (Moving Average)
    ma = np.mean(list(st.session_state.prices)[-20:]) if len(st.session_state.prices)>20 else price
    if price > ma: votes['Trend'] = "BUY"
    else: votes['Trend'] = "SELL"
    
    # Global Agent
    if brain_memory["global_sentiment"] == "BULLISH": votes['Global'] = "BUY"
    elif brain_memory["global_sentiment"] == "BEARISH": votes['Global'] = "SELL"
    else: votes['Global'] = "WAIT"
    
    # Chaos Agent (Filter)
    votes['Chaos'] = "GO" if ent < 1.5 else "NO_TRADE"
    
    # 4. Weighted Scoring
    buy_score = 0
    sell_score = 0
    
    if votes['Physics'] == "BUY": buy_score += weights['Physics']
    if votes['Trend'] == "BUY": buy_score += weights['Trend']
    if votes['Global'] == "BUY": buy_score += weights['Global']
    
    if votes['Physics'] == "SELL": sell_score += weights['Physics']
    if votes['Trend'] == "SELL": sell_score += weights['Trend']
    if votes['Global'] == "SELL": sell_score += weights['Global']
    
    threshold = 2.0
    
    # 5. Signal Generation (Only if No Position)
    if buy_score > threshold and votes['Chaos'] == "GO" and not st.session_state.position and not st.session_state.pending_signal:
        opt = brain.get_auto_strike(price, "BUY")
        st.session_state.pending_signal = {"type": "BUY", "opt": opt}
        speak_jarvis(f"Boss! Buy Signal on {opt}. Approve?")
        send_telegram(f"BUY ALERT: {opt}")
        
    elif sell_score > threshold and votes['Chaos'] == "GO" and not st.session_state.position and not st.session_state.pending_signal:
        opt = brain.get_auto_strike(price, "SELL")
        st.session_state.pending_signal = {"type": "SELL", "opt": opt}
        speak_jarvis(f"Boss! Sell Signal on {opt}. Approve?")
        send_telegram(f"SELL ALERT: {opt}")

    # 6. APPROVAL POPUP (Operator Mode)
    if st.session_state.pending_signal:
        sig = st.session_state.pending_signal
        with approval_ph.container():
            st.markdown(f"<div class='approval-box'>⚠️ AUTHORIZE: {sig['opt']}</div>", unsafe_allow_html=True)
            c_y, c_n = st.columns(2)
            if c_y.button("✅ EXECUTE", key="ex"):
                st.session_state.position = {"type": sig['type'], "entry": price, "opt": sig['opt']}
                st.session_state.trailing_high = 0.0 # Reset Trailing
                st.session_state.pending_signal = None
                add_log(f"ORDER PLACED: {sig['opt']} @ {price}", "warn")
                speak_jarvis("Order Executed.")
                send_telegram("ORDER EXECUTED")
                st.rerun()
            if c_n.button("❌ ABORT", key="ab"):
                st.session_state.pending_signal = None
                st.rerun()

    # 7. ZERO LOSS ENGINE (The Protection)
    if st.session_state.position:
        pos = st.session_state.position
        # Simulated PnL (50 Qty)
        current_pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        
        # Track Highest PnL
        if current_pnl > st.session_state.trailing_high:
            st.session_state.trailing_high = current_pnl
            
        high = st.session_state.trailing_high
        exit = False
        reason = ""
        
        # --- LOGIC ---
        # A. Zero Loss Trigger (Profit > 500 -> Move SL to +200)
        if high > 500 and current_pnl < 200:
            exit = True
            reason = "ZERO LOSS HIT (+200 Brokerage Covered)"
            
        # B. Trailing Profit (Lock 80% if profit > 1000)
        if high > 1000 and current_pnl < (high * 0.8):
            exit = True
            reason = f"TRAILING PROFIT BOOKED"
            
        # C. Hard Stop Loss (Initial Safety)
        if current_pnl < -300:
            exit = True
            reason = "HARD STOP LOSS HIT"
            
        if exit:
            brain.update_learning("WIN" if current_pnl > 0 else "LOSS")
            brain_memory["total_pnl"] += current_pnl
            save_memory(brain_memory)
            st.session_state.position = None
            speak_jarvis(f"Trade Closed. {reason}")
            add_log(f"EXIT: {reason} | PNL: {current_pnl}", "danger")
            send_telegram(f"EXIT: {current_pnl}")
            st.rerun()

    # 8. UPDATE VISUALS
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        def styler(v): return "agent-buy" if v=="BUY" else "agent-sell" if v=="SELL" else "agent-wait"
        
        cc1.markdown(f"<div class='agent-card {styler(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {styler(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card {styler(votes['Global'])}'>GLOBAL<br>{votes['Global']}</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card agent-wait'>CHAOS<br>{ent:.2f}</div>", unsafe_allow_html=True)

    # Metrics
    price_ph.metric("NIFTY 50", f"{price:,.2f}")
    vel_ph.metric("VELOCITY", f"{v:.2f}")
    acc_ph.metric("ACCEL", f"{a:.2f}")
    chaos_ph.metric("CHAOS", f"{ent:.2f}")
    
    # PnL Big Display
    val = brain_memory["total_pnl"]
    pnl_ph.markdown(f"<div style='background:#fff; border:4px solid #000; padding:10px; text-align:center;'><h1 style='color:{'green' if val>=0 else 'red'}; margin:0;'>₹{val:,.2f}</h1></div>", unsafe_allow_html=True)

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='black', width=3), name='Price'))
    # Future Prediction Dots
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+5], 
                             y=[price, future_targets[0]], line=dict(color='green', dash='dot'), name='Bull Path'))
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+5], 
                             y=[price, future_targets[1]], line=dict(color='red', dash='dot'), name='Bear Path'))
    
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)

    # Logs
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    
    # STABLE REFRESH (3 Seconds) - No Shaking!
    time.sleep(3)
    if not st.session_state.pending_signal: st.rerun()
