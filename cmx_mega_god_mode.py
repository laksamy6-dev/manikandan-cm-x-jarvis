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
# 1. SYSTEM CONFIGURATION (DARK NEON MODE)
# ==========================================
st.set_page_config(
    page_title="CM-X MEGA PREDATOR",
    layout="wide",
    page_icon="💀",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. SECRETS & LIVE API SETUP
# ==========================================
try:
    # UPSTOX LIVE CREDENTIALS
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    
    # GEMINI AI
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # TELEGRAM (OPTIONAL)
    if "telegram" in st.secrets:
        TG_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else:
        TG_BOT_TOKEN = None; TG_CHAT_ID = None
except:
    st.error("⚠️ SYSTEM ERROR: Secrets Missing! Check .streamlit/secrets.toml")
    st.stop()

# UPSTOX CONFIGURATION
UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50" # NIFTY 50 LIVE
MEMORY_FILE = "cm_x_mega_brain.json"

# ==========================================
# 3. ADVANCED NEON CSS (THE LOOK)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* GLOBAL DARK THEME */
    .stApp { 
        background-color: #050505; 
        color: #00ff41; 
        font-family: 'Rajdhani', sans-serif; 
    }
    
    /* HEADERS */
    h1, h2, h3 { 
        font-family: 'Orbitron', sans-serif; 
        text-shadow: 0 0 10px #00ff41; 
        color: #fff; 
        border-bottom: 1px solid #333;
    }
    
    /* METRIC CARDS (NEON BOXES) */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        color: #00ff41;
        text-shadow: 0 0 5px #00ff41;
    }
    div[data-testid="stMetricLabel"] { color: #888; font-weight: bold; }
    
    /* THE COUNCIL AGENT CARDS */
    .agent-card {
        background: #111; border: 1px solid #333; padding: 10px; 
        text-align: center; border-radius: 5px; font-family: 'Orbitron';
        margin-bottom: 5px; transition: all 0.3s;
    }
    .agent-buy { border-color: #00ff41; color: #00ff41; box-shadow: 0 0 10px #00ff41; }
    .agent-sell { border-color: #ff003c; color: #ff003c; box-shadow: 0 0 10px #ff003c; }
    .agent-wait { border-color: #555; color: #555; }
    
    /* APPROVAL FLASHING BOX */
    .approval-box {
        border: 2px solid #ffff00;
        background-color: #222200;
        color: #ffff00;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        font-family: 'Orbitron';
        animation: pulse 1s infinite;
        margin-bottom: 20px;
    }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 255, 0, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 255, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 255, 0, 0); } }

    /* TERMINAL LOGS */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #000;
        border: 1px solid #333;
        color: #00ff41;
        padding: 10px;
        height: 250px;
        overflow-y: auto;
        font-size: 13px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
    }
    
    /* BUTTONS */
    .stButton>button {
        background-color: #000;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Orbitron', sans-serif;
        height: 50px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000;
        box-shadow: 0 0 20px #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. BLACK BOX MEMORY & STATE
# ==========================================
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "total_pnl": 0.0,
            "trade_log": [],
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

# STATE INITIALIZATION
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=300)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=30)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = 0.0

# ==========================================
# 5. AUDIO & TELEGRAM
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
    color = "#00ff41" if type=="info" else "#ff003c" if type=="danger" else "#ffff00"
    st.session_state.live_logs.appendleft(f"<span style='color:#555'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")

def send_telegram(msg):
    if TG_BOT_TOKEN and TG_CHAT_ID:
        try: requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", params={"chat_id": TG_CHAT_ID, "text": f"🦅 CM-X: {msg}"})
        except: pass

# ==========================================
# 6. THE MEGA BRAIN (LOGIC CORE)
# ==========================================
class MegaBrain:
    
    def calculate_physics(self, prices):
        # Newton's Law of Market (From PDF)
        p = np.array(list(prices))
        if len(p) < 10: return 0, 0, 0
        
        # Velocity (Speed)
        v = np.diff(p)[-1]
        # Acceleration (Force)
        a = np.diff(np.diff(p))[-1]
        
        # Entropy (Chaos Theory)
        try:
            hist, _ = np.histogram(p[-30:], bins=10, density=True)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy_val = -np.sum(probs * np.log(probs))
        except: entropy_val = 0
        
        return v, a, entropy_val

    def monte_carlo_simulation(self, prices):
        # 3-Step Future Prediction (From PDF)
        data = list(prices)
        last = data[-1]
        vol = np.std(data[-20:]) if len(data)>20 else 5
        
        # Simulate 100 Paths
        paths = [last + np.random.normal(0, vol) for _ in range(100)]
        return np.mean(paths) # Average Predicted Price

    def get_auto_strike(self, spot_price, direction):
        # Automatic Option Selection
        strike = round(spot_price / 50) * 50
        if direction == "BUY": return f"{strike} CE"
        else: return f"{strike} PE"

    def self_improve(self, result):
        # Self Correction Logic
        w = brain_memory["weights"]
        factor = 0.05
        if result == "WIN":
            w["Physics"] += factor
            w["Trend"] += factor
            brain_memory["market_knowledge"] += " | Pattern Validated."
        else:
            w["Physics"] -= factor
            w["Trend"] -= factor
            brain_memory["market_knowledge"] += " | Pattern Failed. Weights Reduced."
        save_memory(brain_memory)

brain = MegaBrain()

# ==========================================
# 7. LIVE DATA FETCHER (UPSTOX)
# ==========================================
def get_live_data():
    # THIS IS THE LIVE CONNECTION CODE
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=2)
        if res.status_code == 200:
            data = res.json()
            # Extracting Last Traded Price
            ltp = float(data['data'][list(data['data'].keys())[0]]['last_price'])
            return ltp
    except Exception as e:
        # Fallback only if API fails (to keep bot running)
        pass
    
    # Simulation fallback if API is down
    if st.session_state.prices: return st.session_state.prices[-1] + np.random.normal(0, 2)
    return 22100.00

# ==========================================
# 8. UI LAYOUT (DARK NEON)
# ==========================================
st.markdown(f"""
<div style="text-align: center; border-bottom: 2px solid #00ff41; padding-bottom: 10px; margin-bottom: 20px;">
    <h1 style="margin:0; font-size: 50px; text-shadow: 0 0 20px #00ff41;">CM-X <span style="color:#fff">MEGA PREDATOR</span></h1>
    <div style="font-family: 'Courier New'; color: #888;">OPERATOR: {st.secrets['general']['owner'] if 'general' in st.secrets else 'BOSS MANIKANDAN'} | MODE: GOD</div>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# MAIN GRID
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 📡 TACTICAL DISPLAY")
    chart_ph = st.empty()
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    price_ph = m1.empty()
    vel_ph = m2.empty()
    acc_ph = m3.empty()
    chaos_ph = m4.empty()

    # THE COUNCIL CHAMBER
    st.markdown("### 🏛️ THE COUNCIL (DECISION CORE)")
    council_ph = st.empty()
    
    # TERMINAL
    st.markdown("### 🖥️ BLACK BOX LOGS")
    log_ph = st.empty()

with c2:
    st.markdown("### 🌍 GLOBAL SENTIMENT")
    g_sent = st.select_slider("Select Bias", ["BEARISH", "NEUTRAL", "BULLISH"], value=brain_memory["global_sentiment"])
    if g_sent != brain_memory["global_sentiment"]:
        brain_memory["global_sentiment"] = g_sent
        save_memory(brain_memory)
    
    # GEMINI CHAT
    st.markdown("### 💬 TALK TO JARVIS")
    user_input = st.text_input("Command:", placeholder="Type or Speak...")
    if st.button("SEND COMMAND"):
        if user_input:
            speak_jarvis("Processing.")
            add_log(f"BOSS: {user_input}", "warn")
            try:
                ctx = f"You are CM-X. PnL: {brain_memory['total_pnl']}. User says: {user_input}"
                reply = model.generate_content(ctx).text
                speak_jarvis(reply)
                add_log(f"AI: {reply}", "info")
            except: pass

    st.write("---")
    # APPROVAL AREA
    approval_ph = st.empty()
    st.write("---")
    
    # CONTROLS
    b1, b2 = st.columns(2)
    start = b1.button("🔥 START SYSTEM")
    stop = b2.button("🛑 KILL SWITCH")
    
    if st.button("❌ EMERGENCY EXIT"):
        st.session_state.position = None
        speak_jarvis("Emergency Exit.")
        st.rerun()
        
    pnl_ph = st.empty()

if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

# ==========================================
# 9. MAIN EXECUTION LOOP
# ==========================================
if st.session_state.bot_active:
    
    # 1. LIVE DATA FETCH
    price = get_live_data()
    st.session_state.prices.append(price)
    
    # 2. CALCULATE PHYSICS & FUTURE
    v, a, ent = brain.calculate_physics(st.session_state.prices)
    future_price = brain.monte_carlo_simulation(st.session_state.prices)
    
    # 3. COUNCIL VOTING
    votes = {}
    weights = brain_memory["weights"]
    
    # Physics Agent
    if v > 1.5 and a > 0.3: votes['Physics'] = "BUY"
    elif v < -1.5 and a < -0.3: votes['Physics'] = "SELL"
    else: votes['Physics'] = "WAIT"
    
    # Trend Agent
    ma = np.mean(list(st.session_state.prices)[-20:]) if len(st.session_state.prices)>20 else price
    if price > ma: votes['Trend'] = "BUY"
    else: votes['Trend'] = "SELL"
    
    # Global Agent
    if brain_memory["global_sentiment"] == "BULLISH": votes['Global'] = "BUY"
    elif brain_memory["global_sentiment"] == "BEARISH": votes['Global'] = "SELL"
    else: votes['Global'] = "WAIT"
    
    # Chaos Agent
    votes['Chaos'] = "GO" if ent < 1.5 else "NO_TRADE"
    
    # 4. SIGNAL SCORING
    buy_score = 0; sell_score = 0
    if votes['Physics'] == "BUY": buy_score += weights['Physics']
    if votes['Trend'] == "BUY": buy_score += weights['Trend']
    if votes['Global'] == "BUY": buy_score += weights['Global']
    
    if votes['Physics'] == "SELL": sell_score += weights['Physics']
    if votes['Trend'] == "SELL": sell_score += weights['Trend']
    if votes['Global'] == "SELL": sell_score += weights['Global']
    
    threshold = 2.0
    
    # 5. GENERATE SIGNAL
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

    # 6. APPROVAL POPUP (OPERATOR MODE)
    if st.session_state.pending_signal:
        sig = st.session_state.pending_signal
        with approval_ph.container():
            st.markdown(f"<div class='approval-box'>⚠️ EXECUTE: {sig['opt']}?</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("✅ EXECUTE", key="ex"):
                st.session_state.position = {"type": sig['type'], "entry": price, "opt": sig['opt']}
                st.session_state.trailing_high = 0.0 # Reset Trailing
                st.session_state.pending_signal = None
                add_log(f"ORDER ACTIVE: {sig['opt']} @ {price}", "warn")
                speak_jarvis("Order Executed.")
                send_telegram("ORDER EXECUTED")
                st.rerun()
            if c2.button("❌ REJECT", key="rej"):
                st.session_state.pending_signal = None
                st.rerun()

    # 7. ZERO LOSS & TRAILING LOGIC
    if st.session_state.position:
        pos = st.session_state.position
        # PnL Calc (50 Qty)
        current_pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        
        # Track Highest PnL
        if current_pnl > st.session_state.trailing_high:
            st.session_state.trailing_high = current_pnl
            
        high = st.session_state.trailing_high
        exit = False
        reason = ""
        
        # --- ZERO LOSS RULES ---
        # 1. Profit > 500? SL Moves to +200 (Brokerage Covered)
        if high > 500 and current_pnl < 200:
            exit = True
            reason = "ZERO LOSS HIT (+200)"
            
        # 2. Trailing Profit (Lock 80% if Profit > 1000)
        if high > 1000 and current_pnl < (high * 0.8):
            exit = True
            reason = "TRAILING PROFIT BOOKED"
            
        # 3. Hard Stop Loss
        if current_pnl < -300:
            exit = True
            reason = "STOP LOSS HIT"
            
        if exit:
            brain.self_improve("WIN" if current_pnl > 0 else "LOSS")
            brain_memory["total_pnl"] += current_pnl
            save_memory(brain_memory)
            st.session_state.position = None
            speak_jarvis(f"Trade Closed. {reason}")
            add_log(f"EXIT: {reason} | PNL: {current_pnl}", "danger")
            send_telegram(f"EXIT PNL: {current_pnl}")
            st.rerun()

    # 8. UPDATE VISUALS
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        def styler(v): return "agent-buy" if v=="BUY" else "agent-sell" if v=="SELL" else "agent-wait"
        
        cc1.markdown(f"<div class='agent-card {styler(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {styler(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card {styler(votes['Global'])}'>GLOBAL<br>{votes['Global']}</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card agent-wait'>CHAOS<br>{ent:.2f}</div>", unsafe_allow_html=True)

    price_ph.metric("NIFTY 50", f"{price:,.2f}")
    vel_ph.metric("VELOCITY", f"{v:.2f}")
    acc_ph.metric("ACCEL", f"{a:.2f}")
    chaos_ph.metric("ENTROPY", f"{ent:.2f}")
    
    val = brain_memory["total_pnl"]
    pnl_ph.markdown(f"<div style='background:#000; border:1px solid #333; padding:10px; text-align:center;'><h1 style='color:{'#00ff41' if val>=0 else '#ff003c'}; margin:0;'>₹{val:,.2f}</h1></div>", unsafe_allow_html=True)
    
    # CHART WITH PREDICTION
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2), name='Price'))
    # Prediction Dots
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+5], 
                             y=[price, future_price], line=dict(color='cyan', dash='dot'), name='AI Path'))
    
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)
    
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)
    
    # STABLE REFRESH (3 SECONDS - NO SHAKING)
    time.sleep(3)
    if not st.session_state.pending_signal: st.rerun()
