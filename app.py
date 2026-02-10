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
    page_title="CM-X COGNITIVE AUTOMATON",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# --- 2. SECRETS & API ---
try:
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    if "telegram" in st.secrets:
        TG_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else: TG_BOT_TOKEN = None; TG_CHAT_ID = None
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    pass # Fallback for simulation if keys are missing

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
MEMORY_FILE = "cm_x_brain_core.json"

# --- 3. THE GROWING BRAIN (MEMORY) ---
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        return {
            "total_pnl": 0.0,
            "wins": 0, "losses": 0,
            "trade_log": [],
            # இதுதான் வளரும் மூளை (Self-Improving Weights)
            "weights": {"Physics": 1.5, "Trend": 1.0, "Global": 1.2, "Chaos": 0.8},
            "global_sentiment": "NEUTRAL",
            "market_knowledge": "Market initialized. Waiting for patterns."
        }
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return init_memory()

def save_memory(mem):
    with open(MEMORY_FILE, 'w') as f: json.dump(mem, f, indent=4)

brain_memory = init_memory()

# --- 4. STATE MANAGEMENT ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=300)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = 0.0

# --- 5. AUDIO & ALERTS ---
def speak_jarvis(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "jarvis_speak.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        md = f"""<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.session_state.audio_html = md
    except: pass

def add_log(msg, type="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    color = "#000" if type=="info" else "#b30000" if type=="danger" else "#cc7a00"
    st.session_state.live_logs.appendleft(f"<span style='color:#555'>[{ts}]</span> <span style='color:{color}; font-weight:bold'>{msg}</span>")

def send_telegram(msg):
    if TG_BOT_TOKEN:
        try: requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", params={"chat_id": TG_CHAT_ID, "text": f"🧬 CM-X: {msg}"})
        except: pass

# --- 6. ADVANCED FORMULAS (PHYSICS + QUANTUM) ---
class CognitiveEngine:
    
    def calculate_newton_metrics(self, prices):
        # வேகம் (Velocity) மற்றும் முடுக்கம் (Acceleration)
        p = np.array(list(prices))
        if len(p) < 10: return 0, 0, 0
        v = np.diff(p)[-1] 
        a = np.diff(np.diff(p))[-1]
        
        # Chaos Theory (Entropy) - சந்தை குழப்பமா இருக்கா?
        hist, _ = np.histogram(p[-20:], bins=10, density=True)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy_val = -np.sum(probs * np.log(probs))
        
        return v, a, entropy_val

    def monte_carlo_forecast(self, prices):
        # எதிர்கால கணிப்பு (Next 3 Candles)
        data = list(prices)
        last = data[-1]
        vol = np.std(data[-20:]) if len(data)>20 else 5
        # 100 முறை சிமுலேஷன் செய்து சராசரியை எடுக்கும்
        paths = [last + np.random.normal(0, vol) for _ in range(100)]
        return np.mean(paths)

    def get_best_option_strike(self, spot_price, direction):
        # தானியங்கி ஸ்ட்ரைக் தேர்வு (Nifty 50)
        # 22130 -> 22150 (Nearest 50)
        strike = round(spot_price / 50) * 50
        if direction == "BUY": return f"{strike} CE" # Call Option
        else: return f"{strike} PE" # Put Option

    def self_correct(self, result):
        # வெற்றி தோல்வியை வைத்து மூளையை திருத்துதல்
        w = brain_memory["weights"]
        learning_rate = 0.05
        
        if result == "WIN":
            w["Physics"] += learning_rate
            w["Trend"] += learning_rate
            brain_memory["market_knowledge"] += " | Pattern Success."
        else:
            w["Physics"] -= learning_rate
            w["Trend"] -= learning_rate
            brain_memory["market_knowledge"] += " | Pattern Failed. Adjusting weights."
        
        save_memory(brain_memory)

brain = CognitiveEngine()

def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: 
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

# --- 7. SMOOTH UI STYLING (DIM & NEON) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    /* BACKGROUND: Dim Dark Blue-Grey (Not Black, Not Light) */
    .stApp { 
        background-color: #1e293b; 
        color: #e2e8f0; 
        font-family: 'Roboto Mono', monospace; 
    }
    
    /* TEXT STYLES */
    h1, h2, h3 { color: #f8fafc; text-shadow: 0 0 5px rgba(255,255,255,0.3); }
    
    /* METRIC CARDS (Smooth Round) */
    div[data-testid="stMetric"] {
        background-color: #0f172a; /* Darker Inner Box */
        border: 1px solid #334155;
        border-radius: 15px; /* Rounded Corners */
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 14px; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #f1f5f9; font-size: 26px; font-weight: bold; }

    /* NEON ACCENTS */
    .neon-green { color: #4ade80; text-shadow: 0 0 8px #4ade80; font-weight:bold; }
    .neon-red { color: #f87171; text-shadow: 0 0 8px #f87171; font-weight:bold; }
    .neon-orange { color: #fbbf24; text-shadow: 0 0 8px #fbbf24; font-weight:bold; }

    /* AGENT CARDS */
    .agent-card {
        background: #1e293b; border: 1px solid #475569; 
        padding: 10px; text-align: center; border-radius: 12px;
        font-size: 14px; color: white; margin-bottom: 5px;
    }
    
    /* BUTTONS */
    .stButton>button {
        background: #334155; color: white; border: 1px solid #94a3b8;
        border-radius: 8px; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover {
        background: #475569; border-color: white;
    }
    
    /* LOG BOX */
    .log-box {
        font-family: 'Courier New'; font-size: 12px;
        background: #020617; color: #4ade80;
        padding: 10px; border-radius: 8px; border: 1px solid #1e293b;
        height: 200px; overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 8. LAYOUT DASHBOARD ---
st.markdown("### 🧬 CM-X COGNITIVE AUTOMATON")
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])

with c1:
    # CHART & METRICS
    chart_ph = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    
    # 4-CHAMBER COUNCIL
    st.markdown("#### 🏛️ THE COUNCIL (DECISION CORE)")
    council_ph = st.empty()
    
    # LOGS
    st.markdown("#### 🖥️ MEMORY LOGS")
    log_ph = st.empty()

with c2:
    # GLOBAL SETTING
    st.markdown("#### 🌍 GLOBAL IMPACT")
    g_sent = st.select_slider("Market Sentiment", ["BEARISH", "NEUTRAL", "BULLISH"], value=brain_memory["global_sentiment"])
    if g_sent != brain_memory["global_sentiment"]:
        brain_memory["global_sentiment"] = g_sent
        save_memory(brain_memory)

    # GEMINI LIVE
    st.markdown("#### 💬 TALK TO BRAIN")
    u_msg = st.text_input("Ask Question:", placeholder="E.g., What did you learn today?")
    if st.button("ASK"):
        if u_msg:
            # Context Aware Reply
            ctx = f"You are CM-X. PnL: {brain_memory['total_pnl']}. Knowledge: {brain_memory['market_knowledge']}. User: {u_msg}"
            try:
                reply = model.generate_content(ctx).text
                speak_jarvis(reply)
                add_log(f"AI: {reply}", "warn")
            except: pass

    st.write("---")
    approval_ph = st.empty() # The Execution Area
    st.write("---")
    
    b1, b2 = st.columns(2)
    start = b1.button("🔥 START SYSTEM")
    stop = b2.button("🛑 STOP SYSTEM")
    
    pnl_ph = st.empty()

# --- 9. INTELLIGENCE LOOP ---
if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

if st.session_state.bot_active:
    
    # 1. LIVE DATA FEED
    price = get_live_data()
    st.session_state.prices.append(price)
    
    # 2. PHYSICS CALCULATION
    v, a, ent = brain.calculate_newton_metrics(st.session_state.prices)
    future = brain.monte_carlo_forecast(st.session_state.prices)
    
    # 3. MARKET STATE (Scalping or Trending?)
    market_mode = "TREND"
    if ent > 1.2: market_mode = "SCALP (ROCKET)" # High Entropy = Volatile
    
    # 4. COUNCIL VOTING
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
    
    # Chaos Agent
    votes['Chaos'] = "GO" # Always active unless extreme risk
    
    # 5. SIGNAL FUSION (Weighted)
    buy_power = 0
    sell_power = 0
    
    if votes['Physics'] == "BUY": buy_power += weights['Physics']
    if votes['Trend'] == "BUY": buy_power += weights['Trend']
    if votes['Global'] == "BUY": buy_power += weights['Global']
    
    if votes['Physics'] == "SELL": sell_power += weights['Physics']
    if votes['Trend'] == "SELL": sell_power += weights['Trend']
    if votes['Global'] == "SELL": sell_power += weights['Global']
    
    # Threshold
    threshold = 2.0
    
    # 6. TRIGGER SIGNAL
    if buy_power > threshold and not st.session_state.position and not st.session_state.pending_signal:
        opt = brain.get_best_option_strike(price, "BUY")
        st.session_state.pending_signal = {"type": "BUY", "opt": opt}
        speak_jarvis(f"Boss! Buy Signal on {opt}. Approve?")
        send_telegram(f"BUY ALERT: {opt}")
        
    elif sell_power > threshold and not st.session_state.position and not st.session_state.pending_signal:
        opt = brain.get_best_option_strike(price, "SELL")
        st.session_state.pending_signal = {"type": "SELL", "opt": opt}
        speak_jarvis(f"Boss! Sell Signal on {opt}. Approve?")
        send_telegram(f"SELL ALERT: {opt}")

    # 7. EXECUTION (User Approval)
    if st.session_state.pending_signal:
        sig = st.session_state.pending_signal
        with approval_ph.container():
            st.markdown(f"<div class='approval-box'>⚠️ EXECUTE: {sig['opt']}?</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("✅ YES", key="yes"):
                st.session_state.position = {"type": sig['type'], "entry": price, "opt": sig['opt']}
                st.session_state.trailing_high = 0.0
                st.session_state.pending_signal = None
                add_log(f"ORDER ACTIVE: {sig['opt']} @ {price}", "warn")
                speak_jarvis("Order Executed.")
                st.rerun()
            if c2.button("❌ NO", key="no"):
                st.session_state.pending_signal = None
                st.rerun()

    # 8. TRADE MANAGEMENT (ZERO LOSS LOGIC)
    if st.session_state.position:
        pos = st.session_state.position
        # Simulated PnL (50 Qty)
        current_pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        
        # Track Highest PnL for Trailing
        if current_pnl > st.session_state.trailing_high:
            st.session_state.trailing_high = current_pnl
        
        high = st.session_state.trailing_high
        exit_trade = False
        reason = ""
        
        # --- ZERO LOSS FORMULA ---
        # 1. If Profit > 500, Move SL to Breakeven+Brokerage (200)
        if high > 500 and current_pnl < 200:
            exit_trade = True
            reason = "ZERO LOSS HIT (+200)"
            
        # 2. Trailing Profit (Lock 70% of gains if huge move)
        if high > 1000 and current_pnl < (high * 0.7):
            exit_trade = True
            reason = f"TRAILING PROFIT BOOKED ({int(high*0.7)})"
            
        # 3. Hard Stop Loss (Initial)
        if current_pnl < -300:
            exit_trade = True
            reason = "HARD STOP LOSS (-300)"
            
        if exit_trade:
            st.session_state.position = None
            brain_memory["total_pnl"] += current_pnl
            if current_pnl > 0: brain.self_correct("WIN")
            else: brain.self_correct("LOSS")
            
            speak_jarvis(f"Trade Closed. {reason}")
            add_log(f"EXIT: {reason} | PNL: {current_pnl}", "warn")
            st.rerun()

    # 9. VISUALS
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        def style(v): return "buy" if v=="BUY" else "sell" if v=="SELL" else "wait"
        cc1.markdown(f"<div class='agent-card {style(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {style(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card {style(votes['Global'])}'>GLOBAL<br>{votes['Global']}</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card wait'>MODE<br>{market_mode}</div>", unsafe_allow_html=True)

    m1.metric("NIFTY 50", f"{price:,.2f}")
    m2.metric("VELOCITY", f"{v:.2f}")
    m3.metric("CHAOS", f"{ent:.2f}")
    
    pnl = brain_memory["total_pnl"]
    pnl_ph.markdown(f"<h1 style='text-align:center; color:{'green' if pnl>=0 else 'red'}'>PNL: ₹{pnl:,.2f}</h1>", unsafe_allow_html=True)
    
    # Chart with Prediction
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='black', width=3), name='History'))
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+5], 
                             y=[price, future], line=dict(color='blue', dash='dot'), name='AI Predict'))
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)
    
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div style="height:200px; overflow-y:auto; border:2px solid #000; padding:5px;">{log_html}</div>', unsafe_allow_html=True)

    time.sleep(8)
    if not st.session_state.pending_signal: st.rerun()
