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
from scipy.stats import entropy as scipy_entropy

# --- 1. SYSTEM CONFIGURATION (PDF Page 1) [cite: 19-25] ---
st.set_page_config(
    page_title="AETHER: FUSION GOD MODE",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# --- 2. GLOBAL CONSTANTS (PDF Page 2) [cite: 27-31] ---
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 300
TELEGRAM_INTERVAL = 120 # 2 Minutes
KILL_SWITCH_LOSS = -2000
TRADE_QUANTITY = 50
TIMEZONE = pytz.timezone('Asia/Kolkata')

# --- 3. SECRETS & API SETUP (PDF Page 2-3) [cite: 34-58] ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    if "telegram" in st.secrets:
        TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else:
        TELEGRAM_BOT_TOKEN = None
        TELEGRAM_CHAT_ID = None
        
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro') # Fast model for live trading
    
except Exception as e:
    st.error(f"SYSTEM FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = 'https://api.upstox.com/v2/market-quote/ltp'
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. ADVANCED CYBERPUNK STYLING (DIM & NEON) (PDF Page 3-9) [cite: 62-211] ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css?family=Fira+Code&display=swap');
    
    /* BASE APP STYLING - Dim Dark Blue-Grey (Eye Friendly) */
    .stApp {
        background-color: #050505; 
        color: #e2e8f0;
        font-family: 'Fira Code', monospace;
    }
    
    /* HEADINGS */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #f8fafc;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #00ff41; /* Neon Green */
        font-family: 'Orbitron', sans-serif;
        font-size: 26px;
        text-shadow: 0 0 5px #00ff41;
    }
    div[data-testid="stMetricLabel"] { color: #888; font-weight: bold; }
    
    /* AGENT CARDS (THE COUNCIL) */
    .agent-card {
        background: #111; border: 1px solid #333; padding: 10px;
        text-align: center; border-radius: 5px; margin-bottom: 5px;
        font-family: 'Orbitron'; font-size: 12px; color: #fff;
    }
    .BUY { border-color: #00ff41; color: #00ff41; box-shadow: 0 0 8px #00ff41; }
    .SELL { border-color: #ff003c; color: #ff003c; box-shadow: 0 0 8px #ff003c; }
    .WAIT { border-color: #fbbf24; color: #fbbf24; }
    .GO { border-color: #c084fc; color: #c084fc; } /* Chaos GO */

    /* TERMINAL LOG BOX */
    .terminal-box {
        font-family: 'Fira Code', monospace;
        background-color: #000;
        color: #00ff41;
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        border: 1px solid #333;
        border-radius: 5px;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.8);
    }
    .log-time { color: #555; margin-right: 10px; }
    .log-buy { color: #00ff41; font-weight: bold; }
    .log-sell { color: #ff003c; font-weight: bold; }
    .log-ai { color: #c084fc; font-style: italic; }
    
    /* APPROVAL BOX */
    .approval-box {
        border: 2px solid #fbbf24; background-color: #222200;
        color: #fbbf24; padding: 15px; text-align: center;
        font-family: 'Orbitron'; animation: pulse 1s infinite;
        border-radius: 10px; margin-bottom: 10px;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    /* BUTTONS */
    .stButton>button {
        background-color: #000; color: #00ff41; border: 1px solid #00ff41;
        font-family: 'Orbitron'; height: 50px; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00ff41; color: #000; box-shadow: 0 0 15px #00ff41;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. MEMORY SYSTEM (PDF Page 9-11) [cite: 213-256] ---
def init_brain():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "total_pnl": 0.0,
            "wins": 0, "losses": 0,
            "trade_log": [],
            "weights": {"Physics": 1.5, "Trend": 1.0, "Global": 1.2, "Chaos": 0.8, "WinProb": 1.0},
            "global_sentiment": "NEUTRAL",
            "market_knowledge": "System Initialized.",
            "pending_signal": None,
            "trailing_high": 0.0,
            "last_tg_time": time.time(),
            "bot_active_on_exit": False
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return init_brain()

def save_brain(mem_data):
    with open(MEMORY_FILE, 'w') as f: json.dump(mem_data, f, indent=4)

brain_memory = init_brain()

# --- 6. SESSION STATE (PDF Page 11) [cite: 259-269] ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = brain_memory.get("bot_active_on_exit", False)
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = brain_memory.get("pending_signal", None)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=50)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = brain_memory.get("trailing_high", 0.0)
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = brain_memory.get("last_tg_time", time.time())

# --- 7. AUDIO & LOGS & TELEGRAM (PDF Page 11-13) [cite: 271-305] ---
def speak_aether(text):
    try:
        brain_memory["market_knowledge"] = f"AETHER: {text}"
        add_log(f"AETHER: {text}", "log-ai")
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "aether.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

def add_log(msg, css_class="log-info"):
    ts = datetime.now(TIMEZONE).strftime("%H:%M:%S")
    entry = f"<div class='log-row'><span class='log-time'>{ts}</span> <span class='{css_class}'>{msg}</span></div>"
    st.session_state.live_logs.appendleft(entry)

def send_telegram_report(msg):
    if not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🦅 CM-X: {msg}"}, timeout=5)
    except: pass

# --- 8. COGNITIVE ENGINE (MATH CORE) (PDF Page 13-17) [cite: 307-388] ---
class CognitiveEngine:
    def calculate_newton_metrics(self, prices_deque):
        p = np.array(list(prices_deque))
        if len(p) < 10: return 0.0, 0.0, 0.0
        v = p[-1] - p[-2]
        a = (p[-1] - p[-2]) - (p[-2] - p[-3])
        # Entropy
        if len(p) >= 20:
            hist, _ = np.histogram(p[-20:], bins=10, density=True)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            ent = scipy_entropy(probs)
        else: ent = 0.0
        return v, a, ent

    def monte_carlo_simulation(self, prices_deque, num_sims=100):
        p = np.array(list(prices_deque))
        if len(p) < 20: return 0.5
        last = p[-1]
        returns = np.diff(p)/p[:-1]
        mu = np.mean(returns); sigma = np.std(returns)
        bull_paths = 0
        for _ in range(num_sims):
            sim_p = last
            for _ in range(10): # 10 steps
                sim_p *= (1 + np.random.normal(mu, sigma))
            if sim_p > last: bull_paths += 1
        return bull_paths / num_sims

    def rocket_formula(self, v, vol_curr, vol_avg):
        if vol_avg == 0: vol_avg = 1
        ratio = abs(vol_curr/vol_avg)
        return v * math.log(ratio + 1)

    def get_best_option_strike(self, spot, direction):
        strike = round(spot/50)*50
        return f"{strike} CE" if direction == "BUY" else f"{strike} PE"

    def self_correct(self, result):
        w = brain_memory["weights"]
        lr = 0.05
        if result == "WIN":
            for k in w: w[k] = min(2.0, w[k] + lr)
            brain_memory["market_knowledge"] += " | Pattern Validated."
        else:
            for k in w: w[k] = max(0.5, w[k] - lr)
            brain_memory["market_knowledge"] += " | Pattern Failed."
        save_brain(brain_memory)

aether_engine = CognitiveEngine()

# --- 9. AI CONSULTATION (PDF Page 17-18)  ---
def consult_ai(query, price=None, v=None):
    if not GEMINI_API_KEY: return "AI Offline."
    ctx = f"You are AETHER. PnL: {brain_memory['total_pnl']}. Price: {price}. Velocity: {v}. User: {query}"
    try:
        return gemini_model.generate_content(ctx).text
    except: return "Connection Error."

# --- 10. LIVE DATA FETCH (PDF Page 18-20) [cite: 420-461] ---
def get_live_market_data():
    if not UPSTOX_ACCESS_TOKEN:
        # Simulation Fallback
        last = st.session_state.prices[-1] if st.session_state.prices else 22100.0
        return last + np.random.normal(0, 5), "SIMULATING"
    
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            # Handle key variations
            k = REQ_INSTRUMENT_KEY
            if k in data: return float(data[k]['last_price']), "CONNECTED"
            k = k.replace('|', ':')
            if k in data: return float(data[k]['last_price']), "CONNECTED"
            return float(data[list(data.keys())[0]]['last_price']), "CONNECTED"
    except: pass
    return None, "ERROR"

# --- 11. UI LAYOUT (PDF Page 20-24) [cite: 463-543] ---
st.markdown(f"""
<div style="text-align: center; border-bottom: 2px solid #00ff41; padding-bottom: 10px;">
    <h1>AETHER: FUSION GOD MODE</h1>
    <p style="color:#888;">OPERATOR: {OWNER_NAME} | MEMORY: ACTIVE</p>
</div>
""", unsafe_allow_html=True)
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

c1, c2 = st.columns([2.5, 1])

with c1:
    st.subheader("📡 QUANTUM TRAJECTORY")
    chart_ph = st.empty()
    
    m1, m2, m3, m4, m5 = st.columns(5)
    p_met = m1.empty(); v_met = m2.empty(); a_met = m3.empty()
    e_met = m4.empty(); w_met = m5.empty()
    
    st.write("---")
    st.subheader("🏛️ THE COUNCIL CHAMBER")
    council_ph = st.empty()
    
    st.write("---")
    st.subheader("🖥️ NEURAL LOGS")
    log_ph = st.empty()

with c2:
    st.subheader("🌍 GLOBAL CONTROL")
    curr_sent = brain_memory.get("global_sentiment", "NEUTRAL")
    new_sent = st.select_slider("Sentiment", ["BEARISH", "NEUTRAL", "BULLISH"], value=curr_sent)
    if new_sent != curr_sent:
        brain_memory["global_sentiment"] = new_sent
        save_brain(brain_memory)
    
    st.write("---")
    ai_in = st.text_input("Consult Aether:")
    if st.button("ASK"):
        if ai_in:
            p = st.session_state.prices[-1] if st.session_state.prices else 0
            rep = consult_ai(ai_in, p)
            speak_aether(rep)
            
    st.write("---")
    pnl_ph = st.empty()
    
    b1, b2 = st.columns(2)
    start = b1.button("🔥 INITIATE")
    stop = b2.button("🛑 KILL SWITCH")
    
    if st.button("❌ EMERGENCY EXIT"):
        st.session_state.position = None
        speak_aether("Emergency Exit.")
        st.rerun()
        
    approval_ph = st.empty()

if start: 
    st.session_state.bot_active = True
    brain_memory["bot_active_on_exit"] = True
    save_brain(brain_memory)
    st.rerun()
if stop:
    st.session_state.bot_active = False
    brain_memory["bot_active_on_exit"] = False
    save_brain(brain_memory)
    st.rerun()

# --- 12. MAIN LOOP (PDF Page 25-40) [cite: 574-917] ---
if st.session_state.bot_active:
    
    # A. FETCH DATA
    price, status = get_live_market_data()
    if price: st.session_state.prices.append(price)
    
    # B. CALCULATE METRICS
    v, a, ent = aether_engine.calculate_newton_metrics(st.session_state.prices)
    win_prob = aether_engine.monte_carlo_simulation(st.session_state.prices)
    
    # C. COUNCIL VOTING
    votes = {}
    w = brain_memory["weights"]
    
    # Physics
    if v > 1.5 and a > 0.3: votes['Physics'] = "BUY"
    elif v < -1.5 and a < -0.3: votes['Physics'] = "SELL"
    else: votes['Physics'] = "WAIT"
    
    # Trend
    if len(st.session_state.prices) > 20:
        ma = np.mean(list(st.session_state.prices)[-20:])
        if price > ma: votes['Trend'] = "BUY"
        else: votes['Trend'] = "SELL"
    
    # Global
    gs = brain_memory["global_sentiment"]
    if gs == "BULLISH": votes['Global'] = "BUY"
    elif gs == "BEARISH": votes['Global'] = "SELL"
    else: votes['Global'] = "WAIT"
    
    # WinProb
    if win_prob > 0.6: votes['WinProb'] = "BUY"
    elif win_prob < 0.4: votes['WinProb'] = "SELL"
    else: votes['WinProb'] = "WAIT"
    
    # Chaos (Rocket Mode)
    mode = "TREND"
    if ent > 1.2: 
        mode = "ROCKET (SCALP)"
        votes['Chaos'] = "GO"
    else: votes['Chaos'] = "GO"
    
    # D. FUSION SCORING
    buy_score = 0; sell_score = 0
    for ag, vote in votes.items():
        if ag in w:
            if vote == "BUY": buy_score += w[ag]
            elif vote == "SELL": sell_score += w[ag]
            
    # E. SIGNAL GENERATION
    threshold = 2.0
    if buy_score > threshold and not st.session_state.position and not st.session_state.pending_signal:
        opt = aether_engine.get_best_option_strike(price, "BUY")
        st.session_state.pending_signal = {"type": "BUY", "opt": opt}
        speak_aether(f"Boss! Buy Signal on {opt}. Score {buy_score:.1f}")
        send_telegram_report(f"BUY ALERT: {opt}")
        st.rerun()
        
    elif sell_score > threshold and not st.session_state.position and not st.session_state.pending_signal:
        opt = aether_engine.get_best_option_strike(price, "SELL")
        st.session_state.pending_signal = {"type": "SELL", "opt": opt}
        speak_aether(f"Boss! Sell Signal on {opt}. Score {sell_score:.1f}")
        send_telegram_report(f"SELL ALERT: {opt}")
        st.rerun()

    # F. APPROVAL UI
    if st.session_state.pending_signal:
        sig = st.session_state.pending_signal
        with approval_ph.container():
            st.markdown(f"<div class='approval-box'>⚠️ EXECUTE {sig['type']} {sig['opt']}?</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("✅ YES", key="y"):
                st.session_state.position = {"type": sig['type'], "entry": price, "opt": sig['opt'], "qty": TRADE_QUANTITY}
                st.session_state.trailing_high = 0.0
                st.session_state.pending_signal = None
                brain_memory["position"] = st.session_state.position
                save_brain(brain_memory)
                speak_aether("Order Executed.")
                add_log("Trade Active", "log-buy")
                st.rerun()
            if c2.button("❌ NO", key="n"):
                st.session_state.pending_signal = None
                st.rerun()

    # G. TRADE MANAGEMENT (ZERO LOSS)
    if st.session_state.position:
        pos = st.session_state.position
        curr_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
        
        if curr_pnl > st.session_state.trailing_high:
            st.session_state.trailing_high = curr_pnl
            
        high = st.session_state.trailing_high
        exit = False; reason = ""
        
        if high > 500 and curr_pnl < 200: # Zero Loss
            exit = True; reason = "ZERO LOSS HIT"
        elif high > 1000 and curr_pnl < (high * 0.7): # Trailing
            exit = True; reason = "TRAILING PROFIT"
        elif curr_pnl < -300: # Hard SL
            exit = True; reason = "STOP LOSS"
            
        if exit:
            st.session_state.position = None
            brain_memory["total_pnl"] += curr_pnl
            brain_memory["position"] = None
            aether_engine.self_correct("WIN" if curr_pnl>0 else "LOSS")
            speak_aether(f"Trade Closed. {reason}")
            send_telegram_report(f"EXIT: {curr_pnl}")
            st.rerun()

    # H. VISUALS UPDATE
    p_met.metric("NIFTY", f"{price:,.2f}")
    v_met.metric("VELOCITY", f"{v:.2f}")
    a_met.metric("ACCEL", f"{a:.2f}")
    e_met.metric("CHAOS", f"{ent:.2f}")
    w_met.metric("WIN%", f"{win_prob*100:.0f}%")
    
    with council_ph.container():
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        def col(v): return v if v in ["BUY", "SELL", "WAIT", "GO"] else "WAIT"
        cc1.markdown(f"<div class='agent-card {col(votes.get('Physics'))}'>PHYSICS<br>{votes.get('Physics')}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {col(votes.get('Trend'))}'>TREND<br>{votes.get('Trend')}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card {col(votes.get('Global'))}'>GLOBAL<br>{votes.get('Global')}</div>", unsafe_allow_html=True)
        cc4.markdown(f"<div class='agent-card {col(votes.get('WinProb'))}'>WIN%<br>{votes.get('WinProb')}</div>", unsafe_allow_html=True)
        cc5.markdown(f"<div class='agent-card {col(votes.get('Chaos'))}'>MODE<br>{mode}</div>", unsafe_allow_html=True)

    val = brain_memory["total_pnl"]
    pnl_ph.markdown(f"<h2 style='text-align:center; color:{'#00ff41' if val>=0 else '#ff003c'}'>PNL: ₹{val:,.2f}</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
    if st.session_state.position:
        fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="#fbbf24")
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)
    
    l_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f"<div class='terminal-box'>{l_html}</div>", unsafe_allow_html=True)
    
    time.sleep(1) # [cite: 916]
    if not st.session_state.pending_signal: st.rerun()
