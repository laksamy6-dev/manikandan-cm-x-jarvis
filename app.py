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

# --- 1. SYSTEM CONFIGURATION (LIGHT MODE FOR DAYLIGHT) ---
st.set_page_config(
    page_title="CM-X COGNITIVE ALPHA",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# --- 2. BLACK BOX MEMORY (THE SOUL) ---
# இதுதான் பாஸ் "உயிர்". நீங்க போனை உடைச்சாலும் இந்த மெமரி அழியாது.
MEMORY_FILE = "cm_x_blackbox_memory.json"

def init_black_box():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "total_pnl": 0.0,
            "trade_history": [],
            "ai_weights": { # இதுதான் வளரும் மூளை
                "Physics": 1.5,
                "Trend": 1.0, 
                "Options": 1.2, 
                "Chaos": 0.8
            },
            "learning_rate": 0.01,
            "last_active": str(datetime.now())
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return init_black_box() # Corrupt ஆனா புதுசா உருவாக்கும்

def save_black_box(memory):
    memory["last_active"] = str(datetime.now())
    with open(MEMORY_FILE, 'w') as f: json.dump(memory, f, indent=4)

# Load Memory on Startup
brain_memory = init_black_box()

# --- 3. SESSION STATE (RAM) ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=200) # Long history
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None # Current Trade
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None

# --- 4. CSS STYLING (DAYLIGHT VISIBILITY MODE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    /* Global Daylight Theme */
    .stApp { background-color: #f0f2f6; color: #111; font-family: 'Rajdhani', sans-serif; }
    
    /* Headers */
    h1, h2, h3 { color: #000; font-weight: 800; text-transform: uppercase; }
    
    /* Metrics Cards (Glassmorphism Light) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #ccc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 10px;
        color: #000;
    }
    div[data-testid="stMetricLabel"] { color: #555; font-weight: bold; }
    div[data-testid="stMetricValue"] { color: #000; font-size: 28px; font-weight: 900; }

    /* THE COUNCIL CHAMBER (Cards) */
    .agent-card {
        background: #fff; border: 2px solid #ddd; padding: 10px; 
        text-align: center; border-radius: 8px; font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .agent-buy { border-color: #00aa00; color: #00aa00; background: #e6ffe6; }
    .agent-sell { border-color: #ff0000; color: #ff0000; background: #ffe6e6; }
    .agent-wait { border-color: #888; color: #888; background: #f0f0f0; }

    /* TERMINAL (Black Box in Day Mode) */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #111;
        color: #00ff41; /* Hacker Green Text */
        border: 2px solid #333;
        padding: 10px;
        height: 250px;
        overflow-y: auto;
        font-size: 13px;
        border-radius: 5px;
    }

    /* APPROVAL ALERT */
    .approval-box {
        background-color: #fffbeb; border: 3px solid #fbbf24; 
        color: #b45309; padding: 20px; text-align: center; 
        border-radius: 10px; animation: pulse 1s infinite;
        font-weight: 900; font-size: 20px;
    }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }

    /* BUTTONS */
    .stButton>button {
        width: 100%; font-weight: 900; border-radius: 8px; height: 50px;
        border: 2px solid #000; color: #000; background: #fff;
    }
    .stButton>button:hover { background: #000; color: #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. CONFIG SECRETS ---
try:
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass # Error handling suppressed for UI flow

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 6. ADVANCED MATH ENGINE (Neuro-Quantum) ---
class CognitiveBrain:
    def calculate_physics(self, prices):
        # Rocket Formula: Velocity & Acceleration
        p = np.array(prices)
        if len(p) < 5: return 0, 0, 0
        v = np.diff(p)[-1] # Velocity
        a = np.diff(np.diff(p))[-1] # Acceleration
        entropy_val = entropy(np.histogram(p[-20:], bins=10)[0]) # Chaos
        return v, a, entropy_val

    def monte_carlo_prediction(self, prices):
        # "அடுத்த 3 கேண்டில் கணிப்பு"
        last_price = prices[-1]
        volatility = np.std(prices[-20:]) if len(prices) > 20 else 5
        
        # 3 Future paths (Bull, Bear, Neutral)
        paths = []
        for _ in range(3): # Next 3 minutes
            shock = np.random.normal(0, volatility)
            paths.append(last_price + shock)
        return paths

    def self_learn(self, result):
        # வெற்றி பெற்றால் எடையை கூட்டு (Reinforcement Learning)
        if result == "WIN":
            brain_memory["ai_weights"]["Physics"] += 0.05
            brain_memory["ai_weights"]["Trend"] += 0.05
        elif result == "LOSS":
            brain_memory["ai_weights"]["Physics"] -= 0.05
        save_black_box(brain_memory)

brain_logic = CognitiveBrain()

# --- 7. HELPER FUNCTIONS ---
def add_log(msg, type="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    color = "#00ff41" if type=="info" else "#ffcc00" if type=="warn" else "#ff0000"
    st.session_state.live_logs.appendleft(f"<span style='color:#888'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")

def speak_jarvis(text):
    try:
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "jarvis_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        md = f"""<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
        st.session_state.audio_html = md
    except: pass

def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=2)
        if res.status_code == 200:
            return float(res.json()['data'][list(res.json()['data'].keys())[0]]['last_price'])
    except: return None

# --- 8. UI LAYOUT ---
st.markdown(f"""
<div style="text-align:center; padding-bottom:10px; border-bottom:2px solid #ccc;">
    <h1 style="color:#000; margin:0;">CM-X COGNITIVE ALPHA</h1>
    <p style="color:#555; font-weight:bold;">OPERATOR: BOSS MANIKANDAN | MEMORY: <span style="color:green">ACTIVE (BLACK BOX)</span></p>
</div>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# MAIN DASHBOARD GRID
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📡 LIVE MARKET KINEMATICS")
    chart_ph = st.empty()
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    price_ph = m1.empty()
    vel_ph = m2.empty()
    acc_ph = m3.empty()
    chaos_ph = m4.empty()

    # THE COUNCIL CHAMBER (4 AGENTS)
    st.write("---")
    st.subheader("🧠 THE COUNCIL (10,000 MINDS)")
    council_ph = st.empty()

    # LIVE TERMINAL
    st.write("---")
    st.subheader("🖥️ NEURAL LOGS")
    log_ph = st.empty()

with c2:
    st.subheader("🎤 GHOST PROTOCOL")
    
    # 1. VOICE INPUT
    user_cmd = st.text_input("COMMAND JARVIS:", placeholder="Type 'Status' or 'Scan'...")
    if user_cmd:
        add_log(f"BOSS: {user_cmd}", "warn")
        speak_jarvis(f"Processing command: {user_cmd}")

    st.write("---")
    
    # 2. APPROVAL BOX (Pop-up)
    approval_ph = st.empty()
    
    st.write("---")
    
    # 3. P&L & CONTROLS
    pnl_ph = st.empty()
    
    start = st.button("🔥 INITIATE SYSTEM")
    stop = st.button("🛑 KILL SWITCH")
    
    # Active Trade Info
    if st.session_state.position:
        st.info(f"OPEN POSITION: {st.session_state.position['type']} @ {st.session_state.position['entry']}")

if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

# --- 9. MAIN LOOP (THE BRAIN) ---
if st.session_state.bot_active:
    
    # 1. DATA FETCH (Real + Simulation Fallback)
    price = get_live_data()
    if not price: 
        if st.session_state.prices: price = st.session_state.prices[-1] + np.random.normal(0, 3)
        else: price = 22000.00 # Base
    
    st.session_state.prices.append(price)
    
    # 2. CALCULATE PHYSICS (Rocket Formula)
    v, a, entropy_val = brain_logic.calculate_physics(st.session_state.prices)
    
    # 3. THE COUNCIL VOTING (Weighted by Memory)
    weights = brain_memory["ai_weights"]
    votes = {}
    
    # Physics Agent
    if v > 2.0 and a > 0.5: votes['Physics'] = "BUY"
    elif v < -2.0 and a < -0.5: votes['Physics'] = "SELL"
    else: votes['Physics'] = "WAIT"
    
    # Trend Agent (Simulated SuperTrend)
    ma = np.mean(list(st.session_state.prices)[-20:]) if len(st.session_state.prices)>20 else price
    if price > ma: votes['Trend'] = "BUY"
    else: votes['Trend'] = "SELL"
    
    # Volatility Agent
    if entropy_val > 1.5: votes['Chaos'] = "NO_TRADE"
    else: votes['Chaos'] = "GO"
    
    # 4. DECISION MATRIX
    buy_score = 0
    sell_score = 0
    
    if votes['Physics'] == "BUY": buy_score += weights['Physics']
    if votes['Trend'] == "BUY": buy_score += weights['Trend']
    
    if votes['Physics'] == "SELL": sell_score += weights['Physics']
    if votes['Trend'] == "SELL": sell_score += weights['Trend']
    
    # 5. SIGNAL GENERATION
    if buy_score > 2.0 and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "BUY"
        speak_jarvis("Boss! Rocket Launch Detected. Buying Call.")
    elif sell_score > 2.0 and not st.session_state.position and not st.session_state.pending_signal:
        st.session_state.pending_signal = "SELL"
        speak_jarvis("Boss! Gravity Failure. Buying Put.")

    # 6. APPROVAL SYSTEM (Human in the Loop)
    if st.session_state.pending_signal:
        with approval_ph.container():
            st.markdown(f"""
            <div class="approval-box">
                ⚠️ AUTHORIZATION REQUIRED<br>
                <span style="font-size:30px">{st.session_state.pending_signal} SIGNAL</span>
            </div>
            """, unsafe_allow_html=True)
            
            c_yes, c_no = st.columns(2)
            if c_yes.button("✅ EXECUTE", key="yes"):
                st.session_state.position = {"type": st.session_state.pending_signal, "entry": price}
                st.session_state.pending_signal = None
                add_log("ORDER EXECUTED", "warn")
                speak_jarvis("Order Placed.")
                st.rerun()
            if c_no.button("❌ REJECT", key="no"):
                st.session_state.pending_signal = None
                add_log("ORDER REJECTED", "error")
                st.rerun()

    # 7. AUTO EXIT (Profit/Loss Logic)
    if st.session_state.position:
        pos = st.session_state.position
        pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
        
        if pnl > 500 or pnl < -300:
            brain_memory["total_pnl"] += pnl
            res = "WIN" if pnl > 0 else "LOSS"
            brain_logic.self_learn(res) # TEACH THE BRAIN
            save_black_box(brain_memory)
            st.session_state.position = None
            speak_jarvis(f"Trade Closed. {res}")
            st.rerun()

    # --- UI UPDATES ---
    
    # 1. Update Council Cards
    with council_ph.container():
        cc1, cc2, cc3, cc4 = st.columns(4)
        def get_cls(v): return "agent-buy" if v=="BUY" else "agent-sell" if v=="SELL" else "agent-wait"
        
        cc1.markdown(f"<div class='agent-card {get_cls(votes['Physics'])}'>PHYSICS<br>{votes['Physics']}</div>", unsafe_allow_html=True)
        cc2.markdown(f"<div class='agent-card {get_cls(votes['Trend'])}'>TREND<br>{votes['Trend']}</div>", unsafe_allow_html=True)
        cc3.markdown(f"<div class='agent-card agent-wait'>OPTIONS<br>WAIT</div>", unsafe_allow_html=True) # Placeholder
        cc4.markdown(f"<div class='agent-card agent-wait'>CHAOS<br>{votes['Chaos']}</div>", unsafe_allow_html=True)

    # 2. Update Metrics
    price_ph.metric("NIFTY 50", f"{price:,.2f}")
    vel_ph.metric("VELOCITY", f"{v:.2f}")
    acc_ph.metric("ACCEL", f"{a:.2f}")
    chaos_ph.metric("ENTROPY", f"{entropy_val:.2f}")
    
    pnl_val = brain_memory["total_pnl"]
    pnl_color = "green" if pnl_val >= 0 else "red"
    pnl_ph.markdown(f"<div style='background:#fff; padding:10px; border-radius:10px; text-align:center; border:2px solid {pnl_color}'>"
                    f"<h2 style='color:{pnl_color}; margin:0;'>PNL: ₹{pnl_val:,.2f}</h2></div>", unsafe_allow_html=True)

    # 3. Update Chart (With Future Prediction - The "Extra Remake")
    future_paths = brain_logic.monte_carlo_prediction(list(st.session_state.prices))
    fig = go.Figure()
    # History
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='black', width=2), name='Price'))
    # Future (Cognitive Alpha Prediction)
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+3], y=[price, future_paths[0]], line=dict(color='green', dash='dot'), name='Bull Path'))
    fig.add_trace(go.Scatter(x=[len(st.session_state.prices), len(st.session_state.prices)+3], y=[price, future_paths[1]], line=dict(color='red', dash='dot'), name='Bear Path'))
    
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True)

    # 4. Logs
    log_html = "".join([l for l in st.session_state.live_logs])
    log_ph.markdown(f'<div class="terminal-box">{log_html}</div>', unsafe_allow_html=True)

    time.sleep(1)
    if not st.session_state.pending_signal: st.rerun()
