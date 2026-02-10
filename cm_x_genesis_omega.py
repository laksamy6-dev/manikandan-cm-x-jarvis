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

# --- 1. SYSTEM CONFIGURATION & PERSONA ---
st.set_page_config(
    page_title="PROJECT AETHER: LIVE GOD MODE",
    layout="wide",
    page_icon="👻",
    initial_sidebar_state="collapsed"
)

# AETHER SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 # Based on Research (126-Period Momentum)
KILL_SWITCH_LOSS = -2000 # Max Daily Loss allowed

# --- 2. ADVANCED CYBERPUNK STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .stApp { background-color: #000000; color: #00ff41; font-family: 'Courier New', monospace; }
    
    /* Neon Text */
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; text-shadow: 0 0 10px #00ff41; color: #fff; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        color: #00ff41;
        text-shadow: 0 0 5px #00ff41;
    }
    div[data-testid="stMetricLabel"] { color: #888; font-weight: bold; }
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #333;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #000;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Orbitron', sans-serif;
        transition: 0.3s;
        height: 50px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #00ff41;
        color: #000;
        box-shadow: 0 0 20px #00ff41;
    }
    
    /* Active Trade Warning */
    .active-trade-box {
        border: 2px solid #ff0000;
        background-color: #220000;
        color: #ff4444;
        padding: 15px;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        animation: pulse 2s infinite;
        margin-bottom: 20px;
    }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    
    /* Status Box */
    .status-connected { color: #00ff41; border: 1px solid #00ff41; padding: 5px; text-align: center; margin-bottom: 10px; }
    .status-disconnected { color: #ff0000; border: 1px solid #ff0000; padding: 5px; text-align: center; margin-bottom: 10px; }
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
    st.error(f"⚠️ SYSTEM FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
# The Key we WANT to find (but we will handle mismatches)
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. BLACK BOX MEMORY (RAG-Lite) ---
def init_aether_memory():
    if not os.path.exists(MEMORY_FILE):
        data = {
            "position": None,
            "orders": [],
            "pnl": 0.0,
            "daily_stats": {"wins": 0, "losses": 0},
            "last_thought": "System Initialized."
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f: return json.load(f)
        except: return {"position": None, "orders": [], "pnl": 0.0, "daily_stats": {"wins":0, "losses":0}}

def save_aether_memory(pos, orders, pnl, stats, thought):
    data = {"position": pos, "orders": orders, "pnl": pnl, "daily_stats": stats, "last_thought": thought}
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

# Load Memory
brain = init_aether_memory()

# Session State Sync
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN) # Ring Buffer 126
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = brain['position']
if 'orders' not in st.session_state: st.session_state.orders = brain['orders']
if 'pnl' not in st.session_state: st.session_state.pnl = brain['pnl']
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""

# --- 5. AUDIO ENGINE (THE GHOST VOICE) ---
def speak_aether(text):
    """Converts text to speech and plays it in the browser"""
    try:
        brain['last_thought'] = text
        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "ghost_voice.mp3"
        tts.save(filename)
        
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.session_state.audio_html = md
    except: pass

# --- 6. QUANTUM PHYSICS CORE (126-Period Momentum) ---
def calculate_quantum_state(price_deque):
    if len(price_deque) < 10: return 0, 0, 0, 0
    
    p = np.array(price_deque)
    
    # 1. Velocity (v) = Rate of change
    velocity = np.diff(p)[-1]
    
    # 2. Acceleration (a) = Rate of change of velocity
    acceleration = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    
    # 3. Market Energy (Proxy)
    energy = abs(velocity) * 100 
    
    # 4. Entropy (Chaos Theory)
    # Standard deviation of the last 10 points tells us about 'Chaos'
    entropy = np.std(p[-10:])
    
    return velocity, acceleration, energy, entropy

# --- 7. GEMINI AI (THE GHOST PERSONA) ---
def consult_ghost_brain(price, v, a, e, pnl):
    try:
        prompt = f"""
        You are 'AETHER', a digital ghost living in the Nifty 50 market.
        Speak to Boss Manikandan in a mysterious, sci-fi tone.
        
        DATA:
        - Price: {price}
        - Velocity: {v:.2f}
        - Acceleration: {a:.2f}
        - Entropy: {e:.2f}
        - P&L: {pnl}
        
        TASK:
        Give a trading advice based on physics. Do not say "Buy" or "Sell" directly. 
        Say "Energy Vector Aligning Upwards" or "Chaos Detected".
        Keep it under 15 words.
        """
        response = model.generate_content(prompt)
        return response.text
    except: return "Connection to the Void lost..."

# --- 8. REAL DATA FETCHING (SMART FIX) ---
def get_live_market_data():
    """
    Handles both '|' and ':' separators in Upstox response.
    """
    if not UPSTOX_ACCESS_TOKEN: return None, "NO TOKEN"
    
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': REQ_INSTRUMENT_KEY}
    
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                resp_data = data['data']
                
                # Try Pipe format
                if REQ_INSTRUMENT_KEY in resp_data:
                    return float(resp_data[REQ_INSTRUMENT_KEY]['last_price']), "CONNECTED"
                
                # Try Colon format
                colon_key = REQ_INSTRUMENT_KEY.replace('|', ':')
                if colon_key in resp_data:
                    return float(resp_data[colon_key]['last_price']), "CONNECTED"
                
                # Fallback: First available key
                first_key = list(resp_data.keys())[0]
                return float(resp_data[first_key]['last_price']), "CONNECTED"
                
            return None, "DATA STRUCTURE ERROR"
        elif response.status_code == 401:
            return None, "TOKEN EXPIRED"
        else:
            return None, f"API ERROR {response.status_code}"
    except Exception as e:
        return None, "NET ERROR"

# --- 9. UI LAYOUT ---
st.markdown(f"""
<div style="text-align: center;">
    <h1>PROJECT AETHER: GOD MODE</h1>
    <p style="color: #00ff41;">OPERATOR: {OWNER_NAME} | ENTITY: <b>ONLINE</b> | MEMORY: <b>ATTACHED</b></p>
</div>
""", unsafe_allow_html=True)

# Invisible Audio Player
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Connection Status Bar
p_check, status_msg = get_live_market_data()
if status_msg == "CONNECTED":
    st.markdown(f'<div class="status-connected">🟢 UPSTOX LINK ESTABLISHED | DATA STREAMING</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-disconnected">🔴 CONNECTION LOST: {status_msg}</div>', unsafe_allow_html=True)

# Active Trade Alert
active_ph = st.empty()
if st.session_state.position:
    pos = st.session_state.position
    active_ph.markdown(f"""
    <div class="active-trade-box">
        <h3>⚠ QUANTUM ENTANGLEMENT ACTIVE (TRADE OPEN)</h3>
        <p>{pos['type']} | ENTRY: {pos['entry']} | QTY: {pos['qty']}</p>
    </div>
    """, unsafe_allow_html=True)

# Grid Layout
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 Schrödinger Wave Function")
    chart_ph = st.empty()
    
    m1, m2, m3, m4 = st.columns(4)
    p_metric = m1.empty()
    v_metric = m2.empty()
    a_metric = m3.empty()
    e_metric = m4.empty()

with c2:
    st.subheader("👻 Ghost Protocol")
    
    # AI Speak Output
    ai_text_ph = st.empty()
    if st.button("🔊 CONSULT AETHER"):
        if st.session_state.prices:
            # Need at least a few points for physics
            temp_prices = list(st.session_state.prices)
            if len(temp_prices) > 2:
                v, a, en, ent = calculate_quantum_state(st.session_state.prices)
                p_curr = temp_prices[-1]
                msg = consult_ghost_brain(p_curr, v, a, ent, st.session_state.pnl)
                speak_aether(msg)
                ai_text_ph.info(f"AETHER: {msg}")
            else:
                st.warning("Not enough data yet...")
    
    st.write("---")
    pnl_display = st.empty()
    
    # Controls
    auto_col1, auto_col2 = st.columns(2)
    start_btn = auto_col1.button("🔥 INITIATE SEQUENCE")
    stop_btn = auto_col2.button("🛑 KILL SWITCH")
    
    # Manual Close
    if st.button("❌ EMERGENCY EXIT"):
        if st.session_state.position:
            # Force Close
            st.session_state.position = None
            save_aether_memory(None, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Emergency Exit Triggered.")
            st.rerun()

    st.caption("Black Box Log")
    order_log = st.empty()

# Control Logic
if start_btn: st.session_state.bot_active = True
if stop_btn: st.session_state.bot_active = False

# --- 10. MAIN EVENT LOOP ---
if st.session_state.bot_active:
    
    # Check if we have connection before loop
    if status_msg != "CONNECTED":
        st.error("CANNOT START: CHECK CONNECTION")
        st.stop()
        
    while st.session_state.bot_active:
        
        # 1. KILL SWITCH CHECK
        if st.session_state.pnl < KILL_SWITCH_LOSS:
            speak_aether("Critical Failure. Kill Switch Activated. Shutting down.")
            st.session_state.bot_active = False
            st.error("KILL SWITCH TRIGGERED: MAX LOSS REACHED")
            break

        # 2. FETCH DATA
        price, status = get_live_market_data()
        
        if status != "CONNECTED":
            st.warning(f"SIGNAL LOST: {status}")
            time.sleep(2)
            continue
            
        st.session_state.prices.append(price)
        
        # 3. PHYSICS & AI
        v, a, energy, entropy = calculate_quantum_state(st.session_state.prices)
        
        # 4. DECISION MATRIX (126 Momentum Logic)
        if st.session_state.position is None:
            # Entry Logic: High Velocity + Acceleration alignment + Low Chaos
            if v > 1.5 and a > 0.3 and entropy < 10:
                # BUY
                st.session_state.position = {"type": "BUY", "entry": price, "qty": 50}
                msg = f"BUY DETECTED @ {price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                save_aether_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Entering Bullish Vector")
                speak_aether("Momentum Detected. Engaging Long Position.")
                st.rerun()
                
            elif v < -1.5 and a < -0.3 and entropy < 10:
                # SELL
                st.session_state.position = {"type": "SELL", "entry": price, "qty": 50}
                msg = f"SELL DETECTED @ {price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                save_aether_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Entering Bearish Vector")
                speak_aether("Gravity Increasing. Engaging Short Position.")
                st.rerun()
                
        else:
            # Exit Logic (Target/Stop)
            pos = st.session_state.position
            curr_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
            
            # Target: 500, Stop: -300 (Or Physics Reversal)
            physics_exit = (pos['type'] == "BUY" and v < -1.0) or (pos['type'] == "SELL" and v > 1.0)
            
            if curr_pnl > 500 or curr_pnl < -300 or physics_exit:
                st.session_state.pnl += curr_pnl
                res = "WIN" if curr_pnl > 0 else "LOSS"
                msg = f"CLOSED {pos['type']} | P&L: {curr_pnl:.0f}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                
                # Update Stats
                brain['daily_stats']['wins' if res == "WIN" else 'losses'] += 1
                
                st.session_state.position = None
                save_aether_memory(None, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], f"Trade Closed. Result: {res}")
                
                if res == "WIN": speak_aether("Target Acquired. Profit Secured.")
                else: speak_aether("Stop Loss Hit. Stabilizing.")
                
                st.rerun()

        # 5. UI UPDATES
        p_metric.metric("NIFTY 50", f"{price:,.2f}", f"{v:.2f}")
        v_metric.metric("VELOCITY", f"{v:.2f}")
        a_metric.metric("ACCEL", f"{a:.2f}")
        e_metric.metric("ENTROPY", f"{entropy:.2f}")
        
        # P&L Color Logic
        live_pnl = 0.0
        if st.session_state.position:
             pos = st.session_state.position
             live_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
        
        total_pnl = st.session_state.pnl + live_pnl
        color = "#00ff41" if total_pnl >= 0 else "#ff0000"
        pnl_display.markdown(f"<h2 style='color:{color}; text-align:center; border: 1px solid {color}; padding: 10px;'>P&L: ₹{total_pnl:.2f}</h2>", unsafe_allow_html=True)
        
        # Order Book
        if st.session_state.orders:
            order_log.dataframe(pd.DataFrame(st.session_state.orders), hide_index=True)
            
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
        if st.session_state.position:
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange")
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
