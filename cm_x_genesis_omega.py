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
    page_title="PROJECT AETHER: GOD MODE",
    layout="wide",
    page_icon="👻",
    initial_sidebar_state="collapsed"
)

# AETHER SYSTEM CONSTANTS
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 126 
KILL_SWITCH_LOSS = -2000 

# --- 2. ADVANCED CYBERPUNK STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
    
    .stApp { background-color: #050505; color: #00ff41; font-family: 'Courier New', monospace; }
    
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
    
    /* Live Terminal Log */
    .terminal-box {
        font-family: 'Fira Code', monospace;
        background-color: #000;
        border: 1px solid #333;
        color: #00ff41;
        padding: 10px;
        height: 200px;
        overflow-y: auto;
        font-size: 14px;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.2);
    }
    .log-time { color: #888; margin-right: 10px; }
    .log-info { color: #00ff41; }
    .log-warn { color: #ffff00; }
    .log-danger { color: #ff0000; }
    
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
    
    /* Input Box Styling */
    .stTextInput>div>div>input {
        background-color: #111;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Fira Code', monospace;
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
    model = genai.GenerativeModel('gemini-1.5-flash')
    
except Exception as e:
    st.error(f"⚠️ SYSTEM FAILURE: Secrets Error - {e}")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
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
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = brain['position']
if 'orders' not in st.session_state: st.session_state.orders = brain['orders']
if 'pnl' not in st.session_state: st.session_state.pnl = brain['pnl']
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=20) # Keep last 20 logs

# --- 5. LOGGING SYSTEM (NEW FEATURE) ---
def add_log(msg, type="info"):
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%H:%M:%S")
    color_class = "log-info"
    if type == "warn": color_class = "log-warn"
    if type == "danger": color_class = "log-danger"
    
    log_entry = f"<span class='log-time'>[{timestamp}]</span> <span class='{color_class}'>{msg}</span>"
    st.session_state.live_logs.appendleft(log_entry)

# --- 6. AUDIO ENGINE (THE GHOST VOICE) ---
def speak_aether(text):
    """Converts text to speech and plays it in the browser"""
    try:
        brain['last_thought'] = text
        add_log(f"JARVIS: {text}", "warn")
        
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

# --- 7. QUANTUM PHYSICS CORE (126-Period Momentum) ---
def calculate_quantum_state(price_deque):
    if len(price_deque) < 10: return 0, 0, 0, 0
    
    p = np.array(price_deque)
    velocity = np.diff(p)[-1]
    acceleration = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    energy = abs(velocity) * 100 
    entropy = np.std(p[-10:])
    
    return velocity, acceleration, energy, entropy

# --- 8. GEMINI AI (THE GHOST PERSONA) ---
def consult_ghost_brain(query, price):
    try:
        prompt = f"""
        You are 'AETHER', a high-frequency trading AI.
        Current Price: {price}.
        User asks: {query}.
        Reply shortly and boldly as a trading assistant.
        """
        response = model.generate_content(prompt)
        return response.text
    except: return "Connection to the Void lost..."

# --- 9. REAL DATA FETCHING ---
def get_live_market_data():
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
                # Fallback
                first_key = list(resp_data.keys())[0]
                return float(resp_data[first_key]['last_price']), "CONNECTED"
                
            return None, "DATA STRUCTURE ERROR"
        elif response.status_code == 401:
            return None, "TOKEN EXPIRED"
        else:
            return None, f"API ERROR {response.status_code}"
    except Exception as e:
        return None, "NET ERROR"

# --- 10. UI LAYOUT ---
st.markdown(f"""
<div style="text-align: center;">
    <h1>PROJECT AETHER: GOD MODE</h1>
    <p style="color: #00ff41;">OPERATOR: {OWNER_NAME} | ENTITY: <b>ONLINE</b> | MEMORY: <b>ATTACHED</b></p>
</div>
""", unsafe_allow_html=True)

# Invisible Audio Player
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

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
    
    # LIVE TERMINAL LOG
    st.write("---")
    st.subheader("🖥️ LIVE SYSTEM TERMINAL")
    log_ph = st.empty()

with c2:
    st.subheader("👻 Ghost Protocol")
    
    # --- NEW FEATURE: USER INPUT BAR ---
    st.markdown("##### 🗣️ TALK TO JARVIS")
    user_query = st.text_input("Message:", placeholder="Ask anything...", label_visibility="collapsed")
    if st.button("SEND MESSAGE"):
        if user_query:
            # Get latest price for context
            curr_p = st.session_state.prices[-1] if st.session_state.prices else 0
            
            add_log(f"BOSS: {user_query}", "info")
            reply = consult_ghost_brain(user_query, curr_p)
            speak_aether(reply)
            
    st.write("---")
    pnl_display = st.empty()
    
    # Controls
    auto_col1, auto_col2 = st.columns(2)
    start_btn = auto_col1.button("🔥 INITIATE SEQUENCE")
    stop_btn = auto_col2.button("🛑 KILL SWITCH")
    
    if st.button("❌ EMERGENCY EXIT"):
        if st.session_state.position:
            st.session_state.position = None
            save_aether_memory(None, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Emergency Exit.")
            add_log("EMERGENCY EXIT TRIGGERED", "danger")
            st.rerun()

    st.caption("Black Box Log")
    order_log = st.empty()

# Control Logic
if start_btn: 
    st.session_state.bot_active = True
    add_log("SYSTEM SEQUENCE INITIATED", "info")
if stop_btn: 
    st.session_state.bot_active = False
    add_log("KILL SWITCH ENGAGED - STOPPING", "danger")

# --- 11. MAIN EVENT LOOP ---
if st.session_state.bot_active:
    
    # Connection Check
    p, status = get_live_market_data()
    if status != "CONNECTED":
        st.error("CANNOT START: CHECK CONNECTION")
        st.stop()
        
    while st.session_state.bot_active:
        
        # 1. UPDATE LOGS
        log_content = "".join([f"<div>{l}</div>" for l in st.session_state.live_logs])
        log_ph.markdown(f'<div class="terminal-box">{log_content}</div>', unsafe_allow_html=True)

        # 2. KILL SWITCH CHECK
        if st.session_state.pnl < KILL_SWITCH_LOSS:
            speak_aether("Critical Failure. Kill Switch Activated.")
            st.session_state.bot_active = False
            add_log("MAX LOSS REACHED. SHUTTING DOWN.", "danger")
            break

        # 3. FETCH DATA
        price, status = get_live_market_data()
        
        if status != "CONNECTED":
            add_log(f"Signal Lost: {status}", "danger")
            time.sleep(2)
            continue
            
        st.session_state.prices.append(price)
        
        # 4. PHYSICS
        v, a, energy, entropy = calculate_quantum_state(st.session_state.prices)
        
        # Log Logic (Every few ticks to avoid spam)
        if len(st.session_state.prices) % 5 == 0:
            add_log(f"Scanning: P={price} V={v:.2f} A={a:.2f} Ent={entropy:.2f}")
        
        # 5. DECISION MATRIX
        if st.session_state.position is None:
            # BUY
            if v > 1.5 and a > 0.3 and entropy < 10:
                st.session_state.position = {"type": "BUY", "entry": price, "qty": 50}
                msg = f"BUY DETECTED @ {price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                save_aether_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Bullish Entry")
                speak_aether("Momentum Detected. Buying.")
                add_log("EXECUTING BUY ORDER", "warn")
                st.rerun()
                
            # SELL
            elif v < -1.5 and a < -0.3 and entropy < 10:
                st.session_state.position = {"type": "SELL", "entry": price, "qty": 50}
                msg = f"SELL DETECTED @ {price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                save_aether_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], "Bearish Entry")
                speak_aether("Gravity Increasing. Selling.")
                add_log("EXECUTING SELL ORDER", "warn")
                st.rerun()
                
        else:
            # Exit Logic
            pos = st.session_state.position
            curr_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
            
            # Physics Reversal
            physics_exit = (pos['type'] == "BUY" and v < -1.0) or (pos['type'] == "SELL" and v > 1.0)
            
            if curr_pnl > 500 or curr_pnl < -300 or physics_exit:
                st.session_state.pnl += curr_pnl
                res = "WIN" if curr_pnl > 0 else "LOSS"
                msg = f"CLOSED {pos['type']} | P&L: {curr_pnl:.0f}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                brain['daily_stats']['wins' if res == "WIN" else 'losses'] += 1
                
                st.session_state.position = None
                save_aether_memory(None, st.session_state.orders, st.session_state.pnl, brain['daily_stats'], f"Trade Closed. {res}")
                
                if res == "WIN": speak_aether("Profit Secured.")
                else: speak_aether("Stop Loss Hit.")
                add_log(f"POSITION CLOSED. RESULT: {res}", "warn")
                st.rerun()

        # 6. UI UPDATES
        p_metric.metric("NIFTY 50", f"{price:,.2f}", f"{v:.2f}")
        v_metric.metric("VELOCITY", f"{v:.2f}")
        a_metric.metric("ACCEL", f"{a:.2f}")
        e_metric.metric("ENTROPY", f"{entropy:.2f}")
        
        live_pnl = 0.0
        if st.session_state.position:
             pos = st.session_state.position
             live_pnl = (price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - price) * pos['qty']
        
        total_pnl = st.session_state.pnl + live_pnl
        color = "#00ff41" if total_pnl >= 0 else "#ff0000"
        pnl_display.markdown(f"<h2 style='color:{color}; text-align:center; border: 1px solid {color}; padding: 10px;'>P&L: ₹{total_pnl:.2f}</h2>", unsafe_allow_html=True)
        
        if st.session_state.orders:
            order_log.dataframe(pd.DataFrame(st.session_state.orders), hide_index=True)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
        if st.session_state.position:
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange")
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        # --- FIX: ADDED KEY TO PREVENT CRASH ---
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
        
        time.sleep(1)
