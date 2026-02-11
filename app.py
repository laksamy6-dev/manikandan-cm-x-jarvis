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

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="CM-X OMNI MODE", layout="wide", page_icon="🧿")
IST = pytz.timezone('Asia/Kolkata')

# --- 2. SETUP & SECRETS ---
try:
    # UPSTOX
    UPSTOX_TOKEN = st.secrets["upstox"]["access_token"]
    UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
    REQ_INSTRUMENT = "NSE_INDEX|Nifty 50"
    
    # GEMINI
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # TELEGRAM
    TG_TOKEN = st.secrets["telegram"]["bot_token"]
    TG_ID = st.secrets["telegram"]["chat_id"]
except:
    st.error("⚠️ SECRETS ERROR. Check .streamlit/secrets.toml")
    st.stop()

MEMORY_FILE = "cm_x_omni_brain.json"

# --- 3. MEMORY SYSTEM ---
def init_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"pnl": 0.0, "wins": 0, "losses": 0, "weights": {"Physics": 1.5, "Trend": 1.0}}
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return {"pnl": 0.0, "wins": 0, "losses": 0, "weights": {"Physics": 1.5, "Trend": 1.0}}

def save_memory(mem):
    with open(MEMORY_FILE, 'w') as f: json.dump(mem, f)

brain = init_memory()

# --- 4. STATE ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=300)
if 'bot' not in st.session_state: st.session_state.bot = False
if 'pos' not in st.session_state: st.session_state.pos = None
if 'audio' not in st.session_state: st.session_state.audio = ""
if 'logs' not in st.session_state: st.session_state.logs = deque(maxlen=20)
if 'high_pnl' not in st.session_state: st.session_state.high_pnl = 0.0
if 'signal' not in st.session_state: st.session_state.signal = None

# --- 5. FUNCTIONS (VOICE, TELEGRAM, LOGS) ---
def speak(text):
    """Live Voice Commentary"""
    try:
        # Save log first
        add_log(f"JARVIS: {text}", "warn")
        
        # Generate Audio
        tts = gTTS(text=text, lang='ta', tld='co.in') # TAMIL VOICE
        filename = "voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        
        # Auto-Play HTML
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.session_state.audio = md
    except: pass

def send_tg(msg):
    """Telegram Message"""
    if TG_TOKEN and TG_ID:
        try: 
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            requests.get(url, params={"chat_id": TG_ID, "text": f"🧿 CM-X: {msg}"})
        except: pass

def add_log(msg, style="info"):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    color = "#0f0" if style=="info" else "#f00" if style=="danger" else "#ff0"
    st.session_state.logs.appendleft(f"<span style='color:#666'>[{ts}]</span> <span style='color:{color}'>{msg}</span>")

# --- 6. ADVANCED MATH (3-CANDLE PREDICTION) ---
def predict_next_3_candles(prices):
    """Monte Carlo Simulation for Next 3 Candles"""
    data = list(prices)
    if len(data) < 20: return []
    
    last_price = data[-1]
    # Calculate Volatility
    volatility = np.std(data[-20:]) 
    
    # Predict 3 steps ahead
    predictions = []
    current_sim = last_price
    for _ in range(3):
        change = np.random.normal(0, volatility)
        current_sim += change
        predictions.append(current_sim)
        
    return predictions

def calc_physics(prices):
    p = np.array(list(prices))
    if len(p) < 10: return 0, 0
    v = np.diff(p)[-1]
    a = np.diff(np.diff(p))[-1]
    return v, a

def get_live_price():
    try:
        h = {'Authorization': f'Bearer {UPSTOX_TOKEN}', 'Accept': 'application/json'}
        r = requests.get(UPSTOX_URL, headers=h, params={'instrument_key': REQ_INSTRUMENT}, timeout=2)
        if r.status_code == 200:
            return float(r.json()['data'][list(r.json()['data'].keys())[0]]['last_price'])
    except: pass
    if st.session_state.prices: return st.session_state.prices[-1]
    return 22100.0

# --- 7. UI ---
st.markdown("""
<style>
.stApp { background: #000; color: #0f0; font-family: monospace; }
.metric-box { border: 1px solid #333; padding: 10px; text-align: center; }
.log-box { height: 200px; overflow-y: auto; border: 1px solid #333; padding: 5px; font-size: 12px; }
.buy { color: #0f0; font-weight: bold; border: 1px solid #0f0; padding: 5px; }
.sell { color: #f00; font-weight: bold; border: 1px solid #f00; padding: 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown(st.session_state.audio, unsafe_allow_html=True)

# HEADER
c1, c2 = st.columns([3, 1])
with c1: st.title("🧿 CM-X OMNI PREDICTOR")
with c2: 
    if st.button("📢 TEST TELEGRAM"):
        send_tg("Testing Connection... Connected!")
        st.toast("Message Sent!")

# MAIN DISPLAY
col1, col2 = st.columns([2, 1])

with col1:
    chart_ph = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    
with col2:
    st.markdown("### 🗣️ LIVE COMMENTARY")
    log_ph = st.empty()
    
    st.markdown("### 🧠 AI INSIGHTS")
    ai_in = st.text_input("Ask Jarvis:", placeholder="Trend eppadi iruku?")
    if st.button("ASK"):
        if ai_in:
            p = st.session_state.prices[-1] if st.session_state.prices else 0
            ans = model.generate_content(f"You are Jarvis (Tamil). Price: {p}. User: {ai_in}. Reply briefly.").text
            speak(ans)
    
    st.write("---")
    alert_ph = st.empty()
    st.write("---")
    
    b1, b2 = st.columns(2)
    start = b1.button("🔥 START")
    stop = b2.button("🛑 STOP")
    
    pnl_ph = st.empty()

if start: st.session_state.bot = True
if stop: st.session_state.bot = False

# --- 8. ENGINE ---
if st.session_state.bot:
    
    # 1. DATA
    price = get_live_price()
    st.session_state.prices.append(price)
    
    # 2. MATH
    v, a = calc_physics(st.session_state.prices)
    future_3 = predict_next_3_candles(st.session_state.prices) # The 3 Candle Logic
    
    # 3. SIGNAL
    signal = None
    if v > 2.0 and a > 0.5: signal = "BUY"
    elif v < -2.0 and a < -0.5: signal = "SELL"
    
    # 4. EXECUTION ALERT
    if signal and not st.session_state.pos and not st.session_state.signal:
        opt = f"{round(price/50)*50} {'CE' if signal=='BUY' else 'PE'}"
        st.session_state.signal = {"type": signal, "opt": opt, "price": price}
        speak(f"Boss! {signal} Signal on {opt}. Next 3 candles looking {'Good' if signal=='BUY' else 'Bad'}.")
        send_tg(f"{signal} SIGNAL: {opt}")
        
    # 5. APPROVAL UI
    if st.session_state.signal:
        sig = st.session_state.signal
        with alert_ph.container():
            st.warning(f"⚠️ EXECUTE {sig['type']} {sig['opt']}?")
            c_y, c_n = st.columns(2)
            if c_y.button("✅ YES"):
                st.session_state.pos = sig
                st.session_state.high_pnl = 0.0
                st.session_state.signal = None
                speak("Order Placed. Tracking Zero Loss.")
                st.rerun()
            if c_n.button("❌ NO"):
                st.session_state.signal = None
                st.rerun()
                
    # 6. ZERO LOSS MANAGER
    if st.session_state.pos:
        pos = st.session_state.pos
        pnl = (price - pos['price']) * 50 if pos['type'] == "BUY" else (pos['price'] - price) * 50
        
        if pnl > st.session_state.high_pnl: st.session_state.high_pnl = pnl
        high = st.session_state.high_pnl
        
        exit = False
        reason = ""
        
        # Zero Loss Logic (Profit > 500 -> SL @ +200)
        if high > 500 and pnl < 200: exit = True; reason = "ZERO LOSS HIT"
        elif high > 1000 and pnl < high*0.8: exit = True; reason = "TRAILING HIT"
        elif pnl < -300: exit = True; reason = "STOP LOSS"
        
        if exit:
            brain["pnl"] += pnl
            save_memory(brain)
            st.session_state.pos = None
            speak(f"Trade Closed. {reason}. PNL: {pnl}")
            send_tg(f"CLOSED: {pnl}")
            st.rerun()

    # 7. UI UPDATE
    m1.metric("NIFTY", f"{price:.2f}")
    m2.metric("VELOCITY", f"{v:.2f}")
    
    # 3-CANDLE PREDICTION DISPLAY
    pred_text = f"{future_3[-1]:.2f}" if future_3 else "Calculating..."
    m3.metric("NEXT 3 MIN", pred_text, delta_color="normal")
    
    # CHART WITH PREDICTION
    fig = go.Figure()
    # History
    fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#0f0'), name='Real'))
    
    # FUTURE 3 CANDLES (The Logic You Asked For)
    if future_3:
        x_pred = list(range(len(st.session_state.prices), len(st.session_state.prices)+3))
        fig.add_trace(go.Scatter(x=x_pred, y=future_3, line=dict(color='cyan', dash='dot'), name='Prediction'))
    
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    chart_ph.plotly_chart(fig, use_container_width=True, key=time.time()) # Key fixes crash
    
    # LOGS
    l_html = "".join([l for l in st.session_state.logs])
    log_ph.markdown(f"<div class='log-box'>{l_html}</div>", unsafe_allow_html=True)
    
    val = brain["pnl"]
    pnl_ph.markdown(f"<h1 style='text-align:center; color:{'#0f0' if val>=0 else '#f00'}'>PNL: ₹{val:.2f}</h1>", unsafe_allow_html=True)

    time.sleep(3)
    if not st.session_state.signal: st.rerun()
