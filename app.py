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
from scipy.stats import entropy as scipy_entropy

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="CM-X MEGA PREDATOR",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# TIMEZONE (IST)
IST = pytz.timezone('Asia/Kolkata')

# --- 2. SECRETS & API ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    if "telegram" in st.secrets:
        TG_TOKEN = st.secrets["telegram"]["bot_token"]
        TG_ID = st.secrets["telegram"]["chat_id"]
    else: TG_TOKEN = None; TG_ID = None
        
    genai.configure(api_key=GEMINI_API_KEY)
    
except Exception as e:
    st.error(f"SECRETS ERROR: {e}")
    st.stop()

UPSTOX_URL = 'https://api.upstox.com/v2/market-quote/ltp'
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 300

# --- 3. STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css?family=Fira+Code&display=swap');
    .stApp { background-color: #050505; color: #e2e8f0; font-family: 'Fira Code', monospace; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #f8fafc; text-shadow: 0 0 10px rgba(0, 255, 65, 0.3); }
    div[data-testid="stMetric"] { background-color: #0a0a0a; border: 1px solid #333; border-radius: 10px; padding: 10px; box-shadow: 0 0 10px rgba(0, 255, 65, 0.1); }
    div[data-testid="stMetricValue"] { color: #00ff41; font-family: 'Orbitron'; font-size: 26px; text-shadow: 0 0 5px #00ff41; }
    .terminal-box { font-family: 'Fira Code', monospace; background-color: #000; color: #00ff41; padding: 15px; height: 250px; overflow-y: auto; border: 1px solid #333; }
    .stButton>button { background-color: #000; color: #00ff41; border: 1px solid #00ff41; font-family: 'Orbitron'; height: 50px; width: 100%; transition: 0.3s; }
    .stButton>button:hover { background-color: #00ff41; color: #000; box-shadow: 0 0 15px #00ff41; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. MEMORY ---
def init_brain():
    if not os.path.exists(MEMORY_FILE):
        return {"total_pnl": 0.0, "wins": 0, "losses": 0, "order_book": [], "weights": {"Physics": 1.5, "Trend": 1.0, "Global": 1.2, "Chaos": 0.8}, "global_sentiment": "NEUTRAL"}
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except: return init_brain()

def save_brain(mem):
    with open(MEMORY_FILE, 'w') as f: json.dump(mem, f, indent=4)

brain_memory = init_brain()

# --- 5. STATE ---
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = False
if 'position' not in st.session_state: st.session_state.position = None
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = None
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=50)
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = 0.0

# --- 6. AUDIO & LOGS ---
def speak_aether(text):
    try:
        add_log(f"JARVIS: {text}", "log-ai")
        tts = gTTS(text=text, lang='ta', tld='co.in') 
        filename = "aether_voice.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except: pass

def add_log(msg, css_class="log-info"):
    ts = datetime.now(IST).strftime("%H:%M:%S")
    color = "#0f0"
    if "SELL" in msg: color = "#f00"
    if "JARVIS" in msg: color = "#fc0"
    entry = f"<span style='color:#666'>[{ts}]</span> <span style='color:{color}'>{msg}</span>"
    st.session_state.live_logs.appendleft(entry)

def send_telegram(msg):
    if TG_TOKEN and TG_ID:
        try: requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", params={"chat_id": TG_ID, "text": f"🧬 CM-X: {msg}"})
        except: pass

# --- 7. SMART AI ENGINE (AUTO-SWITCH FIX) ---
def ask_gemini(prompt):
    """Tries multiple models to avoid 404 Errors"""
    models_to_try = ['gemini-2.0-flash-lite', 'gemini-2.0-flash']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            continue # Try next model
            
    return "AI Error: All models busy."

# --- 8. COGNITIVE ENGINE ---
class CognitiveEngine:
    def calculate_newton_metrics(self, prices_deque):
        p = np.array(list(prices_deque))
        if len(p) < 10: return 0.0, 0.0, 0.0
        v = p[-1] - p[-2]
        a = (p[-1] - p[-2]) - (p[-2] - p[-3])
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
            for _ in range(5):
                sim_p *= (1 + np.random.normal(mu, sigma))
            if sim_p > last: bull_paths += 1
        return bull_paths / num_sims

    def get_best_option_strike(self, spot, direction):
        strike = round(spot/50)*50
        return f"{strike} CE" if direction == "BUY" else f"{strike} PE"

aether_engine = CognitiveEngine()

# --- 9. LIVE DATA ---
def get_live_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    try:
        res = requests.get(UPSTOX_URL, headers=headers, params={'instrument_key': REQ_INSTRUMENT_KEY}, timeout=3)
        if res.status_code == 200:
            data = res.json()['data']
            key = list(data.keys())[0]
            return float(data[key]['last_price'])
    except: pass
    return None

# --- 10. UI LAYOUT ---
st.markdown(f"""
<div style="text-align: center; border-bottom: 2px solid #00ff41; padding-bottom: 10px;">
    <h1>AETHER: FUSION GOD MODE</h1>
    <p style="color:#888;">OPERATOR: {OWNER_NAME} | SYSTEM: ONLINE</p>
</div>
""", unsafe_allow_html=True)
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

c1, c2 = st.columns([2.5, 1])

with c1:
    st.subheader("📡 QUANTUM TRAJECTORY")
    chart_ph = st.empty()
    
    m1, m2, m3, m4 = st.columns(4)
    p_met = m1.empty(); v_met = m2.empty(); a_met = m3.empty(); e_met = m4.empty()
    
    st.subheader("🏛️ THE COUNCIL CHAMBER")
    council_ph = st.empty()
    
    st.subheader("🖥️ NEURAL LOGS")
    log_ph = st.empty()
    
    # --- ORDER BOOK (FIXED) ---
    st.subheader("📖 ORDER BOOK")
    if brain_memory["order_book"]:
        # Show last 5 trades reversely
        order_df = pd.DataFrame(brain_memory["order_book"][::-1])
        st.dataframe(order_df, use_container_width=True, hide_index=True)
    else:
        st.info("No Trades Yet")

with c2:
    st.subheader("👻 JARVIS LINK")
    user_q = st.text_input("Consult Jarvis:", placeholder="Type here...")
    if st.button("SEND MESSAGE"):
        if user_q:
            add_log(f"BOSS: {user_q}", "info")
            p = st.session_state.prices[-1] if st.session_state.prices else 0
            # Call Smart AI Switcher
            ans = ask_gemini(f"You are Jarvis (Tamil). Price: {p}. User: {user_q}. Reply briefly in Tamil.")
            speak_aether(ans)

    st.write("---")
    curr_sent = brain_memory.get("global_sentiment", "NEUTRAL")
    new_sent = st.select_slider("Global Sentiment", ["BEARISH", "NEUTRAL", "BULLISH"], value=curr_sent)
    if new_sent != curr_sent:
        brain_memory["global_sentiment"] = new_sent
        save_brain(brain_memory)
        
    pnl_ph = st.empty()
    
    b1, b2 = st.columns(2)
    start = b1.button("🔥 INITIATE")
    stop = b2.button("🛑 KILL SWITCH")
    
    approval_ph = st.empty()

if start: st.session_state.bot_active = True
if stop: st.session_state.bot_active = False

# --- 11. MAIN LOOP ---
if st.session_state.bot_active:
    
    price = get_live_market_data()
    if price:
        st.session_state.prices.append(price)
        
        v, a, ent = aether_engine.calculate_newton_metrics(st.session_state.prices)
        
        # VOTES
        votes = {}
        if v > 1.5 and a > 0.3: votes['Physics'] = "BUY"
        elif v < -1.5 and a < -0.3: votes['Physics'] = "SELL"
        else: votes['Physics'] = "WAIT"
        
        if len(st.session_state.prices) > 20:
            ma = np.mean(list(st.session_state.prices)[-20:])
            if price > ma: votes['Trend'] = "BUY"
            else: votes['Trend'] = "SELL"
            
        gs = brain_memory["global_sentiment"]
        if gs == "BULLISH": votes['Global'] = "BUY"
        elif gs == "BEARISH": votes['Global'] = "SELL"
        else: votes['Global'] = "WAIT"
        
        # SIGNAL
        if ent > 1.5: votes['Chaos'] = "RISKY"
        else:
            votes['Chaos'] = "GO"
            buy_score = list(votes.values()).count("BUY")
            sell_score = list(votes.values()).count("SELL")
            
            if buy_score >= 2 and not st.session_state.position and not st.session_state.pending_signal:
                opt = aether_engine.get_best_option_strike(price, "BUY")
                st.session_state.pending_signal = {"type": "BUY", "opt": opt}
                speak_aether(f"Boss! Buy Signal on {opt}. Approval venum.")
                send_telegram(f"BUY ALERT: {opt}")
                
            elif sell_score >= 2 and not st.session_state.position and not st.session_state.pending_signal:
                opt = aether_engine.get_best_option_strike(price, "SELL")
                st.session_state.pending_signal = {"type": "SELL", "opt": opt}
                speak_aether(f"Boss! Sell Signal on {opt}. Approval venum.")
                send_telegram(f"SELL ALERT: {opt}")

        # APPROVAL
        if st.session_state.pending_signal:
            sig = st.session_state.pending_signal
            with approval_ph.container():
                st.warning(f"⚠️ EXECUTE {sig['opt']}?")
                c1, c2 = st.columns(2)
                if c1.button("✅ YES"):
                    st.session_state.position = {"type": sig['type'], "entry": price, "opt": sig['opt']}
                    st.session_state.trailing_high = 0.0
                    st.session_state.pending_signal = None
                    
                    # Log Order
                    order = {"Time": datetime.now(IST).strftime("%H:%M:%S"), "Type": sig['type'], "Price": price, "Status": "OPEN", "PNL": 0}
                    brain_memory["order_book"].append(order)
                    save_brain(brain_memory)
                    
                    speak_aether("Order Placed.")
                    st.rerun()
                if c2.button("❌ NO"):
                    st.session_state.pending_signal = None
                    st.rerun()

        # TRAILING & EXIT
        if st.session_state.position:
            pos = st.session_state.position
            pnl = (price - pos['entry']) * 50 if pos['type'] == "BUY" else (pos['entry'] - price) * 50
            
            if pnl > st.session_state.trailing_high: st.session_state.trailing_high = pnl
            high = st.session_state.trailing_high
            
            exit = False; reason = ""
            if high > 500 and pnl < 200: exit = True; reason = "ZERO LOSS HIT"
            elif high > 1000 and pnl < high*0.7: exit = True; reason = "TRAILING HIT"
            elif pnl < -300: exit = True; reason = "STOP LOSS"
            
            if exit:
                st.session_state.position = None
                brain_memory["total_pnl"] += pnl
                
                # Update Order Book
                if brain_memory["order_book"]:
                    brain_memory["order_book"][-1]["Status"] = "CLOSED"
                    brain_memory["order_book"][-1]["PNL"] = pnl
                
                save_brain(brain_memory)
                speak_aether(f"Trade Closed. {reason}. PNL: {pnl}")
                send_telegram(f"EXIT: {pnl}")
                st.rerun()

        # UI UPDATE
        p_met.metric("NIFTY 50", f"{price:,.2f}", f"{v:.2f}")
        v_met.metric("VELOCITY", f"{v:.2f}")
        a_met.metric("ACCEL", f"{a:.2f}")
        e_met.metric("ENTROPY", f"{ent:.2f}")
        
        with council_ph.container():
            cols = st.columns(4)
            i = 0
            for k, v in votes.items():
                cols[i].markdown(f"<div style='border:1px solid #333; text-align:center; color:{'#0f0' if v=='BUY' else '#f00'};'>{k}<br>{v}</div>", unsafe_allow_html=True)
                i+=1
                if i>=4: break

        val = brain_memory["total_pnl"]
        pnl_ph.markdown(f"<h2 style='text-align:center; color:{'#0f0' if val>=0 else '#f00'}'>PNL: ₹{val:,.2f}</h2>", unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#00ff41', width=2)))
        if st.session_state.position:
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange")
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")
        
        l_html = "".join([l for l in st.session_state.live_logs])
        log_ph.markdown(f"<div class='terminal-box'>{l_html}</div>", unsafe_allow_html=True)
        
        time.sleep(1)
        st.rerun()
