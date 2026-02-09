import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import requests
import json
import os
import math
import base64
from datetime import datetime
import pytz
from gtts import gTTS
import plotly.graph_objects as go

# ==========================================
# 1. CONFIGURATION & HACKER UI
# ==========================================
st.set_page_config(
    page_title="CM-X: LIVE MARKET GOD",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', monospace; }
    .holo-card { background: rgba(10, 20, 30, 0.95); border: 1px solid #00f3ff; border-radius: 8px; padding: 15px; margin-bottom: 10px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.15); }
    .metric-val { font-size: 24px; font-weight: bold; }
    .text-green { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    .text-red { color: #ff0000; text-shadow: 0 0 10px #ff0000; }
    .text-cyan { color: #00f3ff; text-shadow: 0 0 10px #00f3ff; }
    .terminal-box { background: #050505; border-left: 3px solid #00ff00; padding: 10px; height: 200px; overflow-y: auto; font-size: 11px; color: #00ff00; font-family: 'Courier New'; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. UPSTOX LIVE CONNECTION (THE MISSING LINK)
# ==========================================
def get_live_price():
    """
    இதுதான் ரியல் மார்க்கெட் கனெக்ஷன்.
    Upstox API-யை கூப்பிட்டு Nifty 50 விலையை எடுக்கும்.
    """
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            url = "https://api.upstox.com/v2/market-quote/ltp"
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {st.secrets["UPSTOX_ACCESS_TOKEN"]}'
            }
            # Nifty 50 Instrument Key
            params = {'instrument_key': 'NSE_INDEX|Nifty 50'}
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if data['status'] == 'success':
                # சரியான விலையை எடு
                price = data['data']['NSE_INDEX:Nifty 50']['last_price']
                return float(price), "🟢 LIVE"
            else:
                return None, "🔴 API FAIL"
    except Exception as e:
        return None, f"🔴 ERROR: {str(e)}"
    
    return None, "🔴 NO TOKEN"

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def get_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def send_telegram(msg):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            requests.get(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_BOT_TOKEN']}/sendMessage",
                         params={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg})
    except: pass

def play_voice(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("voice.mp3")
        with open("voice.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        os.remove("voice.mp3")
    except: pass

# ==========================================
# 4. MATH & BRAIN CORE
# ==========================================
class KalmanFilter:
    def __init__(self):
        self.last_estimate = 0
        self.err_estimate = 1
        self.q = 0.01
    def update(self, measurement):
        if self.last_estimate == 0: self.last_estimate = measurement
        kalman_gain = self.err_estimate / (self.err_estimate + 1)
        current_estimate = self.last_estimate + kalman_gain * (measurement - self.last_estimate)
        self.err_estimate = (1.0 - kalman_gain) * self.err_estimate + abs(self.last_estimate - current_estimate) * self.q
        self.last_estimate = current_estimate
        return current_estimate

def calculate_physics(history):
    if len(history) < 3: return 0, 0
    v = history[-1] - history[-2]
    a = v - (history[-2] - history[-3])
    return v, a

def calculate_entropy(data):
    if len(data) < 10: return 0
    hist, _ = np.histogram(data, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def run_monte_carlo(price, volatility, sims=1000):
    wins = 0
    vol = 0.01 if volatility == 0 else volatility
    for _ in range(sims):
        sim_p = price * math.exp((0 - 0.5 * vol**2) + vol * np.random.normal())
        if sim_p > price: wins += 1
    return (wins / sims) * 100

def predict_future(history):
    if len(history) < 20: return history[-1]
    y = np.array(history[-20:])
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    return np.poly1d(z)(len(y) + 10)

# ==========================================
# 5. MEMORY
# ==========================================
MEMORY_FILE = "cm_x_blackbox.json"

def load_blackbox():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f: return json.load(f)
    return {
        "wins": 0, "losses": 0, "accuracy": 0.0, 
        "weights": {"PHYSICS": 1.5, "MATH": 1.2, "FUTURE": 1.5, "CHAOS": 1.0},
        "last_pred": {}, "lessons": []
    }

def save_blackbox(mem):
    with open(MEMORY_FILE, "w") as f: json.dump(mem, f, indent=4)

def self_construction_loop(current_price, memory):
    last = memory.get("last_pred", {})
    if not last.get("time"): return memory
    try:
        last_time = datetime.strptime(last["time"], '%Y-%m-%d %H:%M:%S')
        now = get_time().replace(tzinfo=None)
        if (now - last_time).total_seconds() / 60 >= 1:
            old_price = last["price"]
            decision = last["decision"]
            result = "NEUTRAL"
            if "BUY" in decision: result = "WIN" if current_price > old_price else "LOSS"
            elif "SELL" in decision: result = "WIN" if current_price < old_price else "LOSS"
            
            if result == "WIN":
                memory["wins"] += 1; memory["weights"]["FUTURE"] += 0.05
            elif result == "LOSS":
                memory["losses"] += 1; memory["weights"]["FUTURE"] -= 0.05
            
            total = memory["wins"] + memory["losses"]
            memory["accuracy"] = round((memory["wins"]/total)*100, 1) if total > 0 else 0
            memory["last_pred"] = {}
            save_blackbox(memory)
    except: pass
    return memory

class ChellakiliBrain:
    def __init__(self):
        self.model = None
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            self.model = genai.GenerativeModel('gemini-2.0-flash')

    def consult_10000_traders(self, prompt):
        if not self.model: return "AI Offline"
        try:
            full_prompt = f"Act as Collective Intelligence of 10,000 Elite Traders. You are Chellakili. {prompt}. Answer in Tanglish."
            res = self.model.generate_content(full_prompt)
            return res.text
        except: return "Neural Link Unstable"

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

# Init
if 'auth' not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    pwd = st.text_input("🔐 ACCESS CODE", type="password")
    if st.button("UNLOCK"):
        if pwd == "boss": st.session_state.auth = True; st.rerun()
    st.stop()

if 'history' not in st.session_state: st.session_state.history = []
if 'memory' not in st.session_state: st.session_state.memory = load_blackbox()
if 'kf' not in st.session_state: st.session_state.kf = KalmanFilter()
if 'brain' not in st.session_state: st.session_state.brain = ChellakiliBrain()

# Sidebar
with st.sidebar:
    st.header("⚙️ CONTROLS")
    speed_mode = st.toggle("⚡ Hyper-Speed", value=True)
    voice_on = st.toggle("🔊 Voice", value=True)
    if st.button("Test Telegram"): send_telegram("Live Market Online.")

# --- GET LIVE DATA ---
live_price, status = get_live_price()

# இணைப்பு கிடைக்கவில்லை என்றால் பழைய முறையை (Simulation) பயன்படுத்து
if live_price is None:
    # Fallback to simulation if market closed or token expired
    if len(st.session_state.history) > 0:
        live_price = st.session_state.history[-1] + np.random.randint(-10, 15)
    else:
        live_price = 19500.0 # Default start
    status_color = "text-red"
    status_text = f"SIMULATION ({status})"
else:
    status_color = "text-green"
    status_text = f"LIVE MARKET ({status})"

# Update History
st.session_state.history.append(live_price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# Header
c1, c2 = st.columns([3, 1])
with c1: st.markdown("<h1>🦁 CM-X <span style='color:#00f3ff'>LIVE GOD MODE</span></h1>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='holo-card' style='text-align:center;'><span class='{status_color}'>{status_text}</span><br>{get_time().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

# --- CALCULATIONS ---
kf_price = st.session_state.kf.update(live_price)
vel, acc = calculate_physics(st.session_state.history)
entropy = calculate_entropy(st.session_state.history[-20:])
volatility = np.std(st.session_state.history[-10:]) / np.mean(st.session_state.history[-10:]) if len(st.session_state.history) > 10 else 0.01
mc_win = run_monte_carlo(live_price, volatility)
future_price = predict_future(st.session_state.history)

st.session_state.memory = self_construction_loop(live_price, st.session_state.memory)

# --- VOTING ---
score = 0
weights = st.session_state.memory["weights"]
votes = []

if vel > 1.0 and acc > 0: score += weights["PHYSICS"]; votes.append("PHY: BUY")
elif vel < -1.0 and acc < 0: score -= weights["PHYSICS"]; votes.append("PHY: SELL")

if future_price > live_price + 3 and mc_win > 60: score += weights["FUTURE"]; votes.append("MATH: BUY")
elif future_price < live_price - 3 and mc_win < 40: score -= weights["FUTURE"]; votes.append("MATH: SELL")

if entropy > 1.8: score -= 5; votes.append("CHAOS: TRAP")

decision = "WAIT"
if score > 2.0: decision = "BUY CE 🚀"
elif score < -2.0: decision = "BUY PE 🩸"

# --- DISPLAY ---
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f"<div class='holo-card'><small>PRICE</small><br><div class='metric-val text-cyan'>{live_price}</div></div>", unsafe_allow_html=True)
with m2: st.markdown(f"<div class='holo-card'><small>VELOCITY</small><br><div class='metric-val text-green'>{vel:.2f}</div></div>", unsafe_allow_html=True)
with m3: st.markdown(f"<div class='holo-card'><small>ENTROPY</small><br><div class='metric-val {'text-red' if entropy>1.8 else 'text-green'}'>{entropy:.2f}</div></div>", unsafe_allow_html=True)
with m4: st.markdown(f"<div class='holo-card'><small>DECISION</small><br><div class='metric-val text-cyan'>{decision}</div></div>", unsafe_allow_html=True)

g1, g2 = st.columns([2, 1])
with g1:
    st.markdown("### 📈 QUANTUM CHART")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.history[-30:], mode='lines', name='Price', line=dict(color='#00f3ff')))
    fig.add_trace(go.Scatter(y=[kf_price]*30, mode='lines', name='Kalman', line=dict(color='yellow', dash='dot')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.markdown("### 🖥️ BLACK BOX LOGS")
    log_txt = f"""
    > [SOURCE] {status_text}
    > [PHYSICS] Vel: {vel:.2f} | Acc: {acc:.2f}
    > [MATH] Future: {future_price:.1f} | Win%: {mc_win:.0f}%
    > [MEMORY] {st.session_state.memory['accuracy']}% Acc
    > [VOTES] {votes}
    > [SCORE] {score:.2f}
    """
    st.markdown(f"<div class='terminal-box'>{log_txt.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

# --- ACTION ---
if decision != "WAIT":
    if not st.session_state.memory["last_pred"].get("time"):
        st.session_state.memory["last_pred"] = {
            "price": live_price, 
            "time": get_time().replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S'),
            "decision": decision
        }
        save_blackbox(st.session_state.memory)
    
    if "GEMINI_API_KEY" in st.secrets:
        if st.button("🤖 CONSULT CHELLAKILI"):
            with st.spinner("Connecting..."):
                prompt = f"Signal: {decision}. Price: {live_price}. Velocity: {vel}. Mode: {status_text}."
                ai_reply = st.session_state.brain.consult_10000_traders(prompt)
                st.info(f"🦁 {ai_reply}")
                if voice_on: play_voice(ai_reply)
                send_telegram(f"🔥 {decision}\nPrice: {live_price}\nAI: {ai_reply}")

# Refresh Speed
refresh_rate = 1 if speed_mode else 3
time.sleep(refresh_rate)
st.rerun()
    
