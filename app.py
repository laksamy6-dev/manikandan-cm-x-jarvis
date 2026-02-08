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

# ==========================================
# 1. CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(
    page_title="CM-X GENESIS: TIME TRAVELER",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_indian_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

# ==========================================
# 2. MEMORY SYSTEM (SELF-LEARNING)
# ==========================================
MEMORY_FILE = "cm_x_time_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": 0.0,
            "last_prediction": {"price": 0, "time": None, "type": None},
            "weights": { 
                "PHYSICS": 1.5, 
                "MATH": 1.2, 
                "CHAOS": 1.0, 
                "FUTURE": 1.5 
            },
            "lessons": []
        }

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def minute_learning_loop(current_price, memory):
    """நிமிடத்திற்கு ஒரு முறை தன்னைத்தானே சரிபார்க்கும் லூப்"""
    last_pred = memory.get("last_prediction", {})
    if not last_pred.get("time"): return memory
    
    try:
        last_time = datetime.strptime(last_pred["time"], '%Y-%m-%d %H:%M:%S')
        # Timezone aware comparison fix
        now = get_indian_time().replace(tzinfo=None)
        diff_minutes = (now - last_time).total_seconds() / 60
    except: return memory

    if diff_minutes >= 1:
        old_price = last_pred["price"]
        pred_type = last_pred["type"]
        
        result = "NEUTRAL"
        if pred_type == "BUY": result = "WIN" if current_price > old_price else "LOSS"
        elif pred_type == "SELL": result = "WIN" if current_price < old_price else "LOSS"
        
        msg = ""
        if result == "WIN":
            memory["wins"] += 1
            memory["weights"]["FUTURE"] += 0.02
            msg = f"✅ Prediction Correct. Boosting Future Agent."
        elif result == "LOSS":
            memory["losses"] += 1
            memory["weights"]["FUTURE"] -= 0.02
            msg = f"❌ Prediction Failed. Tuning Logic."
        
        total = memory["wins"] + memory["losses"]
        memory["accuracy"] = round((memory["wins"] / total) * 100, 2) if total > 0 else 0
        
        if result != "NEUTRAL":
            memory["lessons"].append(f"[{now.strftime('%H:%M')}] {msg}")
            if len(memory["lessons"]) > 20: memory["lessons"].pop(0)
            
            # Reset
            memory["last_prediction"] = {"price": 0, "time": None, "type": None}
            save_memory(memory)
            
    return memory

# ==========================================
# 3. MATH CORE (PHYSICS, CHAOS, MATH)
# ==========================================
class KalmanFilter:
    def __init__(self):
        self.last_estimate = 0
        self.err_estimate = 1
        self.q = 0.01
    def update(self, measurement):
        kalman_gain = self.err_estimate / (self.err_estimate + 1)
        current_estimate = self.last_estimate + kalman_gain * (measurement - self.last_estimate)
        self.err_estimate = (1.0 - kalman_gain) * self.err_estimate + abs(self.last_estimate - current_estimate) * self.q
        self.last_estimate = current_estimate
        return current_estimate

def calculate_physics(history):
    if len(history) < 3: return 0, 0
    velocity = history[-1] - history[-2]
    acceleration = velocity - (history[-2] - history[-3])
    return velocity, acceleration

def calculate_entropy(data):
    if len(data) < 10: return 0
    hist, _ = np.histogram(data, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def run_monte_carlo(price, volatility, sims=500):
    wins = 0
    if volatility <= 0: volatility = 0.01
    for _ in range(sims):
        sim_price = price * math.exp((0 - 0.5 * volatility**2) + volatility * np.random.normal())
        if sim_price > price: wins += 1
    return (wins / sims) * 100

def predict_next_10_min(history):
    if len(history) < 20: return history[-1]
    y = np.array(history[-20:])
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1) 
    p = np.poly1d(z)
    future_index = len(y) + 10 
    return p(future_index)

# ==========================================
# 4. COMMUNICATION (Telegram & Voice)
# ==========================================
def send_telegram(msg):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            token = st.secrets["TELEGRAM_BOT_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            params = {"chat_id": chat_id, "text": msg}
            requests.get(url, params=params)
    except: pass

def play_voice(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("voice_future.mp3")
        with open("voice_future.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        os.remove("voice_future.mp3")
    except: pass

# ==========================================
# 5. THE SUPREME COUNCIL
# ==========================================
def convene_council(price, history, kf_price, weights):
    votes = []
    score = 0
    
    # 1. Physics Agent
    vel, acc = calculate_physics(history)
    p_vote = 0
    if vel > 1.5 and acc > 0: p_vote = 1
    elif vel < -1.5 and acc < 0: p_vote = -1
    score += p_vote * weights.get("PHYSICS", 1.5)
    votes.append(f"Physics (v={vel:.1f}): {'BUY' if p_vote==1 else 'SELL'}")

    # 2. Chaos Agent
    entropy = calculate_entropy(history[-20:])
    if entropy > 1.8:
        score -= 5
        votes.append(f"Chaos: ⚠️ TRAP (Entropy {entropy:.2f})")
    else:
        votes.append("Chaos: SAFE")

    # 3. Math Agent
    volatility = np.std(history[-10:]) / np.mean(history[-10:]) if len(history) > 10 else 0.01
    mc_win = run_monte_carlo(price, volatility)
    m_vote = 1 if (mc_win > 60 and price > kf_price) else (-1 if (mc_win < 40 and price < kf_price) else 0)
    score += m_vote * weights.get("MATH", 1.2)
    votes.append(f"Math (Win% {mc_win:.0f}): {'BUY' if m_vote==1 else 'SELL'}")

    # 4. Future Agent
    future_price = predict_next_10_min(history)
    f_vote = 0
    if future_price > price + 5: f_vote = 1
    elif future_price < price - 5: f_vote = -1
    score += f_vote * weights.get("FUTURE", 1.5)
    votes.append(f"Future (Target {future_price:.1f}): {'BUY' if f_vote==1 else 'SELL'}")

    return score, votes, vel, entropy, mc_win, future_price

# ==========================================
# 6. APP UI & LOGIC
# ==========================================
if "auth" not in st.session_state: st.session_state.auth = False
if "memory" not in st.session_state: st.session_state.memory = load_memory()
if "kf" not in st.session_state: st.session_state.kf = KalmanFilter()

if not st.session_state.auth:
    pwd = st.text_input("🔐 ENTER ACCESS CODE:", type="password")
    if st.button("UNLOCK"):
        if pwd == "boss": st.session_state.auth = True; st.rerun()
    st.stop()

with st.sidebar:
    st.title("⚙️ CONTROLS")
    auto_ref = st.toggle("🔄 MICRO-SCALPING MODE", value=True)
    voice_on = st.toggle("🔊 VOICE", value=True)
    if st.button("Test Telegram"): send_telegram("CM-X Time Traveler Online!")
    st.metric("AI Accuracy", f"{st.session_state.memory.get('accuracy', 0)}%")

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🦁 CM-X TIME TRAVELER")
    st.caption("Physics | Chaos | Monte Carlo | Future Prediction")
with c2:
    st.markdown(f"**IST:** {get_indian_time().strftime('%H:%M:%S')}")

# Data Simulation (Replace with Live Feed later)
if "history" not in st.session_state: st.session_state.history = [19500]*50
raw_price = st.session_state.history[-1] + np.random.randint(-15, 20)
kf_price = st.session_state.kf.update(raw_price)
st.session_state.history.append(raw_price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# 1. MINUTE LEARNING LOOP
st.session_state.memory = minute_learning_loop(raw_price, st.session_state.memory)

# 2. COUNCIL MEETING
score, votes, vel, entropy, mc_win, future_target = convene_council(raw_price, st.session_state.history, kf_price, st.session_state.memory["weights"])

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("PRICE", raw_price)
m2.metric("FUTURE (10m)", f"{future_target:.1f}", delta=f"{future_target-raw_price:.1f}")
m3.metric("ENTROPY", f"{entropy:.2f}", delta="TRAP" if entropy > 1.8 else "SAFE", delta_color="inverse")
m4.metric("SCORE", f"{score:.1f}")

# Decision
final_decision = "WAIT"
if score > 3.0: final_decision = "🚀 MICRO-BUY"
elif score < -3.0: final_decision = "🩸 MICRO-SELL"

# Memory Update
if final_decision != "WAIT" and st.session_state.memory["last_prediction"]["price"] == 0:
    st.session_state.memory["last_prediction"] = {
        "price": raw_price,
        "time": get_indian_time().replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S'),
        "type": "BUY" if "BUY" in final_decision else "SELL"
    }
    save_memory(st.session_state.memory)

# Charts
c_chart, c_log = st.columns([2, 1])
with c_chart:
    st.line_chart(st.session_state.history[-30:])
with c_log:
    for v in votes: st.code(v)
    st.markdown(f"### VERDICT: {final_decision}")

# GEMINI BRAIN INTEGRATION
if "GEMINI_API_KEY" in st.secrets and final_decision != "WAIT":
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if st.button("🤖 VALIDATE PREDICTION"):
            with st.spinner("Calculating Spacetime Probability..."):
                prompt = f"""
                Act as CM-X Time Traveler.
                Data: Price={raw_price}, Vel={vel}, Entropy={entropy}.
                Decision: {final_decision}.
                Explain logic in Tanglish.
                """
                res = model.generate_content(prompt)
                st.success(f"🦁 **CM-X:** {res.text}")
                
                # Notifications
                if voice_on: play_voice(f"Prediction Verified. {res.text[:50]}")
                send_telegram(f"⏳ FUTURE ALERT: {final_decision}\nTarget: {future_target}\nPrice: {raw_price}")
    except Exception as e:
        st.error(f"AI Error: {e}")

# AUTO REFRESH
if auto_ref:
    time.sleep(1)
    st.rerun()
    
