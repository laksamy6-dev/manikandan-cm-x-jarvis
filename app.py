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
from datetime import datetime, timedelta
import pytz
from gtts import gTTS

# ==========================================
# 1. CONFIGURATION & TIME SETUP
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
# 2. MEMORY SYSTEM (SELF-IMPROVEMENT MINUTE-BY-MINUTE)
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
            "weights": { # ஏஜெண்டுகளின் பவர்
                "PHYSICS": 1.5,
                "MATH": 1.2,
                "CHAOS": 1.0,
                "FUTURE": 1.5 # புதுசு: எதிர்கால கணிப்பு ஏஜெண்ட்
            },
            "lessons": []
        }

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def minute_learning_loop(current_price, memory):
    """
    நிமிடத்திற்கு ஒரு முறை தன்னைத்தானே சரிபார்க்கும் லூப்.
    கடந்த கணிப்பு சரியாக இருந்தால் எடையை ஏற்று, தவறாக இருந்தால் குறை.
    """
    last_pred = memory.get("last_prediction", {})
    if not last_pred.get("time"): return memory
    
    # நேரம் ஒப்பீடு (String to Time object)
    try:
        last_time = datetime.strptime(last_pred["time"], '%Y-%m-%d %H:%M:%S')
        now = get_indian_time().replace(tzinfo=None) # Compare naive times
        diff_minutes = (now - last_time).total_seconds() / 60
    except: return memory

    # 1 நிமிடம் ஆகிவிட்டதா?
    if diff_minutes >= 1:
        # கணிப்பு சரிபார்ப்பு
        old_price = last_pred["price"]
        pred_type = last_pred["type"] # BUY or SELL
        
        result = "NEUTRAL"
        if pred_type == "BUY":
            result = "WIN" if current_price > old_price else "LOSS"
        elif pred_type == "SELL":
            result = "WIN" if current_price < old_price else "LOSS"
        
        if result == "WIN":
            memory["wins"] += 1
            memory["weights"]["FUTURE"] += 0.02 # எதிர்கால கணிப்பு பவர் அப்
            msg = f"✅ Learning: Prediction correct. Boosting Future Agent."
        elif result == "LOSS":
            memory["losses"] += 1
            memory["weights"]["FUTURE"] -= 0.02 # தப்பு பண்ணா பவர் குறை
            msg = f"❌ Learning: Prediction failed. Tuning Logic."
        
        # Accuracy Update
        total = memory["wins"] + memory["losses"]
        memory["accuracy"] = round((memory["wins"] / total) * 100, 2) if total > 0 else 0
        
        if result != "NEUTRAL":
            memory["lessons"].append(f"[{now.strftime('%H:%M')}] {msg}")
            if len(memory["lessons"]) > 20: memory["lessons"].pop(0)
            
            # Reset Prediction Check
            memory["last_prediction"] = {"price": 0, "time": None, "type": None}
            save_memory(memory)
            
    return memory

# ==========================================
# 3. MATH CORE (PHYSICS, CHAOS, MONTE CARLO, KALMAN)
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
    for _ in range(sims):
        sim_price = price * math.exp((0 - 0.5 * volatility**2) + volatility * np.random.normal())
        if sim_price > price: wins += 1
    return (wins / sims) * 100

# === NEW: FUTURE PREDICTOR (LINEAR REGRESSION) ===
def predict_next_10_min(history):
    """அடுத்த 10 நிமிட விலையை கணிக்கும் கணித முறை"""
    if len(history) < 20: return history[-1]
    
    # எளிய Trend Projection (Numpy Polyfit)
    y = np.array(history[-20:]) # கடைசி 20 புள்ளிகள்
    x = np.arange(len(y))
    
    # கோடு போடுதல் (Slope & Intercept)
    z = np.polyfit(x, y, 1) 
    p = np.poly1d(z)
    
    # அடுத்த 10 ஸ்டெப் கணிப்பு
    future_index = len(y) + 10 
    future_price = p(future_index)
    
    return future_price

# ==========================================
# 4. COMMUNICATION
# ==========================================
def send_telegram(msg):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            token = st.secrets["TELEGRAM_BOT_TOKEN"]
            chat_id = st.secrets["TELEGRAM_CHAT_ID"]
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", params={"chat_id": chat_id, "text": msg})
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
# 5. THE SUPREME COUNCIL (UPDATED WITH FUTURE SIGHT)
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
        score -= 5 # TRAP
        votes.append(f"Chaos: ⚠️ TRAP (Entropy {entropy:.2f})")
    else:
        votes.append("Chaos: SAFE")

    # 3. Math Agent
    volatility = np.std(history[-10:]) / np.mean(history[-10:])
    mc_win = run_monte_carlo(price, volatility)
    m_vote = 1 if (mc_win > 60 and price > kf_price) else (-1 if (mc_win < 40 and price < kf_price) else 0)
    score += m_vote * weights.get("MATH", 1.2)
    votes.append(f"Math (Win% {mc_win:.0f}): {'BUY' if m_vote==1 else 'SELL'}")

    # 4. NEW: Future Agent (10-Min Prediction)
    future_price = predict_next_10_min(history)
    f_vote = 0
    if future_price > price + 5: f_vote = 1 # 10 நிமிஷத்துல 5 ரூபா ஏறும்
    elif future_price < price - 5: f_vote = -1
    
    # மைக்ரோ ஸ்கால்பிங் லாஜிக் (Micro Scalp)
    f_weight = weights.get("FUTURE", 1.5)
    score += f_vote * f_weight
    votes.append(f"Future (Target {future_price:.1f}): {'BUY' if f_vote==1 else 'SELL'}")

    return score, votes, vel, entropy, mc_win, future_price

# ==========================================
# 6. APP LOGIC & UI
# ==========================================

# Auth & Init
if "auth" not in st.session_state: st.session_state.auth = False
if "memory" not in st.session_state: st.session_state.memory = load_memory()
if "kf" not in st.session_state: st.session_state.kf = KalmanFilter()

if not st.session_state.auth:
    pwd = st.text_input("🔐 ENTER ACCESS CODE:", type="password")
    if st.button("UNLOCK"):
        if pwd == "boss": st.session_state.auth = True; st.rerun()
    st.stop()

# Sidebar
with st.sidebar:
    st.title("⚙️ CONTROLS")
    auto_ref = st.toggle("🔄 MICRO-SCALPING MODE", value=True) # Default ON
    voice_on = st.toggle("🔊 VOICE", value=True)
    st.markdown("---")
    st.metric("Self-Learning Accuracy", f"{st.session_state.memory.get('accuracy', 0)}%")

# Header
c1, c2 = st.columns([3, 1])
with c1:
    st.title("🦁 CM-X TIME TRAVELER")
    st.caption("Physics | Chaos | Monte Carlo | Future Prediction")
with c2:
    st.markdown(f"**IST:** {get_indian_time().strftime('%H:%M:%S')}")

# Data Simulation (Replace with Live Feed in Production)
if "history" not in st.session_state: st.session_state.history = [19500]*50
raw_price = st.session_state.history[-1] + np.random.randint(-15, 20)
kf_price = st.session_state.kf.update(raw_price)
st.session_state.history.append(raw_price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# 1. MINUTE LEARNING LOOP (BACKGROUND PROCESS)
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

# Update Memory with Prediction (For Self-Learning Loop)
if final_decision != "WAIT" and st.session_state.memory["last_prediction"]["price"] == 0:
    st.session_state.memory["last_prediction"] = {
        "price": raw_price,
        "time": get_indian_time().strftime('%Y-%m-%d %H:%M:%S'),
        "type": "BUY" if "BUY" in final_decision else "SELL"
    }
    save_memory(st.session_state.memory)

# Charts
c_chart, c_log = st.columns([2, 1])
with c_chart:
    st.subheader("📈 Time Traveler Chart")
    # Show Past History + Future Point
    chart_data = pd.DataFrame({"History": st.session_state.history[-30:]})
    st.line_chart(chart_data)
    # Note: Streamlit line chart simple plotting. For advanced future line, we use metrics.

with c_log:
    st.subheader("🧠 Brain Log")
    for v in votes: st.code(v)
    
    st.markdown("---")
    if len(st.session_state.memory["lessons"]) > 0:
        st.caption("Recent Self-Improvements:")
        st.text(st.session_state.memory["lessons"][-1])

# GEMINI BRAIN
if "GEMINI_API_KEY" in st.secrets and final_decision != "WAIT":
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    if st.button("🤖 VALIDATE PREDICTION"):
        with st.spinner("Calculating Spacetime Probability..."):
            prompt = f"""
            Act as CM-X Time Traveler.
            Current Price: {raw_price}.
            Predicted Price (10min): {future_target}.
            Physics Velocity: {vel}.
            Entropy: {entropy}.
            
            System says: {final_decision}.
            
            Verify this micro-scalp trade. Is the prediction logical based on Momentum (Velocity)?
            Answer in Tanglish.
            """
            try:
                res = model.generate_content(prompt)
                st.success(f"🦁 **CM-X:** {res.text}")
                if voice_on: play_voice(f"Prediction Verified. {res.text[:50]}")
                send_telegram(f"⏳ FUTURE ALERT: {final_decision}\nTarget: {future_target}\nPrice: {raw_price}")
            except: pass

# AUTO REFRESH (MICRO-SECOND SPEED SIMULATION)
if auto_ref:
    time.sleep(1) # For real micro-scalping, reduce this to 0.5 or 0.1 with proper API
    st.rerun()
    
