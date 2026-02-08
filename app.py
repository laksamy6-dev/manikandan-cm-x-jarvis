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
# 1. CONFIGURATION & DARK THEME (Hacker UI)
# ==========================================
st.set_page_config(
    page_title="CM-X: TIME EMPEROR",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hacker / Iron Man Style CSS
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #111 0%, #000 100%);
        color: #00f3ff;
        font-family: 'Courier New', monospace;
    }
    
    /* Neon Text */
    h1, h2, h3 { color: #fff; text-shadow: 0 0 10px #00f3ff; }
    
    /* Neon Cards */
    .neon-card {
        background: rgba(20, 20, 30, 0.7);
        border: 1px solid #00f3ff;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Metrics Colors */
    .val-green { color: #00ff00; text-shadow: 0 0 10px #00ff00; font-size: 24px; font-weight: bold; }
    .val-red { color: #ff0000; text-shadow: 0 0 10px #ff0000; font-size: 24px; font-weight: bold; }
    .val-cyan { color: #00f3ff; text-shadow: 0 0 10px #00f3ff; font-size: 24px; font-weight: bold; }
    .val-amber { color: #ffcc00; text-shadow: 0 0 10px #ffcc00; font-size: 24px; font-weight: bold; }

    /* Terminal Box */
    .terminal {
        background: #000;
        border-left: 3px solid #00f3ff;
        padding: 10px;
        color: #00f3ff;
        font-size: 12px;
        height: 200px;
        overflow-y: auto;
        font-family: 'Courier New';
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ADVANCED BRAIN LOGIC (எதுவும் மிஸ் ஆகல!)
# ==========================================

def get_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

# --- A. MEMORY SYSTEM (Teacher & Student) ---
MEMORY_FILE = "cm_x_god_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f: return json.load(f)
    return {"wins": 0, "losses": 0, "accuracy": 0.0, "last_pred": {}, "weights": {"PHYSICS": 1.5, "MATH": 1.2, "FUTURE": 1.5}, "lessons": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f: json.dump(mem, f, indent=4)

def self_learning_loop(current_price, memory):
    last = memory.get("last_pred", {})
    if not last.get("time"): return memory
    
    try:
        last_time = datetime.strptime(last["time"], '%Y-%m-%d %H:%M:%S')
        now = get_time().replace(tzinfo=None)
        if (now - last_time).total_seconds() / 60 >= 1: # 1 Minute Check
            old_price = last["price"]
            decision = last["decision"]
            result = "NEUTRAL"
            
            if "BUY" in decision: result = "WIN" if current_price > old_price else "LOSS"
            elif "SELL" in decision: result = "WIN" if current_price < old_price else "LOSS"
            
            if result == "WIN":
                memory["wins"] += 1
                memory["weights"]["FUTURE"] += 0.05
                memory["lessons"].append(f"[{now.strftime('%H:%M')}] WIN: Strategy Boosted.")
            elif result == "LOSS":
                memory["losses"] += 1
                memory["weights"]["FUTURE"] -= 0.05
                memory["lessons"].append(f"[{now.strftime('%H:%M')}] LOSS: Logic Tuned.")
            
            total = memory["wins"] + memory["losses"]
            memory["accuracy"] = round((memory["wins"]/total)*100, 1) if total > 0 else 0
            memory["last_pred"] = {}
            save_memory(memory)
    except: pass
    return memory

# --- B. MATH CORE (Physics, Chaos, Monte Carlo, Future) ---
def calculate_physics(hist):
    if len(hist) < 3: return 0, 0
    v = hist[-1] - hist[-2]
    a = v - (hist[-2] - hist[-3])
    return v, a

def calculate_entropy(data):
    if len(data) < 10: return 0
    hist, _ = np.histogram(data, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def run_monte_carlo(price, vol, sims=500):
    wins = 0
    vol = 0.01 if vol == 0 else vol
    for _ in range(sims):
        sim_p = price * math.exp((0 - 0.5 * vol**2) + vol * np.random.normal())
        if sim_p > price: wins += 1
    return (wins / sims) * 100

def predict_future(hist): # Linear Regression
    if len(hist) < 20: return hist[-1]
    y = np.array(hist[-20:])
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    return np.poly1d(z)(len(y) + 10) # Next 10 mins

# --- C. COMMUNICATION ---
def send_telegram(msg):
    try:
        if "TELEGRAM_BOT_TOKEN" in st.secrets:
            requests.get(f"https://api.telegram.org/bot{st.secrets['TELEGRAM_BOT_TOKEN']}/sendMessage", 
                         params={"chat_id": st.secrets['TELEGRAM_CHAT_ID'], "text": msg})
    except: pass

# ==========================================
# 3. UI DASHBOARD (Hacker Style)
# ==========================================

# Auth
if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    pwd = st.text_input("🔐 ENTER CODE", type="password")
    if st.button("UNLOCK"): 
        if pwd == "boss": st.session_state.auth = True; st.rerun()
    st.stop()

# Load Memory & History
if "memory" not in st.session_state: st.session_state.memory = load_memory()
if "history" not in st.session_state: st.session_state.history = [19500]*50

# --- SIMULATION ---
raw_price = st.session_state.history[-1] + np.random.randint(-15, 20)
st.session_state.history.append(raw_price)
if len(st.session_state.history) > 50: st.session_state.history.pop(0)

# --- BRAIN CALCULATIONS ---
st.session_state.memory = self_learning_loop(raw_price, st.session_state.memory)
vel, acc = calculate_physics(st.session_state.history)
entropy = calculate_entropy(st.session_state.history[-20:])
future_price = predict_future(st.session_state.history)
volatility = np.std(st.session_state.history[-10:]) / np.mean(st.session_state.history[-10:])
mc_win = run_monte_carlo(raw_price, volatility)

# --- DECISION LOGIC ---
score = 0
weights = st.session_state.memory["weights"]

# 1. Physics
if vel > 1.5 and acc > 0: score += weights["PHYSICS"]
elif vel < -1.5 and acc < 0: score -= weights["PHYSICS"]

# 2. Future & Math
if future_price > raw_price + 5 and mc_win > 60: score += weights["FUTURE"]
elif future_price < raw_price - 5 and mc_win < 40: score -= weights["FUTURE"]

# 3. Chaos Trap
if entropy > 1.8: score -= 5 # Trap

final_decision = "WAIT"
if score > 2.5: final_decision = "BUY CE 🚀"
elif score < -2.5: final_decision = "BUY PE 🩸"

# Update Memory for Learning
if final_decision != "WAIT" and not st.session_state.memory["last_pred"].get("time"):
    st.session_state.memory["last_pred"] = {
        "price": raw_price, 
        "time": get_time().replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S'),
        "decision": final_decision
    }
    save_memory(st.session_state.memory)

# ==========================================
# 4. DISPLAY (THE GOD MODE UI)
# ==========================================
c1, c2 = st.columns([3, 1])
with c1: st.markdown("## 🦁 CM-X <span style='color:white'>TIME EMPEROR</span>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='neon-card val-amber'>{get_time().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f"<div class='neon-card'><small>PRICE</small><br><div class='val-cyan'>{raw_price}</div></div>", unsafe_allow_html=True)
with m2: st.markdown(f"<div class='neon-card'><small>FUTURE (10m)</small><br><div class='val-amber'>{future_price:.1f}</div></div>", unsafe_allow_html=True)
with m3: 
    ent_color = "val-red" if entropy > 1.8 else "val-green"
    st.markdown(f"<div class='neon-card'><small>ENTROPY</small><br><div class='{ent_color}'>{entropy:.2f}</div></div>", unsafe_allow_html=True)
with m4: 
    dec_color = "val-green" if "BUY" in final_decision else ("val-red" if "SELL" in final_decision else "val-cyan")
    st.markdown(f"<div class='neon-card' style='border:2px solid white'><small>DECISION</small><br><div class='{dec_color}'>{final_decision}</div></div>", unsafe_allow_html=True)

# Chart & Terminal
g1, g2 = st.columns([2, 1])
with g1:
    st.markdown("### 📈 QUANTUM CHART")
    st.line_chart(st.session_state.history[-30:])
with g2:
    st.markdown("### 🖥️ NEURAL TERMINAL")
    logs = f"""
    > [PHYSICS] Vel: {vel} | Acc: {acc}
    > [MATH] Monte Carlo Win%: {mc_win:.1f}%
    > [FUTURE] Predicted Target: {future_price:.1f}
    > [MEMORY] Accuracy: {st.session_state.memory['accuracy']}%
    > [STATUS] System Learning...
    """
    st.markdown(f"<div class='terminal'>{logs.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

# Gemini & Telegram
st.markdown("---")
if "GEMINI_API_KEY" in st.secrets and final_decision != "WAIT":
    if st.button("🤖 ACTIVATE JARVIS"):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            with st.spinner("Connecting..."):
                prompt = f"Act as CM-X. Decision: {final_decision}. Price: {raw_price}. Future: {future_price}. Explain in Tanglish."
                res = model.generate_content(prompt)
                st.success(f"🦁 JARVIS: {res.text}")
                send_telegram(f"🔥 {final_decision}\nPrice: {raw_price}\nAI: {res.text}")
        except: st.error("AI Error")

# Auto Refresh
time.sleep(1)
st.rerun()
  
