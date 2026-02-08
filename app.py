%%writefile CM_X_Dashboard.py
# -*- coding: utf-8 -*-
# ==============================================================================
#   PROJECT: CM-X JARVIS: THE OMEGA EMPEROR (MASTER V8.0) - CONSOLIDATED DASHBOARD
#   OWNER: Boss Manikandan
#   AI CORE: JARVIS (Neuro-Quantum + Cognitive Alpha + Physics Engine) & CHELLAKILI (Gemini Integration)
#   PURPOSE: The Ultimate Autonomous Trading Intelligence & Interactive Dashboard
# ==============================================================================

import os
import time
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv # Uncommented as per analysis
import math
import base64
import pytz
from gtts import gTTS # For text-to-speech

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="CM-X JARVIS: WAR ROOM", layout="wide", page_icon="🦁")

# --- 2. SECURITY VAULT (Secrets Management) ---
# If running locally without secrets.toml, use .env or defaults
load_dotenv() # Uncommented as per analysis
UPSTOX_KEY = st.secrets.get("UPSTOX_API_KEY", os.getenv("UPSTOX_API_KEY")) # Fallback to os.getenv for local dev
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN"))
CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID"))
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

# --- 3. UI STYLING (THEME: OMEGA) ---
st.markdown("""
    <style>
    .stApp { backgrh4ound-color: #000000; color: #00f3ff; font-family: 'Courier New', monospace; }
    .fire-text { background: linear-gradient(to top, #ff0000, #ff8800, #ffff00); -webkit-background-clip: text; color: transparent; font-weight: 900; animation: flicker 1.5s infinite alternate; } @keyframes flicker { 0% { opacity: 1; text-shadow: 0 0 10px red; } 100% { opacity: 0.8; text-shadow: 0 0 20px orange; } }
    .holo-card { background: rgba(10, 20, 30, 0.8); border: 1px solid rgba(0, 243, 255, 0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px; color: #00f3ff; }
    .stButton>button { background: linear-gradient(45deg, #00f3ff, #0066ff); color: black; font-weight: bold; border: none; width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 4. HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1>CM-X <span style='color:#00f3ff'>JARVIS</span></h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='fire-text'>CREATOR: BOSS MANIKANDAN</h3>", unsafe_allow_html=True)
with c2:
    st.success("🟢 SYSTEM ONLINE")

# --- UTILITY FUNCTIONS ---
def get_indian_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def send_telegram_msg(message):
    try:
        if TELEGRAM_TOKEN and CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {"chat_id": CHAT_ID, "text": message}
            requests.get(url, params=params)
            return True
    except Exception as e:
        st.error(f"Telegram Error: {e}")
    return False

def play_voice(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("voice_output.mp3")
        with open("voice_output.mp3", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}"></audio>', unsafe_allow_html=True)
        os.remove("voice_output.mp3")
    except Exception as e:
        st.error(f"Voice playback error: {e}")

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🔐 SECURITY VAULT")
    # Note: GEMINI_KEY is now loaded via st.secrets.get() at the top
    # The text_input remains for local testing if not in .streamlit/secrets.toml
    if not GEMINI_KEY:
        GEMINI_KEY = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("--- --- --- --- --- --- ---")
    st.markdown("### 📡 DATA FEED")
    refresh_rate = st.slider("Refresh Rate (Sec)", 1, 5, 1)

    st.markdown("--- --- --- --- --- --- ---")
    st.header("📡 NETWORK TEST")
    if st.button("Test Telegram Connection"):
        if send_telegram_msg("Vanakkam Boss! Jarvis Online. 🦁"):
            st.success("Telegram message sent!")
        else:
            st.error("Failed to send Telegram message. Check token/chat ID.")

# --- 6. NEURO-QUANTUM BRAIN (from LYHVXrYI8v4R) ---
class NeuroQuantumBrain:
    def __init__(self):
        self.physics_engine = True

    def calculate_physics(self, price_history):
        if len(price_history) < 5: return 0, 0
        velocity = price_history[-1] - price_history[-2]
        acceleration = velocity - (price_history[-2] - price_history[-3])
        return velocity, acceleration

    def analyze_market(self, price, price_history):
        vel, acc = self.calculate_physics(price_history)
        decision = "WAIT"
        reason = "Scanning..."

        # QUANTUM LOGIC
        if vel > 5 and acc > 2:
            decision = "BUY CE"
            reason = "🚀 Quantum Velocity Breakout!"
        elif vel < -5 and acc < -2:
            decision = "BUY PE"
            reason = "🩸 Gravity Crash Detected!"

        return decision, reason, vel

# --- KALMAN FILTER (from YwqHPCVc8xNU) ---
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

# --- MATH CORE FUNCTIONS (from YwqHPCVc8xNU) ---
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

def predict_next_10_min(history):
    if len(history) < 20: return history[-1]
    y = np.array(history[-20:])
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    future_index = len(y) + 10
    future_price = p(future_index)
    return future_price

# --- CHELLAKILI BRAIN (from MFnbQ5pv9sBX) ---
class ChellakiliBrain:
    def __init__(self):
        global GEMINI_KEY # Ensure it uses the global GEMINI_KEY
        if GEMINI_KEY:
            genai.configure(api_key=GEMINI_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def analyze_market(self, price, rsi, trend, fiis_data, physics_velocity):
        if not self.model:
            return '{"decision": "WAIT", "reason": "Error: Gemini API Key not configured."}'

        prompt = f"""
        You are an elite scalping trader named 'Chellakili' operating in Indian Nifty 50 Options.
        Your goal is ultra-short-term profit based on Physics and Data.

        LIVE MARKET DATA:
        - Current Price: {price}
        - RSI (Momentum): {rsi}
        - SuperTrend: {trend}
        - Physics Velocity: {physics_velocity} (Positive means speeding up, Negative means slowing down)
        - FII Activity: {fiis_data}

        INSTRUCTIONS:
        Analyze these factors together.
        If Velocity is dropping but Price is rising, it's a trap (Divergence).
        If RSI > 80, it's Overbought.

        OUTPUT FORMAT (JSON only):
        {{
            "decision": "BUY_CE" or "BUY_PE" or "WAIT",
            "confidence": "HIGH" or "LOW",
            "reason": "One line explanation in Tamil logic (Tanglish)"
        }}
        """
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.replace('```json', '').replace('```', '').strip()
            return result_text
        except Exception as e:
            return f'{{"decision": "WAIT", "reason": "AI Error: {str(e)}"}}'

# --- 7. MAIN LOGIC LOOP (Combined & Updated) ---
if 'price' not in st.session_state: st.session_state.price = 19500.0
if 'vwap' not in st.session_state: st.session_state.vwap = 19500.0
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Price', 'VWAP'])
if 'jarvis_brain' not in st.session_state: st.session_state.jarvis_brain = NeuroQuantumBrain()
if 'chellakili_brain' not in st.session_state: st.session_state.chellakili_brain = ChellakiliBrain()
if 'kf' not in st.session_state: st.session_state.kf = KalmanFilter()

# Simulate Data
move = np.random.randint(-10, 15)
st.session_state.price += move
st.session_state.vwap = (st.session_state.vwap * 19 + st.session_state.price) / 20

# Brain Analysis (JARVIS original)
jarvis_decision, jarvis_reason, jarvis_vel = st.session_state.jarvis_brain.analyze_market(st.session_state.price, st.session_state.history['Price'].tolist() if not st.session_state.history.empty else [st.session_state.price])

# Kalman Filter Update
kf_price = st.session_state.kf.update(st.session_state.price)

# Chellakili Brain Analysis (Simulated RSI, Trend, FII for demo)
# For a real application, you'd calculate these from actual market data
simulated_rsi = np.random.uniform(30, 70)
simulated_trend = "UP" if jarvis_vel > 0 else ("DOWN" if jarvis_vel < 0 else "FLAT")
simulated_fiis_data = np.random.choice(["BUYING", "SELLING", "NEUTRAL"])

chellakili_output = st.session_state.chellakili_brain.analyze_market(
    price=st.session_state.price,
    rsi=simulated_rsi,
    trend=simulated_trend,
    fiis_data=simulated_fiis_data,
    physics_velocity=jarvis_vel
)

try:
    chellakili_data = json.loads(chellakili_output)
    chellakili_decision = chellakili_data.get("decision", "WAIT")
    chellakili_reason = chellakili_data.get("reason", "")
except json.JSONDecodeError:
    chellakili_decision = "WAIT"
    chellakili_reason = f"Chellakili JSON Error: {chellakili_output}"

# Update History
new_row = pd.DataFrame({'Time': [get_indian_time().strftime("%H:%M:%S")], 'Price': [st.session_state.price], 'VWAP': [st.session_state.vwap]})
st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(50)

# --- 8. DASHBOARD LAYOUT ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("**📡 LIVE MARKET FEED**")
    st.video("https://www.w3schools.com/html/mov_bbb.mp4", format="video/mp4", start_time=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['Price'], mode='lines', name='Price', line=dict(color='#00f3ff')))
    fig.add_trace(go.Scatter(x=st.session_state.history['Time'], y=st.session_state.history['VWAP'], mode='lines', name='VWAP', line=dict(color='#ff9900', dash='dash')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#00f3ff'), height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # NEW SECTION: Indian Share Market Theories & Fundamentals
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("📚 Indian Share Market Theories & Fundamentals")
    st.markdown("""
    - **Option Chain Analysis**: Understanding OI, PCR for sentiment.
    - **Technical Indicators**: RSI, MACD, Moving Averages for trend and momentum.
    - **Price Action**: Candlestick patterns, support/resistance levels.
    - **Global Cues**: Impact of DII/FII data, global market sentiment.
    - **News & Events**: Macroeconomic data, corporate earnings, geopolitical events.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="holo-card" style="text-align:center;">', unsafe_allow_html=True)
    st.metric(label="NIFTY 50 (Current)", value=f"{st.session_state.price:.2f}", delta=f"{move:.2f}")
    st.markdown(f"**VELOCITY:** <span style='color:{'#10b981' if jarvis_vel > 0 else '#ef4444'}; font-size:1.2rem; font-weight:bold;'>{jarvis_vel:.2f}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # JARVIS CHAT
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("🤖 ASK JARVIS")
    user_query = st.text_input("Command:", placeholder="e.g. Trend Analysis")
    if st.button("ACTIVATE"):
        if GEMINI_KEY and user_query:
            try:
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Act as JARVIS for Boss Manikandan. Market Price: {st.session_state.price}. Query: {user_query}")
                st.success(f"🦁 **JARVIS:** {response.text}")

                if st.sidebar.toggle("🔊 Voice Output (JARVIS)", value=False): # New toggle for Jarvis voice
                    play_voice(response.text)
            except Exception as e:
                st.error(f"AI Error: {e}")
        else:
            st.warning("Please enter Gemini API Key or a query.")
    st.markdown('</div>', unsafe_allow_html=True)

    # CHELLAKILI'S LATEST INSIGHT
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("🧠 CHELLAKILI'S INSIGHT")
    st.info(f"**DECISION:** {chellakili_decision}")
    st.info(f"**REASON:** {chellakili_reason}")
    if st.sidebar.toggle("🔊 Voice Output (CHELLAKILI)", value=False): # New toggle for Chellakili voice
        play_voice(f"Chellakili says: {chellakili_reason}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 9. SIGNAL ALERT (Combined Jarvis & Chellakili) ---
final_decision_combined = "WAIT"
final_reason_combined = ""

if jarvis_decision != "WAIT":
    final_decision_combined = jarvis_decision
    final_reason_combined = jarvis_reason

if chellakili_decision != "WAIT" and final_decision_combined == "WAIT":
    final_decision_combined = chellakili_decision
    final_reason_combined = chellakili_reason

# Prioritize based on some logic, or combine if both suggest same direction
if jarvis_decision == chellakili_decision and jarvis_decision != "WAIT":
    final_decision_combined = jarvis_decision
    final_reason_combined = f"{jarvis_reason} | {chellakili_reason}"
elif chellakili_decision != "WAIT" and jarvis_decision == "WAIT":
    final_decision_combined = chellakili_decision
    final_reason_combined = chellakili_reason

if final_decision_combined != "WAIT":
    st.error(f"⚠️ CONSOLIDATED SIGNAL: {final_decision_combined} | {final_reason_combined}")
    if st.button(f"✅ APPROVE {final_decision_combined}"):
        st.toast("Order Executed!", icon="🚀")
        send_telegram_msg(f"🚀 ALERT: {final_decision_combined} @ {st.session_state.price:.2f} | Reason: {final_reason_combined}")

# --- 10. EMBED INTERACTIVE HTML DASHBOARD ---
st.markdown("--- --- --- --- --- --- ---")
st.subheader("🌐 CM-X GENESIS: Interactive Dashboard")

try:
    with open("CM_X_Ultimate_Brahmastra_AI_FIXED_V3.html", "r") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)
except FileNotFoundError:
    st.error("Error: CM_X_Ultimate_Brahmastra_AI_FIXED_V3.html not found. Please ensure it's in the same directory.")

# Auto-Refresh
time.sleep(refresh_rate)
st.rerun()
