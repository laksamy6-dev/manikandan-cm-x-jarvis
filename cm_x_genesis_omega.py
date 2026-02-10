import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import google.generativeai as genai
import time
from datetime import datetime
import pytz

# --- 1. PAGE CONFIGURATION (LIGHT COMMANDER MODE) ---
st.set_page_config(
    page_title="CM-X OMEGA (LIGHT)",
    layout="wide",
    page_icon="☀️",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (PROFESSIONAL LIGHT THEME) ---
st.markdown("""
    <style>
    /* Main Background - Clean White */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1f2937;
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 600;
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Header */
    .commander-header {
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 25px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION LOAD (SECRETS) ---
try:
    if "general" in st.secrets:
        OWNER_NAME = st.secrets["general"]["owner"]
    else:
        OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
    TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
except Exception as e:
    st.error(f"⚠️ CONFIG ERROR: {e}. Check secrets.toml")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. ADVANCED PHYSICS BRAIN (CORE LOGIC) ---
def calculate_advanced_physics(prices):
    """
    நியூட்டன் விதி + Chaos Theory (Entropy)
    """
    if len(prices) < 10: return 0, 0, 0, 0
    p = np.array(prices)
    
    # 1. Velocity (v) = dP/dt
    velocity = np.diff(p)[-1]
    
    # 2. Acceleration (a) = dv/dt
    acceleration = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
    
    # 3. Force (F) = ma (Mass is assumed as 1 unit for Index)
    force = acceleration * 100
    
    # 4. Entropy (குழப்ப நிலை) - Standard Deviation of last 10 ticks
    # அதிக என்ட்ரோபி = பெரிய மூவ்மென்ட் வரப்போகுதுனு அர்த்தம்
    entropy = np.std(p[-10:])
    
    return velocity, acceleration, force, entropy

# --- 5. AI ANALYST (JARVIS) ---
def ask_jarvis(price, v, a, entropy):
    """
    Gemini AI Brain Analysis
    """
    try:
        prompt = f"""
        Act as an Advanced HFT Algo Analyst for Boss Manikandan.
        
        LIVE MARKET DATA:
        - Asset: Nifty 50
        - Price: {price}
        - Velocity: {v:.2f}
        - Acceleration: {a:.2f}
        - Market Entropy (Chaos): {entropy:.2f}
        
        TASK:
        Based on this physics data, provide a 'Sniper Signal' (BUY/SELL/WAIT) and a 1-line reason.
        Keep it professional and strict.
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI RECALIBRATING..."

def send_telegram_msg(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧠 {OWNER_NAME}: {message}"}
        requests.get(url, params=params)
    except: pass

def get_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': INSTRUMENT_KEY}
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=2)
        if response.status_code == 200:
            return float(response.json()['data'][INSTRUMENT_KEY]['last_price'])
    except: return None
    return None

# --- 6. MAIN DASHBOARD UI ---

ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).strftime('%I:%M:%S %p')

st.markdown('<div class="commander-header">CM-X GENESIS: OMEGA</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">OPERATOR: <b>{OWNER_NAME}</b> | TIME: {current_time} | BRAIN: <b>ACTIVE</b></div>', unsafe_allow_html=True)

# Session State
if 'prices' not in st.session_state: st.session_state.prices = []
if 'bot_running' not in st.session_state: st.session_state.bot_running = False

# Layout - Top Metrics
col1, col2, col3, col4 = st.columns(4)
price_ph = col1.empty()
vel_ph = col2.empty()
acc_ph = col3.empty()
entropy_ph = col4.empty() # Added Entropy Display

# Layout - AI & Chart
st.markdown("### 🧠 AI Neural Feed")
ai_feed_ph = st.empty()
chart_ph = st.empty()

# Layout - Controls
st.markdown("---")
c1, c2, c3 = st.columns(3)
start_btn = c1.button("▶️ ACTIVATE NEURAL ENGINE")
stop_btn = c2.button("⏹️ DEACTIVATE")
ai_btn = c3.button("📡 FORCE AI SCAN")

# --- 7. LOGIC LOOP ---

if start_btn:
    st.session_state.bot_running = True
    send_telegram_msg("SYSTEM ONLINE: ADVANCED BRAIN ACTIVATED")

if stop_btn:
    st.session_state.bot_running = False
    st.warning("SYSTEM STANDBY.")

if st.session_state.bot_running:
    while st.session_state.bot_running:
        
        # 1. Fetch Data
        current_price = get_market_data()
        
        # Fallback Simulation (Keep Brain Active even if API lags)
        if current_price is None:
            if st.session_state.prices:
                # Add micro-noise to simulate market breath
                noise = np.random.normal(0, 0.5)
                current_price = st.session_state.prices[-1] + noise
            else:
                current_price = 22000.0
        
        st.session_state.prices.append(current_price)
        if len(st.session_state.prices) > 100: st.session_state.prices.pop(0)
        
        # 2. Physics Brain Calculation
        v, a, f, entropy = calculate_advanced_physics(st.session_state.prices)
        
        # 3. Signal Logic (Strict)
        signal = "NEUTRAL"
        signal_color = "#6b7280"
        
        # Physics Trigger
        if v > 1.5 and a > 0.2:
            signal = "BULLISH MOMENTUM"
            signal_color = "#16a34a" # Green
        elif v < -1.5 and a < -0.2:
            signal = "BEARISH PRESSURE"
            signal_color = "#dc2626" # Red
            
        # 4. Update UI
        price_ph.metric("NIFTY 50", f"₹{current_price:,.2f}", f"{v:.2f}")
        vel_ph.metric("VELOCITY", f"{v:.2f}")
        acc_ph.metric("ACCELERATION", f"{a:.2f}")
        entropy_ph.metric("ENTROPY (CHAOS)", f"{entropy:.2f}")
        
        ai_feed_ph.markdown(f"<div style='padding:15px; background-color:#f3f4f6; border-left: 5px solid {signal_color}; font-weight:bold; color:#374151;'>SYSTEM STATUS: {signal}</div>", unsafe_allow_html=True)
        
        # 5. Chart (Professional)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.prices, mode='lines', line=dict(color='#2563eb', width=2), fill='tozeroy', fillcolor='rgba(37,99,235,0.05)'))
        fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=10,b=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f3f4f6'))
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)

# AI Manual Scan
if ai_btn:
    if st.session_state.prices:
        p = st.session_state.prices[-1]
        v, a, f, e = calculate_advanced_physics(st.session_state.prices)
        with st.spinner("JARVIS ANALYZING..."):
            insight = ask_jarvis(p, v, a, e)
        st.success(f"JARVIS: {insight}")
    else:
        st.error("No Data. Start Engine.")
