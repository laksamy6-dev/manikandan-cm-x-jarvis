import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import google.generativeai as genai
import time
from datetime import datetime
import pytz

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CM-X OMEGA (REAL-TIME)",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #111827; font-weight: 800; }
    .status-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 20px; }
    .connected { background-color: #dcfce7; color: #166534; border: 1px solid #166534; }
    .disconnected { background-color: #fee2e2; color: #991b1b; border: 1px solid #991b1b; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CONFIGURATION LOAD ---
try:
    if "general" in st.secrets:
        OWNER_NAME = st.secrets["general"]["owner"]
    else:
        OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
except Exception as e:
    st.error(f"⚠️ CONFIG ERROR: {e}")
    st.stop()

# UPSTOX API SETUP
UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
# குறியீடு சரியாக இருக்க வேண்டும் (Case Sensitive)
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. REAL DATA FETCHING (DEBUG MODE) ---
def get_real_market_data():
    """
    உண்மையான மார்க்கெட் டேட்டாவை எடுக்கும். 
    பிரச்சனை இருந்தால், எரர் மெசேஜ் கொடுக்கும்.
    """
    if not UPSTOX_ACCESS_TOKEN: 
        return None, "NO TOKEN FOUND"
    
    headers = {
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    params = {'instrument_key': INSTRUMENT_KEY}
    
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            # சரியான டேட்டா வருகிறதா என சரிபார்த்தல்
            if 'data' in data and INSTRUMENT_KEY in data['data']:
                price = data['data'][INSTRUMENT_KEY]['last_price']
                return float(price), "CONNECTED"
            else:
                return None, f"DATA ERROR: {data}"
        elif response.status_code == 401:
            return None, "TOKEN EXPIRED (401)"
        else:
            return None, f"API ERROR: {response.status_code}"
            
    except Exception as e:
        return None, f"NETWORK ERROR: {e}"

# --- 5. PHYSICS BRAIN ---
def calculate_physics(prices):
    if len(prices) < 5: return 0, 0, 0
    p = np.array(prices)
    v = np.gradient(p)[-1]
    a = np.gradient(np.gradient(p))[-1]
    f = a * 100
    return v, a, f

def ask_jarvis(price, v, a):
    try:
        prompt = f"Nifty 50 is at {price}. Velocity {v:.2f}. Trend analysis in 5 words?"
        response = model.generate_content(prompt)
        return response.text
    except: return "AI SLEEPING..."

# --- 6. UI LAYOUT ---

st.markdown(f"<h1 style='text-align:center;'>CM-X GENESIS: OMEGA (REAL)</h1>", unsafe_allow_html=True)

# Session State
if 'prices' not in st.session_state: st.session_state.prices = []
if 'bot_running' not in st.session_state: st.session_state.bot_running = False
if 'connection_status' not in st.session_state: st.session_state.connection_status = "WAITING..."

# Status Indicator
status_placeholder = st.empty()

col1, col2, col3, col4 = st.columns(4)
price_ph = col1.empty()
vel_ph = col2.empty()
ai_ph = col3.empty()
chart_ph = st.empty()

# Controls
c1, c2 = st.columns(2)
if c1.button("🔥 CONNECT TO LIVE MARKET"):
    st.session_state.bot_running = True
if c2.button("🛑 DISCONNECT"):
    st.session_state.bot_running = False

# --- 7. MAIN LOOP ---
if st.session_state.bot_running:
    
    # லூப் தொடங்குவதற்கு முன் டெஸ்ட் கால்
    price, status = get_real_market_data()
    st.session_state.connection_status = status
    
    # Status Display
    if status == "CONNECTED":
        status_placeholder.markdown(f'<div class="status-box connected">🟢 SYSTEM ONLINE | UPSTOX CONNECTED</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f'<div class="status-box disconnected">🔴 CONNECTION FAILED: {status}</div>', unsafe_allow_html=True)
        st.error(f"காரணம்: {status}. தயவுசெய்து புது டோக்கன் போடவும்!")
        st.session_state.bot_running = False # Stop if error
        st.stop()

    # Loop Starts
    while st.session_state.bot_running:
        
        # 1. Fetch Real Data
        current_price, status = get_real_market_data()
        
        if current_price:
            st.session_state.prices.append(current_price)
            if len(st.session_state.prices) > 50: st.session_state.prices.pop(0)
            
            # 2. Physics
            v, a, f = calculate_physics(st.session_state.prices)
            
            # 3. AI (Every 10th tick to save quota)
            ai_insight = "..."
            if len(st.session_state.prices) % 10 == 0:
                ai_insight = ask_jarvis(current_price, v, a)
            
            # 4. Update UI
            price_ph.metric("NIFTY 50", f"₹{current_price:,.2f}", f"{v:.2f}")
            vel_ph.metric("VELOCITY", f"{v:.2f}")
            ai_ph.metric("AI VIEW", ai_insight)
            
            # 5. Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.prices, mode='lines', line=dict(color='#2563eb', width=2)))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
            chart_ph.plotly_chart(fig, use_container_width=True)
            
        else:
            status_placeholder.markdown(f'<div class="status-box disconnected">🔴 SIGNAL LOST: {status}</div>', unsafe_allow_html=True)
            break
            
        time.sleep(1)
