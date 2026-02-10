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
    page_title="CM-X OMEGA (LIVE)",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (Light Theme) ---
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
# Request Key (Pipe format)
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. REAL DATA FETCHING (SMART FIX) ---
def get_real_market_data():
    """
    Handles both '|' and ':' separators in Upstox response.
    """
    if not UPSTOX_ACCESS_TOKEN: 
        return None, "NO TOKEN FOUND"
    
    headers = {
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    params = {'instrument_key': REQ_INSTRUMENT_KEY}
    
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data:
                resp_data = data['data']
                
                # CASE 1: Check for "NSE_INDEX:Nifty 50" (Colon format)
                colon_key = REQ_INSTRUMENT_KEY.replace('|', ':')
                
                # CASE 2: Check for "NSE_INDEX|Nifty 50" (Pipe format)
                pipe_key = REQ_INSTRUMENT_KEY
                
                price = None
                
                if colon_key in resp_data:
                    price = resp_data[colon_key]['last_price']
                elif pipe_key in resp_data:
                    price = resp_data[pipe_key]['last_price']
                else:
                    # Fallback: Take the first key available
                    first_key = list(resp_data.keys())[0]
                    price = resp_data[first_key]['last_price']
                
                return float(price), "CONNECTED"
            else:
                return None, f"DATA STRUCTURE ERROR: {data}"
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
        prompt = f"Analysis for Nifty 50 at {price}. V:{v:.2f}, A:{a:.2f}. Trend in 3 words?"
        response = model.generate_content(prompt)
        return response.text
    except: return "AI ANALYZING..."

# --- 6. UI LAYOUT ---

st.markdown(f"<h1 style='text-align:center;'>CM-X GENESIS: OMEGA (LIVE)</h1>", unsafe_allow_html=True)

# Session State
if 'prices' not in st.session_state: st.session_state.prices = []
if 'bot_running' not in st.session_state: st.session_state.bot_running = False

# Status Indicator
status_placeholder = st.empty()

col1, col2, col3, col4 = st.columns(4)
price_ph = col1.empty()
vel_ph = col2.empty()
ai_ph = col3.empty()
chart_ph = st.empty()

# Controls
c1, c2 = st.columns(2)
start = c1.button("🔥 CONNECT LIVE")
stop = c2.button("🛑 DISCONNECT")

if start: st.session_state.bot_running = True
if stop: st.session_state.bot_running = False

# --- 7. MAIN LOOP ---
if st.session_state.bot_running:
    
    # Initial Check
    price, status = get_real_market_data()
    
    if status == "CONNECTED":
        status_placeholder.markdown(f'<div class="status-box connected">🟢 UPSTOX CONNECTED | DATA FLOWING</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown(f'<div class="status-box disconnected">🔴 ERROR: {status}</div>', unsafe_allow_html=True)
        # Don't stop immediately, try looping to see if it recovers or shows detailed error
        if "TOKEN" in status:
            st.stop()

    # Loop
    while st.session_state.bot_running:
        
        current_price, status = get_real_market_data()
        
        if current_price:
            st.session_state.prices.append(current_price)
            if len(st.session_state.prices) > 50: st.session_state.prices.pop(0)
            
            v, a, f = calculate_physics(st.session_state.prices)
            
            ai_insight = "..."
            if len(st.session_state.prices) % 10 == 0:
                ai_insight = ask_jarvis(current_price, v, a)
            
            price_ph.metric("NIFTY 50", f"₹{current_price:,.2f}", f"{v:.2f}")
            vel_ph.metric("VELOCITY", f"{v:.2f}")
            ai_ph.metric("AI BRAIN", ai_insight)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.prices, mode='lines', line=dict(color='#2563eb', width=2)))
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
            chart_ph.plotly_chart(fig, use_container_width=True)
            
        else:
            status_placeholder.markdown(f'<div class="status-box disconnected">🔴 SIGNAL LOST: {status}</div>', unsafe_allow_html=True)
            time.sleep(2) # Wait before retry
            
        time.sleep(1)
