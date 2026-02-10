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
    page_title="CM-X COMMANDER (ULTIMATE)",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="collapsed"
)

# --- 2. PROFESSIONAL DASHBOARD CSS ---
st.markdown("""
    <style>
    /* Global Settings */
    .stApp { background-color: #f8f9fa; color: #212529; }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* P&L Box */
    .pnl-box {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .profit { background-color: #dcfce7; color: #166534; border: 2px solid #166534; }
    .loss { background-color: #fee2e2; color: #991b1b; border: 2px solid #991b1b; }
    .neutral { background-color: #e2e8f0; color: #475569; border: 2px solid #475569; }

    /* Order Buttons */
    .stButton>button {
        height: 60px;
        font-weight: bold;
        font-size: 18px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    /* Header */
    .header-style {
        text-align: center;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
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

# UPSTOX API
UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 4. DATA ENGINE (FIXED) ---
def get_real_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None, "NO TOKEN"
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': REQ_INSTRUMENT_KEY}
    
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                # Key Matching Logic
                colon_key = REQ_INSTRUMENT_KEY.replace('|', ':')
                pipe_key = REQ_INSTRUMENT_KEY
                
                if colon_key in data['data']: price = data['data'][colon_key]['last_price']
                elif pipe_key in data['data']: price = data['data'][pipe_key]['last_price']
                else: price = data['data'][list(data['data'].keys())[0]]['last_price']
                
                return float(price), "CONNECTED"
            return None, "DATA ERROR"
        return None, "API ERROR"
    except: return None, "NET ERROR"

# --- 5. PHYSICS BRAIN (CORE) ---
def calculate_brain_metrics(prices):
    if len(prices) < 10: return 0, 0, 0, 0
    p = np.array(prices)
    v = np.gradient(p)[-1]
    a = np.gradient(np.gradient(p))[-1]
    entropy = np.std(p[-10:]) # Chaos measure
    
    # Signal Logic
    signal = "WAIT"
    if v > 1.5 and a > 0.2: signal = "BUY CALL"
    elif v < -1.5 and a < -0.2: signal = "BUY PUT"
    
    return v, a, entropy, signal

def ask_jarvis_brain(price, v, a, entropy):
    try:
        prompt = f"Nifty 50: {price}. V:{v:.1f}, A:{a:.1f}, Entropy:{entropy:.1f}. Trade Advice (Sniper style)?"
        response = model.generate_content(prompt)
        return response.text
    except: return "AI BRAIN BUSY..."

# --- 6. SESSION STATE (MEMORY) ---
if 'prices' not in st.session_state: st.session_state.prices = []
if 'bot_running' not in st.session_state: st.session_state.bot_running = False
if 'orders' not in st.session_state: st.session_state.orders = [] # Order Book
if 'position' not in st.session_state: st.session_state.position = None # Current Trade
if 'pnl' not in st.session_state: st.session_state.pnl = 0.0 # Total P&L

# --- 7. UI LAYOUT ---

# Header
st.markdown(f"""
<div class="header-style">
    <h1>CM-X GENESIS: ULTIMATE COMMANDER</h1>
    <p>OPERATOR: <b>{OWNER_NAME}</b> | BRAIN: <b>ACTIVE</b> | MODE: <b>LIVE MARKET</b></p>
</div>
""", unsafe_allow_html=True)

# Main Grid
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📈 Live Market Kinematics")
    # Chart Placeholder
    chart_ph = st.empty()
    
    # Physics Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    price_ph = m1.empty()
    vel_ph = m2.empty()
    acc_ph = m3.empty()
    entropy_ph = m4.empty()

with col_right:
    st.markdown("### 🎮 Command Center")
    
    # P&L Display
    pnl_ph = st.empty()
    
    # AI Feed
    st.info("🤖 JARVIS FEED:")
    ai_ph = st.empty()
    
    st.markdown("---")
    # Trading Controls
    qty = st.number_input("Lot Size (Qty)", min_value=50, value=50, step=50)
    
    b1, b2 = st.columns(2)
    buy_btn = b1.button("🚀 BUY CALL", type="primary")
    sell_btn = b2.button("🔻 BUY PUT", type="secondary")
    
    close_btn = st.button("❌ CLOSE POSITION (PANIC)")

    st.markdown("---")
    st.markdown("### 📜 Order Book")
    order_book_ph = st.empty()

# Controls (Start/Stop)
st.markdown("---")
c1, c2 = st.columns(2)
start = c1.button("🔥 CONNECT SYSTEM")
stop = c2.button("🛑 SHUTDOWN")

if start: st.session_state.bot_running = True
if stop: st.session_state.bot_running = False

# --- 8. TRADE EXECUTION LOGIC ---
def execute_trade(action, price):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.orders.insert(0, {"Time": timestamp, "Type": action, "Price": price, "Qty": qty})
    
    if action == "CLOSE":
        if st.session_state.position:
            entry_price = st.session_state.position['price']
            trade_qty = st.session_state.position['qty']
            type = st.session_state.position['type']
            
            # Calculate P&L
            if type == "BUY CALL": pnl = (price - entry_price) * trade_qty
            else: pnl = (entry_price - price) * trade_qty
            
            st.session_state.pnl += pnl
            st.session_state.position = None
            st.toast(f"POSITION CLOSED! P&L: ₹{pnl:.2f}", icon="💰")
            
    else:
        if st.session_state.position is None:
            st.session_state.position = {"type": action, "price": price, "qty": qty}
            st.toast(f"ORDER EXECUTED: {action} @ {price}", icon="🚀")
        else:
            st.toast("ALREADY IN A POSITION!", icon="⚠️")

# Handle Button Clicks (Outside Loop)
if st.session_state.prices:
    curr_p = st.session_state.prices[-1]
    if buy_btn: execute_trade("BUY CALL", curr_p)
    if sell_btn: execute_trade("BUY PUT", curr_p)
    if close_btn: execute_trade("CLOSE", curr_p)

# --- 9. MAIN LOOP ---
if st.session_state.bot_running:
    
    # Check Connection
    p, status = get_real_market_data()
    if status != "CONNECTED":
        st.error(f"CONNECTION FAILED: {status}")
        st.stop()
        
    while st.session_state.bot_running:
        
        # 1. Fetch
        current_price, status = get_real_market_data()
        if not current_price: 
            time.sleep(1)
            continue
            
        st.session_state.prices.append(current_price)
        if len(st.session_state.prices) > 100: st.session_state.prices.pop(0)
        
        # 2. Physics Calc
        v, a, entropy, signal = calculate_brain_metrics(st.session_state.prices)
        
        # 3. AI Scan (Every 15 ticks)
        if len(st.session_state.prices) % 15 == 0:
            insight = ask_jarvis_brain(current_price, v, a, entropy)
            ai_ph.markdown(f"**{insight}**")
        
        # 4. Update Metrics
        price_ph.metric("NIFTY 50", f"₹{current_price:,.2f}", f"{v:.2f}")
        vel_ph.metric("VELOCITY", f"{v:.2f}")
        acc_ph.metric("ACCEL", f"{a:.2f}")
        entropy_ph.metric("ENTROPY", f"{entropy:.2f}")
        
        # 5. P&L Monitor (Real-Time)
        live_pnl = 0.0
        pnl_class = "neutral"
        
        if st.session_state.position:
            entry = st.session_state.position['price']
            q = st.session_state.position['qty']
            t = st.session_state.position['type']
            
            if t == "BUY CALL": live_pnl = (current_price - entry) * q
            else: live_pnl = (entry - current_price) * q
            
            pnl_class = "profit" if live_pnl > 0 else "loss"
            pnl_text = f"RUNNING: ₹{live_pnl:.2f}"
        else:
            pnl_text = f"BOOKED: ₹{st.session_state.pnl:.2f}"
            pnl_class = "profit" if st.session_state.pnl >= 0 else "loss"

        pnl_ph.markdown(f'<div class="pnl-box {pnl_class}">{pnl_text}</div>', unsafe_allow_html=True)
        
        # 6. Order Book Display
        if st.session_state.orders:
            df = pd.DataFrame(st.session_state.orders)
            order_book_ph.dataframe(df, height=200, use_container_width=True)
        else:
            order_book_ph.info("No Orders Yet")

        # 7. Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.prices, mode='lines', line=dict(color='#2563eb', width=2), name="Price"))
        
        # Add Trade Markers
        if st.session_state.position:
            entry_line = [st.session_state.position['price']] * len(st.session_state.prices)
            fig.add_trace(go.Scatter(y=entry_line, mode='lines', line=dict(color='orange', dash='dash'), name="Entry"))

        fig.update_layout(height=450, margin=dict(l=0,r=0,t=10,b=10), template="plotly_white")
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1) # 1 Second Refresh
