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

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CM-X GOD MODE (PERMANENT)",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# --- 2. PERMANENT MEMORY SYSTEM (BLACK BOX) ---
MEMORY_FILE = "cm_x_god_memory.json"

def init_memory():
    """மூளையை உருவாக்குதல் அல்லது பழைய நினைவுகளை மீட்டெடுத்தல்"""
    if not os.path.exists(MEMORY_FILE):
        # புது மெமரி
        default_data = {
            "position": None,      # Current Active Trade
            "orders": [],          # Order History
            "pnl": 0.0,            # Total Profit/Loss
            "last_price": 0.0      # Last seen price
        }
        with open(MEMORY_FILE, 'w') as f:
            json.dump(default_data, f)
        return default_data
    else:
        # பழைய மெமரி லோடிங்
        try:
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"position": None, "orders": [], "pnl": 0.0, "last_price": 0.0}

def save_memory(position, orders, pnl, last_price):
    """நடக்கும் ஒவ்வொரு விஷயத்தையும் மூளையில் பதிவு செய்தல்"""
    data = {
        "position": position,
        "orders": orders,
        "pnl": pnl,
        "last_price": last_price
    }
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f)

# Load Memory on Startup
brain_data = init_memory()

# --- 3. SESSION STATE SYNC (CRITICAL) ---
# Streamlit Session State-ஐ Permanent Memory உடன் இணைத்தல்
if 'prices' not in st.session_state: st.session_state.prices = []
if 'bot_active' not in st.session_state: st.session_state.bot_active = False

# பழைய கணக்கு வழக்குகள் (Restore from JSON)
if 'position' not in st.session_state: st.session_state.position = brain_data['position']
if 'orders' not in st.session_state: st.session_state.orders = brain_data['orders']
if 'pnl' not in st.session_state: st.session_state.pnl = brain_data['pnl']

# --- 4. ADVANCED CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; color: #212529; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 800; color: #0f172a; }
    .pnl-box { font-size: 28px; font-weight: bold; text-align: center; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .profit { background-color: #dcfce7; color: #166534; border: 2px solid #166534; }
    .loss { background-color: #fee2e2; color: #991b1b; border: 2px solid #991b1b; }
    .active-trade { background-color: #fff7ed; border: 2px solid #ea580c; padding: 10px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. CONFIG LOAD ---
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"
    
    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
except:
    st.error("⚠️ SECRETS ERROR")
    st.stop()

UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 6. CORE FUNCTIONS ---
def get_live_data():
    if not UPSTOX_ACCESS_TOKEN: return None, "NO TOKEN"
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': REQ_INSTRUMENT_KEY}
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                colon_key = REQ_INSTRUMENT_KEY.replace('|', ':')
                pipe_key = REQ_INSTRUMENT_KEY
                if colon_key in data['data']: price = data['data'][colon_key]['last_price']
                elif pipe_key in data['data']: price = data['data'][pipe_key]['last_price']
                else: price = data['data'][list(data['data'].keys())[0]]['last_price']
                return float(price), "OK"
    except: pass
    return None, "ERROR"

def calculate_physics(prices):
    if len(prices) < 10: return 0, 0, 0, 0
    p = np.array(prices)
    v = np.gradient(p)[-1]
    a = np.gradient(np.gradient(p))[-1]
    entropy = np.std(p[-10:])
    return v, a, a*100, entropy

def ask_jarvis(price, v, a, e):
    try:
        prompt = f"Nifty: {price}. V:{v:.1f}, A:{a:.1f}, Chaos:{e:.1f}. Action (BUY/SELL/WAIT)?"
        response = model.generate_content(prompt)
        return response.text
    except: return "AI THINKING..."

# --- 7. UI LAYOUT ---
st.markdown(f"<h2 style='text-align:center; border-bottom: 3px solid #2563eb;'>CM-X GOD MODE: PERMANENT MEMORY</h2>", unsafe_allow_html=True)
st.caption(f"COMMANDER: {OWNER_NAME} | MEMORY: LOADED FROM DISK | SYSTEM: READY")

# ACTIVE POSITION DISPLAY (Top Priority)
active_ph = st.empty()
if st.session_state.position:
    pos = st.session_state.position
    active_ph.markdown(f"""
    <div class="active-trade">
        <h3>🚨 ACTIVE TRADE RUNNING!</h3>
        <p><b>TYPE:</b> {pos['type']} | <b>ENTRY:</b> {pos['entry']} | <b>QTY:</b> {pos['qty']}</p>
        <p><i>Do not close this tab until trade is finished.</i></p>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    chart_ph = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    p_metric = m1.empty()
    v_metric = m2.empty()
    a_metric = m3.empty()
    e_metric = m4.empty()

with col2:
    pnl_display = st.empty()
    ai_box = st.empty()
    
    st.markdown("---")
    auto_c1, auto_c2 = st.columns(2)
    start_btn = auto_c1.button("🔥 START AUTO")
    stop_btn = auto_c2.button("🛑 STOP AUTO")
    
    # Manual Override
    if st.button("❌ FORCE CLOSE POSITION"):
        if st.session_state.position:
            # Emergency Close
            last_p = st.session_state.prices[-1] if st.session_state.prices else 0
            if last_p > 0:
                pos = st.session_state.position
                pnl = (last_p - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - last_p) * pos['qty']
                
                st.session_state.pnl += pnl
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": f"FORCE CLOSE @ {last_p} | P&L: {pnl:.1f}"})
                st.session_state.position = None
                
                # SAVE TO DISK
                save_memory(None, st.session_state.orders, st.session_state.pnl, last_p)
                st.rerun()

    st.caption("📜 Order History (Black Box)")
    order_log = st.empty()

# Logic Controls
if start_btn: st.session_state.bot_active = True
if stop_btn: st.session_state.bot_active = False

# --- 8. THE BRAIN LOOP ---
if st.session_state.bot_active:
    
    # Initial Fetch
    price, status = get_live_data()
    if status != "OK":
        st.error(f"CONNECTION ERROR: {status}")
        st.stop()

    while st.session_state.bot_active:
        
        # 1. LIVE DATA
        current_price, status = get_live_data()
        if not current_price: 
            time.sleep(1)
            continue
            
        st.session_state.prices.append(current_price)
        if len(st.session_state.prices) > 100: st.session_state.prices.pop(0)
        
        # 2. PHYSICS
        v, a, f, entropy = calculate_physics(st.session_state.prices)
        
        # 3. AI LOGIC (Every 10 ticks)
        ai_msg = "..."
        if len(st.session_state.prices) % 10 == 0:
            ai_msg = ask_jarvis(current_price, v, a, entropy)
            
        # 4. TRADING LOGIC (MEMORY AWARE)
        # ஏற்கனவே பொசிஷன் இருக்கான்னு பாரு. இருந்தா, புது ஆர்டர் போடாதே!
        
        if st.session_state.position is None:
            # --- SEARCHING FOR NEW TRADE ---
            if v > 1.5 and a > 0.2 and entropy < 10: # BUY Logic
                st.session_state.position = {"type": "BUY", "entry": current_price, "qty": 50}
                msg = f"BUY OPEN @ {current_price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                # IMPORTANT: SAVE IMMEDIATELY
                save_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, current_price)
                st.rerun() # Refresh UI to show Active Box
                
            elif v < -1.5 and a < -0.2 and entropy < 10: # SELL Logic
                st.session_state.position = {"type": "SELL", "entry": current_price, "qty": 50}
                msg = f"SELL OPEN @ {current_price}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                # IMPORTANT: SAVE IMMEDIATELY
                save_memory(st.session_state.position, st.session_state.orders, st.session_state.pnl, current_price)
                st.rerun()

        else:
            # --- MANAGING EXISTING TRADE (Even after refresh) ---
            pos = st.session_state.position
            curr_pnl = 0.0
            
            if pos['type'] == "BUY": curr_pnl = (current_price - pos['entry']) * pos['qty']
            else: curr_pnl = (pos['entry'] - current_price) * pos['qty']
            
            # Exit Logic (Target 500 / Stop 250)
            if curr_pnl > 500 or curr_pnl < -250:
                st.session_state.pnl += curr_pnl
                msg = f"CLOSED {pos['type']} @ {current_price} | P&L: {curr_pnl:.1f}"
                st.session_state.orders.insert(0, {"time": str(datetime.now().time())[:8], "msg": msg})
                st.session_state.position = None
                
                # SAVE CLOSE STATE
                save_memory(None, st.session_state.orders, st.session_state.pnl, current_price)
                st.rerun()

        # 5. UI UPDATES
        p_metric.metric("NIFTY 50", f"₹{current_price:,.2f}", f"{v:.2f}")
        v_metric.metric("VELOCITY", f"{v:.2f}")
        a_metric.metric("ACCEL", f"{a:.2f}")
        e_metric.metric("ENTROPY", f"{entropy:.2f}")
        
        ai_box.info(f"AI BRAIN: {ai_msg}")
        
        # P&L Display (Realtime)
        running_pnl = 0.0
        if st.session_state.position:
            pos = st.session_state.position
            if pos['type'] == "BUY": running_pnl = (current_price - pos['entry']) * pos['qty']
            else: running_pnl = (pos['entry'] - current_price) * pos['qty']
            
        total_disp = st.session_state.pnl + running_pnl
        color = "green" if total_disp >= 0 else "red"
        pnl_display.markdown(f"<div class='pnl-box' style='color:{color}; border:2px solid {color};'>TOTAL P&L: ₹{total_disp:.2f}</div>", unsafe_allow_html=True)
        
        # Order Book
        if st.session_state.orders:
            order_log.dataframe(pd.DataFrame(st.session_state.orders), hide_index=True)
            
        # Chart with Entry Line
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.prices, mode='lines', line=dict(color='#2563eb', width=2)))
        if st.session_state.position:
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="orange", annotation_text="ENTRY")
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
        chart_ph.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)
