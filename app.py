import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import json
import os
import math
from datetime import datetime
import pytz
from collections import deque

# --- 1. SYSTEM CONFIGURATION & UI SETUP ---
st.set_page_config(
    page_title="CM-X: SHADOW EMPEROR",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="collapsed"
)

# --- 2. THE "SHADOW EMPEROR" CSS (DESIGN ENGINE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');
    
    /* GLOBAL THEME */
    .stApp {
        background-color: #050505;
        color: #00f3ff;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* NEON TEXT & HEADERS */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
    }
    
    /* METRIC CARDS (GLASS NEUMORPHISM) */
    div[data-testid="stMetric"] {
        background: rgba(10, 20, 30, 0.8);
        border: 1px solid rgba(0, 243, 255, 0.2);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.3);
        border-color: rgba(0, 243, 255, 0.6);
    }
    div[data-testid="stMetricLabel"] { color: #888; font-size: 12px; letter-spacing: 1px; }
    div[data-testid="stMetricValue"] { 
        color: #00f3ff; font-family: 'Orbitron', sans-serif; font-size: 24px; text-shadow: 0 0 5px #00f3ff;
    }

    /* CUSTOM TERMINAL LOG */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #000;
        border-left: 3px solid #00f3ff;
        color: #00f3ff;
        padding: 15px;
        height: 250px;
        overflow-y: auto;
        font-size: 12px;
        box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8);
    }
    .log-time { color: #555; font-weight: bold; margin-right: 8px; }
    .log-warn { color: #ff9900; }
    .log-danger { color: #ff003c; text-shadow: 0 0 5px #ff003c; }
    .log-success { color: #00ff41; text-shadow: 0 0 5px #00ff41; }

    /* ANIMATED RADAR (CSS) */
    .radar {
        width: 80px; height: 80px;
        border: 2px solid #ff003c; border-radius: 50%;
        position: relative; margin: 0 auto;
        background: radial-gradient(circle, rgba(255,0,60,0.1) 0%, rgba(0,0,0,0) 70%);
        box-shadow: 0 0 15px rgba(255, 0, 60, 0.3);
    }
    .radar::after {
        content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(255, 0, 60, 0.5) 60deg, transparent 61deg);
        animation: spin 2s linear infinite;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE BRAIN (Logic Core) ---
class QuantumBrain:
    def __init__(self):
        self.memory_file = "cm_x_core_memory.json"
        self.load_memory()
        
    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {"pnl": 0.0, "trades": [], "wins": 0, "losses": 0}

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)

    def calculate_metrics(self, prices):
        if len(prices) < 20: return 0, 0, 0, 0
        
        p = np.array(prices)
        
        # 1. PHYSICS (Velocity & Acceleration)
        velocity = np.diff(p)[-1]
        acceleration = np.diff(np.diff(p))[-1] if len(p) > 2 else 0
        
        # 2. ENTROPY (Chaos)
        log_returns = np.diff(np.log(p))
        hist, _ = np.histogram(log_returns, bins=10, density=True)
        probs = hist / hist.sum()
        probs = probs[probs > 0] # Filter zeros
        entropy = -np.sum(probs * np.log(probs))
        entropy_norm = min(max(entropy, 0), 1) # Normalize 0-1
        
        # 3. HURST EXPONENT (Trend Strength)
        # Simplified calculation for speed
        lags = range(2, 10)
        tau = [np.sqrt(np.std(np.subtract(p[lag:], p[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0
        
        return velocity, acceleration, entropy_norm, hurst

# --- 4. SESSION STATE INITIALIZATION ---
if 'brain' not in st.session_state:
    st.session_state.brain = QuantumBrain()
if 'prices' not in st.session_state:
    st.session_state.prices = deque(maxlen=100)
    # Seed with dummy data for display
    start_price = 21500
    for _ in range(50):
        st.session_state.prices.append(start_price + np.random.normal(0, 5))
        
if 'logs' not in st.session_state:
    st.session_state.logs = deque(maxlen=20)
if 'active_trade' not in st.session_state:
    st.session_state.active_trade = None

def add_log(msg, type="info"):
    t = datetime.now().strftime("%H:%M:%S")
    css = "log-info"
    if type == "danger": css = "log-danger"
    if type == "success": css = "log-success"
    if type == "warn": css = "log-warn"
    st.session_state.logs.appendleft(f"<span class='log-time'>[{t}]</span> <span class='{css}'>{msg}</span>")

# --- 5. MAIN DASHBOARD LAYOUT ---
# Header
c1, c2, c3 = st.columns([1, 4, 1])
with c1:
    st.markdown("### 🦅 CM-X")
with c2:
    st.markdown("<h1 style='text-align: center;'>SHADOW EMPEROR <span style='font-size:14px; color:#ff003c'>[PREDATOR MODE]</span></h1>", unsafe_allow_html=True)
with c3:
    st.metric("NET P&L", f"₹{st.session_state.brain.memory['pnl']:.2f}")

st.divider()

# Main Grid
col_left, col_mid, col_right = st.columns([1, 2, 1])

# --- LEFT COLUMN: QUANTUM METRICS ---
with col_left:
    st.markdown("### 🧠 NEURAL CORTEX")
    
    # Live Data Simulation
    current_price = st.session_state.prices[-1]
    # Simulate new tick
    new_price = current_price + np.random.normal(0, 2)
    st.session_state.prices.append(new_price)
    
    # Calculate Metrics
    v, a, ent, hurst = st.session_state.brain.calculate_metrics(list(st.session_state.prices))
    
    st.metric("VELOCITY (v)", f"{v:.2f}", delta=f"{a:.2f} (a)")
    st.metric("ENTROPY (Chaos)", f"{ent:.2f}", delta_color="inverse")
    st.metric("HURST (Trend)", f"{hurst:.2f}")
    
    if hurst > 0.6:
        st.success("STRONG TREND DETECTED")
    elif ent > 0.7:
        st.error("HIGH CHAOS - STAY AWAY")
    else:
        st.info("MARKET RANGING")

# --- MIDDLE COLUMN: LIVE CHART & ACTION ---
with col_mid:
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(st.session_state.prices), 
        mode='lines', 
        name='NIFTY 50',
        line=dict(color='#00f3ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 243, 255, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        height=350,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # DECISION LOGIC
    decision = "WAIT"
    if not st.session_state.active_trade:
        if v > 1.5 and hurst > 0.55 and ent < 0.6:
            decision = "BUY CE 🚀"
            if st.button(f"EXECUTE {decision}", type="primary"):
                st.session_state.active_trade = {"type": "CE", "entry": new_price}
                add_log(f"ENTRY TAKEN: CE @ {new_price}", "success")
                st.rerun()
        elif v < -1.5 and hurst > 0.55 and ent < 0.6:
            decision = "BUY PE 🩸"
            if st.button(f"EXECUTE {decision}", type="primary"):
                st.session_state.active_trade = {"type": "PE", "entry": new_price}
                add_log(f"ENTRY TAKEN: PE @ {new_price}", "danger")
                st.rerun()
        else:
            st.button("SCANNING FOR TARGET...", disabled=True)
    else:
        # Exit Logic
        trade = st.session_state.active_trade
        pnl = (new_price - trade['entry']) if trade['type'] == "CE" else (trade['entry'] - new_price)
        pnl = pnl * 50 # Lot size
        
        st.info(f"OPEN POSITION: {trade['type']} | P&L: ₹{pnl:.2f}")
        
        if st.button("CLOSE POSITION"):
            st.session_state.brain.memory['pnl'] += pnl
            st.session_state.brain.save_memory()
            st.session_state.active_trade = None
            add_log(f"POSITION CLOSED | P&L: {pnl:.2f}", "warn")
            st.rerun()

# --- RIGHT COLUMN: THREAT RADAR & LOGS ---
with col_right:
    st.markdown("### 📡 THREAT RADAR")
    
    # Trap Detector Animation
    is_trap = ent > 0.8
    radar_color = "#ff003c" if is_trap else "#00ff41"
    radar_status = "TRAP DETECTED" if is_trap else "SAFE ZONE"
    
    st.markdown(f"""
        <div class="radar" style="border-color: {radar_color}; box-shadow: 0 0 15px {radar_color};"></div>
        <div style="text-align:center; margin-top:10px; font-weight:bold; color:{radar_color}">{radar_status}</div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🖥️ SYSTEM LOGS")
    log_html = "".join([f"<div style='border-bottom:1px dashed #333; margin-bottom:5px;'>{l}</div>" for l in st.session_state.logs])
    st.markdown(f"<div class='terminal-box'>{log_html}</div>", unsafe_allow_html=True)

# Auto Refresh for Simulation Effect
time.sleep(1)
st.rerun()
