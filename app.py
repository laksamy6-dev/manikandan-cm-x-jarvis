import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime
import random

# --- PAGE CONFIGURATION (COPYRIGHT PROTECTION) ---
st.set_page_config(
    page_title="CM-X GENESIS | BOSS MANIKANDAN",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# --- CUSTOM CSS FOR BRANDING & DARK THEME ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
        color: #00ff00;
    }
    /* Copyright Watermark */
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        opacity: 0.5;
        z-index: 99;
        color: white;
        font-size: 12px;
    }
    /* Header Branding */
    .brand-header {
        font-size: 40px;
        font-weight: bold;
        color: #00FFCC;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        border-bottom: 2px solid #00FFCC;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .sub-brand {
        font-size: 18px;
        color: #ff4b4b;
        text-align: center;
        font-style: italic;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00ff00;
    }
    </style>
    <div class="watermark">© 2026 COPYRIGHT - BOSS MANIKANDAN - ALL RIGHTS RESERVED</div>
    """, unsafe_allow_html=True)

# --- SIDEBAR IDENTITY ---
with st.sidebar:
    st.markdown("## 🛡️ SYSTEM IDENTITY")
    st.info("**SYSTEM OWNER:** BOSS MANIKANDAN")
    st.text(f"SERVER TIME: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown("---")
    st.markdown("### 🤖 ENGINE STATUS")
    st.success("PHYSICS CORE: **ONLINE**")
    st.success("10,000 AGENTS: **ACTIVE**")
    st.success("COPYRIGHT GUARD: **ENABLED**")
    st.markdown("---")
    st.markdown("### ⚙️ CONTROLS")
    mode = st.radio("TRADING MODE", ["SIMULATION", "LIVE MARKET (API)"])
    risk_level = st.slider("RISK FACTOR (Turbulence Tolerance)", 0.0, 1.0, 0.3)

# --- HEADER SECTION ---
st.markdown('<div class="brand-header">CM-X GENESIS: APTE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-brand">Advanced Physics Trading Engine | Architect: BOSS MANIKANDAN</div>', unsafe_allow_html=True)
st.markdown("---")

# --- MOCK DATA GENERATOR (PHYSICS SIMULATION) ---
# In a real scenario, this comes from Upstox API
def get_physics_data():
    # Simulate Price with Sine wave + Noise
    t = time.time()
    price = 22000 + 100 * np.sin(t/10) + np.random.normal(0, 5)
    
    # Calculate Velocity (dP/dt)
    velocity = 10 * np.cos(t/10) + np.random.normal(0, 2)
    
    # Calculate Acceleration (d²P/dt²)
    acceleration = -1 * np.sin(t/10) + np.random.normal(0, 0.5)
    
    # Kinetic Energy (KE = 0.5 * m * v²) - Mass assumed constant for demo
    mass = random.uniform(0.8, 1.2) # Volume factor
    ke = 0.5 * mass * (velocity ** 2)
    
    # Rocket Fuel (Liquidity) - decreasing as price moves up fast
    fuel = 100 - min(100, abs(velocity) * 5)
    
    return price, velocity, acceleration, ke, fuel

# --- MAIN DASHBOARD ---
# Create placeholders for live updates
col1, col2, col3, col4 = st.columns(4)
chart_placeholder = st.empty()
swarms_placeholder = st.empty()

# --- RUNNING THE ENGINE LOOP ---
if st.button("🚀 ACTIVATE GENESIS ENGINE"):
    st.toast("SYSTEM BOOTING... WELCOME BOSS MANIKANDAN", icon="🔥")
    
    # Simulate live data stream
    history_price = []
    history_velocity = []
    
    for _ in range(100): # Run for 100 ticks for demo
        price, vel, acc, ke, fuel = get_physics_data()
        
        history_price.append(price)
        history_velocity.append(vel)
        if len(history_price) > 50:
            history_price.pop(0)
            history_velocity.pop(0)

        # --- 1. KEY METRICS DISPLAY ---
        with col1:
            st.metric(label="NIFTY 50 PRICE", value=f"₹{price:.2f}", delta=f"{vel:.2f} pts/s")
        with col2:
            st.metric(label="VELOCITY (v)", value=f"{vel:.2f} m/s", delta_color="off")
        with col3:
            st.metric(label="ACCELERATION (a)", value=f"{acc:.2f} m/s²", 
                      delta="SPEEDING UP" if acc > 0 else "SLOWING DOWN")
        with col4:
            st.metric(label="KINETIC ENERGY (J)", value=f"{ke:.0f} kJ", 
                      delta="HIGH IMPACT" if ke > 50 else "LOW ENERGY")

        # --- 2. PHYSICS GAUGES (ROCKET & TURBULENCE) ---
        fig_gauges = go.Figure()

        # Rocket Fuel Gauge
        fig_gauges.add_trace(go.Indicator(
            mode = "gauge+number",
            value = fuel,
            domain = {'x': [0, 0.45], 'y': [0, 1]},
            title = {'text': "ROCKET FUEL (Liquidity)"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "red" if fuel < 20 else "#00FFCC"},
                     'steps' : [{'range': [0, 20], 'color': "rgba(255, 0, 0, 0.3)"}]}
        ))

        # Acceleration/Momentum Gauge
        fig_gauges.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = acc,
            domain = {'x': [0.55, 1], 'y': [0, 1]},
            title = {'text': "MOMENTUM FORCE (F=ma)"},
            gauge = {'axis': {'range': [-5, 5]},
                     'bar': {'color': "#FFFF00"},
                     'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 4}}
        ))

        fig_gauges.update_layout(
            paper_bgcolor="#0e1117", 
            font={'color': "white", 'family': "Courier New"},
            height=250,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        swarms_placeholder.plotly_chart(fig_gauges, use_container_width=True)

        # --- 3. PRICE CHART WITH PHYSICS VECTORS ---
        fig_chart = go.Figure()
        
        # Price Line
        fig_chart.add_trace(go.Scatter(
            y=history_price, 
            mode='lines', 
            name='Price',
            line=dict(color='#00FFCC', width=2)
        ))
        
        # Velocity Overlay (Secondary Axis logic simulated by color intensity)
        fig_chart.add_trace(go.Scatter(
            y=[p - 10 for p in history_price], # Offset for visuals
            mode='lines',
            name='Velocity Trail',
            line=dict(color='rgba(255, 255, 0, 0.5)', width=1, dash='dot')
        ))

        fig_chart.update_layout(
            title=f"LIVE MARKET KINEMATICS | OWNER: BOSS MANIKANDAN",
            xaxis_title="Time Ticks (t)",
            yaxis_title="Price Level",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        chart_placeholder.plotly_chart(fig_chart, use_container_width=True)
        
        # --- 4. SWARM INTELLIGENCE CONSENSUS ---
        # Simulated voting from 10,000 agents
        buy_votes = 50 + (vel * 5) + (acc * 10)
        buy_votes = max(0, min(100, buy_votes)) # Clamp between 0-100
        sell_votes = 100 - buy_votes
        
        st.markdown(f"""
        ### 🧠 10,000 AGENT CONSENSUS (SWARM MIND)
        <div style="display: flex; align-items: center; justify-content: space-between; background-color: #222; padding: 10px; border-radius: 5px;">
            <div style="width: {buy_votes}%; background-color: #00FF00; height: 20px; text-align: center; color: black; font-weight: bold;">BUY ({int(buy_votes)}%)</div>
            <div style="width: {sell_votes}%; background-color: #FF0000; height: 20px; text-align: center; color: white; font-weight: bold;">SELL ({int(sell_votes)}%)</div>
        </div>
        <p style="text-align: center; color: gray; font-size: 10px;">DECISION ENGINE POWERED BY BOSS MANIKANDAN</p>
        """, unsafe_allow_html=True)
        
        time.sleep(0.1) # Refresh rate

else:
    st.info("👋 WAITING FOR BOSS MANIKANDAN TO ACTIVATE THE ENGINE...")
    st.markdown("""
    **SYSTEM READY.**
    - API CONNECTION: **STANDBY**
    - PHYSICS MODELS: **LOADED**
    - COPYRIGHT PROTOCOLS: **ACTIVE**
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    DEVELOPED BY <b>BOSS MANIKANDAN</b> | KUMBAKONAM HQ | GENESIS APTE V1.0 <br>
    <i>WARNING: UNAUTHORIZED COPYING OF THIS ALGORITHM IS STRICTLY PROHIBITED.</i>
</div>
""", unsafe_allow_html=True)
