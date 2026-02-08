import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
from datetime import datetime

# --- 1. PAGE CONFIGURATION (அழகுபடுத்துதல்) ---
st.set_page_config(
    page_title="CM-X GENESIS AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (நம்ம கெத்துக்காக) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #4b4b4b; text-align: center; }
    .big-font { font-size: 24px !important; font-weight: bold; color: #4caf50; }
    .status-up { color: #00ff00; font-weight: bold; }
    .status-down { color: #ff0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROLS (கட்டுப்பாட்டு அறை) ---
with st.sidebar:
    st.title("🤖 COMMAND CENTER")
    st.markdown("---")
    
    # API Keys
    st.subheader("🔑 Secret Keys")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Paste AIza...")
    upstox_id = st.text_input("Upstox ID", value="BOSS_MANI")
    
    st.markdown("---")
    
    # Controls
    st.subheader("⚙️ System Controls")
    mode = st.radio("Trading Mode", ["Paper Trading", "LIVE EXECUTION"], index=0)
    qty = st.slider("Quantity (Lots)", 1, 10, 2)
    
    st.markdown("---")
    status_placeholder = st.empty()
    if st.button("🔴 STOP BOT"):
        st.session_state.running = False
        status_placeholder.error("SYSTEM HALTED!")

# --- 3. SESSION STATE (ஞாபக சக்தி) ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Price', 'VWAP'])
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'running' not in st.session_state:
    st.session_state.running = False

# --- 4. MAIN DASHBOARD UI ---

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 CM-X GENESIS: BRAHMASTRA AI")
    st.caption(f"System Status: ONLINE | Mode: {mode}")
with col2:
    if st.button("▶️ START ENGINE", type="primary"):
        st.session_state.running = True

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
placeholder_price = m1.empty()
placeholder_rsi = m2.empty()
placeholder_trend = m3.empty()
placeholder_pcr = m4.empty()

# Chart Area
st.subheader("📈 Live Market Scanner")
chart_placeholder = st.empty()

# AI Chat Area
st.markdown("---")
st.subheader("💬 Ask Chellakili (AI Assistant)")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input for AI
if prompt := st.chat_input("பாஸ், என்ன சந்தேகம்? கேளுங்க..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gemini Response Logic
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
            
            # Context for AI
            context = f"You are a trading assistant named Chellakili. Current Market Price is volatile. User asks: {prompt}. Answer in Tamil/Tanglish briefly."
            
            response = model.generate_content(context)
            ai_reply = response.text
        except Exception as e:
            ai_reply = f"AI Error: {str(e)}"
    else:
        ai_reply = "பாஸ், சைடுல அந்த Gemini API Key-ஐ போடுங்க, அப்போ தான் நான் பேசுவேன்!"

    # Add AI message
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    with st.chat_message("assistant"):
        st.markdown(ai_reply)

# --- 5. SIMULATION LOOP (உயிர் ஓட்டம்) ---
if st.session_state.running:
    # Simulating Data Stream
    new_price = 19500 + np.random.randint(-10, 10)
    rsi_val = 50 + np.random.randint(-5, 5)
    
    # Update Metrics
    placeholder_price.metric("Nifty 50", f"₹{new_price}", "12.5")
    placeholder_rsi.metric("RSI (14)", f"{rsi_val}", "-2.1")
    
    trend_color = "🟢 UP" if new_price > 19500 else "🔴 DOWN"
    placeholder_trend.metric("SuperTrend", trend_color)
    placeholder_pcr.metric("PCR Value", "1.12", "Bullish")
    
    # Update Chart Data
    now = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame({'Time': [now], 'Price': [new_price], 'VWAP': [new_price - 5]})
    st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(50) # Keep last 50
    
    # Draw Chart
    chart_data = st.session_state.history.set_index('Time')
    chart_placeholder.line_chart(chart_data[['Price', 'VWAP']])
    
    # Auto Rerun to simulate live feed (Small Hack for Streamlit)
    time.sleep(1)
    st.rerun()
