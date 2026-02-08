import streamlit as st
import pandas as pd
import numpy as np
import time
import google.generativeai as genai
import datetime

# --- 1. PAGE CONFIGURATION (பக்கம் அமைப்பு) ---
st.set_page_config(
    page_title="CM-X JARVIS: WAR ROOM",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (சைபர் பங்க் தீம்) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000000 100%);
        color: #00f3ff;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Courier New', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #ffffff;
        text-shadow: 0 0 10px #00f3ff;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(20, 20, 30, 0.8);
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.05);
        border-color: #00f3ff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        height: 50px;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #0072ff;
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #111;
        color: #00f3ff;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SECURITY SYSTEM (பாதுகாப்பு) ---
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("## 🔐 IDENTITY VERIFICATION")
            password = st.text_input("ENTER ACCESS CODE:", type="password")
            if st.button("UNLOCK SYSTEM"):
                if password == "boss":  # Password
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("⛔ ACCESS DENIED: Intruder Alert!")
        return False
    return True

if not check_login():
    st.stop()

# --- 4. SIDEBAR CONFIG (கண்ட்ரோல் ரூம்) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.markdown("### ⚙️ SYSTEM CONFIG")
    
    # API Keys Management
    gemini_key = st.text_input("Gemini API Key", type="password", help="Enter Google AI Key")
    
    st.markdown("---")
    st.markdown("### 📡 DATA STREAM")
    auto_refresh = st.checkbox("🔄 AUTO-REFRESH MODE", value=False)
    refresh_rate = st.slider("Rate (Seconds)", 1, 10, 2)
    
    if st.button("🚪 LOGOUT"):
        st.session_state.authenticated = False
        st.rerun()

# --- 5. MAIN DASHBOARD ---
# Header
c1, c2 = st.columns([4, 1])
with c1:
    st.title("🦁 CM-X JARVIS: WAR ROOM")
    st.caption(f"OPERATOR: BOSS MANIKANDAN | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with c2:
    st.metric(label="SYSTEM STATUS", value="ONLINE", delta="STABLE")

st.markdown("---")

# --- 6. LIVE MARKET DATA (SIMULATION) ---
# நிஜ மார்க்கெட் டேட்டா வரும்போது இங்கே Upstox Logic போடலாம்.
current_price = 24500 + np.random.randint(-50, 50)
vwap_val = current_price + np.random.randint(-20, 20)
rsi_val = 50 + np.random.randint(-10, 10)

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("NIFTY 50", f"₹{current_price}", f"{np.random.randint(-10, 10)} pts")
m2.metric("VWAP LEVEL", f"₹{vwap_val}", delta_color="off")
m3.metric("RSI INDICATOR", f"{rsi_val}", "NEUTRAL" if 40 < rsi_val < 60 else "ALERT")
m4.metric("VOLATILITY", "HIGH", "CRITICAL", delta_color="inverse")

# --- 7. DECISION ENGINE (மூளை) ---
st.subheader("🧠 JARVIS INTELLIGENCE CORE")

col_chart, col_ai = st.columns([2, 1])

with col_chart:
    # Live Chart Simulation
    chart_data = pd.DataFrame({
        'Price': np.random.randn(50).cumsum() + current_price,
        'VWAP': np.random.randn(50).cumsum() + vwap_val
    })
    st.line_chart(chart_data, color=["#00f3ff", "#f59e0b"])

with col_ai:
    st.markdown("#### 🛡️ STRATEGY SIGNAL")
    
    # Logic
    signal = "WAIT & WATCH"
    color = "gray"
    
    if current_price > vwap_val + 20 and rsi_val > 55:
        signal = "🚀 BUY CALL (CE)"
        color = "green"
    elif current_price < vwap_val - 20 and rsi_val < 45:
        signal = "🩸 BUY PUT (PE)"
        color = "red"
        
    st.markdown(f"""
        <div style="background-color:{color}; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="margin:0; color:white;">{signal}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Gemini AI Chat
    user_query = st.text_input("🤖 COMMAND JARVIS:", placeholder="Type your query...")
    
    if user_query:
        if not gemini_key:
            st.warning("⚠️ Please enter Gemini API Key in Sidebar")
        else:
            try:
                with st.spinner("Analyzing..."):
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"You are Jarvis, an AI assistant for Stock Trader Boss Manikandan. Keep answers short and technical. Query: {user_query}")
                    st.success(f"🦁 **JARVIS:** {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# --- 8. TRADE EXECUTION ---
with st.expander("⚡ INSTANT EXECUTION TERMINAL", expanded=True):
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        qty = st.number_input("QUANTITY (Lots)", min_value=1, value=1)
    with ec2:
        if st.button("✅ EXECUTE TRADE"):
            st.toast(f"Order Placed for {qty} Lots! [SIMULATION]", icon="🦅")
    with ec3:
        if st.button("❌ EMERGENCY EXIT"):
            st.toast("ALL POSITIONS CLOSED!", icon="🔥")

# --- 9. AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
            
