import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import hmac
import time
from datetime import datetime
import plotly.graph_objects as go

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="CM-X JARVIS: SHADOW EMPEROR",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SECURITY LOCK (பாதுகாப்பு பூட்டு) ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        # பாஸ்வேர்ட்: "boss" (இதை நீங்க secrets-ல மாத்திக்கலாம்)
        if hmac.compare_digest(st.session_state["password"], st.secrets["system"]["admin_password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Login Screen Styling
    st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #00f3ff; }
        .stTextInput > div > div > input { color: #00f3ff; background-color: #111; border: 1px solid #00f3ff; }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: #00f3ff;'>🦁 CM-X SECURITY GATE</h1>", unsafe_allow_html=True)
    st.text_input("ENTER ACCESS CODE:", type="password", on_change=password_entered, key="password")
    
    if "password_correct" in st.session_state:
        st.error("⛔ ACCESS DENIED: தவறான கடவுச்சொல்!")
    return False

# Secrets இல்லாத பட்சத்தில், லோக்கல் ரன்னுக்கு ஒரு தற்காலிக வழி
if "system" not in st.secrets:
    st.warning("⚠️ Secrets not found. Using default password: 'boss'")
    st.secrets["system"] = {"admin_password": "boss"}

if not check_password():
    st.stop()

# --- 3. UI THEME (HOLOGRAPHIC & FIRE) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #050505; }
    
    /* Fire Text Animation */
    .fire-text {
        background: linear-gradient(to top, #ff0000, #ff8800, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 1.5rem;
        text-shadow: 0 0 10px rgba(255, 69, 0, 0.6);
        animation: flicker 1.5s infinite alternate;
    }
    @keyframes flicker {
        0% { opacity: 1; text-shadow: 0 0 10px red; }
        100% { opacity: 0.8; text-shadow: 0 0 20px orange; }
    }

    /* Holographic Cards */
    .holo-card {
        background: rgba(10, 20, 30, 0.8);
        border: 1px solid rgba(0, 243, 255, 0.3);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        color: #00f3ff;
    }

    /* Video Maximize Button Style (Simulated via CSS) */
    .video-container { position: relative; border: 2px solid #f59e0b; border-radius: 10px; overflow: hidden; }
    
    /* Jarvis Button */
    .stButton>button {
        background: linear-gradient(45deg, #00f3ff, #0066ff);
        color: black;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. HEADER & IDENTITY ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown("# CM-X <span style='color:#00f3ff'>JARVIS</span>", unsafe_allow_html=True)
    st.markdown("<div class='fire-text'>CREATOR: BOSS MANIKANDAN</div>", unsafe_allow_html=True)
with col_head2:
    st.markdown("### 🟢 ONLINE")
    st.markdown(f"**MODE:** {st.secrets.get('system', {}).get('mode', 'SHADOW_EMPEROR')}")

# --- 5. INITIALIZE SESSION STATE ---
if 'price' not in st.session_state: st.session_state['price'] = 19500.0
if 'vwap' not in st.session_state: st.session_state['vwap'] = 19500.0
if 'data_history' not in st.session_state: 
    st.session_state['data_history'] = pd.DataFrame(columns=['Time', 'Price', 'VWAP'])

# --- 6. MAIN DASHBOARD LAYOUT ---
# Left: Video & Chart | Right: AI & Controls
col_left, col_right = st.columns([2, 1])

with col_left:
    # --- VIDEO FEED (With Maximize) ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("**📡 LIVE MARKET FEED**")
    # Using a placeholder video or trading view iframe
    # Note: Streamlit's native video player has a fullscreen button built-in
    st.video("https://www.w3schools.com/html/mov_bbb.mp4", format="video/mp4", start_time=0) 
    st.caption("Feed Source: Upstox Secure Stream")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- LIVE CHART (Plotly for interactivity) ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    
    # Simulate Data Update
    move = np.random.randint(-10, 15)
    st.session_state['price'] += move
    st.session_state['vwap'] = (st.session_state['vwap'] * 19 + st.session_state['price']) / 20
    
    # Add to history
    now_str = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame({'Time': [now_str], 'Price': [st.session_state['price']], 'VWAP': [st.session_state['vwap']]})
    st.session_state['data_history'] = pd.concat([st.session_state['data_history'], new_row]).tail(50)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state['data_history']['Time'], y=st.session_state['data_history']['Price'], mode='lines', name='Price', line=dict(color='#00f3ff')))
    fig.add_trace(go.Scatter(x=st.session_state['data_history']['Time'], y=st.session_state['data_history']['VWAP'], mode='lines', name='VWAP', line=dict(color='#ff9900', dash='dash')))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#00f3ff'), height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # --- METRICS ---
    st.markdown('<div class="holo-card" style="text-align:center;">', unsafe_allow_html=True)
    st.metric(label="NIFTY 50", value=f"{st.session_state['price']:.2f}", delta=f"{move:.2f}")
    
    velocity = move
    vel_color = "#10b981" if velocity > 0 else "#ef4444"
    st.markdown(f"**VELOCITY:** <span style='color:{vel_color}; font-size:1.2rem; font-weight:bold;'>{velocity:.2f}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- JARVIS AI CHAT ---
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.subheader("🤖 ASK JARVIS")
    
    user_query = st.text_input("Command:", placeholder="e.g., Analyze Trend")
    
    if st.button("ACTIVATE JARVIS"):
        if not user_query:
            st.warning("Please enter a command.")
        else:
            api_key = st.secrets["api_keys"]["Gemini_api_key"]
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Act as JARVIS for Boss Manikandan. Market Price: {st.session_state['price']}. User asks: {user_query}. Keep it robotic and short."
                    with st.spinner("Processing..."):
                        response = model.generate_content(prompt)
                    
                    st.success(f"🦁 **JARVIS:** {response.text}")
                    
                    # Voice Output (Javascript Injection)
                    # This simple script will make the browser speak the response
                    js_code = f"""
                        <script>
                            var msg = new SpeechSynthesisUtterance("{response.text.replace('"', '')}");
                            window.speechSynthesis.speak(msg);
                        </script>
                    """
                    st.components.v1.html(js_code, height=0)
                    
                except Exception as e:
                    st.error(f"AI Error: {e}")
            else:
                st.error("API Key Missing in Secrets!")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. AUTO REFRESH (Simulating Live Feed) ---
# In a real app, use st.empty() or streamlit-autorefresh
time.sleep(1)
st.rerun()
