import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import google.genai as genai # Modified import statement
import time
from datetime import datetime
import pytz
import json
import os
from gtts import gTTS
import base64
from collections import deque
import math
import random
from scipy.stats import entropy as scipy_entropy # Renamed to avoid conflict

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="AETHER: FUSION GOD MODE",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="collapsed"
)

# --- 2. GLOBAL SYSTEM CONSTANTS ---
MEMORY_FILE = "cm_x_aether_memory.json"
MAX_HISTORY_LEN = 300  # From KVicPlrusRE5, more history for better analysis
TELEGRAM_INTERVAL = 120 # 2 Minutes
KILL_SWITCH_LOSS = -2000 # Max loss for bot shutdown
TRADE_QUANTITY = 50 # Default trade quantity

# --- 3. SECRETS & API ---
# Load secrets with fallback for simulation
try:
    if "general" in st.secrets: OWNER_NAME = st.secrets["general"]["owner"]
    else: OWNER_NAME = "BOSS MANIKANDAN"

    UPSTOX_ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    # Check for telegram secrets, make optional for local testing
    if "telegram" in st.secrets:
        TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["bot_token"]
        TELEGRAM_CHAT_ID = st.secrets["telegram"]["chat_id"]
    else:
        TELEGRAM_BOT_TOKEN = None
        TELEGRAM_CHAT_ID = None

    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro') # Using 1.5-pro for better advice
except Exception as e:
    st.error(f"⚠️ SYSTEM FAILURE: Secrets Error - {e}")
    # Provide dummy values for local development without secrets
    OWNER_NAME = "SIMULATION MODE"
    UPSTOX_ACCESS_TOKEN = None
    GEMINI_API_KEY = None
    TELEGRAM_BOT_TOKEN = None
    TELEGRAM_CHAT_ID = None
    st.warning("Running in simulation mode due to missing API keys. Data will be random.")
    # st.stop() # Commented out to allow running without secrets

UPSTOX_URL = 'https://api.upstox.com/v2/market-quote/ltp'
REQ_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
TIMEZONE = pytz.timezone('Asia/Kolkata') # Consistent timezone for logging

# --- 4. ADVANCED CYBERPUNK STYLING (DIM & NEON) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* BASE APP STYLING */
    .stApp {
        background-color: #1e293b; /* Dim Dark Blue-Grey */
        color: #e2e8f0; /* Off-white for general text */
        font-family: 'Roboto Mono', monospace;
    }

    /* HEADINGS */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #f8fafc; /* White */
        text-shadow: 0 0 5px rgba(255,255,255,0.3);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* METRIC CARDS (Smooth Round with Neon Accents) */
    div[data-testid="stMetric"] {
        background-color: #0f172a; /* Darker Inner Box */
        border: 1px solid #334155;
        border-radius: 15px; /* Rounded Corners */
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-family: 'Orbitron', sans-serif;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8; /* Light blue-grey */
        font-size: 14px;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        color: #f1f5f9; /* Off-white */
        font-size: 26px;
        font-weight: bold;
        text-shadow: 0 0 5px rgba(255,255,255,0.1);
    }
    /* Metric Delta colors (green, red) */
    div[data-testid="stMetricDelta"] svg { display: none; } /* Hide default arrow */
    div[data-testid="stMetricDelta"] {
        font-weight: bold;
        color: #4ade80; /* Neon Green */
        text-shadow: 0 0 5px #4ade80;
    }
    div[data-testid="stMetricDelta"][data-testid="stMetricDelta"] {
        color: #f87171; /* Neon Red */
        text-shadow: 0 0 5px #f87171;
    }

    /* NEON ACCENTS */
    .neon-green { color: #4ade80; text-shadow: 0 0 8px #4ade80; font-weight:bold; }
    .neon-white { color: #f8fafc; text-shadow: 0 0 8px #f8fafc; font-weight:bold; }
    .neon-red { color: #f87171; text-shadow: 0 0 8px #f87171; font-weight:bold; }
    .neon-orange { color: #fbbf24; text-shadow: 0 0 8px #fbbf24; font-weight:bold; }
    .neon-purple { color: #c084fc; text-shadow: 0 0 8px #c084fc; font-weight:bold; }

    /* TERMINAL LOG BOX */
    .terminal-box {
        font-family: 'Fira Code', monospace;
        background-color: #020617; /* Very dark blue-black */
        color: #4ade80; /* Neon green default log text */
        padding: 15px;
        height: 250px; /* Adjusted height */
        overflow-y: auto;
        font-size: 14px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.1);
        line-height: 1.4;
    }
    .log-row { border-bottom: 1px solid rgba(51, 65, 85, 0.2); padding: 2px 0; }
    .log-time { color: #64748b; margin-right: 10px; font-weight:normal; } /* Subtle timestamp */
    .log-info { color: #4ade80; } /* Neon Green */
    .log-warn { color: #fbbf24; } /* Neon Orange */
    .log-danger { color: #f87171; } /* Neon Red */
    .log-buy { color: #34d399; font-weight: bold; } /* Stronger green for buys */
    .log-sell { color: #ef4444; font-weight: bold; } /* Stronger red for sells */
    .log-ai { color: #a78bfa; font-style: italic; } /* Purple for AI */


    /* BUTTONS */
    .stButton>button {
        background: #334155; /* Dark blue-grey */
        color: #e2e8f0; /* Off-white */
        border: 1px solid #94a3b8;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
        height: 50px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: #475569; /* Lighter blue-grey on hover */
        border-color: #e2e8f0;
        box-shadow: 0 0 10px rgba(226, 232, 240, 0.3);
    }

    /* ACTIVE TRADE WARNING */
    .active-trade-box {
        background-color: rgba(248, 113, 113, 0.1); /* Very subtle red background */
        border: 2px solid #f87171; /* Neon Red border */
        color: #f8fafc; /* White text */
        padding: 15px;
        text-align: center;
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        margin-bottom: 20px;
        animation: pulse-red 2s infinite;
        box-shadow: 0 0 15px rgba(248, 113, 113, 0.5);
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 5px rgba(248, 113, 113, 0.5); }
        50% { box-shadow: 0 0 20px rgba(248, 113, 113, 0.9); }
        100% { box-shadow: 0 0 5px rgba(248, 113, 113, 0.5); }
    }

    /* STATUS BOXES */
    .status-online {
        background-color: rgba(74, 222, 128, 0.1); border: 1px solid #4ade80;
        color: #4ade80; padding: 8px; border-radius: 5px; text-align: center;
        font-weight: bold; margin-bottom: 10px; text-shadow: 0 0 5px #4ade80;
    }
    .status-offline {
        background-color: rgba(248, 113, 113, 0.1); border: 1px solid #f87171;
        color: #f87171; padding: 8px; border-radius: 5px; text-align: center;
        font-weight: bold; margin-bottom: 10px; text-shadow: 0 0 5px #f87171;
    }

    /* AGENT CARDS (Council) */
    .agent-card {
        background: #0f172a; border: 1px solid #334155;
        padding: 8px; text-align: center; border-radius: 8px;
        font-size: 13px; color: #e2e8f0; margin-bottom: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .agent-card.BUY { background-color: rgba(74, 222, 128, 0.2); border-color: #4ade80; } /* Neon Green */
    .agent-card.SELL { background-color: rgba(248, 113, 113, 0.2); border-color: #f87171; } /* Neon Red */
    .agent-card.WAIT { background-color: rgba(251, 191, 36, 0.1); border-color: #fbbf24; } /* Neon Orange */
    .agent-card.GO { background-color: rgba(192, 132, 252, 0.1); border-color: #c084fc; } /* Neon Purple */

    .approval-box {
        background-color: rgba(251, 191, 36, 0.1); border: 2px solid #fbbf24;
        color: #fbbf24; padding: 10px; text-align: center; border-radius: 8px;
        font-weight: bold; margin-bottom: 10px; animation: pulse-orange 1.5s infinite;
    }
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 5px rgba(251, 191, 36, 0.5); }
        50% { box_shadow: 0 0 15px rgba(251, 191, 36, 0.9); }
        100% { box_shadow: 0 0 5px rgba(251, 191, 36, 0.5); }
    }

    </style>
    """, unsafe_allow_html=True)

# --- 5. THE GROWING BRAIN (MEMORY) ---
def init_brain():
    """Initializes or loads the persistent memory for AETHER."""
    if not os.path.exists(MEMORY_FILE):
        data = {
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "trade_log": [], # Detailed log of closed trades
            "weights": {"Physics": 1.5, "Trend": 1.0, "Global": 1.2, "Chaos": 0.8, "WinProb": 1.0}, # Self-improving weights
            "global_sentiment": "NEUTRAL", # User-controlled global market sentiment
            "market_knowledge": "System Initialized. Observing quantum fluctuations.",
            # Session-like data that needs to be restored on restart
            "position": None, # Current open position (type, entry, qty, opt)
            "pending_signal": None, # A signal awaiting user approval
            "trailing_high": 0.0, # Used for trailing stop-loss logic
            "last_tg_time": time.time(), # Last telegram report timestamp
            "bot_active_on_exit": False # To remember bot state across reruns
        }
        with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
        return data
    else:
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                # Ensure new keys are added if memory file is older
                data.setdefault("weights", {"Physics": 1.5, "Trend": 1.0, "Global": 1.2, "Chaos": 0.8, "WinProb": 1.0})
                data.setdefault("global_sentiment", "NEUTRAL")
                data.setdefault("market_knowledge", "System Initialized. Observing quantum fluctuations.")
                data.setdefault("pending_signal", None)
                data.setdefault("trailing_high", 0.0)
                data.setdefault("last_tg_time", time.time())
                data.setdefault("bot_active_on_exit", False)
                data.setdefault("trade_log", [])
                return data
        except Exception as e:
            st.error(f"Error loading memory file: {e}. Reinitializing memory.")
            return init_brain() # Reinitialize on error

def save_brain(mem_data):
    """Saves the current state of AETHER's memory to a file."""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(mem_data, f, indent=4)

brain_memory = init_brain()

# --- 6. SESSION STATE SYNCHRONIZATION ---
# Initialize session state variables, often mirroring persistent memory
if 'prices' not in st.session_state: st.session_state.prices = deque(maxlen=MAX_HISTORY_LEN)
if 'bot_active' not in st.session_state: st.session_state.bot_active = brain_memory.get("bot_active_on_exit", False)
if 'position' not in st.session_state: st.session_state.position = brain_memory.get("position", None)
if 'pending_signal' not in st.session_state: st.session_state.pending_signal = brain_memory.get("pending_signal", None)
if 'audio_html' not in st.session_state: st.session_state.audio_html = ""
if 'live_logs' not in st.session_state: st.session_state.live_logs = deque(maxlen=50) # The CSV style logs
if 'last_tg_time' not in st.session_state: st.session_state.last_tg_time = brain_memory.get("last_tg_time", time.time())
if 'trailing_high' not in st.session_state: st.session_state.trailing_high = brain_memory.get("trailing_high", 0.0)

# --- 7. AUDIO & LOGGING SYSTEM ---
def speak_aether(text):
    """Converts text to speech and plays it in the browser, also logs the speech."""
    try:
        # Update market knowledge with what AETHER just said
        brain_memory["market_knowledge"] = f"AETHER: {text}"
        add_log(f"AETHER: {text}", "log-ai")

        tts = gTTS(text=text, lang='en', tld='co.in')
        filename = "aether_speak.mp3"
        tts.save(filename)
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.session_state.audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    except Exception as e:
        add_log(f"Audio Error: {e}", "log-danger")
        st.session_state.audio_html = "" # Clear audio if error

def add_log(msg, css_class="log-info"):
    """Adds a timestamped, color-coded message to the live logs deque."""
    ts = datetime.now(TIMEZONE).strftime("%H:%M:%S")
    entry = f"<div class='log-row'><span class='log-time'>{ts}</span> <span class='{css_class}'>{msg}</span></div>"
    st.session_state.live_logs.appendleft(entry)

def send_telegram_report(msg):
    """Sends a message to a Telegram chat if configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # add_log("Telegram keys missing. Report not sent.", "log-warn")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🧬 AETHER REPORT:\n\n{msg}",
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        add_log(f"Telegram Comm Error: {e}", "log-danger")

# --- 8. MATHEMATICAL AND PHYSICS CORE (CognitiveEngine) ---
class CognitiveEngine:
    def calculate_newton_metrics(self, prices_deque):
        """Calculates velocity, acceleration, and entropy (chaos) of prices."""
        p = np.array(list(prices_deque))
        if len(p) < 10: return 0.0, 0.0, 0.0 # Not enough data

        # Velocity (v) = Price Change
        v = p[-1] - p[-2] if len(p) >= 2 else 0.0

        # Acceleration (a) = Change in Velocity
        a = (p[-1] - p[-2]) - (p[-2] - p[-3]) if len(p) >= 3 else 0.0

        # Chaos Theory (Entropy) - using scipy's entropy
        # Use last 20 prices for entropy calculation to capture recent volatility
        if len(p) >= 20:
            hist, _ = np.histogram(p[-20:], bins=10, density=True)
            probs = hist / hist.sum()
            probs = probs[probs > 0] # Remove zero probabilities for log
            entropy_val = scipy_entropy(probs)
        else:
            entropy_val = 0.0 # Default if not enough data

        return v, a, entropy_val

    def monte_carlo_simulation(self, prices_deque, num_sims=100, steps=10):
        """
        Performs Monte Carlo simulation to forecast price direction and estimate win probability.
        Returns the probability of the next price being higher than the last.
        """
        p_array = np.array(list(prices_deque))
        if len(p_array) < 20: # Need sufficient historical data for meaningful simulation
            return 0.5 # Neutral probability

        last_price = p_array[-1]
        returns = np.diff(p_array) / p_array[:-1]
        mu = np.mean(returns) # Mean daily return
        sigma = np.std(returns) # Volatility

        bullish_paths = 0
        for _ in range(num_sims):
            sim_price = last_price
            for _ in range(steps):
                # Random shock based on historical mean and std dev
                shock = np.random.normal(mu, sigma)
                sim_price = sim_price * (1 + shock)
            if sim_price > last_price:
                bullish_paths += 1
        return bullish_paths / num_sims # Probability of an upward movement

    def monte_carlo_forecast(self, prices_deque, forecast_steps=5):
        """
        Forecasts the price for the next few steps using Monte Carlo,
        returning the average of the simulated end prices.
        """
        p_array = np.array(list(prices_deque))
        if len(p_array) < 20:
            return p_array[-1] if len(p_array) > 0 else 0.0

        last_price = p_array[-1]
        returns = np.diff(p_array) / p_array[:-1]
        mu = np.mean(returns)
        sigma = np.std(returns)

        simulated_prices = []
        for _ in range(50): # Run 50 simulations for the forecast
            sim_path_price = last_price
            for _ in range(forecast_steps):
                shock = np.random.normal(mu, sigma)
                sim_path_price = sim_path_price * (1 + shock)
            simulated_prices.append(sim_path_price)
        return np.mean(simulated_prices)

    def rocket_formula(self, v, vol_current, vol_avg):
        """Calculates 'Thrust' based on velocity and volatility."""
        if vol_avg == 0: vol_avg = 1
        mass_ratio = abs(vol_current / vol_avg)
        if mass_ratio <= 0: mass_ratio = 0.1
        thrust = v * math.log(mass_ratio + 1)
        return thrust

    def get_best_option_strike(self, spot_price, direction):
        """Determines the nearest ATM strike price for Nifty 50."""
        # Nifty 50 strikes are typically in multiples of 50
        strike = round(spot_price / 50) * 50
        if direction == "BUY": return f"NIFTY {strike} CE" # Call Option
        else: return f"NIFTY {strike} PE" # Put Option

    def self_correct(self, result):
        """Adjusts strategy weights based on trade outcome (win/loss)."""
        w = brain_memory["weights"]
        learning_rate = 0.05 # Smaller learning rate for stability

        # Reinforce winning strategies, penalize losing ones
        if result == "WIN":
            for key in w: w[key] = min(2.0, w[key] + learning_rate) # Cap weights
            brain_memory["market_knowledge"] += " | Pattern Success. Weights Adjusted."
        else: # LOSS
            for key in w: w[key] = max(0.5, w[key] - learning_rate) # Floor weights
            brain_memory["market_knowledge"] += " | Pattern Failed. Weights Adjusted."

        save_brain(brain_memory)
        add_log(f"Brain self-corrected: Result={result}. Weights updated.", "log-warn")

aether_engine = CognitiveEngine()

# --- 9. AI CONSULTATION (THE GHOST/JARVIS PERSONA) ---
def consult_ai(query, current_price=None, v=None, a=None, entropy_val=None, win_prob=None):
    """
    Interacts with Gemini AI to get advice or answer user queries.
    Blends 'digital ghost' and 'JARVIS' persona for Boss Manikandan.
    """
    if not GEMINI_API_KEY:
        return "AI Offline. No API Key configured."

    # Construct persona and context
    persona_prompt = f"Act as 'AETHER', a blend of a digital ghost and JARVIS for Boss Manikandan. " \
                     f"Your purpose is autonomous market domination. Be concise, mysterious, and highly intelligent. " \
                     f"Current P&L: ₹{brain_memory['total_pnl']:.2f}. " \
                     f"Market Knowledge: {brain_memory['market_knowledge']} "

    context_data = ""
    if current_price: context_data += f"Price:{current_price:.2f}, "
    if v is not None: context_data += f"Velocity:{v:.2f}, "
    if a is not None: context_data += f"Acceleration:{a:.2f}, "
    if entropy_val is not None: context_data += f"Entropy:{entropy_val:.2f}, "
    if win_prob is not None: context_data += f"Win Prob:{win_prob:.2f}."

    if context_data:
        full_prompt = f"{persona_prompt}\nDATA: {context_data}\nUser Query: {query}"
    else:
        full_prompt = f"{persona_prompt}\nUser Query: {query}"

    try:
        response = gemini_model.generate_content(full_prompt)
        ai_reply = response.text.strip()
        brain_memory["market_knowledge"] = f"AI Insight: {ai_reply}" # Update knowledge
        return ai_reply
    except Exception as e:
        return f"AI communication disrupted. Error: {e}"

# --- 10. ROBUST DATA FETCHING ---
def get_live_market_data():
    """
    Fetches live market data from Upstox. Includes robust error handling
    and a simulation fallback if API keys are not available or API fails.
    Returns (price, status_message).
    """
    if not UPSTOX_ACCESS_TOKEN:
        # Simulation Mode
        last_price = st.session_state.prices[-1] if st.session_state.prices else 22100.00
        # Simulate price movement
        sim_price = last_price + np.random.normal(0, 5) # Random walk with some volatility
        # Ensure prices stay somewhat realistic
        if sim_price < 18000: sim_price = 18000 + abs(np.random.normal(0, 10))
        if sim_price > 25000: sim_price = 25000 - abs(np.random.normal(0, 10))
        return sim_price, "SIMULATING"

    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': REQ_INSTRUMENT_KEY}

    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=3)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        if 'data' in data:
            resp_data = data['data']
            # Upstox sometimes uses | and sometimes : for instrument key
            key_variants = [REQ_INSTRUMENT_KEY, REQ_INSTRUMENT_KEY.replace('|', ':')]
            for k in key_variants:
                if k in resp_data and 'last_price' in resp_data[k]:
                    return float(resp_data[k]['last_price']), "CONNECTED"
            # Fallback if specific key not found, try first available instrument
            if resp_data:
                first_key = list(resp_data.keys())[0]
                if 'last_price' in resp_data[first_key]:
                    return float(resp_data[first_key]['last_price']), "CONNECTED"
            return None, "DATA STRUCTURE ERROR: Last price not found."
        else:
            return None, f"API RESPONSE ERROR: {data.get('message', 'Unknown error in data.')}"

    except requests.exceptions.Timeout:
        return None, "NET ERROR: Request timed out."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return None, "TOKEN EXPIRED: Unauthorized access."
        return None, f"API ERROR: HTTP {e.response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "NET ERROR: Could not connect to Upstox API."
    except Exception as e:
        return None, f"UNEXPECTED ERROR: {e}"

# --- 11. UI DASHBOARD LAYOUT ---
st.markdown(f"""
<div style="text-align: center; padding-bottom: 10px; border-bottom: 2px solid #334155;">
    <h1>AETHER: FUSION GOD MODE</h1>
    <p class="neon-white">OPERATOR: {OWNER_NAME} | BRAIN: <span class="neon-purple">ACTIVE</span> | LR: {brain_memory['weights']['Physics']:.2f} (avg)</p>
</div>
""", unsafe_allow_html=True)

# Invisible Audio Player
st.markdown(st.session_state.audio_html, unsafe_allow_html=True)

# Connection Status
price_check_val, price_check_status = get_live_market_data()
if price_check_status == "CONNECTED":
    st.markdown('<div class="status-online">🟢 UPSTOX CONNECTED | DATA FLOWING</div>', unsafe_allow_html=True)
elif price_check_status == "SIMULATING":
    st.markdown('<div class="status-online">🟡 SIMULATION MODE | GENERATING DATA</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-offline">🔴 {price_check_status} | CHECK TOKEN/CONNECTION</div>', unsafe_allow_html=True)

# Active Trade Warning
active_trade_placeholder = st.empty()
if st.session_state.position:
    pos = st.session_state.position
    active_trade_placeholder.markdown(f"""
    <div class="active-trade-box">
        <h3>⚠ QUANTUM ENTANGLEMENT ACTIVE (TRADE OPEN)</h3>
        <p>{pos['type']} {pos['opt']} | ENTRY: {pos['entry']:.2f} | QTY: {pos['qty']} </p>
    </div>
    """, unsafe_allow_html=True)

# Main Grid Layout
col_main, col_sidebar = st.columns([2.5, 1])

with col_main:
    st.subheader("📈 Quantum Trajectory Analysis")
    chart_placeholder = st.empty()

    # Metrics Row
    metric_cols = st.columns(5)
    price_metric = metric_cols[0].empty()
    velocity_metric = metric_cols[1].empty()
    accel_metric = metric_cols[2].empty()
    entropy_metric = metric_cols[3].empty()
    winprob_metric = metric_cols[4].empty()

    st.write("---")
    st.subheader("🏛️ THE COUNCIL (DECISION CORE)")
    council_placeholder = st.empty()

    st.write("---")
    st.subheader("🖥️ NEURAL CORE LOGS")
    log_placeholder = st.empty()

with col_sidebar:
    st.subheader("🧠 CONTROL DECK")

    # Global Sentiment Slider
    current_global_sentiment = brain_memory.get("global_sentiment", "NEUTRAL")
    new_global_sentiment = st.select_slider(
        "Market Sentiment Overwrite",
        options=["BEARISH", "NEUTRAL", "BULLISH"],
        value=current_global_sentiment,
        key="global_sentiment_slider"
    )
    if new_global_sentiment != current_global_sentiment:
        brain_memory["global_sentiment"] = new_global_sentiment
        save_brain(brain_memory)
        add_log(f"Global sentiment set to {new_global_sentiment}", "log-warn")

    # AI Consultation
    st.write("---")
    ai_consult_input = st.text_input("Consult AETHER:", placeholder="e.g., Is Nifty bullish?", key="ai_input")
    ai_reply_placeholder = st.empty()

    consult_button = st.button("🔊 CONSULT AETHER", key="consult_aether_btn")
    if consult_button and ai_consult_input:
        curr_p = st.session_state.prices[-1] if st.session_state.prices else 0.0
        v_curr, a_curr, ent_curr = aether_engine.calculate_newton_metrics(st.session_state.prices)
        prob_curr = aether_engine.monte_carlo_simulation(st.session_state.prices)
        reply = consult_ai(ai_consult_input, curr_p, v_curr, a_curr, ent_curr, prob_curr)
        speak_aether(reply) # Speak the AI's reply
        ai_reply_placeholder.info(f"AETHER: {reply}")
    elif consult_button and not ai_consult_input:
        ai_reply_placeholder.warning("Please enter a question for AETHER.")

    st.write("---")
    pnl_display_placeholder = st.empty() # For Total P&L
    st.write("---")

    # Control Buttons
    control_cols = st.columns(2)
    start_button = control_cols[0].button("🔥 INITIATE SEQUENCE", key="start_bot")
    stop_button = control_cols[1].button("🛑 KILL SWITCH", key="stop_bot")

    if st.button("❌ EMERGENCY EXIT (Close Position)", key="emergency_exit_btn"):
        if st.session_state.position:
            # Force close the position
            st.session_state.position = None
            brain_memory["position"] = None
            brain_memory["pending_signal"] = None
            brain_memory["bot_active_on_exit"] = False # Deactivate bot
            save_brain(brain_memory)
            st.session_state.bot_active = False # Ensure bot is off
            add_log("EMERGENCY EXIT TRIGGERED! Position forcibly closed.", "log-danger")
            speak_aether("Emergency protocol initiated. Position purged.")
        else:
            add_log("No active position to emergency exit.", "log-warn")
        st.rerun() # Rerun to update UI immediately

    pending_approval_placeholder = st.empty() # For signal approval buttons

# Update bot_active state
if start_button:
    st.session_state.bot_active = True
    brain_memory["bot_active_on_exit"] = True
    save_brain(brain_memory)
    add_log("SYSTEM SEQUENCE INITIATED", "log-info")
    speak_aether("Systems online. Engaging quantum processors.")
    st.rerun() # Rerun to activate loop
if stop_button:
    st.session_state.bot_active = False
    brain_memory["bot_active_on_exit"] = False
    save_brain(brain_memory)
    add_log("KILL SWITCH ENGAGED - Stopping all processes.", "log-danger")
    speak_aether("Critical systems offline. Standby.")
    st.rerun() # Rerun to deactivate loop

# --- 12. MAIN EXECUTION LOOP ---
if st.session_state.bot_active:
    # Initial connection check if bot is active
    if price_check_status not in ["CONNECTED", "SIMULATING"]:
        st.session_state.bot_active = False # Deactivate bot if connection is bad
        brain_memory["bot_active_on_exit"] = False
        save_brain(brain_memory)
        add_log(f"Bot deactivated due to: {price_check_status}", "log-danger")
        speak_aether("Critical data link compromised. System shutdown.")
        st.rerun()

    while st.session_state.bot_active:
        # --- A. PRE-LOOP UI UPDATES (no reruns needed) ---
        # Update P&L display
        current_live_pnl = 0.0
        if st.session_state.position:
            pos = st.session_state.position
            current_price = st.session_state.prices[-1] if st.session_state.prices else pos['entry']
            current_live_pnl = (current_price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - current_price) * pos['qty']

        total_display_pnl = brain_memory["total_pnl"] + current_live_pnl
        pnl_color = "neon-green" if total_display_pnl >= 0 else "neon-red"
        pnl_display_placeholder.markdown(f"""
            <h2 style='text-align:center;' class='{pnl_color}'>
                TOTAL P&L: ₹{total_display_pnl:,.2f}
                <br> <span style='font-size:0.7em; color:gray'>Open P&L: ₹{current_live_pnl:,.2f}</span>
            </h2>
            """, unsafe_allow_html=True)

        # Update log display
        log_html_content = "".join([l for l in st.session_state.live_logs])
        log_placeholder.markdown(f'<div class="terminal-box">{log_html_content}</div>', unsafe_allow_html=True)

        # Update active trade warning
        if st.session_state.position:
            pos = st.session_state.position
            active_trade_placeholder.markdown(f"""
            <div class="active-trade-box">
                <h3>⚠ QUANTUM ENTANGLEMENT ACTIVE (TRADE OPEN)</h3>
                <p>{pos['type']} {pos['opt']} | ENTRY: {pos['entry']:.2f} | QTY: {pos['qty']} </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            active_trade_placeholder.empty()

        # --- B. KILL SWITCH CHECK ---
        if brain_memory["total_pnl"] < KILL_SWITCH_LOSS:
            speak_aether(f"Critical Failure. Max loss of {KILL_SWITCH_LOSS} breached. Kill Switch Activated.")
            st.session_state.bot_active = False
            brain_memory["bot_active_on_exit"] = False
            save_brain(brain_memory)
            add_log(f"MAX LOSS ({KILL_SWITCH_LOSS}) REACHED. SHUTTING DOWN SYSTEM.", "log-danger")
            st.rerun() # Rerun to stop the loop and update UI

        # --- C. DATA FETCHING ---
        current_price, status = get_live_market_data()

        if status not in ["CONNECTED", "SIMULATING"]:
            add_log(f"Data feed lost: {status}. Attempting reconnect...", "log-danger")
            time.sleep(2) # Wait before retrying
            continue
        elif status == "SIMULATING" and not UPSTOX_ACCESS_TOKEN and random.random() < 0.01: # 1% chance to warn user in simulation
             add_log("Running in simulation. Real-time data requires Upstox API key.", "log-warn")

        st.session_state.prices.append(current_price)

        # Ensure enough data for calculations
        if len(st.session_state.prices) < 30:
            add_log(f"Collecting initial data ({len(st.session_state.prices)}/{30})...", "log-info")
            # Update metrics with current price even if other metrics are 0
            price_metric.metric("NIFTY 50", f"{current_price:,.2f}")
            velocity_metric.metric("VELOCITY", "0.00")
            accel_metric.metric("ACCEL", "0.00")
            entropy_metric.metric("CHAOS", "0.00")
            winprob_metric.metric("WIN PROB", "50%")
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#4ade80', width=2)))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            time.sleep(1)
            continue

        # --- D. METRIC CALCULATIONS ---
        v, a, entropy_val = aether_engine.calculate_newton_metrics(st.session_state.prices)
        win_prob = aether_engine.monte_carlo_simulation(st.session_state.prices)
        future_forecast = aether_engine.monte_carlo_forecast(st.session_state.prices)

        # Update metrics (no rerun)
        price_metric.metric("NIFTY 50", f"{current_price:,.2f}", f"{v:+.2f}") # Delta is velocity
        velocity_metric.metric("VELOCITY", f"{v:.2f}")
        accel_metric.metric("ACCEL", f"{a:.2f}")
        entropy_metric.metric("CHAOS", f"{entropy_val:.2f}")
        winprob_metric.metric("WIN PROB", f"{win_prob*100:.0f}%",
                              delta_color="normal" if win_prob >= 0.5 else "inverse")

        # --- E. COUNCIL VOTING ---
        votes = {}
        weights = brain_memory["weights"]

        # 1. Physics Agent (Velocity & Acceleration)
        if v > 1.5 and a > 0.3: votes['Physics'] = "BUY"
        elif v < -1.5 and a < -0.3: votes['Physics'] = "SELL"
        else: votes['Physics'] = "WAIT"

        # 2. Trend Agent (Simple Moving Average - last 20 periods)
        ma_period = 20
        if len(st.session_state.prices) >= ma_period:
            ma = np.mean(list(st.session_state.prices)[-ma_period:])
            if current_price > ma: votes['Trend'] = "BUY"
            elif current_price < ma: votes['Trend'] = "SELL"
            else: votes['Trend'] = "WAIT"
        else: votes['Trend'] = "WAIT"

        # 3. Global Sentiment Agent (User controlled)
        global_s = brain_memory["global_sentiment"]
        if global_s == "BULLISH": votes['Global'] = "BUY"
        elif global_s == "BEARISH": votes['Global'] = "SELL"
        else: votes['Global'] = "WAIT"

        # 4. Win Probability Agent
        if win_prob > 0.6: votes['WinProb'] = "BUY"
        elif win_prob < 0.4: votes['WinProb'] = "SELL"
        else: votes['WinProb'] = "WAIT"

        # 5. Chaos Agent (High entropy implies scalping/caution)
        market_mode = "TREND"
        if entropy_val > 1.2 and abs(v) < 1.0: # High chaos but low velocity
            market_mode = "SCALP (ROCKET)"
            votes['Chaos'] = "WAIT" # Be cautious
        elif entropy_val > 1.5:
             market_mode = "EXTREME CHAOS"
             votes['Chaos'] = "WAIT" # Extreme caution
        else:
             votes['Chaos'] = "GO" # Allow trading if not too chaotic

        # Update Council Display
        with council_placeholder.container():
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            # Helper for styling based on vote
            def get_vote_class(vote):
                if vote == "BUY": return "BUY"
                if vote == "SELL": return "SELL"
                if vote == "WAIT": return "WAIT"
                if vote == "GO": return "GO"
                return "WAIT"

            cc1.markdown(f"<div class='agent-card {get_vote_class(votes.get('Physics', 'WAIT'))}'>PHYSICS<br>{votes.get('Physics', 'WAIT')}</div>", unsafe_allow_html=True)
            cc2.markdown(f"<div class='agent-card {get_vote_class(votes.get('Trend', 'WAIT'))}'>TREND<br>{votes.get('Trend', 'WAIT')}</div>", unsafe_allow_html=True)
            cc3.markdown(f"<div class='agent-card {get_vote_class(votes.get('Global', 'WAIT'))}'>GLOBAL<br>{votes.get('Global', 'WAIT')}</div>", unsafe_allow_html=True)
            cc4.markdown(f"<div class='agent-card {get_vote_class(votes.get('WinProb', 'WAIT'))}'>PROB<br>{votes.get('WinProb', 'WAIT')}</div>", unsafe_allow_html=True)
            cc5.markdown(f"<div class='agent-card {get_vote_class(votes.get('Chaos', 'WAIT'))}'>MODE<br>{market_mode}</div>", unsafe_allow_html=True)

        # --- F. SIGNAL FUSION (Weighted Decision) ---
        buy_strength = 0
        sell_strength = 0

        # Sum weighted votes
        for agent, vote in votes.items():
            if agent in weights: # Ensure agent has a weight
                if vote == "BUY": buy_strength += weights[agent]
                elif vote == "SELL": sell_strength += weights[agent]

        # Threshold for action
        action_threshold = 2.0 # Minimum combined weight to trigger a signal

        # --- G. TRIGGER SIGNAL & USER APPROVAL ---
        # Only consider new signals if no position is open and no signal is pending approval
        if not st.session_state.position and not st.session_state.pending_signal:
            if buy_strength >= action_threshold:
                option_strike = aether_engine.get_best_option_strike(current_price, "BUY")
                st.session_state.pending_signal = {"type": "BUY", "opt": option_strike, "strength": buy_strength}
                brain_memory["pending_signal"] = st.session_state.pending_signal
                save_brain(brain_memory)
                speak_aether(f"Boss! Buy Signal detected on {option_strike}. Strength {buy_strength:.1f}. Approve?")
                send_telegram_report(f"🚨 **BUY SIGNAL:** Nifty {option_strike} @ {current_price:.2f} (Strength: {buy_strength:.1f})")
                add_log(f"BUY SIGNAL: {option_strike} @ {current_price:.2f} (Strength: {buy_strength:.1f})", "log-buy")
                st.rerun() # Rerun to display approval buttons

            elif sell_strength >= action_threshold:
                option_strike = aether_engine.get_best_option_strike(current_price, "SELL")
                st.session_state.pending_signal = {"type": "SELL", "opt": option_strike, "strength": sell_strength}
                brain_memory["pending_signal"] = st.session_state.pending_signal
                save_brain(brain_memory)
                speak_aether(f"Boss! Sell Signal detected on {option_strike}. Strength {sell_strength:.1f}. Approve?")
                send_telegram_report(f"🚨 **SELL SIGNAL:** Nifty {option_strike} @ {current_price:.2f} (Strength: {sell_strength:.1f})")
                add_log(f"SELL SIGNAL: {option_strike} @ {current_price:.2f} (Strength: {sell_strength:.1f})", "log-sell")
                st.rerun() # Rerun to display approval buttons

        # Display approval buttons if a signal is pending
        if st.session_state.pending_signal:
            sig = st.session_state.pending_signal
            with pending_approval_placeholder.container():
                st.markdown(f"<div class='approval-box'>⚠️ {sig['type']} SIGNAL: {sig['opt']} @ {current_price:.2f}. Execute?</div>", unsafe_allow_html=True)
                approve_col, reject_col = st.columns(2)
                if approve_col.button(f"✅ APPROVE {sig['type']}", key="approve_signal"):
                    st.session_state.position = {
                        "type": sig['type'],
                        "entry": current_price,
                        "qty": TRADE_QUANTITY,
                        "opt": sig['opt']
                    }
                    brain_memory["position"] = st.session_state.position
                    st.session_state.trailing_high = 0.0 # Reset trailing high for new trade
                    brain_memory["trailing_high"] = 0.0
                    st.session_state.pending_signal = None
                    brain_memory["pending_signal"] = None
                    save_brain(brain_memory)
                    add_log(f"ORDER EXECUTED: {sig['type']} {sig['opt']} @ {current_price:.2f}", "log-buy" if sig['type']=="BUY" else "log-sell")
                    speak_aether(f"Order executed: {sig['type']} {sig['opt']}. Committing resources.")
                    st.rerun() # Rerun to clear approval buttons and show active trade
                if reject_col.button(f"❌ REJECT {sig['type']}", key="reject_signal"):
                    st.session_state.pending_signal = None
                    brain_memory["pending_signal"] = None
                    save_brain(brain_memory)
                    add_log(f"SIGNAL REJECTED: {sig['type']} {sig['opt']}.", "log-warn")
                    speak_aether("Signal rejected. Recalibrating.")
                    st.rerun() # Rerun to clear approval buttons
        else:
            pending_approval_placeholder.empty() # Clear approval buttons if no signal

        # --- H. TRADE MANAGEMENT (ZERO LOSS LOGIC with Trailing Stop) ---
        if st.session_state.position:
            pos = st.session_state.position
            pnl_current_trade = (current_price - pos['entry']) * pos['qty'] if pos['type'] == "BUY" else (pos['entry'] - current_price) * pos['qty']

            # Update Trailing High (only if in profit)
            if pnl_current_trade > st.session_state.trailing_high:
                st.session_state.trailing_high = pnl_current_trade
                brain_memory["trailing_high"] = st.session_state.trailing_high
                # add_log(f"Trailing High updated to {st.session_state.trailing_high:.0f}", "log-info")

            exit_condition = False
            exit_reason = ""
            pnl_final_on_exit = 0.0

            # Dynamic Stop Loss and Profit Targets
            # Base SL: -300, Base Target: +500 (adjust these as needed)
            hard_stop_loss = -300
            initial_profit_target = 500

            # 1. Trailing Stop Loss Logic (after some profit is made)
            # If current PnL drops significantly from the trailing high
            trailing_stop_percent = 0.5 # Exit if PnL drops 50% from high
            if st.session_state.trailing_high > initial_profit_target: # Only activate trailing if significant profit was reached
                if pnl_current_trade < st.session_state.trailing_high * trailing_stop_percent:
                    exit_condition = True
                    exit_reason = f"TRAILING STOP HIT (from {st.session_state.trailing_high:.0f})"
                    pnl_final_on_exit = pnl_current_trade

            # 2. Hard Stop Loss
            if pnl_current_trade < hard_stop_loss:
                exit_condition = True
                exit_reason = f"HARD STOP LOSS HIT ({hard_stop_loss})"
                pnl_final_on_exit = pnl_current_trade

            # 3. Profit Target
            if pnl_current_trade >= initial_profit_target:
                exit_condition = True
                exit_reason = f"PROFIT TARGET REACHED ({initial_profit_target})"
                pnl_final_on_exit = pnl_current_trade

            # 4. Physics Reversal Exit
            # If current position is BUY and velocity turns significantly negative
            # Or current position is SELL and velocity turns significantly positive
            physics_reversal_threshold = 1.5 # e.g., if velocity drops below -1.5 for a BUY
            if pos['type'] == "BUY" and v < -physics_reversal_threshold:
                 if not exit_condition: # Only if not already exiting for other reasons
                    exit_condition = True
                    exit_reason = f"PHYSICS REVERSAL (Velocity {v:.2f})"
                    pnl_final_on_exit = pnl_current_trade # Use current pnl

            elif pos['type'] == "SELL" and v > physics_reversal_threshold:
                 if not exit_condition:
                    exit_condition = True
                    exit_reason = f"PHYSICS REVERSAL (Velocity {v:.2f})"
                    pnl_final_on_exit = pnl_current_trade # Use current pnl

            # Execute Exit
            if exit_condition:
                trade_result_type = "WIN" if pnl_final_on_exit >= 0 else "LOSS"
                brain_memory["total_pnl"] += pnl_final_on_exit
                brain_memory["wins" if trade_result_type == "WIN" else "losses"] += 1
                brain_memory["trade_log"].insert(0, { # Prepend to log
                    "time": datetime.now(TIMEZONE).strftime("%H:%M:%S"),
                    "type": pos['type'],
                    "opt": pos['opt'],
                    "entry": pos['entry'],
                    "exit_price": current_price,
                    "pnl": pnl_final_on_exit,
                    "result": trade_result_type,
                    "reason": exit_reason
                })

                aether_engine.self_correct(trade_result_type) # Update learning weights

                st.session_state.position = None
                brain_memory["position"] = None
                st.session_state.trailing_high = 0.0
                brain_memory["trailing_high"] = 0.0
                save_brain(brain_memory)

                speak_aether(f"Position Closed. {trade_result_type}. {exit_reason}.")
                add_log(f"POSITION CLOSED: {pos['type']} {pos['opt']} | P&L: {pnl_final_on_exit:.2f} ({trade_result_type}) - {exit_reason}", "log-buy" if trade_result_type=="WIN" else "log-danger")
                st.rerun() # Rerun to update UI for closed position

        # --- I. TELEGRAM REPORTING ---
        if time.time() - st.session_state.last_tg_time > TELEGRAM_INTERVAL:
            report_msg = (
                f"⏰ {datetime.now(TIMEZONE).strftime('%H:%M')}\n"
                f"💰 NIFTY: {current_price:,.2f}\n"
                f"🚀 VELOCITY: {v:.2f} | ACCEL: {a:.2f}\n"
                f"🎲 WIN PROB: {win_prob*100:.0f}% | CHAOS: {entropy_val:.2f}\n"
                f"💸 TOTAL P&L: ₹{brain_memory['total_pnl']:.2f}\n"
                f"📈 WINS: {brain_memory['wins']} | LOSSES: {brain_memory['losses']}\n"
                f"🧠 KNOWLEDGE: {brain_memory['market_knowledge']}"
            )
            if st.session_state.position:
                pos = st.session_state.position
                report_msg += f"\n🔥 ACTIVE: {pos['type']} {pos['opt']} @ {pos['entry']:.2f} (Open P&L: {current_live_pnl:.2f})"
            send_telegram_report(report_msg)
            st.session_state.last_tg_time = time.time()
            brain_memory["last_tg_time"] = st.session_state.last_tg_time
            save_brain(brain_memory)
            add_log("Telegram report dispatched.", "log-info")

        # --- J. CHART UPDATE ---
        fig = go.Figure()
        # Historical prices
        fig.add_trace(go.Scatter(y=list(st.session_state.prices), mode='lines', line=dict(color='#4ade80', width=2),
                                 name='Nifty Live')) # Neon Green
        # Predicted future price (simple dot for now)
        fig.add_trace(go.Scatter(x=[len(st.session_state.prices) - 1, len(st.session_state.prices)],
                                 y=[current_price, future_forecast],
                                 mode='lines+markers',
                                 marker=dict(color='#c084fc', size=8), # Neon Purple
                                 line=dict(color='#c084fc', dash='dot'),
                                 name='AI Forecast'))

        if st.session_state.position:
            # Entry point
            fig.add_hline(y=st.session_state.position['entry'], line_dash="dash", line_color="#fbbf24", # Neon Orange
                          annotation_text="ENTRY", annotation_position="bottom right")

        fig.update_layout(
            height=250,
            margin=dict(l=0,r=0,t=0,b=0),
            template="plotly_dark", # Dark theme for cyberpunk feel
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(51, 65, 85, 0.5)', zeroline=False), # Subtle grid
            showlegend=False
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # --- K. SLEEP ---
        time.sleep(3) # Update every second (adjust as needed for API limits or responsiveness)

# --- 13. POST-LOOP UI (when bot is inactive) ---
if not st.session_state.bot_active:
    st.subheader("📚 AETHER's Trade History")
    if brain_memory["trade_log"]:
        # Convert trade_log to DataFrame and display
        trade_df = pd.DataFrame(brain_memory["trade_log"])
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet.")

    # Final P&L display
    final_pnl_color = "neon-green" if brain_memory["total_pnl"] >= 0 else "neon-red"
    st.markdown(f"""
        <h2 style='text-align:center;' class='{final_pnl_color}'>
            FINAL P&L: ₹{brain_memory["total_pnl"]:.2f}
        </h2>
        """, unsafe_allow_html=True)
    st.write(f"Total Wins: {brain_memory['wins']}, Total Losses: {brain_memory['losses']}")
    st.info("AETHER is in standby mode. Activate to initiate sequence.")
