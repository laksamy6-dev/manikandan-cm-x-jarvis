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

        full_prompt = f"{persona_prompt}\nDAT
