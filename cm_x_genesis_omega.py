import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import toml
import google.generativeai as genai
import threading
import time

# --- 1. CONFIGURATION LOAD (ரகசிய தகவல்கள்) ---
try:
    secrets = toml.load(".streamlit/secrets.toml")
    
    # SYSTEM INFO
    OWNER_NAME = secrets["general"]["owner"]
    
    # UPSTOX
    UPSTOX_ACCESS_TOKEN = secrets["upstox"]["access_token"]
    
    # GEMINI AI
    GEMINI_API_KEY = secrets["gemini"]["api_key"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    # TELEGRAM
    TELEGRAM_BOT_TOKEN = secrets["telegram"]["bot_token"]
    TELEGRAM_CHAT_ID = secrets["telegram"]["chat_id"]
    
except Exception as e:
    print(f"⚠️ CONFIG ERROR: {e}")
    UPSTOX_ACCESS_TOKEN = None

# Upstox API Endpoint
UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

# --- 2. HELPER FUNCTIONS (உதவியாளர்கள்) ---

# Telegram Sender
def send_telegram_msg(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 {OWNER_NAME}: {message}"}
        requests.get(url, params=params)
    except Exception as e:
        print(f"Telegram Error: {e}")

# Gemini AI Analyst (JARVIS)
def ask_jarvis(price, velocity, acceleration):
    try:
        prompt = f"""
        Act as JARVIS, an AI Trading Assistant for Boss Manikandan.
        Current Nifty 50 Price: {price}
        Velocity: {velocity:.2f}
        Acceleration: {acceleration:.2f}
        
        Analyze this physics data. Is the market gaining momentum (Bullish) or losing it (Bearish)?
        Reply in 1 short sentence (Max 10 words).
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI SYSTEM BUSY..."

# --- 3. PHYSICS ENGINE ---
class PhysicsEngine:
    def calculate(self, prices):
        if len(prices) < 5: return 0, 0, 0
        p = np.array(prices)
        v = np.gradient(p)[-1] # Velocity
        a = np.gradient(np.gradient(p))[-1] # Acceleration
        f = a * 100 # Force
        return v, a, f

# --- 4. APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

physics = PhysicsEngine()
price_history = []
last_signal = "WAIT" # To prevent spamming Telegram

# --- 5. DATA FETCHING ---
def get_market_data():
    if not UPSTOX_ACCESS_TOKEN: return None
    headers = {'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}', 'Accept': 'application/json'}
    params = {'instrument_key': INSTRUMENT_KEY}
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=2)
        if response.status_code == 200:
            return float(response.json()['data'][INSTRUMENT_KEY]['last_price'])
    except:
        return None
    return None

# --- 6. LAYOUT ---
app.layout = dbc.Container([
    
    # Header
    dbc.Row([
        dbc.Col(html.H1("CM-X GENESIS: OMEGA LIVE", className="text-center text-info glitch-effect"), width=12),
        dbc.Col(html.P(f"COMMANDER: {OWNER_NAME} | SYSTEM: ONLINE", className="text-center text-muted"), width=12),
    ], className="mb-4 mt-4"),

    # Metrics
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("NIFTY 50 (LIVE)"), dbc.CardBody(html.H2(id="live-price", className="text-warning"))], color="dark", inverse=True, className="border-warning"), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("VELOCITY"), dbc.CardBody(html.H2(id="live-velocity", className="text-success"))], color="dark", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("ACCELERATION"), dbc.CardBody(html.H2(id="live-accel", className="text-danger"))], color="dark", inverse=True), width=3),
        dbc.Col(dbc.Card([dbc.CardHeader("JARVIS AI OPINION"), dbc.CardBody(html.H4(id="ai-insight", children="INITIALIZING...", className="text-info"))], color="dark", inverse=True, className="border-info"), width=3),
    ], className="mb-4"),

    # Chart
    dbc.Row([dbc.Col(dcc.Graph(id='live-chart', animate=False), width=12)]),

    # Controls
    html.Hr(),
    dbc.Row([
        dbc.Col(dbc.Button("🤖 ASK JARVIS (MANUAL SCAN)", id="btn-ask-ai", color="primary", size="lg", className="w-100"), width=4),
        dbc.Col(dbc.Button("🚀 BUY CONFIRM", id="btn-buy", color="success", size="lg", className="w-100"), width=4),
        dbc.Col(dbc.Button("🔻 SELL CONFIRM", id="btn-sell", color="danger", size="lg", className="w-100"), width=4),
    ], className="mb-5"),

    dcc.Interval(id='interval-component', interval=2000, n_intervals=0), # 2 Seconds Update
    html.Div(id="hidden-div", style={"display": "none"}) # For background tasks

], fluid=True)

# --- 7. CALLBACKS ---

@app.callback(
    [Output('live-price', 'children'),
     Output('live-velocity', 'children'),
     Output('live-accel', 'children'),
     Output('live-chart', 'figure'),
     Output('ai-insight', 'children')],
    [Input('interval-component', 'n_intervals'), Input('btn-ask-ai', 'n_clicks')]
)
def update_system(n, ai_click):
    global price_history, last_signal
    
    # 1. Fetch Data
    current_price = get_market_data()
    
    # Fallback if market closed or API error
    if current_price is None:
        if price_history: current_price = price_history[-1]
        else: current_price = 22000.0 # Default
    
    price_history.append(current_price)
    if len(price_history) > 100: price_history.pop(0)
    
    # 2. Physics
    v, a, f = physics.calculate(price_history)
    
    # 3. Auto-Signal Logic
    signal = "NEUTRAL"
    if v > 3.0 and a > 0.5: signal = "BUY"
    elif v < -3.0 and a < -0.5: signal = "SELL"
    
    # 4. Telegram Alert (Only on status change)
    if signal != last_signal and signal != "NEUTRAL":
        msg = f"MARKET ALERT! Signal: {signal} | Price: {current_price} | Velocity: {v:.2f}"
        # Run Telegram in background thread to not block UI
        threading.Thread(target=send_telegram_msg, args=(msg,)).start()
        last_signal = signal

    # 5. JARVIS AI (Only updates every 10 ticks OR on button click to save API quota)
    ctx = callback_context
    ai_text = dash.no_update
    if (ctx.triggered and "btn-ask-ai" in ctx.triggered[0]['prop_id']) or (n % 10 == 0):
        ai_text = ask_jarvis(current_price, v, a)
        
    # 6. Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price_history, mode='lines', line=dict(color='#00ff00', width=2), name='Nifty 50', fill='tozeroy', fillcolor='rgba(0,255,0,0.1)'))
    fig.update_layout(template="plotly_dark", height=450, title="CM-X OMEGA LIVE KINEMATICS", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return f"₹{current_price:,.2f}", f"{v:.2f}", f"{a:.2f}", fig, ai_text

# --- 8. RUN ---
if __name__ == '__main__':
    # Send Startup Msg
    send_telegram_msg("SYSTEM ONLINE: CM-X OMEGA LIVE")
    app.run_server(debug=True, port=8050)
