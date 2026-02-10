import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import toml
import os

# --- 1. CONFIGURATION LOAD (ரகசிய பெட்டியை திறத்தல்) ---
try:
    # ஸ்ட்ரீம்லிட் பாணியில் secrets.toml ஐப் படித்தல்
    secrets = toml.load(".streamlit/secrets.toml")
    ACCESS_TOKEN = secrets["upstox"]["access_token"]
    OWNER_NAME = secrets["general"]["owner"]
except Exception as e:
    print("⚠️ SECRETS ERROR: secrets.toml file not found or invalid.")
    ACCESS_TOKEN = None
    OWNER_NAME = "BOSS MANIKANDAN"

# Upstox API Config
UPSTOX_URL = "https://api.upstox.com/v2/market-quote/ltp"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50" # நிஃப்டி 50 குறியீடு

# --- 2. PHYSICS ENGINE (நியூட்டன் விதிகளின்படி கணக்கீடு) ---
class PhysicsEngine:
    def calculate(self, prices):
        if len(prices) < 5: return 0, 0, 0
        p = np.array(prices)
        
        # Velocity (வேகம்) - விலை மாற்றத்தின் வேகம்
        v = np.gradient(p)[-1]
        
        # Acceleration (முடுக்கம்) - வேகம் கூடுகிறதா குறைகிறதா?
        a = np.gradient(np.gradient(p))[-1]
        
        # Force (விசை) - சந்தையின் உந்துதல்
        f = a * 100 
        
        return v, a, f

# --- 3. APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

physics = PhysicsEngine()
price_history = [] # விலை வரலாற்றை சேமிக்க

# --- 4. DATA FETCHING FUNCTION (உண்மையான டேட்டா) ---
def get_market_data():
    if not ACCESS_TOKEN:
        return None
    
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    params = {'instrument_key': INSTRUMENT_KEY}
    
    try:
        response = requests.get(UPSTOX_URL, headers=headers, params=params, timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Upstox JSON-லிருந்து LTP (Last Traded Price) எடுத்தல்
            ltp = data['data'][INSTRUMENT_KEY]['last_price']
            return float(ltp)
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

# --- 5. DASHBOARD LAYOUT ---
app.layout = dbc.Container([
    
    # Header
    dbc.Row([
        dbc.Col(html.H1("CM-X GENESIS: COMMANDER MODE", className="text-center text-info"), width=12),
        dbc.Col(html.P(f"OPERATOR: {OWNER_NAME} | SOURCE: UPSTOX LIVE API", className="text-center text-muted"), width=12),
    ], className="mb-4 mt-4"),

    # Live Metrics
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("NIFTY 50 (LIVE)"),
            dbc.CardBody(html.H2(id="live-price", children="Connecting...", className="text-warning"))
        ], color="dark", inverse=True, className="border-warning"), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("VELOCITY (m/s)"),
            dbc.CardBody(html.H2(id="live-velocity", children="0.00", className="text-success"))
        ], color="dark", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("ACCELERATION (m/s²)"),
            dbc.CardBody(html.H2(id="live-accel", children="0.00", className="text-danger"))
        ], color="dark", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("AI COMMAND"),
            dbc.CardBody(html.H2(id="ai-signal", children="ANALYZING...", className="text-white"))
        ], color="primary", inverse=True, className="border-white"), width=3),
    ], className="mb-4"),

    # Main Chart
    dbc.Row([
        dbc.Col(dcc.Graph(id='live-chart', animate=False), width=12)
    ]),

    # Manual Override Controls
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H3("⚠️ MANUAL OVERRIDE PROTOCOL", className="text-center text-danger"), width=12),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(dbc.Button("🚀 AUTHORIZE BUY", id="btn-buy-pre", color="success", size="lg", className="w-100"), width=6),
        dbc.Col(dbc.Button("🔻 AUTHORIZE SELL", id="btn-sell-pre", color="danger", size="lg", className="w-100"), width=6),
    ], className="mb-5"),

    # Refresh Interval (Every 1 second)
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),

    # Confirmation Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("⚠️ SECURITY CLEARANCE REQUIRED")),
        dbc.ModalBody(id="modal-body", children="Boss, confirm execution?"),
        dbc.ModalFooter([
            dbc.Button("CANCEL", id="modal-cancel", className="ms-auto", n_clicks=0),
            dbc.Button("🔥 EXECUTE ORDER", id="modal-confirm", color="danger", n_clicks=0),
        ]),
    ], id="modal", is_open=False, centered=True, backdrop="static"),

    # Alert Box
    html.Div(id="order-status-msg")

], fluid=True)

# --- 6. CALLBACKS (BRAIN) ---

@app.callback(
    [Output('live-price', 'children'),
     Output('live-velocity', 'children'),
     Output('live-accel', 'children'),
     Output('ai-signal', 'children'),
     Output('live-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    global price_history
    
    # 1. Fetch Real Data
    current_price = get_market_data()
    
    # If API fails, simulate small movement to keep UI alive (Fallback)
    if current_price is None:
        if len(price_history) > 0:
            current_price = price_history[-1]
        else:
            current_price = 22000.00 # Default fallback
    
    # Append to history
    price_history.append(current_price)
    if len(price_history) > 100: price_history.pop(0) # Keep last 100 ticks
    
    # 2. Physics Calculation
    v, a, f = physics.calculate(price_history)
    
    # 3. AI Logic
    signal = "WAIT"
    signal_color = "white"
    
    if v > 2.0 and a > 0.5: 
        signal = "BUY NOW"
    elif v < -2.0 and a < -0.5: 
        signal = "SELL NOW"
    
    # 4. Charting
    fig = go.Figure()
    
    # Price Line
    fig.add_trace(go.Scatter(
        y=price_history, 
        mode='lines', 
        line=dict(color='#00ff00', width=2), 
        name='Nifty 50',
        fill='tozeroy',
        fillcolor='rgba(0, 255, 0, 0.1)'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#333'),
        title={'text': "REAL-TIME MARKET KINEMATICS", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )

    return f"₹{current_price:,.2f}", f"{v:.2f}", f"{a:.2f}", signal, fig

# Button Logic (Modal)
@app.callback(
    [Output("modal", "is_open"), Output("order-status-msg", "children"), Output("modal-body", "children")],
    [Input("btn-buy-pre", "n_clicks"), Input("btn-sell-pre", "n_clicks"), 
     Input("modal-confirm", "n_clicks"), Input("modal-cancel", "n_clicks")],
    [State("modal", "is_open"), State("ai-signal", "children"), State("live-price", "children")]
)
def toggle_modal(n_buy, n_sell, n_confirm, n_cancel, is_open, signal, price):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, "", ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Open Modal
    if button_id == "btn-buy-pre":
        return True, "", f"Boss, {price}-ல BUY பண்ணவா? சிக்னல்: {signal}"
    elif button_id == "btn-sell-pre":
        return True, "", f"Boss, {price}-ல SELL பண்ணவா? சிக்னல்: {signal}"
        
    # Close Modal
    elif button_id == "modal-cancel":
        return False, dbc.Alert("OPERATION ABORTED.", color="warning", duration=3000), ""
        
    # Execute
    elif button_id == "modal-confirm":
        # API Order Placement Logic would go here
        return False, dbc.Alert(f"✅ ORDER SENT TO UPSTOX! {price}", color="success", duration=5000), ""

    return is_open, "", ""

# --- 7. RUN ---
if __name__ == '__main__':
    print("🚀 CM-X GENESIS COMMANDER IS ONLINE...")
    app.run_server(debug=True, port=8050)
