import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import time

# --- 1. PHYSICS ENGINE (நம்ம நியூட்டன் மூளை) ---
class PhysicsEngine:
    def calculate(self, prices):
        if len(prices) < 5: return 0, 0, 0
        p = np.array(prices)
        
        # Velocity (வேகம்)
        v = np.gradient(p)[-1]
        
        # Acceleration (முடுக்கம்)
        a = np.gradient(np.gradient(p))[-1]
        
        # Force (விசை) - Mass is assumed constant for now
        f = a * 100 
        
        return v, a, f

# --- 2. APP CONFIGURATION ---
# Dark Theme (Cyberpunk Style)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Global State Simulation (உதாரண டேட்டா)
price_history = [22000]
physics = PhysicsEngine()

# --- 3. LAYOUT DESIGN (DASHBOARD) ---
app.layout = dbc.Container([
    
    # --- HEADER ---
    dbc.Row([
        dbc.Col(html.H1("CM-X GENESIS: COMMANDER MODE", className="text-center text-info"), width=12),
        dbc.Col(html.P("OPERATOR: BOSS MANIKANDAN | STATUS: ONLINE", className="text-center text-muted"), width=12),
    ], className="mb-4 mt-4"),

    # --- LIVE STATS ROW ---
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("LIVE PRICE (NIFTY)"),
            dbc.CardBody(html.H2(id="live-price", className="text-warning"))
        ], color="dark", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("VELOCITY (m/s)"),
            dbc.CardBody(html.H2(id="live-velocity", className="text-success"))
        ], color="dark", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("ACCELERATION (m/s²)"),
            dbc.CardBody(html.H2(id="live-accel", className="text-danger"))
        ], color="dark", inverse=True), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardHeader("AI SUGGESTION"),
            dbc.CardBody(html.H2(id="ai-signal", children="WAIT...", className="text-white"))
        ], color="primary", inverse=True, className="border-white"), width=3),
    ], className="mb-4"),

    # --- MAIN CHART AREA ---
    dbc.Row([
        dbc.Col(dcc.Graph(id='live-chart', animate=False), width=12)
    ]),

    # --- NUCLEAR LAUNCH CODES (MANUAL CONFIRMATION) ---
    html.Hr(),
    dbc.Row([
        dbc.Col(html.H3("⚠️ MANUAL OVERRIDE PROTOCOL", className="text-center text-danger"), width=12),
    ], className="mb-2"),

    dbc.Row([
        # BUY BUTTON
        dbc.Col(dbc.Button("🚀 AUTHORIZE BUY", id="btn-buy-pre", color="success", size="lg", className="w-100"), width=6),
        # SELL BUTTON
        dbc.Col(dbc.Button("🔻 AUTHORIZE SELL", id="btn-sell-pre", color="danger", size="lg", className="w-100"), width=6),
    ], className="mb-5"),

    # --- HIDDEN COMPONENTS ---
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0), # 1 sec update
    html.Div(id='voice-output', style={'display': 'none'}), # For Voice triggers

    # --- CONFIRMATION MODAL (நம்ம செக் பாயிண்ட்) ---
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("⚠️ SECURITY CLEARANCE REQUIRED")),
        dbc.ModalBody("Boss, AI recommends a trade. Do you want to EXECUTE this order? (Real Money Involved)"),
        dbc.ModalFooter([
            dbc.Button("CANCEL", id="modal-cancel", className="ms-auto", n_clicks=0),
            dbc.Button("🔥 EXECUTE ORDER", id="modal-confirm", color="danger", n_clicks=0),
        ]),
    ], id="modal", is_open=False, centered=True, backdrop="static"),

    # Alert Message
    html.Div(id="order-status-msg")

], fluid=True)

# --- 4. CALLBACKS (BRAIN OF THE APP) ---

# A. Live Data Update & Charting
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
    
    # 1. Simulate Price (Random Walk for Demo)
    new_price = price_history[-1] + np.random.randint(-20, 25)
    price_history.append(new_price)
    if len(price_history) > 50: price_history.pop(0)
    
    # 2. Physics Calculation
    v, a, f = physics.calculate(price_history)
    
    # 3. AI Logic
    signal = "WAIT"
    if v > 5 and a > 0: signal = "BUY SIGNAL"
    elif v < -5 and a < 0: signal = "SELL SIGNAL"
    
    # 4. Charting (Dark Mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price_history, mode='lines+markers', line=dict(color='#00ff00', width=2), name='Price'))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return f"₹{new_price}", f"{v:.2f}", f"{a:.2f}", signal, fig

# B. Button Logic (Confirmation Modal)
@app.callback(
    [Output("modal", "is_open"), Output("order-status-msg", "children")],
    [Input("btn-buy-pre", "n_clicks"), Input("btn-sell-pre", "n_clicks"), 
     Input("modal-confirm", "n_clicks"), Input("modal-cancel", "n_clicks")],
    [State("modal", "is_open"), State("ai-signal", "children")]
)
def toggle_modal(n_buy, n_sell, n_confirm, n_cancel, is_open, current_signal):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1. Open Modal on Initial Click
    if button_id in ["btn-buy-pre", "btn-sell-pre"]:
        return True, ""

    # 2. Close on Cancel
    elif button_id == "modal-cancel":
        return False, dbc.Alert("ORDER CANCELLED BY BOSS.", color="warning", duration=3000)

    # 3. Execute on Confirm (Real Logic Goes Here)
    elif button_id == "modal-confirm":
        # இங்கே தான் Upstox API கால் வரும்
        return False, dbc.Alert(f"✅ ORDER EXECUTED SUCCESSFULLY! Signal: {current_signal}", color="success", duration=5000)

    return is_open, ""

# --- 5. RUN SERVER ---
if __name__ == '__main__':
    # debug=True போட்டா தான் Error தெரியும், ஆனா Live-ல False போடணும்.
    app.run_server(debug=True, port=8050)
