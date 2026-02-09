import re
import websocket
import json

with open('app.py', 'r') as f:
    app_py_content = f.read()

# --- 1. Modify ScalpingSniper.enter_trade to store more position details ---
# Ensure instrument_key, quantity are stored for position tracking
app_py_content = re.sub(
    r"""                self.positions.append({'type': type, 'entry': price, 'time': time.time(), 'order_id': api_response.get('order_id')})""",
    r"""                self.positions.append({'type': type, 'entry': price, 'time': time.time(), 'order_id': api_response.get('order_id'),
                                       'instrument_key': instrument_key, 'quantity': quantity})
                # Update session state with current positions for UI display
                st.session_state['open_positions'] = self.positions""",
    app_py_content
)

# --- 2. Modify ScalpingSniper.manage_positions for accurate P&L calculation and session state update ---
# Fix `pos_price` undefined error, ensure Upstox API calls use stored instrument_token/quantity
app_py_content = re.sub(
    r"""                profit = current_price - pos_data['entry']
                else:
                    profit = pos_price - pos_data['entry'] # Assuming pos_price from Upstox response""",
    r"""                profit = current_price - pos_data['entry']
                else:
                    profit = pos_data['entry'] - current_price""", # Fixed to use current_price for PUTs
    app_py_content
)

app_py_content = re.sub(
    r"""                    quantity=50, # Need actual quantity from position
                    product_type='INTRADAY',
                    order_type='MARKET',
                    instrument_token='NSE_EQ|INE000000000', # Need actual instrument token""",
    r"""                    quantity=pos_data['quantity'],
                    product_type='INTRADAY',
                    order_type='MARKET',
                    instrument_token=pos_data['instrument_key'],""",
    app_py_content
)

# Also update session state with current positions after any modifications
app_py_content = re.sub(
    r"""            # Remove closed positions from internal tracking (iterating backwards to avoid index issues)
            for i in sorted(closed_positions_indices, reverse=True):
                del self.positions[i]""",
    r"""            # Remove closed positions from internal tracking (iterating backwards to avoid index issues)
            for i in sorted(closed_positions_indices, reverse=True):
                del self.positions[i]
            st.session_state['open_positions'] = self.positions""",
    app_py_content
)

# --- 3. Modify run_live_bot to store last price and update position data in session state ---
app_py_content = re.sub(
    r"""            all_decisions.append(decision)
            all_confidences.append(conf)
            all_phys_data.append(phys)""",
    r"""            all_decisions.append(decision)
            all_confidences.append(conf)
            all_phys_data.append(phys)

            # Update latest price and open positions in session state
            st.session_state['latest_market_price'] = p
            st.session_state['open_positions'] = bot.positions""",
    app_py_content
)


# --- 4. Add UI for displaying real-time live trading status and open positions ---
# This will be added after the main bot control logic and before the charts

# Find the place where the charts are displayed
charts_display_marker = r"""# Display charts after simulation is complete and results are available"""

# Add the live status and positions display after the bot start/stop buttons and before charts
new_ui_elements = '''
# --- Real-time Live Trading Status and Positions ---
st.subheader("📊 Live Trading Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Bot Status", "🟢 Running" if st.session_state.get('bot_running', False) else "🔴 Stopped")
with col2:
    st.metric("Latest Market Price", f"₹{st.session_state.get('latest_market_price', 'N/A'):.2f}")
with col3:
    if st.session_state.get('scheduled_to_run', False):
        st.info("Waiting for market open...")
    else:
        st.empty() # Clear placeholder if not waiting

st.markdown("#### Open Positions")
if st.session_state.get('open_positions'):
    positions_df = pd.DataFrame(st.session_state['open_positions'])
    positions_df['PnL'] = positions_df.apply(lambda row: (st.session_state.get('latest_market_price', row['entry']) - row['entry']) * row['quantity'] if row['type'] == 'CALL' else (row['entry'] - st.session_state.get('latest_market_price', row['entry'])) * row['quantity'], axis=1)
    st.dataframe(positions_df[['instrument_key', 'type', 'entry', 'quantity', 'PnL']])
else:
    st.info("No open positions currently.")

'''

# Insert new_ui_elements before the charts_display_marker
app_py_content = app_py_content.replace(
    charts_display_marker,
    new_ui_elements + "\n" + charts_display_marker
)

# Final check and write to file
with open('app.py', 'w') as f:
    f.write(app_py_content)

print("Modified app.py to integrate Upstox position management in ScalpingSniper.manage_positions and add live UI elements.")
