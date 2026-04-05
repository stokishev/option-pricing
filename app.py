import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Internal modules
from utils.market_data import DataFetcher
from utils.volatility_surface import generate_volatility_surface
from models.base import OptionSetup
from models.black_scholes import BlackScholes
from models.monte_carlo import MonteCarlo
from models.binomial_tree import BinomialTree

# Configuration
st.set_page_config(page_title="Derivates Structuring", layout="wide", page_icon="📈")

@st.cache_data(ttl=300)
def load_market_data(ticker):
    return DataFetcher.get_market_data(ticker)

# Sidebar
st.sidebar.title("Derivates Structuring")

ticker_input = st.sidebar.text_input("Underlying Ticker", value="AAPL").upper()
market_data = load_market_data(ticker_input)

if market_data:
    spot_price = market_data.spot_price
    hist_vol = market_data.historical_volatility
    st.sidebar.info(f"Spot (S): {spot_price:.2f} | Hist Vol: {hist_vol:.1%}")
else:
    spot_price, hist_vol = 100.0, 0.20
    st.sidebar.warning("Ticker not found. Using default values.")

# Asset & Contract Parameters
strike_price = st.sidebar.number_input("Strike (K)", value=round(spot_price, 2), step=1.0)
expiry_date = st.sidebar.date_input("Expiry", value=datetime.today() + timedelta(days=30))
rfr = st.sidebar.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100

# Cost of Carry: Dividends & Borrow Costs
div_yield = st.sidebar.number_input("Dividend Yield (%)", value=0.5, step=0.1) / 100
borrow_cost = st.sidebar.number_input("Borrow/Repo Cost (%)", value=0.0, step=0.1) / 100
total_q = div_yield + borrow_cost  # Continuous cost of carry

# Volatility configuration
vol_mode = st.sidebar.radio("Volatility Input", ["Historical", "Manual"])
sigma = hist_vol if vol_mode == "Historical" else st.sidebar.slider("IV (%)", 1.0, 150.0, 20.0) / 100

# Timing calculations (Actual/365 convention)
days_to_expiry = (expiry_date - datetime.today().date()).days
T = max(days_to_expiry / 365.0, 0.001)

# Model configuration
model_choice = st.sidebar.selectbox("Pricing Model", 
    ["Black-Scholes (Merton)", "Monte Carlo (Path-Dependent)", "Binomial Tree (CRR)"])

exotic_type, barrier_level = 'european', None
if model_choice == "Monte Carlo (Path-Dependent)":
    mc_sims = st.sidebar.number_input("Simulations", 1000, 100000, 10000, 1000)
    exotic_choice = st.sidebar.selectbox("Style", ["Vanilla", "Asian (Arithmetic)", "Up-and-Out Barrier"])
    if exotic_choice == "Asian (Arithmetic)": exotic_type = 'asian'
    elif exotic_choice == "Up-and-Out Barrier":
        exotic_type = 'up_and_out'
        barrier_level = st.sidebar.number_input("Barrier", value=round(spot_price * 1.2, 2))
elif model_choice == "Binomial Tree (CRR)":
    bt_steps = st.sidebar.number_input("Tree Steps", 10, 1000, 100, 10)

# Main Interface
st.title(f"Structuring Dashboard: {ticker_input}")

tab_pricing, tab_market, tab_simulation, tab_vol = st.tabs([
    "Pricing & Risk", "Market Data", "Monte Carlo Paths", "Volatility Surface"
])

# Setup parameters
setup_call = OptionSetup(S=spot_price, K=strike_price, T=T, r=rfr, sigma=sigma, q=total_q, option_type='call')
setup_put = OptionSetup(S=spot_price, K=strike_price, T=T, r=rfr, sigma=sigma, q=total_q, option_type='put')

with tab_pricing:
    if st.button("Calculate", type="primary"):
        # Model routing
        if model_choice == "Black-Scholes (Merton)":
            m_call, m_put = BlackScholes(setup_call), BlackScholes(setup_put)
        elif model_choice == "Monte Carlo (Path-Dependent)":
            m_call = MonteCarlo(setup_call, mc_sims, exotic_type=exotic_type, barrier=barrier_level)
            m_put = MonteCarlo(setup_put, mc_sims, exotic_type=exotic_type, barrier=barrier_level)
        else:
            m_call, m_put = BinomialTree(setup_call, bt_steps), BinomialTree(setup_put, bt_steps)

        c_price, p_price = m_call.calculate_price(), m_put.calculate_price()
        
        c1, c2 = st.columns(2)
        c1.metric("Call Price", f"{c_price:.4f}")
        c2.metric("Put Price", f"{p_price:.4f}")

        st.subheader("Risk Parameters (Greeks)")
        g_call, g_put = m_call.calculate_greeks(), m_put.calculate_greeks()
        st.table(pd.DataFrame([g_call, g_put], index=["Call", "Put"]))

        if model_choice == "Monte Carlo (Path-Dependent)":
            st.session_state['paths'] = m_call.simulated_paths

with tab_market:
    if market_data:
        fig = go.Figure(data=[go.Candlestick(x=market_data.history.index,
            open=market_data.history['Open'], high=market_data.history['High'],
            low=market_data.history['Low'], close=market_data.history['Close'])])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

with tab_simulation:
    if 'paths' in st.session_state and model_choice == "Monte Carlo (Path-Dependent)":
        p = st.session_state['paths']
        fig_p = go.Figure()
        for i in range(min(100, p.shape[0])):
            fig_p.add_trace(go.Scatter(y=p[i, :], mode='lines', opacity=0.3, showlegend=False))
        if barrier_level: fig_p.add_hline(y=barrier_level, line_dash="dash", line_color="red")
        fig_p.update_layout(template="plotly_dark", title="Sample Paths (N=100)")
        st.plotly_chart(fig_p, width="stretch")
    else:
        st.info("Run Monte Carlo model to visualize paths.")

with tab_vol:
    st.subheader("Implied Volatility Surface")
    if st.button("Generate Surface"):
        try:
            fig_v = generate_volatility_surface(ticker_input)
            st.plotly_chart(fig_v, width="stretch")
        except Exception as e:
            st.error(f"Surface generation failed: {e}")