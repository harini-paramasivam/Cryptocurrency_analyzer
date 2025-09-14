import os
os.environ["STREAMLIT_CONFIG_DIR"] = ".streamlit"

import streamlit as st
from utils.data_fetcher import get_supported_coins, get_market_data
import traceback
import pandas as pd


st.set_page_config(
    page_title="Crypto Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if 'selected_coins' not in st.session_state:
    st.session_state.selected_coins = ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana']
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = '30'
if 'compare_coins' not in st.session_state:
    st.session_state.compare_coins = ['bitcoin', 'ethereum']
if 'prediction_coin' not in st.session_state:
    st.session_state.prediction_coin = 'bitcoin'
if 'prediction_days' not in st.session_state:
    st.session_state.prediction_days = 7
if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = 'ARIMA'

# App title and description
st.title("Cryptocurrency Market Analyzer")
st.markdown("""
This dashboard provides real-time cryptocurrency market analysis, 
historical price tracking, and predictive modeling to help you make informed decisions.
""")

# Display cryptocurrency market dashboard images
col1, col2 = st.columns(2)
with col1:
    st.image("https://images.unsplash.com/photo-1554260570-e9689a3418b8", 
             caption="Market Analysis Dashboard", use_container_width=True)
with col2:
    st.image("https://images.unsplash.com/photo-1488459716781-31db52582fe9", 
             caption="Market Analytics", use_container_width=True)

# Load market data for the main page
try:
    market_data = get_market_data(st.session_state.selected_coins)
    
    # Display market overview
    st.subheader("Market Overview")
    
    # Display market data in a table
    if not market_data.empty:
        # Format the DataFrame for display
        display_df = market_data.copy()
        display_df['price_change_percentage_24h'] = display_df['price_change_percentage_24h'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}" if x is not None else "N/A")
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
        display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'id': 'ID',
            'symbol': 'Symbol',
            'name': 'Name',
            'current_price': 'Current Price (USD)',
            'market_cap': 'Market Cap (USD)',
            'market_cap_rank': 'Market Cap Rank',
            'price_change_percentage_24h': '24h Change',
            'total_volume': '24h Volume (USD)'
        })
        
        st.dataframe(display_df[['Symbol', 'Name', 'Current Price (USD)', '24h Change', 'Market Cap (USD)', 'Market Cap Rank', '24h Volume (USD)']], use_container_width=True)
    else:
        st.error("No market data available. Please check your connection or selected cryptocurrencies.")
    
    # Navigation buttons for different pages
    st.subheader("Navigate to Detailed Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Market Overview", use_container_width=True):
            st.switch_page("pages/market_overview.py")
    
    with col2:
        if st.button("🔍 Cryptocurrency Detail", use_container_width=True):
            st.switch_page("pages/cryptocurrency_detail.py")
    
    with col3:
        if st.button("🔮 Price Prediction", use_container_width=True):
            st.switch_page("pages/price_prediction.py")
    
    with col4:
        if st.button("⚖️ Compare Cryptocurrencies", use_container_width=True):
            st.switch_page("pages/compare_cryptos.py")

except Exception as e:
    st.error(f"Error loading market data: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Display cryptocurrency chart images
st.subheader("Cryptocurrency Market Charts")
col1, col2 = st.columns(2)
with col1:
    st.image("https://images.unsplash.com/photo-1639754390580-2e7437267698",
             caption="Cryptocurrency Trading", use_container_width=True)
    st.image("https://images.unsplash.com/photo-1639987402632-d7273e921454",
             caption="Market Analysis Chart", use_container_width=True)
with col2:
    st.image("https://images.unsplash.com/photo-1639389016105-2fb11199fb6b",
             caption="Cryptocurrency Dashboard", use_container_width=True)
    st.image("https://images.unsplash.com/photo-1640592276475-56a1c277a38f",
             caption="Crypto Trading Analytics", use_container_width=True)

# Sidebar configuration
st.sidebar.title("Dashboard Settings")

# Get supported coins for selection
try:
    all_coins = get_supported_coins()
    if all_coins:
        # Dashboard customization options
        st.sidebar.subheader("Customize Dashboard")
        selected_coins = st.sidebar.multiselect(
            "Select cryptocurrencies to track:",
            options=all_coins,
            default=st.session_state.selected_coins,
            format_func=lambda x: x.title()
        )
        if selected_coins:
            st.session_state.selected_coins = selected_coins
        
        # Timeframe selection
        timeframe_options = {
            '1': 'Last 24 Hours',
            '7': 'Last 7 Days',
            '30': 'Last 30 Days',
            '90': 'Last 90 Days',
            '365': 'Last Year',
            'max': 'Maximum Available'
        }
        selected_timeframe = st.sidebar.selectbox(
            "Select timeframe for analysis:",
            options=list(timeframe_options.keys()),
            format_func=lambda x: timeframe_options[x],
            index=list(timeframe_options.keys()).index(st.session_state.timeframe)
        )
        st.session_state.timeframe = selected_timeframe
        
        # About section
        st.sidebar.subheader("About")
        st.sidebar.info("""
        This Cryptocurrency Market Analyzer provides real-time market data, 
        historical price charts, and predictive modeling to help you make 
        informed investment decisions. Data is sourced from CoinGecko API.
        """)
    else:
        st.sidebar.error("Could not load supported cryptocurrencies.")
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")
    st.sidebar.info("Using default coin selection.")


