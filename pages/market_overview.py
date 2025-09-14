import streamlit as st
import pandas as pd
import numpy as np
from utils.data_fetcher import get_market_data, get_historical_prices, export_data_to_csv
from utils.visualization import plot_price_volume_chart, create_market_metrics_chart

st.set_page_config(
    page_title="Market Overview - Crypto Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Cryptocurrency Market Overview")
st.markdown("""
This page provides an overview of the cryptocurrency market, including market metrics,
historical price charts, and volume analysis.
""")

# Get selected coins from session state
selected_coins = st.session_state.get('selected_coins', ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana'])
timeframe = st.session_state.get('timeframe', '30')

# Market Overview section
st.header("Market Overview")

try:
    # Fetch market data
    market_data = get_market_data(selected_coins)
    
    if not market_data.empty:
        # Display market metrics summary
        st.subheader("Market Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market Cap chart
            fig_market_cap = create_market_metrics_chart(
                market_data, 
                'market_cap', 
                "Market Capitalization (USD)",
                height=400
            )
            st.plotly_chart(fig_market_cap, use_container_width=True)
        
        with col2:
            # 24h Volume chart
            fig_volume = create_market_metrics_chart(
                market_data, 
                'total_volume', 
                "24h Trading Volume (USD)",
                height=400
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart
            fig_price = create_market_metrics_chart(
                market_data, 
                'current_price', 
                "Current Price (USD)",
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # 24h Change chart
            fig_change = create_market_metrics_chart(
                market_data, 
                'price_change_percentage_24h', 
                "24h Price Change (%)",
                height=400
            )
            st.plotly_chart(fig_change, use_container_width=True)
        
        # Display market data table
        st.subheader("Market Data Table")
        
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
        
        # Export market data button
        csv = export_data_to_csv(market_data, "crypto_market_data.csv")
        st.download_button(
            label="Download Market Data as CSV",
            data=csv,
            file_name="crypto_market_data.csv",
            mime="text/csv",
        )
    else:
        st.error("No market data available. Please check your connection or selected cryptocurrencies.")
    
except Exception as e:
    st.error(f"Error loading market data: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Historical Price Charts section
st.header("Historical Price Charts")

# Cryptocurrency selector for historical chart
selected_coin = st.selectbox(
    "Select cryptocurrency for historical analysis:",
    options=selected_coins,
    format_func=lambda x: x.title()
)

try:
    # Fetch historical data for selected coin
    historical_data = get_historical_prices(selected_coin, days=timeframe)
    
    if not historical_data.empty:
        st.subheader(f"{selected_coin.title()} Historical Data")
        
        # Create price and volume chart
        fig = plot_price_volume_chart(historical_data, coin_name=selected_coin.title())
        st.plotly_chart(fig, use_container_width=True)
        
        # Show historical data summary
        st.subheader("Price Summary")
        
        # Calculate summary statistics
        price_min = historical_data['price'].min()
        price_max = historical_data['price'].max()
        price_mean = historical_data['price'].mean()
        price_current = historical_data['price'].iloc[-1]
        price_first = historical_data['price'].iloc[0]
        price_change = ((price_current - price_first) / price_first) * 100
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${price_current:.2f}")
        col2.metric("Period Change", f"{price_change:.2f}%")
        col3.metric("Period High", f"${price_max:.2f}")
        col4.metric("Period Low", f"${price_min:.2f}")
        
        # Export historical data button
        csv = export_data_to_csv(historical_data, f"{selected_coin}_historical_data.csv")
        st.download_button(
            label="Download Historical Data as CSV",
            data=csv,
            file_name=f"{selected_coin}_historical_data.csv",
            mime="text/csv",
        )
    else:
        st.error(f"No historical data available for {selected_coin}.")
    
except Exception as e:
    st.error(f"Error loading historical data: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Sidebar configuration
st.sidebar.title("Market Overview Settings")

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
    index=list(timeframe_options.keys()).index(timeframe)
)
st.session_state.timeframe = selected_timeframe

# Navigation
st.sidebar.subheader("Navigation")
if st.sidebar.button("Return to Home"):
    st.switch_page("app.py")
if st.sidebar.button("Cryptocurrency Detail"):
    st.switch_page("pages/cryptocurrency_detail.py")
if st.sidebar.button("Price Prediction"):
    st.switch_page("pages/price_prediction.py")
if st.sidebar.button("Compare Cryptocurrencies"):
    st.switch_page("pages/compare_cryptos.py")

# About section
st.sidebar.subheader("About")
st.sidebar.info("""
This market overview dashboard provides real-time cryptocurrency market metrics
and historical price charts. Data is sourced from CoinGecko API.
""")