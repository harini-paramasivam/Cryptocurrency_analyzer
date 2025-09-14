import streamlit as st
import pandas as pd
import numpy as np
from utils.data_fetcher import get_coin_details, get_historical_prices, export_data_to_csv
from utils.visualization import plot_price_volume_chart
import plotly.express as px

st.set_page_config(
    page_title="Cryptocurrency Detail - Crypto Market Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Cryptocurrency Detail Analysis")
st.markdown("""
This page provides detailed analysis for a specific cryptocurrency, including price history,
market metrics, and technical indicators.
""")

# Get selected coins from session state
selected_coins = st.session_state.get('selected_coins', ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana'])
timeframe = st.session_state.get('timeframe', '30')

# Cryptocurrency selector
selected_coin = st.selectbox(
    "Select cryptocurrency for detailed analysis:",
    options=selected_coins,
    format_func=lambda x: x.title()
)

try:
    # Fetch coin details
    coin_details = get_coin_details(selected_coin)
    
    if coin_details:
        # Display coin details
        st.header(f"{coin_details.get('name', selected_coin.title())} ({coin_details.get('symbol', '').upper()})")
        
        # Coin info section
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if 'image' in coin_details and 'large' in coin_details['image']:
                st.image(coin_details['image']['large'])
            
            st.subheader("Basic Information")
            st.markdown(f"**Genesis Date:** {coin_details.get('genesis_date', 'N/A')}")
            st.markdown(f"**Market Cap Rank:** #{coin_details.get('market_cap_rank', 'N/A')}")
            
            if 'links' in coin_details and 'homepage' in coin_details['links'] and coin_details['links']['homepage']:
                st.markdown(f"**Website:** [{coin_details['links']['homepage'][0]}]({coin_details['links']['homepage'][0]})")
            
            if 'links' in coin_details and 'blockchain_site' in coin_details['links'] and coin_details['links']['blockchain_site']:
                st.markdown(f"**Explorer:** [{coin_details['links']['blockchain_site'][0]}]({coin_details['links']['blockchain_site'][0]})")
        
        with col2:
            st.subheader("Market Data")
            
            if 'market_data' in coin_details:
                market_data = coin_details['market_data']
                
                current_price = market_data.get('current_price', {}).get('usd', 'N/A')
                if current_price != 'N/A':
                    current_price = f"${current_price:,.2f}"
                st.markdown(f"**Current Price:** {current_price}")
                
                market_cap = market_data.get('market_cap', {}).get('usd', 'N/A')
                if market_cap != 'N/A':
                    market_cap = f"${market_cap:,.0f}"
                st.markdown(f"**Market Cap:** {market_cap}")
                
                total_volume = market_data.get('total_volume', {}).get('usd', 'N/A')
                if total_volume != 'N/A':
                    total_volume = f"${total_volume:,.0f}"
                st.markdown(f"**24h Volume:** {total_volume}")
                
                price_change_24h = market_data.get('price_change_percentage_24h', 'N/A')
                if price_change_24h != 'N/A':
                    price_change_24h = f"{price_change_24h:.2f}%"
                st.markdown(f"**24h Change:** {price_change_24h}")
                
                ath = market_data.get('ath', {}).get('usd', 'N/A')
                if ath != 'N/A':
                    ath = f"${ath:,.2f}"
                st.markdown(f"**All-Time High:** {ath}")
                
                atl = market_data.get('atl', {}).get('usd', 'N/A')
                if atl != 'N/A':
                    atl = f"${atl:,.2f}"
                st.markdown(f"**All-Time Low:** {atl}")
        
        with col3:
            if 'description' in coin_details and 'en' in coin_details['description'] and coin_details['description']['en']:
                st.subheader("Description")
                st.markdown(coin_details['description']['en'][:500] + "..." if len(coin_details['description']['en']) > 500 else coin_details['description']['en'])
        
        # Fetch historical data for selected coin
        historical_data = get_historical_prices(selected_coin, days=timeframe)
        
        if not historical_data.empty:
            st.header("Price History")
            
            # Create price and volume chart
            fig = plot_price_volume_chart(historical_data, coin_name=coin_details.get('name', selected_coin.title()))
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display key metrics
            st.subheader("Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Price metrics
            price_current = historical_data['price'].iloc[-1]
            price_first = historical_data['price'].iloc[0]
            price_change = ((price_current - price_first) / price_first) * 100
            price_high = historical_data['price'].max()
            price_low = historical_data['price'].min()
            
            col1.metric("Current Price", f"${price_current:.2f}")
            col2.metric("Period Change", f"{price_change:.2f}%")
            col3.metric("Period High", f"${price_high:.2f}")
            col4.metric("Period Low", f"${price_low:.2f}")
            
            # Volume metrics
            col1, col2, col3, col4 = st.columns(4)
            
            volume_current = historical_data['volume'].iloc[-1]
            volume_avg = historical_data['volume'].mean()
            volume_high = historical_data['volume'].max()
            
            col1.metric("24h Volume", f"${volume_current:,.0f}")
            col2.metric("Avg. Volume", f"${volume_avg:,.0f}")
            col3.metric("Max Volume", f"${volume_high:,.0f}")
            col4.metric("Vol/Price Ratio", f"{(volume_current/price_current):,.0f}")
            
            # Historical data table
            st.subheader("Historical Data Table")
            
            # Format the DataFrame for display
            display_df = historical_data.copy().reset_index()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.2f}" if x is not None else "N/A")
            display_df['volume'] = display_df['volume'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
            display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
            
            # Rename columns for better display
            display_df = display_df.rename(columns={
                'timestamp': 'Date',
                'price': 'Price (USD)',
                'volume': 'Volume (USD)',
                'market_cap': 'Market Cap (USD)'
            })
            
            st.dataframe(display_df, use_container_width=True)
            
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
    else:
        st.error(f"No details available for {selected_coin}.")
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Sidebar configuration
st.sidebar.title("Cryptocurrency Detail Settings")

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
if st.sidebar.button("Market Overview"):
    st.switch_page("pages/market_overview.py")
if st.sidebar.button("Price Prediction"):
    st.switch_page("pages/price_prediction.py")
if st.sidebar.button("Compare Cryptocurrencies"):
    st.switch_page("pages/compare_cryptos.py")

# About section
st.sidebar.subheader("About")
st.sidebar.info("""
This cryptocurrency detail dashboard provides in-depth analysis for a specific
cryptocurrency, including price history, market metrics, and technical indicators.
Data is sourced from CoinGecko API.
""")