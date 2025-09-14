import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import streamlit as st

# Base URL for CoinGecko API
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Cache time in seconds
CACHE_TIME = 300  # 5 minutes

@st.cache_data(ttl=CACHE_TIME)
def get_supported_coins():
    """
    Get list of supported coins from CoinGecko API
    
    Returns:
        list: List of coin IDs or None if API call fails
    """
    try:
        response = requests.get(f"{COINGECKO_API_URL}/coins/list")
        if response.status_code == 200:
            coins = response.json()
            # Return just the IDs of popular coins to avoid overwhelming the user
            popular_coins = [
                'bitcoin', 'ethereum', 'ripple', 'cardano', 'solana', 
                'polkadot', 'dogecoin', 'avalanche-2', 'chainlink',
                'litecoin', 'polygon', 'binancecoin', 'uniswap',
                'tron', 'stellar', 'cosmos', 'monero', 'algorand', 'tezos'
            ]
            return popular_coins
        else:
            return None
    except Exception as e:
        st.error(f"Error getting supported coins: {str(e)}")
        return None

@st.cache_data(ttl=CACHE_TIME)
def get_market_data(coin_ids):
    """
    Get market data for specified coins from CoinGecko API
    
    Args:
        coin_ids (list): List of coin IDs to fetch data for
        
    Returns:
        pandas.DataFrame: DataFrame containing market data
    """
    try:
        params = {
            'ids': ','.join(coin_ids),
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h'
        }
        
        response = requests.get(f"{COINGECKO_API_URL}/coins/markets", params=params)
        
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TIME)
def get_historical_prices(coin_id, days='30', vs_currency='usd'):
    """
    Get historical price data for a specific coin from CoinGecko API
    
    Args:
        coin_id (str): ID of the coin to fetch data for
        days (str): Number of days of data to fetch, can be '1', '7', '30', '90', '365', 'max'
        vs_currency (str): Currency to compare against, default 'usd'
        
    Returns:
        pandas.DataFrame: DataFrame containing historical price data with datetime index
    """
    try:
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        
        response = requests.get(f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart", params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract price data
            df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], unit='ms')
            
            # Extract volume data
            df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            df_volume['timestamp'] = pd.to_datetime(df_volume['timestamp'], unit='ms')
            
            # Extract market cap data
            df_market_cap = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            df_market_cap['timestamp'] = pd.to_datetime(df_market_cap['timestamp'], unit='ms')
            
            # Merge dataframes
            df = pd.merge(df_price, df_volume, on='timestamp')
            df = pd.merge(df, df_market_cap, on='timestamp')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            return df
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TIME)
def get_coin_details(coin_id):
    """
    Get detailed information about a specific coin from CoinGecko API
    
    Args:
        coin_id (str): ID of the coin to fetch data for
        
    Returns:
        dict: Dictionary containing coin details
    """
    try:
        response = requests.get(f"{COINGECKO_API_URL}/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false")
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching coin details: {str(e)}")
        return {}

def export_data_to_csv(data, filename):
    """
    Export DataFrame to CSV
    
    Args:
        data (pandas.DataFrame): Data to export
        filename (str): Name for the CSV file
        
    Returns:
        bytes: CSV file as bytes for download
    """
    try:
        return data.to_csv().encode('utf-8')
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return