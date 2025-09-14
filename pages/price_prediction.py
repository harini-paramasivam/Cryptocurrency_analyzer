import streamlit as st
import pandas as pd
import numpy as np
from utils.data_fetcher import get_historical_prices, export_data_to_csv
from utils.predictive_models import predict_future_prices, prepare_time_series_data
from utils.visualization import plot_prediction_chart

st.set_page_config(
    page_title="Price Prediction - Crypto Market Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Cryptocurrency Price Prediction")
st.markdown("""
This page provides price predictions for cryptocurrencies using time series forecasting models.
""")

# Get selected coins and prediction settings from session state
selected_coins = st.session_state.get('selected_coins', ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana'])
prediction_coin = st.session_state.get('prediction_coin', 'bitcoin')
prediction_days = st.session_state.get('prediction_days', 7)
prediction_model = st.session_state.get('prediction_model', 'ARIMA')
timeframe = st.session_state.get('timeframe', '30')

# Prediction settings
st.sidebar.title("Prediction Settings")

# Cryptocurrency selector
selected_coin = st.sidebar.selectbox(
    "Select cryptocurrency for prediction:",
    options=selected_coins,
    format_func=lambda x: x.title()
)
st.session_state.prediction_coin = selected_coin

# Model selector
model_type = st.sidebar.selectbox(
    "Select prediction model:",
    options=["ARIMA", "LSTM"],
    index=0 if prediction_model == "ARIMA" else 1
)
st.session_state.prediction_model = model_type

# Forecast horizon selector
forecast_days = st.sidebar.slider(
    "Forecast horizon (days):",
    min_value=1,
    max_value=30,
    value=prediction_days
)
st.session_state.prediction_days = forecast_days

# Timeframe selection for historical data
timeframe_options = {
    '30': 'Last 30 Days',
    '90': 'Last 90 Days',
    '180': 'Last 180 Days',
    '365': 'Last Year',
    'max': 'Maximum Available'
}
selected_timeframe = st.sidebar.selectbox(
    "Select historical data timeframe:",
    options=list(timeframe_options.keys()),
    format_func=lambda x: timeframe_options[x],
    index=0
)

# Main content
st.header(f"{selected_coin.title()} Price Prediction")

# Warning about predictions
st.warning("""
**Disclaimer:** Price predictions are based on historical data and should not be considered as
financial advice. Cryptocurrency markets are highly volatile and unpredictable.
""")

try:
    # Fetch historical data for selected coin
    with st.spinner(f"Fetching historical data for {selected_coin}..."):
        historical_data = get_historical_prices(selected_coin, days=selected_timeframe)
    
    if not historical_data.empty:
        # Display data preparation message
        with st.spinner("Preparing data for modeling..."):
            # Prepare time series data
            prepared_data = prepare_time_series_data(historical_data)
        
        # Model training and prediction
        with st.spinner(f"Training {model_type} model and making predictions..."):
            # Generate predictions
            forecast_df, model_accuracy, model_summary = predict_future_prices(
                prepared_data, 
                model_type, 
                target_column='price',
                forecast_days=forecast_days
            )
        
        if not forecast_df.empty:
            # Display prediction chart
            st.subheader(f"{model_type} Price Prediction for Next {forecast_days} Days")
            
            # Create prediction chart
            fig = plot_prediction_chart(
                historical_data, 
                forecast_df, 
                coin_name=selected_coin.title(),
                model_type=model_type,
                mape=model_accuracy
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction metrics
            st.subheader("Prediction Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            # Current price and forecasted price
            current_price = historical_data['price'].iloc[-1]
            forecasted_end_price = forecast_df['forecast'].iloc[-1]
            price_change = ((forecasted_end_price - current_price) / current_price) * 100
            
            col1.metric(
                "Current Price", 
                f"${current_price:.2f}"
            )
            
            col2.metric(
                f"Predicted Price (Day {forecast_days})", 
                f"${forecasted_end_price:.2f}",
                f"{price_change:.2f}%"
            )
            
            col3.metric(
                "Model Accuracy (MAPE)", 
                f"{model_accuracy:.2f}%",
                "Lower is better"
            )
            
            # Display forecast table
            st.subheader("Forecast Table")
            
            # Format forecast DataFrame for display
            display_df = forecast_df.copy().reset_index()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['forecast'] = display_df['forecast'].apply(lambda x: f"${x:.2f}")
            
            # Rename columns for better display
            display_df = display_df.rename(columns={
                'date': 'Date',
                'forecast': 'Forecasted Price (USD)'
            })
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export forecast data button
            csv = export_data_to_csv(forecast_df, f"{selected_coin}_{model_type}_forecast.csv")
            st.download_button(
                label="Download Forecast Data as CSV",
                data=csv,
                file_name=f"{selected_coin}_{model_type}_forecast.csv",
                mime="text/csv",
            )
            
            # Model details
            with st.expander("Model Details"):
                st.text(model_summary)
        else:
            st.error("Failed to generate predictions. Please try a different model or cryptocurrency.")
    else:
        st.error(f"No historical data available for {selected_coin}.")
    
except Exception as e:
    st.error(f"Error during prediction: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Model information section
st.header("About the Models")

# ARIMA model info
with st.expander("ARIMA Model"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)** is a statistical model used for analyzing and forecasting time series data.
    
    **Strengths:**
    - Works well with stationary time series data
    - Good for short-term forecasting
    - Captures trend and seasonality
    - Requires less data than deep learning models
    
    **Limitations:**
    - Assumes linear relationships
    - May not capture complex patterns
    - Less effective with highly volatile data
    """)

# LSTM model info (PyTorch implementation)
with st.expander("LSTM Model (PyTorch)"):
    st.markdown("""
    **LSTM (Long Short-Term Memory)** is a type of recurrent neural network designed to recognize patterns in sequences of data. This implementation uses PyTorch, a flexible deep learning framework.
    
    **Strengths:**
    - Can capture complex non-linear patterns
    - Better at handling long-term dependencies
    - More robust to noise and volatility
    - Can incorporate multiple features
    - Dynamic computational graph (PyTorch advantage)
    
    **Limitations:**
    - Requires more data for training
    - Computationally intensive
    - Prone to overfitting with limited data
    - Longer training time
    """)

# Navigation
st.sidebar.subheader("Navigation")
if st.sidebar.button("Return to Home"):
    st.switch_page("app.py")
if st.sidebar.button("Market Overview"):
    st.switch_page("pages/market_overview.py")
if st.sidebar.button("Cryptocurrency Detail"):
    st.switch_page("pages/cryptocurrency_detail.py")
if st.sidebar.button("Compare Cryptocurrencies"):
    st.switch_page("pages/compare_cryptos.py")

# About section
st.sidebar.subheader("About")
st.sidebar.info("""
This price prediction dashboard uses time series forecasting models to predict
future cryptocurrency prices based on historical data. The predictions are for
educational purposes only and should not be used for trading decisions.
Data is sourced from CoinGecko API.
""")