import streamlit as st
import pandas as pd
import numpy as np
from utils.data_fetcher import get_market_data, get_historical_prices, export_data_to_csv
from utils.visualization import plot_comparison_chart, plot_normalized_comparison

st.set_page_config(
    page_title="Compare Cryptocurrencies - Crypto Market Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Compare Cryptocurrencies")
st.markdown("""
This page allows you to compare multiple cryptocurrencies based on various metrics
such as price, market cap, and trading volume.
""")

# Get selected coins and comparison settings from session state
selected_coins = st.session_state.get('selected_coins', ['bitcoin', 'ethereum', 'ripple', 'cardano', 'solana'])
compare_coins = st.session_state.get('compare_coins', ['bitcoin', 'ethereum'])
timeframe = st.session_state.get('timeframe', '30')

# Comparison settings
st.sidebar.title("Comparison Settings")

# Cryptocurrencies selector
coins_to_compare = st.sidebar.multiselect(
    "Select cryptocurrencies to compare:",
    options=selected_coins,
    default=compare_coins,
    format_func=lambda x: x.title()
)

if coins_to_compare:
    st.session_state.compare_coins = coins_to_compare
else:
    st.sidebar.error("Please select at least one cryptocurrency")
    coins_to_compare = compare_coins

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
    "Select timeframe for comparison:",
    options=list(timeframe_options.keys()),
    format_func=lambda x: timeframe_options[x],
    index=list(timeframe_options.keys()).index(timeframe)
)
st.session_state.timeframe = selected_timeframe

# Metrics selection
metrics = st.sidebar.multiselect(
    "Select metrics to compare:",
    options=["price", "volume", "market_cap"],
    default=["price"],
    format_func=lambda x: x.title()
)

# Normalization option
normalize = st.sidebar.checkbox("Normalize data (for easier comparison)", value=True)

# Main content
st.header(f"Comparing {', '.join([coin.title() for coin in coins_to_compare])}")

try:
    # Fetch and store historical data for selected coins
    historical_data_dict = {}
    
    with st.spinner("Fetching historical data for selected cryptocurrencies..."):
        for coin in coins_to_compare:
            historical_data = get_historical_prices(coin, days=selected_timeframe)
            if not historical_data.empty:
                historical_data_dict[coin] = historical_data
    
    if historical_data_dict:
        # Compare metrics
        for metric in metrics:
            st.subheader(f"{metric.title()} Comparison")
            
            if normalize:
                # Create normalized comparison chart
                fig = plot_normalized_comparison(
                    historical_data_dict,
                    metric=metric,
                    title=f"Normalized {metric.title()} Comparison (Base: 100)"
                )
            else:
                # Create direct comparison chart
                fig = plot_comparison_chart(
                    historical_data_dict,
                    metric=metric,
                    title=f"{metric.title()} Comparison"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market metrics comparison
        st.subheader("Current Market Metrics Comparison")
        
        # Fetch current market data
        market_data = get_market_data(coins_to_compare)
        
        if not market_data.empty:
            # Format market data for display
            display_df = market_data.copy()
            
            # Select relevant columns for comparison
            columns_to_display = {
                'name': 'Name',
                'current_price': 'Current Price (USD)',
                'price_change_percentage_24h': '24h Change (%)',
                'market_cap': 'Market Cap (USD)',
                'total_volume': '24h Volume (USD)',
                'market_cap_rank': 'Market Cap Rank'
            }
            
            display_df = display_df[columns_to_display.keys()].rename(columns=columns_to_display)
            
            # Format values
            display_df['Current Price (USD)'] = display_df['Current Price (USD)'].apply(lambda x: f"${x:,.2f}" if x is not None else "N/A")
            display_df['24h Change (%)'] = display_df['24h Change (%)'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
            display_df['Market Cap (USD)'] = display_df['Market Cap (USD)'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
            display_df['24h Volume (USD)'] = display_df['24h Volume (USD)'].apply(lambda x: f"${x:,.0f}" if x is not None else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Create combined historical dataset for export
            combined_data = pd.DataFrame()
            
            for coin, data in historical_data_dict.items():
                # For each metric in the data
                for metric in ['price', 'volume', 'market_cap']:
                    if metric in data.columns:
                        column_name = f"{coin}_{metric}"
                        # Add the data to the combined dataframe
                        if combined_data.empty:
                            combined_data = pd.DataFrame(index=data.index)
                        combined_data[column_name] = data[metric]
            
            # Export combined data button
            if not combined_data.empty:
                csv = export_data_to_csv(combined_data, "crypto_comparison_data.csv")
                st.download_button(
                    label="Download Comparison Data as CSV",
                    data=csv,
                    file_name="crypto_comparison_data.csv",
                    mime="text/csv",
                )
        else:
            st.error("No market data available for comparison.")
    else:
        st.error("No historical data available for selected cryptocurrencies.")
    
except Exception as e:
    st.error(f"Error during comparison: {str(e)}")
    st.info("Please check your internet connection and try again.")

# Correlation Analysis
if len(historical_data_dict) > 1 and 'price' in metrics:
    st.header("Price Correlation Analysis")
    
    try:
        # Create a DataFrame with prices for each coin
        price_df = pd.DataFrame()
        
        for coin, data in historical_data_dict.items():
            price_df[coin] = data['price']
        
        # Calculate correlation matrix
        corr_matrix = price_df.corr()
        
        # Display correlation matrix
        st.subheader("Price Correlation Matrix")
        st.dataframe(corr_matrix.style.format("{:.2f}").background_gradient(cmap='coolwarm'), use_container_width=True)
        
        # Create heatmap using Plotly
        import plotly.figure_factory as ff
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            annotation_text=np.around(corr_matrix.values, decimals=2).tolist(),
            colorscale='Viridis'
        )
        
        fig.update_layout(
            title="Cryptocurrency Price Correlation Heatmap",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation interpretation
        st.subheader("Correlation Interpretation")
        st.markdown("""
        **Understanding the correlation coefficient:**
        - **1.00**: Perfect positive correlation
        - **0.70 to 0.99**: Strong positive correlation
        - **0.50 to 0.69**: Moderate positive correlation
        - **0.30 to 0.49**: Weak positive correlation
        - **0.00 to 0.29**: Negligible correlation
        - **-0.29 to 0.00**: Negligible correlation
        - **-0.49 to -0.30**: Weak negative correlation
        - **-0.69 to -0.50**: Moderate negative correlation
        - **-0.99 to -0.70**: Strong negative correlation
        - **-1.00**: Perfect negative correlation
        """)
        
        # Find highest and lowest correlation pairs
        if len(corr_matrix) > 1:
            # Get upper triangle of correlation matrix excluding diagonal
            upper_triangle = np.triu(corr_matrix.values, k=1)
            
            # Find indices of max and min values
            if len(upper_triangle[upper_triangle != 0]) > 0:
                max_idx = np.unravel_index(upper_triangle.argmax(), upper_triangle.shape)
                max_corr = upper_triangle[max_idx]
                max_pair = (corr_matrix.index[max_idx[0]], corr_matrix.columns[max_idx[1]])
                
                min_upper = upper_triangle.copy()
                min_upper[min_upper == 0] = 1  # Replace zeros with ones to exclude them
                min_idx = np.unravel_index(min_upper.argmin(), min_upper.shape)
                min_corr = upper_triangle[min_idx]
                min_pair = (corr_matrix.index[min_idx[0]], corr_matrix.columns[min_idx[1]])
                
                st.markdown(f"""
                **Highest correlation: {max_corr:.2f}** between **{max_pair[0].title()}** and **{max_pair[1].title()}**
                
                **Lowest correlation: {min_corr:.2f}** between **{min_pair[0].title()}** and **{min_pair[1].title()}**
                """)
    
    except Exception as e:
        st.error(f"Error during correlation analysis: {str(e)}")

# Navigation
st.sidebar.subheader("Navigation")
if st.sidebar.button("Return to Home"):
    st.switch_page("app.py")
if st.sidebar.button("Market Overview"):
    st.switch_page("pages/market_overview.py")
if st.sidebar.button("Cryptocurrency Detail"):
    st.switch_page("pages/cryptocurrency_detail.py")
if st.sidebar.button("Price Prediction"):
    st.switch_page("pages/price_prediction.py")

# About section
st.sidebar.subheader("About")
st.sidebar.info("""
This comparison dashboard allows you to compare multiple cryptocurrencies
based on various metrics such as price, volume, and market cap. The correlation
analysis helps identify relationships between different cryptocurrencies.
Data is sourced from CoinGecko API.
""")