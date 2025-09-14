import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_price_chart(df, title="Price Chart", height=500):
    """
    Create a price chart using Plotly
    
    Args:
        df (pandas.DataFrame): DataFrame containing price data with datetime index
        title (str): Title for the chart
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = px.line(
        df, 
        y='price', 
        title=title,
        labels={'price': 'Price (USD)', 'timestamp': 'Date'},
        height=height
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_price_volume_chart(df, coin_name="Cryptocurrency", height=700):
    """
    Create a price and volume chart using Plotly subplots
    
    Args:
        df (pandas.DataFrame): DataFrame containing price and volume data with datetime index
        coin_name (str): Name of the cryptocurrency
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{coin_name} Price (USD)", f"{coin_name} Volume (USD)"),
        row_heights=[0.7, 0.3]
    )
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['price'], 
            name="Price",
            line=dict(color='#00b0f0', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add volume trace
    fig.add_trace(
        go.Bar(
            x=df.index, 
            y=df['volume'], 
            name="Volume",
            marker=dict(color='rgba(0, 176, 240, 0.5)')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text=f"{coin_name} Historical Data",
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add range slider
    fig.update_layout(
        xaxis2=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_prediction_chart(historical_df, forecast_df, coin_name="Cryptocurrency", model_type="ARIMA", mape=None, height=600):
    """
    Create a price prediction chart using Plotly
    
    Args:
        historical_df (pandas.DataFrame): DataFrame containing historical price data with datetime index
        forecast_df (pandas.DataFrame): DataFrame containing forecast data with datetime index
        coin_name (str): Name of the cryptocurrency
        model_type (str): Type of model used for prediction
        mape (float): Mean Absolute Percentage Error of the model
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add historical price trace
    fig.add_trace(
        go.Scatter(
            x=historical_df.index,
            y=historical_df['price'],
            name="Historical Price",
            line=dict(color='#00b0f0', width=2)
        )
    )
    
    # Add forecast trace
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df['forecast'],
            name=f"{model_type} Forecast",
            line=dict(color='#ff7f0e', width=2, dash='dash')
        )
    )
    
    # Update layout
    title_text = f"{coin_name} Price Prediction using {model_type}"
    if mape is not None:
        title_text += f" (MAPE: {mape:.2f}%)"
        
    fig.update_layout(
        title_text=title_text,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_comparison_chart(coin_data_dict, metric='price', title=None, height=600):
    """
    Create a comparison chart for multiple cryptocurrencies
    
    Args:
        coin_data_dict (dict): Dictionary with coin names as keys and DataFrames as values
        metric (str): Metric to compare ('price', 'volume', or 'market_cap')
        title (str): Title for the chart
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for coin_name, df in coin_data_dict.items():
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[metric],
                    name=coin_name.title(),
                    mode='lines'
                )
            )
    
    if title is None:
        title = f"Cryptocurrency {metric.title()} Comparison"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Date",
        yaxis_title=f"{metric.title()} (USD)",
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def plot_normalized_comparison(coin_data_dict, metric='price', title=None, height=600):
    """
    Create a normalized comparison chart for multiple cryptocurrencies
    
    Args:
        coin_data_dict (dict): Dictionary with coin names as keys and DataFrames as values
        metric (str): Metric to compare ('price', 'volume', or 'market_cap')
        title (str): Title for the chart
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    for coin_name, df in coin_data_dict.items():
        if metric in df.columns:
            # Normalize the data (first value = 100)
            normalized = df[metric] / df[metric].iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized,
                    name=coin_name.title(),
                    mode='lines'
                )
            )
    
    if title is None:
        title = f"Normalized {metric.title()} Comparison (Base: 100)"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base: 100)",
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig

def create_market_metrics_chart(market_data, metric, title=None, height=500):
    """
    Create a bar chart for market metrics
    
    Args:
        market_data (pandas.DataFrame): DataFrame containing market data
        metric (str): Metric to display
        title (str): Title for the chart
        height (int): Height of the chart in pixels
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if title is None:
        title = f"Cryptocurrency {metric.replace('_', ' ').title()}"
    
    fig = px.bar(
        market_data,
        x='name',
        y=metric,
        color='name',
        title=title,
        height=height
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Cryptocurrency",
        yaxis_title=f"{metric.replace('_', ' ').title()}",
        template="plotly_dark",
        showlegend=False,
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig