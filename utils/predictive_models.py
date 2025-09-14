import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def prepare_time_series_data(df, target_column='price'):
    """
    Prepare time series data for modeling
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data with datetime index
        target_column (str): Column name for the target variable
        
    Returns:
        pandas.DataFrame: Prepared DataFrame for time series analysis
    """
    # Make a copy to avoid modifying the original
    df_prep = df.copy()
    
    # Ensure the DataFrame is sorted by date
    df_prep = df_prep.sort_index()
    
    # Resample to daily frequency if needed (taking the mean of each day)
    if not df_prep.index.is_unique:
        df_prep = df_prep.resample('D').mean()
    
    # Fill missing values if any
    df_prep = df_prep.fillna(method='ffill')
    
    # Create lag features
    for lag in [1, 2, 3, 7]:
        column_name = f'{target_column}_lag_{lag}'
        df_prep[column_name] = df_prep[target_column].shift(lag)
    
    # Create rolling window features
    for window in [3, 7, 14]:
        # Rolling mean
        df_prep[f'{target_column}_rolling_mean_{window}'] = df_prep[target_column].rolling(window=window).mean()
        # Rolling std
        df_prep[f'{target_column}_rolling_std_{window}'] = df_prep[target_column].rolling(window=window).std()
    
    # Drop rows with NaN values after creating lag features
    df_prep = df_prep.dropna()
    
    return df_prep

def test_stationarity(time_series):
    """
    Test stationarity of time series using Augmented Dickey-Fuller test
    
    Args:
        time_series (pandas.Series): Time series data
        
    Returns:
        tuple: (is_stationary, p_value, test_statistic)
    """
    # Perform ADF test
    result = adfuller(time_series.dropna())
    
    # Extract test results
    test_statistic = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05
    
    return is_stationary, p_value, test_statistic

def fit_arima_model(df, target_column='price', forecast_days=7):
    """
    Fit ARIMA model to time series data and make forecast
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data with datetime index
        target_column (str): Column name for the target variable
        forecast_days (int): Number of days to forecast
        
    Returns:
        tuple: (forecast_df, mape, model_results)
    """
    try:
        # Check if df is empty
        if df.empty:
            st.error("No data available for modeling")
            return pd.DataFrame(), 0, None
        
        # Get the time series data
        time_series = df[target_column]
        
        # Check stationarity
        is_stationary, p_value, _ = test_stationarity(time_series)
        
        # Determine differencing order
        d = 0 if is_stationary else 1
        
        # Split data into train and test sets (last 20% for test)
        train_size = int(len(time_series) * 0.8)
        train, test = time_series[:train_size], time_series[train_size:]
        
        # If test set is empty, use the last 20% of train data for testing
        if len(test) == 0:
            train_size = int(len(train) * 0.8)
            train, test = train[:train_size], train[train_size:]
        
        # Fit ARIMA model
        model = ARIMA(train, order=(5, d, 1))
        model_fit = model.fit()
        
        # Make predictions for test set
        predictions = model_fit.forecast(steps=len(test))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((test - predictions) / test)) * 100
        
        # Forecast for future days
        future_forecast = model_fit.forecast(steps=forecast_days)
        
        # Create DataFrame for forecast results
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': future_forecast
        })
        forecast_df = forecast_df.set_index('date')
        
        return forecast_df, mape, model_fit
    
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        return pd.DataFrame(), 0, None

def create_sequences(data, seq_length):
    """
    Create sequences for LSTM model
    
    Args:
        data (numpy.array): Input data
        seq_length (int): Sequence length
        
    Returns:
        tuple: (X, y) where X is sequence data and y is target
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
            
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def fit_lstm_model(df, target_column='price', forecast_days=7, epochs=50):
    """
    Fit LSTM model to time series data and make forecast using PyTorch
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data with datetime index
        target_column (str): Column name for the target variable
        forecast_days (int): Number of days to forecast
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (forecast_df, mape, model)
    """
    if not PYTORCH_AVAILABLE:
        st.error("PyTorch is not available. Cannot use LSTM model.")
        return pd.DataFrame(), 0, None
    
    try:
        # Check if df is empty
        if df.empty:
            st.error("No data available for modeling")
            return pd.DataFrame(), 0, None
        
        # Get the time series data
        time_series = df[target_column].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(time_series)
        
        # Define sequence length
        seq_length = 30
        
        # Create sequences for LSTM
        if len(scaled_data) <= seq_length:
            st.error(f"Not enough data points. Need at least {seq_length+1} points.")
            return pd.DataFrame(), 0, None
            
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split data into train and test sets (last 20% for test)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create DataLoader for batch processing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Build LSTM model
        input_dim = 1
        hidden_dim = 50
        num_layers = 2
        output_dim = 1
        dropout = 0.2
        
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train model
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Make predictions for test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
        
        # Convert predictions to numpy
        y_pred = y_pred.numpy()
        
        # Inverse transform predictions and actual values
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        # Forecast for future days
        forecast = []
        
        # Get the last sequence from the training data
        current_seq = torch.FloatTensor(scaled_data[-seq_length:]).view(1, seq_length, 1)
        
        model.eval()
        with torch.no_grad():
            for _ in range(forecast_days):
                # Get prediction (next value)
                pred = model(current_seq)
                # Append to forecast list
                forecast.append(pred.item())
                # Update sequence for next prediction
                current_seq = torch.cat([current_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)
        
        # Convert forecast to numpy and inverse transform
        forecast = np.array(forecast).reshape(-1, 1)
        forecast = scaler.inverse_transform(forecast)
        
        # Create DataFrame for forecast results
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast.flatten()
        })
        forecast_df = forecast_df.set_index('date')
        
        return forecast_df, mape, model
    
    except Exception as e:
        st.error(f"Error fitting LSTM model: {str(e)}")
        return pd.DataFrame(), 0, None

def predict_future_prices(df, model_type, target_column='price', forecast_days=7):
    """
    Predict future prices using specified model
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data with datetime index
        model_type (str): Type of model to use ('ARIMA' or 'LSTM')
        target_column (str): Column name for the target variable
        forecast_days (int): Number of days to forecast
        
    Returns:
        tuple: (forecast_df, model_accuracy, model_summary)
    """
    if model_type == 'ARIMA':
        forecast_df, model_accuracy, model = fit_arima_model(df, target_column, forecast_days)
        if model is not None:
            model_summary = model.summary()
        else:
            model_summary = "Model fitting failed"
    elif model_type == 'LSTM' and PYTORCH_AVAILABLE:
        forecast_df, model_accuracy, model = fit_lstm_model(df, target_column, forecast_days)
        if model is not None:
            model_summary = "LSTM model trained successfully with PyTorch"
        else:
            model_summary = "Model fitting failed"
    else:
        st.error(f"Unsupported model type: {model_type}")
        return pd.DataFrame(), 0, "Unsupported model type"
    
    return forecast_df, model_accuracy, model_summary