"""
Technical Indicators Module

This script calculates various technical indicators used for trading decisions.
"""

import numpy as np
import pandas as pd

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
        window (int): The window size for RSI calculation
        
    Returns:
        pandas.Series: RSI values
    """
    # Calculate price changes
    delta = data['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
        fast_period (int): The fast EMA period
        slow_period (int): The slow EMA period
        signal_period (int): The signal line period
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
        period (int): The period for ADX calculation
        
    Returns:
        pandas.Series: ADX values
    """
    # Calculate True Range
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Calculate +DM and -DM
    plus_dm = data['High'].diff()
    minus_dm = data['Low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
    minus_dm = minus_dm.abs().where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0)
    
    # Calculate smoothed TR, +DM, and -DM
    tr_smoothed = true_range.rolling(window=period).sum()
    plus_dm_smoothed = plus_dm.rolling(window=period).sum()
    minus_dm_smoothed = minus_dm.rolling(window=period).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
    minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_ema(data, periods=[9, 21, 50, 200]):
    """
    Calculate Exponential Moving Averages (EMA)
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
        periods (list): List of periods for EMA calculation
        
    Returns:
        dict: Dictionary with EMAs for each period
    """
    emas = {}
    for period in periods:
        emas[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    
    return emas

def add_all_indicators(data):
    """
    Add all technical indicators to the data
    
    Args:
        data (pandas.DataFrame): DataFrame with price data
        
    Returns:
        pandas.DataFrame: DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Add RSI
    df['RSI'] = calculate_rsi(df)
    
    # Add MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)
    
    # Add ADX
    df['ADX'] = calculate_adx(df)
    
    # Add EMAs
    emas = calculate_ema(df)
    for key, value in emas.items():
        df[key] = value
    
    return df

def main():
    """
    Main function to test indicator calculations
    """
    # This is a placeholder for testing
    print("This script calculates technical indicators for trading decisions.")
    print("Import this module in other scripts to use the indicator functions.")

if __name__ == "__main__":
    main()
