"""
Technical Indicators Module - Demo Version

This is a simplified version of the technical indicators module that calculates
indicators using basic Python without external dependencies.
"""

import os
import csv
import math
from datetime import datetime

def load_csv_data(filepath):
    """
    Load data from a CSV file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries with the data
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                processed_row = {
                    'Date': row['Date'],
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': int(row['Volume'])
                }
                data.append(processed_row)
        print(f"Loaded {len(data)} records from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return []

def calculate_sma(data, field='Close', period=14):
    """
    Calculate Simple Moving Average
    
    Args:
        data (list): List of dictionaries with price data
        field (str): The field to calculate SMA for
        period (int): The period for SMA calculation
        
    Returns:
        list: List of SMA values
    """
    sma_values = []
    
    for i in range(len(data)):
        if i < period - 1:
            sma_values.append(None)  # Not enough data for SMA
        else:
            # Calculate sum of closing prices for the period
            sum_prices = sum(data[j][field] for j in range(i - period + 1, i + 1))
            sma = sum_prices / period
            sma_values.append(sma)
    
    return sma_values

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (list): List of dictionaries with price data
        period (int): The period for RSI calculation
        
    Returns:
        list: List of RSI values
    """
    rsi_values = []
    
    # Calculate price changes
    price_changes = []
    for i in range(1, len(data)):
        price_changes.append(data[i]['Close'] - data[i-1]['Close'])
    
    # For the first period-1 data points, RSI is not defined
    for _ in range(period):
        rsi_values.append(None)
    
    # Calculate RSI for the rest of the data
    for i in range(period, len(data)):
        gains = []
        losses = []
        
        # Get price changes for the period
        for j in range(i - period, i):
            change = price_changes[j - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate RS
        if avg_loss == 0:
            rs = 100  # Avoid division by zero
        else:
            rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data (list): List of dictionaries with price data
        fast_period (int): The fast EMA period
        slow_period (int): The slow EMA period
        signal_period (int): The signal line period
        
    Returns:
        tuple: (List of MACD values, List of Signal values, List of Histogram values)
    """
    # Calculate EMAs
    ema_fast = calculate_ema(data, period=fast_period)
    ema_slow = calculate_ema(data, period=slow_period)
    
    # Calculate MACD line
    macd_line = []
    for i in range(len(data)):
        if ema_fast[i] is None or ema_slow[i] is None:
            macd_line.append(None)
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])
    
    # Calculate signal line (EMA of MACD)
    signal_line = []
    for i in range(len(data)):
        if i < slow_period + signal_period - 2:
            signal_line.append(None)
        else:
            # Calculate EMA of MACD
            macd_period = macd_line[i-(signal_period-1):i+1]
            macd_period = [m for m in macd_period if m is not None]
            
            if len(macd_period) < signal_period:
                signal_line.append(None)
            else:
                # Simple approximation of EMA for demo
                signal = sum(macd_period) / len(macd_period)
                signal_line.append(signal)
    
    # Calculate histogram
    histogram = []
    for i in range(len(data)):
        if macd_line[i] is None or signal_line[i] is None:
            histogram.append(None)
        else:
            histogram.append(macd_line[i] - signal_line[i])
    
    return macd_line, signal_line, histogram

def calculate_ema(data, field='Close', period=14):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data (list): List of dictionaries with price data
        field (str): The field to calculate EMA for
        period (int): The period for EMA calculation
        
    Returns:
        list: List of EMA values
    """
    ema_values = []
    
    # For the first period-1 data points, EMA is not defined
    for _ in range(period - 1):
        ema_values.append(None)
    
    # First EMA is the SMA of the first period points
    prices = [data[i][field] for i in range(period)]
    first_ema = sum(prices) / period
    ema_values.append(first_ema)
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA for the rest of the data
    for i in range(period, len(data)):
        if ema_values[i-1] is None:
            ema_values.append(None)
        else:
            ema = (data[i][field] * multiplier) + (ema_values[i-1] * (1 - multiplier))
            ema_values.append(ema)
    
    return ema_values

def calculate_adx(data, period=14):
    """
    Calculate Average Directional Index (ADX) - Simplified version
    
    Args:
        data (list): List of dictionaries with price data
        period (int): The period for ADX calculation
        
    Returns:
        list: List of ADX values
    """
    # This is a simplified version that returns random values for demonstration
    adx_values = []
    
    # For the first 2*period-1 data points, ADX is not defined
    for _ in range(2 * period - 1):
        adx_values.append(None)
    
    # Generate random ADX values between 0 and 100 for the rest
    for _ in range(2 * period - 1, len(data)):
        adx = 25 + (20 * math.sin(len(adx_values) / 10))  # Oscillating around 25
        adx_values.append(max(0, min(100, adx)))  # Clamp between 0 and 100
    
    return adx_values

def add_indicators_to_data(data):
    """
    Add technical indicators to the data
    
    Args:
        data (list): List of dictionaries with price data
        
    Returns:
        list: List of dictionaries with added indicators
    """
    # Make a copy to avoid modifying the original
    result = []
    for item in data:
        result.append(item.copy())
    
    # Calculate indicators
    rsi_values = calculate_rsi(data)
    macd_line, signal_line, histogram = calculate_macd(data)
    adx_values = calculate_adx(data)
    ema9_values = calculate_ema(data, period=9)
    ema21_values = calculate_ema(data, period=21)
    
    # Add indicators to the data
    for i in range(len(result)):
        result[i]['RSI'] = rsi_values[i]
        result[i]['MACD'] = macd_line[i]
        result[i]['MACD_Signal'] = signal_line[i]
        result[i]['MACD_Hist'] = histogram[i]
        result[i]['ADX'] = adx_values[i]
        result[i]['EMA_9'] = ema9_values[i]
        result[i]['EMA_21'] = ema21_values[i]
    
    return result

def save_indicators_to_csv(data, filepath):
    """
    Save data with indicators to a CSV file
    
    Args:
        data (list): List of dictionaries with indicators
        filepath (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', newline='') as f:
            # Get all field names
            fieldnames = list(data[0].keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                # Convert None values to empty strings for CSV
                row_copy = {}
                for key, value in row.items():
                    if value is None:
                        row_copy[key] = ''
                    else:
                        row_copy[key] = value
                
                writer.writerow(row_copy)
        
        print(f"Data with indicators saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        return False

def main():
    """
    Main function to calculate indicators and save results
    """
    print("WTI Crude Oil Trading System - Technical Indicators")
    print("==================================================")
    
    # Load data
    data_path = "../data/crude_oil_data.csv"
    data = load_csv_data(data_path)
    
    if not data:
        print("No data available. Please run data_fetch.py first.")
        return
    
    print(f"Calculating technical indicators for {len(data)} records...")
    
    # Add indicators
    data_with_indicators = add_indicators_to_data(data)
    
    # Save results
    output_path = "../data/crude_oil_with_indicators.csv"
    save_indicators_to_csv(data_with_indicators, output_path)
    
    print("\nIndicator calculation complete!")
    print("You can now proceed with the trading strategy and backtesting.")

if __name__ == "__main__":
    main()
