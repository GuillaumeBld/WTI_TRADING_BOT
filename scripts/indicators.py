import os
import sqlite3
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging for indicators
logging.basicConfig(
    filename="/Users/guillaumebolivard/Documents/School/Loyola_U/Classes/Capstone_MS_Finance/Trading_challenge/trading_bot/data/indicators_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(filepath):
    """
    Load price data from a CSV file using pandas.
    
    Args:
        filepath (str): Absolute path to the CSV file.
        
    Returns:
        DataFrame: Pandas DataFrame with the price data.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["Date"])
        logging.info(f"Loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def calculate_indicators(df):
    """
    Calculate technical indicators using pandas.
    
    Indicators calculated:
      - EMA (9 and 21)
      - RSI (14)
      - MACD, MACD Signal, MACD Histogram
      - ADX (14) using a simplified method
    
    Args:
        df (DataFrame): DataFrame containing at least 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
        
    Returns:
        DataFrame: Original DataFrame with added indicator columns.
    """
    # Ensure the data is sorted by Date
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Convert columns to numeric, coerce errors to NaN
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where 'Close' is NaN (or any key numeric field if desired)
    df.dropna(subset=['Close'], inplace=True)
    
    # EMA calculations using pandas ewm
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    
    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    period = 14
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD calculation
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # ADX calculation (simplified)
    df["prev_Close"] = df["Close"].shift(1)
    df["TR"] = np.maximum(df["High"] - df["Low"],
                          np.maximum((df["High"] - df["prev_Close"]).abs(), (df["Low"] - df["prev_Close"]).abs()))
    df["up_move"] = df["High"] - df["High"].shift(1)
    df["down_move"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["-DM"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
    
    period_adx = 14
    df["+DM_Smooth"] = df["+DM"].rolling(window=period_adx, min_periods=period_adx).sum()
    df["-DM_Smooth"] = df["-DM"].rolling(window=period_adx, min_periods=period_adx).sum()
    df["TR_Sum"] = df["TR"].rolling(window=period_adx, min_periods=period_adx).sum()
    df["+DI"] = 100 * df["+DM_Smooth"] / df["TR_Sum"]
    df["-DI"] = 100 * df["-DM_Smooth"] / df["TR_Sum"]
    df["DX"] = 100 * (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))
    df["ADX"] = df["DX"].rolling(window=period_adx, min_periods=period_adx).mean()
    
    # Clean up temporary columns
    df.drop(columns=["prev_Close", "up_move", "down_move", "+DM", "-DM", "+DM_Smooth", "-DM_Smooth", "TR_Sum", "DX"], inplace=True)
    
    return df

def save_indicators_to_db(df, db_path):
    """
    Store the DataFrame with indicators into an SQLite database.
    
    Args:
        df (DataFrame): DataFrame with computed indicators.
        db_path (str): Absolute path to the SQLite database.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql("market_data", conn, if_exists="replace", index=False)
        conn.close()
        logging.info(f"Indicators stored in SQLite database at {db_path} in table 'market_data'")
        return True
    except Exception as e:
        logging.error(f"Error saving indicators to SQLite: {e}")
        return False

def main():
    logging.info("Starting technical indicators calculation...")
    
    data_csv_path = "/Users/guillaumebolivard/Documents/School/Loyola_U/Classes/Capstone_MS_Finance/Trading_challenge/trading_bot/data/crude_oil_data.csv"
    db_path = "/Users/guillaumebolivard/Documents/School/Loyola_U/Classes/Capstone_MS_Finance/Trading_challenge/trading_bot/data/market_data.db"
    
    df = load_data(data_csv_path)
    if df is None:
        print("Failed to load price data. Please check the CSV file.")
        return
    
    logging.info("Calculating technical indicators using pandas...")
    df_with_indicators = calculate_indicators(df)
    
    # Optionally, save to CSV for verification
    output_csv = "/Users/guillaumebolivard/Documents/School/Loyola_U/Classes/Capstone_MS_Finance/Trading_challenge/trading_bot/data/crude_oil_with_indicators.csv"
    try:
        df_with_indicators.to_csv(output_csv, index=False)
        logging.info(f"Indicators saved to CSV at {output_csv}")
    except Exception as e:
        logging.error(f"Error saving indicators to CSV: {e}")
    
    # Save indicators directly to SQLite
    if save_indicators_to_db(df_with_indicators, db_path):
        print("Indicator calculation complete! Data stored in SQLite.")
    else:
        print("Indicator calculation complete, but failed to store data in SQLite.")
    
    logging.info("Technical indicators calculation complete!")

if __name__ == "__main__":
    main()