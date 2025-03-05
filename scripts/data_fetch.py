import os
import sqlite3
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_data_directory():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bot_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(bot_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def fetch_market_data(days=365, symbol="CL=F"):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=days)
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty:
                df = df.iloc[:-1]
                df.reset_index(inplace=True)
                print("Market data successfully fetched.")
                return df
            print(f"Attempt {attempt + 1}/{retry_attempts}: No data returned. Retrying...")
            time.sleep(3)
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retry_attempts}: Error fetching market data: {e}")
            time.sleep(3)
    
    print("No data could be fetched after multiple attempts.")
    return None

def save_data_to_csv(df, filename="crude_oil_data.csv"):
    data_dir = get_data_directory()
    filepath = os.path.join(data_dir, filename)
    try:
        df.to_csv(filepath, index=False)
        print(f"Data saved to CSV at {filepath}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

def save_data_to_sqlite(df, db_name="market_data.db", table_name="market_data"):
    data_dir = get_data_directory()
    db_path = os.path.join(data_dir, db_name)
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    Date TEXT PRIMARY KEY,
                    Open REAL,
                    High REAL,
                    Low REAL,
                    Close REAL,
                    Volume INTEGER,
                    Sentiment_Score REAL DEFAULT NULL
                )
            ''')
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Data saved to SQLite database at {db_path} in table '{table_name}'")
    except Exception as e:
        print(f"Error saving data to SQLite: {e}")

def main():
    print("WTI Crude Oil Trading Bot - Market Data Fetching")
    print("==================================================")
    
    df = fetch_market_data(days=365, symbol="CL=F")
    if df is None:
        print("Market data could not be fetched. Exiting.")
        return
    
    save_data_to_csv(df)
    save_data_to_sqlite(df)
    print("Data fetching and storage complete!")

if __name__ == "__main__":
    main()
