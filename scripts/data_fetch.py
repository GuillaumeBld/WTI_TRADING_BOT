"""
Data Fetching Module - Demo Version

This is a simplified version of the data fetching module that generates
sample data instead of fetching from external APIs.
"""

import os
import random
from datetime import datetime, timedelta

def generate_sample_data(days=365):
    """
    Generate sample price data for demonstration
    
    Args:
        days (int): Number of days of data to generate
        
    Returns:
        list: List of dictionaries with sample data
    """
    print(f"Generating {days} days of sample data for WTI Crude Oil Futures...")
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    start_price = 75.0  # Starting price around $75
    
    current_price = start_price
    for i in range(days):
        date = start_date + timedelta(days=i)
        
        # Generate random price movement (-2% to +2%)
        price_change = current_price * (random.uniform(-0.02, 0.02))
        current_price += price_change
        
        # Generate OHLC data
        open_price = current_price
        high_price = open_price * (1 + random.uniform(0, 0.015))
        low_price = open_price * (1 - random.uniform(0, 0.015))
        close_price = open_price + random.uniform(low_price - open_price, high_price - open_price)
        
        # Generate volume
        volume = random.randint(100000, 500000)
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    print(f"Successfully generated {len(data)} records of sample data")
    return data

def save_data(data, filename='crude_oil_data.csv', directory='../data'):
    """
    Save the generated data to a CSV file
    
    Args:
        data (list): The data to save
        filename (str): The name of the file to save the data to
        directory (str): The directory to save the file in
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not data:
        print("No data to save")
        return False
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save data to CSV
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, 'w') as f:
            # Write header
            f.write("Date,Open,High,Low,Close,Volume\n")
            
            # Write data
            for row in data:
                f.write(f"{row['Date']},{row['Open']},{row['High']},{row['Low']},{row['Close']},{row['Volume']}\n")
        
        print(f"Data saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def setup_database():
    """
    Set up a SQLite database for storing market data (demo version)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("Database setup placeholder - would create SQLite database")
    return True

def main():
    """
    Main function to generate and save sample data
    """
    print("WTI Crude Oil Trading System - Data Generation")
    print("=============================================")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Save data
    if data:
        save_data(data)
    
    # Setup database
    setup_database()
    
    print("\nData generation complete!")
    print("You can now proceed with the other components of the trading system.")

if __name__ == "__main__":
    main()
