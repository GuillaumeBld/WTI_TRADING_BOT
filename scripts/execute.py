"""
Trade Execution Module - Demo Version

This is a simplified version of the trade execution module that sends
trading signals to a simulated Telegram bot.
"""

import os
import csv
import json
from datetime import datetime

def load_signals(filepath):
    """
    Load trading signals from a CSV file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries with the signals
    """
    signals = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                processed_row = {
                    'Date': row['Date'],
                    'Price': float(row['Price']),
                    'Signal': int(float(row['Signal'])),  # Convert to int
                    'Confidence': float(row['Confidence'])
                }
                signals.append(processed_row)
        print(f"Loaded {len(signals)} trading signals from {filepath}")
        return signals
    except Exception as e:
        print(f"Error loading signals from {filepath}: {e}")
        return []

def get_latest_signals(signals, count=5):
    """
    Get the latest trading signals
    
    Args:
        signals (list): List of dictionaries with trading signals
        count (int): Number of latest signals to return
        
    Returns:
        list: List of latest signals
    """
    if not signals:
        return []
    
    return signals[-count:]

def format_signal_message(signal):
    """
    Format a signal as a Telegram message
    
    Args:
        signal (dict): Signal to format
        
    Returns:
        str: Formatted message
    """
    signal_type = "BUY" if signal['Signal'] == 1 else "SELL" if signal['Signal'] == -1 else "HOLD"
    confidence = signal['Confidence'] * 100
    
    message = "🚨 WTI CRUDE OIL TRADING SIGNAL 🚨\n\n"
    message += f"📅 Date: {signal['Date']}\n"
    message += f"💰 Price: ${signal['Price']:.2f}\n"
    message += f"🔍 Signal: {signal_type}\n"
    message += f"📊 Confidence: {confidence:.1f}%\n\n"
    
    if signal_type == "BUY":
        message += "⚠️ Action: Consider opening a LONG position\n"
        message += f"🛑 Suggested Stop Loss: ${signal['Price'] * 0.98:.2f} (2% below entry)\n"
        message += f"🎯 Suggested Take Profit: ${signal['Price'] * 1.05:.2f} (5% above entry)"
    elif signal_type == "SELL":
        message += "⚠️ Action: Consider opening a SHORT position\n"
        message += f"🛑 Suggested Stop Loss: ${signal['Price'] * 1.02:.2f} (2% above entry)\n"
        message += f"🎯 Suggested Take Profit: ${signal['Price'] * 0.95:.2f} (5% below entry)"
    else:
        message += "⚠️ Action: No action recommended at this time"
    
    return message

def send_to_telegram(message, simulate=True):
    """
    Send a message to Telegram (simulated)
    
    Args:
        message (str): Message to send
        simulate (bool): Whether to simulate sending
        
    Returns:
        bool: True if successful, False otherwise
    """
    if simulate:
        print("\n=== SIMULATED TELEGRAM MESSAGE ===")
        print(message)
        print("=================================\n")
        return True
    else:
        # In a real implementation, this would use the Telegram API
        print("Real Telegram sending not implemented")
        return False

def save_signal_to_log(signal, log_dir="../logs"):
    """
    Save a signal to the log
    
    Args:
        signal (dict): Signal to save
        log_dir (str): Directory to save logs
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file path
        log_file = f"{log_dir}/trade_signals_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing logs if file exists
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        # Add new signal with timestamp
        signal_with_timestamp = signal.copy()
        signal_with_timestamp['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logs.append(signal_with_timestamp)
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        print(f"Signal saved to log: {log_file}")
        return True
    except Exception as e:
        print(f"Error saving signal to log: {e}")
        return False

def main():
    """
    Main function to execute trades
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute trades and send signals to Telegram')
    parser.add_argument('--signals', type=str, default='../data/trading_signals_ml.csv',
                        help='Path to the signals CSV file')
    parser.add_argument('--telegram', action='store_true',
                        help='Send signals to Telegram (simulated)')
    args = parser.parse_args()
    
    print("WTI Crude Oil Trading System - Trade Execution")
    print("==============================================")
    
    # Load signals
    signals = load_signals(args.signals)
    
    if not signals:
        print(f"No signals found in {args.signals}")
        return
    
    # Get latest signals
    latest_signals = get_latest_signals(signals, count=5)
    
    print(f"\nLatest {len(latest_signals)} trading signals:")
    for signal in latest_signals:
        signal_type = "BUY" if signal['Signal'] == 1 else "SELL" if signal['Signal'] == -1 else "HOLD"
        print(f"Date: {signal['Date']}, Price: ${signal['Price']:.2f}, Signal: {signal_type}, Confidence: {signal['Confidence']:.2f}")
    
    # Get the most recent signal
    most_recent_signal = latest_signals[-1]
    
    # Format as Telegram message
    message = format_signal_message(most_recent_signal)
    
    # Send to Telegram (simulated)
    if args.telegram:
        print("\nSending latest signal to Telegram...")
        send_to_telegram(message)
        
        # Save to log
        save_signal_to_log(most_recent_signal)
    
    print("\nTrade execution complete!")
    print("In a real-world scenario, these signals would be sent to Telegram for traders to act upon.")

if __name__ == "__main__":
    main()
