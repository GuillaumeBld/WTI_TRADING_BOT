#!/usr/bin/env python3
"""
Trade Execution Module - Enhanced Telegram-Based Version

This module loads trading signals from a SQLite database,
sends signals via Telegram with robust retry logic,
filters out duplicate or low-confidence signals,
logs trades (including failures) to SQLite,
and sends a daily summary message at 8:00 PM UTC if enabled.
"""

import os
import sqlite3
import argparse
import requests
import time
import json
from datetime import datetime, timedelta, timezone

# -----------------------------
# Global Constants
# -----------------------------
RETRY_COUNT = 3
RETRY_DELAY = 5  # seconds
DUPLICATE_WINDOW_HOURS = 1  # filter duplicates if a similar trade sent within 1 hour

# -----------------------------
# Database Initialization
# -----------------------------
def initialize_database(db_path="market_data.db"):
    """
    Ensure required tables exist. Creates the trading_signals and trade_history tables.
    Inserts sample data into trading_signals if table is empty.
    Also ensures that trade_history has the 'note' column.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create trading_signals table if not exists.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                Date TEXT,
                Price REAL,
                Signal INTEGER,
                Confidence REAL
            )
        """)
        # Create trade_history table if not exists.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_time TEXT,
                trade_type TEXT,
                executed_price REAL,
                message TEXT,
                status TEXT,
                note TEXT
            )
        """)
        conn.commit()
        
        # Check if trade_history has the 'note' column.
        cursor.execute("PRAGMA table_info(trade_history)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'note' not in columns:
            print("Altering trade_history table to add 'note' column...")
            cursor.execute("ALTER TABLE trade_history ADD COLUMN note TEXT")
            conn.commit()
        
        # Insert sample signals if trading_signals is empty.
        cursor.execute("SELECT COUNT(*) FROM trading_signals")
        count = cursor.fetchone()[0]
        if count == 0:
            print("No trading signals found. Inserting sample data...")
            sample_data = [
                ('2025-03-01', 70.50, 1, 0.85),
                ('2025-03-02', 71.00, -1, 0.80),
                ('2025-03-03', 69.75, 1, 0.90)
            ]
            cursor.executemany("INSERT INTO trading_signals (Date, Price, Signal, Confidence) VALUES (?, ?, ?, ?)", sample_data)
            conn.commit()
            print("Sample trading signals inserted.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Signal Loading from SQLite
# -----------------------------
def load_signals(db_path="market_data.db"):
    """
    Load trading signals from the SQLite database sorted in descending order by Date.
    """
    signals = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT Date, Price, Signal, Confidence FROM trading_signals ORDER BY Date DESC"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            signals.append({
                'Date': row[0],
                'Price': float(row[1]),
                'Signal': int(row[2]),
                'Confidence': float(row[3])
            })
        print(f"Loaded {len(signals)} trading signals from database '{db_path}'")
        return signals
    except Exception as e:
        print(f"Error loading signals from database: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Duplicate Check in Trade History
# -----------------------------
def is_duplicate_trade(signal, db_path="market_data.db"):
    """
    Check if a similar trade (same Signal and Price) was sent within the duplicate window.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        window_start = (datetime.now() - timedelta(hours=DUPLICATE_WINDOW_HOURS)).strftime('%Y-%m-%d %H:%M:%S')
        query = """
            SELECT COUNT(*) FROM trade_history 
            WHERE trade_type = ? AND executed_price = ? AND execution_time >= ?
        """
        trade_type = "BUY" if signal['Signal'] == 1 else "SELL" if signal['Signal'] == -1 else "HOLD"
        cursor.execute(query, (trade_type, signal['Price'], window_start))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        print(f"Error checking duplicate trades: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Telegram Message Sender with Retry
# -----------------------------
def send_to_telegram(message, simulate=True):
    """
    Send a message to Telegram with retry logic.
    
    Args:
        message (str): The message to send.
        simulate (bool): If True, simulate sending (print only); if False, send via Telegram API.
        
    Returns:
        tuple: (bool success, str note) where note may contain error info if failed.
    """
    if simulate:
        print("\n=== SIMULATED TELEGRAM MESSAGE ===")
        print(message)
        print("=================================\n")
        return (True, "Simulated message")
    else:
        # Validate Telegram credentials.
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            error_msg = "ERROR: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Set them in your environment variables."
            print(error_msg)
            # Fallback: Save message to a queue file for later execution.
            queue_message(message)
            return (False, error_msg)
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                print(f"Attempt {attempt} to send Telegram message...")
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    print("Telegram message sent successfully.")
                    return (True, "Message sent")
                else:
                    error_info = f"Status code: {response.status_code}, Response: {response.text}"
                    print(f"Failed attempt {attempt}: {error_info}")
            except Exception as e:
                print(f"Exception on attempt {attempt}: {e}")
            time.sleep(RETRY_DELAY)
        return (False, "All retry attempts failed.")

# -----------------------------
# Fallback Queue for Signals
# -----------------------------
def queue_message(message, queue_file="queued_signals.json"):
    """
    Append the message to a JSON queue file for later execution.
    """
    try:
        queued = []
        if os.path.exists(queue_file):
            with open(queue_file, 'r') as f:
                queued = json.load(f)
        queued.append({"timestamp": datetime.now().isoformat(), "message": message})
        with open(queue_file, 'w') as f:
            json.dump(queued, f, indent=4)
        print(f"Message queued in {queue_file}.")
    except Exception as e:
        print(f"Error queuing message: {e}")

# -----------------------------
# Trade Logging in SQLite
# -----------------------------
def log_trade(signal, message, status, note, db_path="market_data.db"):
    """
    Log trade details and Telegram status in the trade_history table.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_type = "BUY" if signal['Signal'] == 1 else "SELL" if signal['Signal'] == -1 else "HOLD"
        cursor.execute("""
            INSERT INTO trade_history (execution_time, trade_type, executed_price, message, status, note)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (execution_time, trade_type, signal['Price'], message, status, note))
        conn.commit()
        print("Trade logged in database.")
    except Exception as e:
        print(f"Error logging trade: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Daily Trade Summary Sender
# -----------------------------
def send_daily_summary(simulate, db_path="market_data.db"):
    """
    Compile and send a daily trade summary message.
    Summary includes: number of trades, most traded direction, win rate, total PnL (if available).
    For this demo, win rate and PnL are not computed and are shown as N/A.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        query = """
            SELECT trade_type FROM trade_history
            WHERE DATE(execution_time) = ?
        """
        cursor.execute(query, (today,))
        rows = cursor.fetchall()
        if not rows:
            summary_text = "No trades executed today."
        else:
            trade_types = [row[0] for row in rows]
            total_trades = len(trade_types)
            buy_count = trade_types.count("BUY")
            sell_count = trade_types.count("SELL")
            most_traded = "BUY" if buy_count >= sell_count else "SELL"
            summary_text = (f"📅 Daily Trade Summary for {today}\n"
                            f"Total Trades: {total_trades}\n"
                            f"Most Traded Direction: {most_traded}\n"
                            f"Win Rate: N/A\n"
                            f"Total PnL: N/A")
        # Send the summary via Telegram.
        success, note = send_to_telegram(summary_text, simulate=simulate)
        if success:
            print("Daily summary sent.")
        else:
            print("Failed to send daily summary:", note)
    except Exception as e:
        print(f"Error generating daily summary: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Signal Message Formatter
# -----------------------------
def format_signal_message(signal):
    """
    Format a trading signal into a human-readable Telegram message.
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
        message += f"🛑 Suggested Stop Loss: ${signal['Price'] * 0.98:.2f}\n"
        message += f"🎯 Suggested Take Profit: ${signal['Price'] * 1.05:.2f}"
    elif signal_type == "SELL":
        message += "⚠️ Action: Consider opening a SHORT position\n"
        message += f"🛑 Suggested Stop Loss: ${signal['Price'] * 1.02:.2f}\n"
        message += f"🎯 Suggested Take Profit: ${signal['Price'] * 0.95:.2f}"
    else:
        message += "⚠️ Action: No action recommended at this time"
    return message

# -----------------------------
# Main Execution Flow
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Execute trades by sending signals via Telegram')
    # Mode flags
    parser.add_argument('--live', action='store_true', help='Enable live Telegram messaging (actual message sending)')
    parser.add_argument('--paper', action='store_true', help='Enable paper mode (simulate Telegram messaging)')
    # New CLI options
    parser.add_argument('--confidence', type=float, default=50.0, help='Minimum confidence threshold (in percent) for sending trades')
    parser.add_argument('--skip-duplicate-filtering', action='store_true', help='Disable duplicate filtering')
    parser.add_argument('--daily-summary', action='store_true', help='Enable sending daily trade summary at 8:00 PM UTC')
    parser.add_argument('--db', type=str, default="market_data.db", help='Path to the SQLite database')
    args = parser.parse_args()

    # Determine mode: default to paper if neither is specified.
    if args.live and args.paper:
        print("Error: Cannot enable both live and paper modes. Please choose one.")
        return
    mode = "live" if args.live else "paper"
    simulate = True if mode == "paper" else False

    print("WTI Crude Oil Trading System - Trade Execution via Telegram")
    print("=============================================================")
    print(f"Mode: {mode.capitalize()}")
    print(f"Minimum Confidence Threshold: {args.confidence}%")
    print(f"Duplicate Filtering: {'Enabled' if not args.skip_duplicate_filtering else 'Disabled'}")
    print(f"Daily Summary: {'Enabled' if args.daily_summary else 'Disabled'}\n")

    # In live mode, ensure Telegram credentials are present.
    if mode == "live":
        if not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")):
            error_msg = "ERROR: Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. Set them in your environment variables."
            print(error_msg)
            print("Falling back to queuing signals for later execution.")
            signals = load_signals(args.db)
            for sig in signals:
                message = format_signal_message(sig)
                queue_message(message)
            return

    # Initialize database (create tables and sample data if needed).
    initialize_database(args.db)
    
    # Load signals from the database.
    signals = load_signals(args.db)
    if not signals:
        print("No trading signals found. Exiting.")
        return

    # Process the most recent signal.
    most_recent_signal = signals[0]

    # Filter based on confidence threshold.
    if (most_recent_signal['Confidence'] * 100) < args.confidence:
        note = f"Signal confidence {most_recent_signal['Confidence']*100:.1f}% is below threshold of {args.confidence}%."
        print("Skipping trade:", note)
        log_trade(most_recent_signal, "", "skipped", note, args.db)
        return

    # If duplicate filtering is enabled, check for duplicate trade.
    if not args.skip_duplicate_filtering:
        if is_duplicate_trade(most_recent_signal, args.db):
            note = "Duplicate signal detected within the filtering window. Trade skipped."
            print(note)
            log_trade(most_recent_signal, "", "skipped", note, args.db)
            return

    # Format the signal message.
    message = format_signal_message(most_recent_signal)
    print("Most Recent Signal:")
    print(f"Date: {most_recent_signal['Date']}, Price: ${most_recent_signal['Price']:.2f}, "
          f"Signal: {most_recent_signal['Signal']}, Confidence: {most_recent_signal['Confidence']:.2f}")
    print("\nFormatted Signal Message:\n")
    print(message)

    # Send the message via Telegram.
    success, note = send_to_telegram(message, simulate=simulate)
    status = "sent" if success else "failed"
    log_trade(most_recent_signal, message, status, note, args.db)

    # Daily Summary: If enabled and current UTC time is past 8:00 PM, send summary.
    now_utc = datetime.now(timezone.utc)
    if args.daily_summary and now_utc.hour >= 20:
        print("\nSending daily trade summary...")
        send_daily_summary(simulate, args.db)

    print("\nTrade execution complete!")

if __name__ == "__main__":
    main()