#!/usr/bin/env python3
"""
Investment Tracker

This module allows you to manually input the details of executed trades.
It updates the portfolio (account balance and open positions) and records
the trades in an SQLite database.

New Features:
  - Position Sizing: Automatically calculates the number of shares to buy
    based on a fixed risk per trade (e.g., 5% of current balance).
  - Max Open Trades Check: Prevents opening new trades if the number of open
    trades equals or exceeds the maximum allowed.
  - Each trade is recorded with a note column for additional comments.
"""

import os
import sqlite3
import time
import logging
from datetime import datetime

# -----------------------------
# Global Constants & Configuration
# -----------------------------
DB_PATH = os.path.join("data", "market_data.db")  # Ensure the data folder exists
INITIAL_BALANCE = 100000.0  # Starting account balance
MAX_OPEN_TRADES = 10        # Maximum number of open trades allowed
RISK_PER_TRADE = 0.05       # Risk 5% of current balance per trade

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Database Initialization
# -----------------------------
def initialize_database(db_path=DB_PATH):
    """
    Ensure that the trade_history table exists with the required columns.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_time TEXT,
                trade_type TEXT,
                executed_price REAL,
                shares REAL,
                cost REAL,
                note TEXT
            )
        """)
        conn.commit()
        logging.info("Database initialized.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def record_trade(trade_type, executed_price, shares, cost, note="", db_path=DB_PATH):
    """
    Record an executed trade in the trade_history table.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO trade_history (execution_time, trade_type, executed_price, shares, cost, note)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (execution_time, trade_type, executed_price, shares, cost, note))
        conn.commit()
        logging.info(f"Recorded trade: {trade_type} {shares} shares at ${executed_price:.2f} for ${cost:.2f}")
    except Exception as e:
        logging.error(f"Error recording trade: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def count_open_trades(db_path=DB_PATH):
    """
    Count the number of open trades recorded in the trade_history.
    For this simple tracker, we assume a BUY trade adds an open trade and a SELL clears all.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Here, we assume that trades without a corresponding SELL are considered "open".
        # For this demo, we will consider the open trades as those recorded in our InvestmentTracker.open_trades list.
        # (In a more advanced system, you would query the database for trades that have not been closed.)
        # For now, we simply return the count from our tracker.
        # This function can be expanded in a production system.
        return None  # Not used directly here
    except Exception as e:
        logging.error(f"Error counting open trades: {e}")
        return 0
    finally:
        if 'conn' in locals():
            conn.close()

# -----------------------------
# Investment Tracker Logic
# -----------------------------
class InvestmentTracker:
    def __init__(self, initial_balance=INITIAL_BALANCE):
        self.balance = initial_balance
        self.open_trades = []  # List to track open positions; each trade is a dict

    def display_portfolio(self):
        print(f"Current Account Balance: ${self.balance:.2f}")
        if self.open_trades:
            print("Open Trades:")
            for trade in self.open_trades:
                print(f"  Date: {trade['Date']} | Type: {trade['trade_type']} | Shares: {trade['shares']} | Entry Price: ${trade['executed_price']:.2f}")
        else:
            print("No open trades.")

    def process_trade(self):
        """
        Manually input trade details with position sizing and max open trades check.
        For BUY trades, the trade size is automatically calculated as RISK_PER_TRADE * current balance.
        """
        trade_type = input("Enter trade type (BUY/SELL): ").strip().upper()
        if trade_type not in ["BUY", "SELL"]:
            print("Invalid trade type. Must be BUY or SELL.")
            return

        # Check if we already have maximum open trades for BUY orders.
        if trade_type == "BUY" and len(self.open_trades) >= MAX_OPEN_TRADES:
            print(f"Maximum open trades reached ({MAX_OPEN_TRADES}). Cannot open new BUY trade.")
            return

        # For position sizing, for BUY trades, automatically determine the amount to invest:
        if trade_type == "BUY":
            # Calculate trade amount as a fixed percentage of the current balance
            trade_amount = self.balance * RISK_PER_TRADE
            print(f"Calculated trade amount based on {RISK_PER_TRADE*100:.1f}% of balance: ${trade_amount:.2f}")
            try:
                # Ask the user to confirm the executed price
                executed_price = float(input("Enter executed price: ").strip())
            except ValueError:
                print("Invalid price entered.")
                return
            # Calculate the number of shares automatically
            shares = trade_amount / executed_price
            cost = executed_price * shares
            note = input("Enter any note for this trade (optional): ").strip()
            execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Deduct cost from balance
            if cost > self.balance:
                print("Insufficient funds for this trade (should not happen with position sizing).")
                return
            self.balance -= cost
            # Record the open trade details
            trade_details = {
                "Date": execution_time,
                "trade_type": trade_type,
                "executed_price": executed_price,
                "shares": shares,
                "cost": cost,
                "note": note
            }
            self.open_trades.append(trade_details)
            # Record the trade in the database
            record_trade(trade_type, executed_price, shares, cost, note)
            print(f"Trade executed: {trade_type} {shares:.2f} shares at ${executed_price:.2f} costing ${cost:.2f}.")
            print(f"Updated balance: ${self.balance:.2f}")
        else:  # For SELL trades, user inputs the details manually
            try:
                executed_price = float(input("Enter executed price for SELL: ").strip())
                shares = float(input("Enter number of shares to SELL: ").strip())
            except ValueError:
                print("Invalid number entered.")
                return
            cost = executed_price * shares  # Revenue in this case
            note = input("Enter any note for this trade (optional): ").strip()
            execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add revenue to balance
            self.balance += cost
            # For simplicity, assume SELL clears all open trades
            self.open_trades = []
            # Record the trade in the database
            record_trade(trade_type, executed_price, shares, cost, note)
            print(f"Trade executed: {trade_type} {shares} shares at ${executed_price:.2f} generating ${cost:.2f}.")
            print(f"Updated balance: ${self.balance:.2f}")

def main():
    # Initialize database
    initialize_database(DB_PATH)

    tracker = InvestmentTracker()
    print("Welcome to the Investment Tracker.")
    tracker.display_portfolio()

    while True:
        print("\nOptions:")
        print("1. Input a new trade (with automatic position sizing for BUY)")
        print("2. Display portfolio")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            tracker.process_trade()
        elif choice == "2":
            tracker.display_portfolio()
        elif choice == "3":
            print("Exiting Investment Tracker.")
            break
        else:
            print("Invalid choice. Please try again.")
        # Add a short pause between operations
        time.sleep(1)

if __name__ == "__main__":
    main()