"""
Trade Execution Module

This module handles the execution of trades based on signals from the strategy module.
It includes risk management, position sizing, and trade logging functionality.
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize the trade history database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        trade_type TEXT,
                        quantity INTEGER,
                        price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        status TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def calculate_position_size(self, signal: Dict, account_balance: float) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Dictionary containing trade signal details
            account_balance: Current account balance
            
        Returns:
            int: Number of shares/contracts to trade
        """
        risk_per_trade = 0.01  # Risk 1% of account per trade
        stop_loss_pct = 0.02   # 2% stop loss
        
        risk_amount = account_balance * risk_per_trade
        price = signal['price']
        stop_loss_price = price * (1 - stop_loss_pct)
        risk_per_share = price - stop_loss_price
        
        if risk_per_share <= 0:
            return 0
            
        position_size = math.floor(risk_amount / risk_per_share)
        return max(1, position_size)  # Minimum 1 share/contract

    def execute_trade(self, signal: Dict, account_balance: float) -> bool:
        """
        Execute a trade based on the signal and account balance.
        
        Args:
            signal: Dictionary containing trade signal details
            account_balance: Current account balance
            
        Returns:
            bool: True if trade was executed successfully, False otherwise
        """
        try:
            # Calculate position size
            quantity = self.calculate_position_size(signal, account_balance)
            if quantity <= 0:
                logger.error("Invalid position size calculated")
                return False
                
            # Calculate stop loss and take profit levels
            price = signal['price']
            stop_loss = price * 0.98  # 2% stop loss
            take_profit = price * 1.05  # 5% take profit
            
            # Log the trade
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_history 
                    (timestamp, trade_type, quantity, price, stop_loss, take_profit, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    signal['type'],
                    quantity,
                    price,
                    stop_loss,
                    take_profit,
                    'executed'
                ))
                conn.commit()
                
            logger.info(f"Executed {signal['type']} trade: {quantity} shares at ${price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve trade history from the database.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of dictionaries containing trade details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM trade_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving trade history: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    executor = TradeExecutor()
    
    # Example signal
    signal = {
        'type': 'buy',
        'price': 70.50,
        'confidence': 0.85
    }
    
    # Execute trade with $100,000 account balance
    success = executor.execute_trade(signal, 100000.0)
    if success:
        print("Trade executed successfully")
    else:
        print("Trade execution failed")
