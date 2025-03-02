"""
Trade Execution Module

This script generates trade signals and sends them to Telegram.
"""

import os
import time
import json
import logging
import pandas as pd
from datetime import datetime
import telepot
from data_fetch import fetch_historical_data
from strategy import TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/execute.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TelegramTradeExecutor:
    """
    Class to generate trade signals and send them to Telegram
    """
    
    def __init__(self, token=None, chat_id=None, config_path='../config/telegram_config.json'):
        """
        Initialize the trade executor
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
            config_path (str): Path to the Telegram configuration file
        """
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        
        # Create logs directory if it doesn't exist
        os.makedirs("../logs", exist_ok=True)
        
        # Try to load configuration from file if not provided
        if (not token or not chat_id) and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.token = config.get('token', token)
                    self.chat_id = config.get('chat_id', chat_id)
                    logger.info("Loaded Telegram configuration from file")
            except Exception as e:
                logger.error(f"Error loading Telegram configuration: {e}")
        
        # Initialize Telegram bot if token is available
        if self.token:
            try:
                self.bot = telepot.Bot(self.token)
                logger.info("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Error initializing Telegram bot: {e}")
                self.bot = None
    
    def save_config(self, config_path='../config/telegram_config.json'):
        """
        Save Telegram configuration to file
        
        Args:
            config_path (str): Path to save the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.token or not self.chat_id:
            logger.error("Token or chat ID not available")
            return False
        
        config = {
            'token': self.token,
            'chat_id': self.chat_id
        }
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def send_message(self, message):
        """
        Send a message to Telegram
        
        Args:
            message (str): The message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.bot or not self.chat_id:
            logger.error("Telegram bot or chat ID not available")
            return False
        
        try:
            self.bot.sendMessage(self.chat_id, message)
            logger.info(f"Message sent to Telegram: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {e}")
            return False
    
    def generate_trade_signal(self, strategy, data):
        """
        Generate a trade signal based on the strategy
        
        Args:
            strategy (TradingStrategy): The trading strategy
            data (pandas.DataFrame): The price data
            
        Returns:
            dict: The trade signal
        """
        # Generate signals
        signals = strategy.generate_signals(data)
        
        if signals is None or signals.empty:
            logger.error("Failed to generate signals")
            return None
        
        # Get the latest signal
        latest_signal = signals.iloc[-1]
        
        # Create trade signal
        trade_signal = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'CL=F',  # WTI Crude Oil Futures
            'price': data['Close'].iloc[-1],
            'signal': 'BUY' if latest_signal['Signal'] > 0 else 'SELL',
            'confidence': float(latest_signal['Prediction']),
            'indicators': {
                'RSI': data['RSI'].iloc[-1] if 'RSI' in data else None,
                'MACD': data['MACD'].iloc[-1] if 'MACD' in data else None,
                'ADX': data['ADX'].iloc[-1] if 'ADX' in data else None
            }
        }
        
        return trade_signal
    
    def format_trade_message(self, trade_signal):
        """
        Format a trade signal as a Telegram message
        
        Args:
            trade_signal (dict): The trade signal
            
        Returns:
            str: The formatted message
        """
        message = f"🚨 *TRADE SIGNAL ALERT* 🚨\n\n"
        message += f"*Symbol:* {trade_signal['symbol']} (WTI Crude Oil Futures)\n"
        message += f"*Action:* {trade_signal['signal']}\n"
        message += f"*Price:* ${trade_signal['price']:.2f}\n"
        message += f"*Confidence:* {trade_signal['confidence']:.2f}\n"
        message += f"*Time:* {trade_signal['timestamp']}\n\n"
        
        message += "*Technical Indicators:*\n"
        
        # Add indicators if available
        indicators = trade_signal['indicators']
        if indicators['RSI'] is not None:
            message += f"RSI: {indicators['RSI']:.2f}\n"
        if indicators['MACD'] is not None:
            message += f"MACD: {indicators['MACD']:.2f}\n"
        if indicators['ADX'] is not None:
            message += f"ADX: {indicators['ADX']:.2f}\n"
        
        message += "\n*Risk Management:*\n"
        
        # Add risk management suggestions
        if trade_signal['signal'] == 'BUY':
            stop_loss = trade_signal['price'] * 0.98  # 2% below current price
            take_profit = trade_signal['price'] * 1.05  # 5% above current price
            message += f"Stop Loss: ${stop_loss:.2f}\n"
            message += f"Take Profit: ${take_profit:.2f}\n"
        else:  # SELL
            stop_loss = trade_signal['price'] * 1.02  # 2% above current price
            take_profit = trade_signal['price'] * 0.95  # 5% below current price
            message += f"Stop Loss: ${stop_loss:.2f}\n"
            message += f"Take Profit: ${take_profit:.2f}\n"
        
        message += "\n*Disclaimer:*\n"
        message += "This is an automated trading signal. Always conduct your own analysis before making trading decisions."
        
        return message
    
    def execute_trade(self, strategy, data):
        """
        Generate a trade signal and send it to Telegram
        
        Args:
            strategy (TradingStrategy): The trading strategy
            data (pandas.DataFrame): The price data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate trade signal
        trade_signal = self.generate_trade_signal(strategy, data)
        
        if trade_signal is None:
            return False
        
        # Format message
        message = self.format_trade_message(trade_signal)
        
        # Send message
        success = self.send_message(message)
        
        # Log trade signal
        log_path = f"../logs/trades_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            # Load existing trades
            trades = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    trades = json.load(f)
            
            # Add new trade
            trades.append(trade_signal)
            
            # Save trades
            with open(log_path, 'w') as f:
                json.dump(trades, f, indent=4)
            
            logger.info(f"Trade signal logged to {log_path}")
        except Exception as e:
            logger.error(f"Error logging trade signal: {e}")
        
        return success

def main():
    """
    Main function to execute trades
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute trades and send signals to Telegram')
    parser.add_argument('--telegram', action='store_true', help='Send signals to Telegram')
    parser.add_argument('--token', type=str, help='Telegram bot token')
    parser.add_argument('--chat_id', type=str, help='Telegram chat ID')
    parser.add_argument('--interval', type=int, default=3600, help='Interval between trades in seconds')
    parser.add_argument('--once', action='store_true', help='Execute once and exit')
    args = parser.parse_args()
    
    # Create executor
    executor = TelegramTradeExecutor(token=args.token, chat_id=args.chat_id)
    
    # Save configuration if provided
    if args.token and args.chat_id:
        executor.save_config()
    
    # Create strategy
    strategy = TradingStrategy()
    strategy.load_model()
    
    if args.once:
        # Execute once
        logger.info("Executing trade once")
        data = fetch_historical_data()
        
        if data is not None:
            if args.telegram:
                executor.execute_trade(strategy, data)
            else:
                trade_signal = executor.generate_trade_signal(strategy, data)
                if trade_signal:
                    logger.info(f"Trade signal: {trade_signal}")
    else:
        # Execute periodically
        logger.info(f"Executing trades every {args.interval} seconds")
        
        while True:
            try:
                data = fetch_historical_data()
                
                if data is not None:
                    if args.telegram:
                        executor.execute_trade(strategy, data)
                    else:
                        trade_signal = executor.generate_trade_signal(strategy, data)
                        if trade_signal:
                            logger.info(f"Trade signal: {trade_signal}")
                
                logger.info(f"Sleeping for {args.interval} seconds")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Execution interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                time.sleep(60)  # Sleep for a minute before retrying

if __name__ == "__main__":
    main()
