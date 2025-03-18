#!/usr/bin/env python
"""
TradingAgent: Adaptive Trade Suggestion System

This agent uses technical analysis, sentiment analysis, and AI-driven suggestions.
It wakes up every configurable interval (set in config.json) and checks for market-triggered events.
Trades are suggested but not executed—logged for manual confirmation via investment_tracker.py.
"""

from smolagents import CodeAgent
import os
import json
import time
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import pipeline
import requests
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration from config.json
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

TRADING_CHECK_INTERVAL = int(config.get("trading_check_interval_seconds", 3600))
COOLDOWN_SECONDS = int(config.get("alert_cooldown_seconds", 3600))
BATCH_WINDOW_SECONDS = int(config.get("alert_batch_window_seconds", 5))

# Environment variables (check .env parsing)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    logging.warning("python-dotenv not installed. Ensure environment variables are set manually.")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Validate environment variables
if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NEWSAPI_KEY]):
    logging.error("Missing required environment variables (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, or NEWSAPI_KEY).")

# Initialize Sentiment Model
sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

MARKET_DB = "data/market_data.db"

def initialize_database():
    """Initialize the database with required tables."""
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
            logging.info("Created 'data' directory.")

        conn = sqlite3.connect(MARKET_DB)
        cursor = conn.cursor()

        # Create account table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                balance REAL DEFAULT 100000.0
            )
        """)
        cursor.execute("INSERT OR IGNORE INTO account (id, balance) VALUES (1, 100000.0)")

        # Create trades table with telegram_message_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                size REAL,
                confidence REAL,
                timestamp TEXT,
                confirmed INTEGER DEFAULT 0,
                telegram_message_id INTEGER
            )
        """)

        # Create trade_history table
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
        logging.info("Database tables initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def check_duplicate_alert(alert_type, alert_message):
    """Check for duplicate alerts within cooldown."""
    conn = sqlite3.connect("alerts_log.db")
    cursor = conn.cursor()
    cooldown_time = (datetime.now() - timedelta(seconds=COOLDOWN_SECONDS)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("SELECT COUNT(*) FROM alerts_log WHERE alert_type = ? AND alert_message = ? AND sent_time >= ?",
                   (alert_type, alert_message, cooldown_time))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def record_alert(alert_type, alert_message):
    """Record an alert with timestamp."""
    conn = sqlite3.connect("alerts_log.db")
    cursor = conn.cursor()
    sent_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO alerts_log (alert_type, alert_message, sent_time) VALUES (?, ?, ?)",
                   (alert_type, alert_message, sent_time))
    conn.commit()
    conn.close()

# Fallback Rule-Based Model
class RuleBasedModel:
    """Fallback model using simple rules with enhanced logic."""
    def __call__(self, features):
        sentiment_score = features.get("sentiment_score", 0)
        # Add market trend check (e.g., 5-day moving average from market_data)
        market_data = features.get("market_data", pd.DataFrame())
        if not market_data.empty:
            ma5 = market_data["Close"].rolling(window=5).mean().iloc[-1]
            current_price = market_data["Close"].iloc[-1]
            price_trend = 1 if current_price > ma5 else -1 if current_price < ma5 else 0
        else:
            price_trend = 0

        # Combine sentiment and price trend
        combined_score = sentiment_score + (price_trend * 0.2)  # Weight price trend less than sentiment
        if combined_score > 0.2:  # Lowered threshold for more signals
            return "BUY"
        elif combined_score < -0.2:  # Lowered threshold for more signals
            return "SELL"
        else:
            return "HOLD"

# Real Trading Model
import joblib

class RealTradingModel:
    def __init__(self, model_path="models/trading_model.pkl"):
        self.model_path = model_path
        self.model = None
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logging.info("Real trading model loaded successfully.")
            else:
                logging.warning(f"Model file {model_path} not found. Using rule-based fallback. Creating models directory if needed.")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
        except Exception as e:
            logging.error(f"Error loading trading model: {e}. Using rule-based fallback.")

    def __call__(self, features):
        if self.model:
            # Prepare features for the model (ensure they match training data)
            feature_vector = pd.DataFrame([{
                "sentiment_score": features.get("sentiment_score", 0),
                "close_price": features.get("market_data", pd.DataFrame())["Close"].iloc[-1] if not features.get("market_data", pd.DataFrame()).empty else 0,
                "volume": features.get("market_data", pd.DataFrame())["Volume"].iloc[-1] if not features.get("market_data", pd.DataFrame()).empty else 0
            }])
            prediction = self.model.predict(feature_vector)[0]
            # Map prediction back to BUY/SELL/HOLD
            signal_map = {1: "BUY", 0: "SELL", -1: "HOLD"}
            return signal_map[prediction]
        else:
            return RuleBasedModel()(features)

class TradingAgent(CodeAgent):
    def __init__(self):
        # Load real model with fallback
        real_model = RealTradingModel(model_path="models/trading_model.pkl")
        try:
            super().__init__(name="TradingAgent", model=real_model, tools=[])
        except TypeError as e:
            logging.warning(f"Error initializing with tools: {e}. Initializing without tools.")
            super().__init__(name="TradingAgent", model=real_model)
        self.knowledge_base = self.load_knowledge_base()
        initialize_database()  # Call the function here
        self.alert_buffer = []
        self.last_batch_flush = datetime.now()
        self.max_open_trades = config.get("max_open_trades", 5)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        self.last_update_id = -1  # Initialize Telegram update ID tracking
        self.last_message_id = None  # Initialize last message ID tracking

    def load_knowledge_base(self):
        """Load static knowledge about WTI crude oil."""
        kb_path = "data/energy_knowledge.json"
        if os.path.exists(kb_path):
            with open(kb_path, "r") as f:
                return json.load(f)
        return {"market": {}, "sentiment": {}}

    def fetch_market_data(self):
        """Fetch real-time WTI crude oil data from Yahoo Finance."""
        try:
            ticker = yf.Ticker("CL=F")
            data = ticker.history(period="1d")
            if data.empty:
                logging.error("No market data returned from Yahoo Finance.")
                return pd.DataFrame()
            logging.info("Market data fetched successfully.")
            return data
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def fetch_news(self):
        """Fetch recent WTI crude oil news."""
        try:
            newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
            articles = newsapi.get_everything(q="WTI crude oil", language="en", page_size=5)["articles"]
            return articles
        except Exception as e:
            logging.error(f"Error fetching news: {e}")
            return []

    def analyze_sentiment(self, articles):
        """Analyze sentiment of news articles."""
        if not articles:
            return 0.0
        sentiments = [sentiment_classifier(article["title"])[0]["label"] for article in articles]
        sentiment_score = sum(1 if s == "POSITIVE" else -1 for s in sentiments) / len(sentiments)
        return sentiment_score

    def get_account_balance(self):
        """Fetch current balance from market_data.db."""
        conn = sqlite3.connect(MARKET_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM account WHERE id = 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 100000.0  # Default $100,000

    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk management."""
        balance = self.get_account_balance()
        risk_amount = balance * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            logging.warning("Stop-loss equals entry price. Defaulting to 1 unit.")
            return 1
        position_size = risk_amount / price_diff
        return max(1, min(position_size, balance / entry_price))

    def generate_trading_signal(self, sentiment_score, market_data):
        logging.debug(f"Generating signal with sentiment_score: {sentiment_score}")
        features = {"sentiment_score": sentiment_score, "market_data": market_data}
        signal = self.model(features)
        logging.debug(f"Generated signal: {signal}")
        if signal in ["BUY", "SELL"] and not market_data.empty:
            entry_price = market_data["Close"].iloc[-1]
            stop_loss = entry_price * (0.98 if signal == "BUY" else 1.02)
            take_profit = entry_price * (1.02 if signal == "BUY" else 0.98)
            size = self.calculate_position_size(entry_price, stop_loss)
            return {
                "Signal": signal,
                "Entry": entry_price,
                "StopLoss": stop_loss,
                "TakeProfit": take_profit,
                "Size": size,
                "Confidence": abs(sentiment_score) if sentiment_score != 0 else 0.5
            }
        return {"Signal": "HOLD", "Confidence": 0.5}
    
    def log_trade(self, signal):
        """Log trade suggestion to SQLite and store Telegram message ID."""
        if signal["Signal"] == "HOLD":
            return "No trade suggested."
        conn = sqlite3.connect(MARKET_DB)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO trades (signal, entry_price, stop_loss, take_profit, size, confidence, timestamp, confirmed, telegram_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """, (signal["Signal"], signal["Entry"], signal["StopLoss"], signal["TakeProfit"],
              signal["Size"], signal["Confidence"], timestamp))
        trade_id = cursor.lastrowid
        conn.commit()

        # Send alert and store message ID
        trade_result = f"Trade suggested: {signal['Signal']} {signal['Size']} units @ {signal['Entry']}"
        message_id = self.send_alert(trade_result, alert_category="trade", priority="high")
        if message_id:
            cursor.execute("UPDATE trades SET telegram_message_id = ? WHERE id = ?", (message_id, trade_id))
            conn.commit()

        conn.close()
        return trade_result

    def send_alert(self, message, alert_category="trade", priority="high"):
        """Handle alerts based on priority."""
        if priority.lower() == "high":
            message_id = self._send_alert_immediate(message, alert_type=alert_category)
            return message_id
        elif priority.lower() == "medium":
            self.alert_buffer.append(message)
            self.flush_alert_buffer(priority="medium")
            return None
        elif priority.lower() == "low":
            logging.info(f"LOW priority alert: {message}")
            return None

    def _send_alert_immediate(self, message, alert_type):
        """Send immediate Telegram alert and store the message ID."""
        if check_duplicate_alert(alert_type, message):
            logging.info(f"Duplicate alert ({alert_type}) skipped.")
            return None
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🚨 {alert_type.upper()} Alert: {message}"}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                message_id = data.get("result", {}).get("message_id")
                logging.info(f"Telegram message sent successfully! Message ID: {message_id}")
                record_alert(alert_type, message)
                return message_id
            else:
                logging.warning(f"Failed to send Telegram message: {response.text}")
                return None
        except Exception as e:
            logging.error(f"Telegram API Error: {e}")
            return None

    def flush_alert_buffer(self, priority):
        """Flush buffered alerts."""
        if self.alert_buffer and (datetime.now() - self.last_batch_flush).total_seconds() >= BATCH_WINDOW_SECONDS:
            combined_message = "\n---\n".join(self.alert_buffer)
            self._send_alert_immediate(combined_message, alert_type="batch_" + priority)
            self.alert_buffer = []
            self.last_batch_flush = datetime.now()

    def check_market_triggers(self):
        """Check for market events."""
        triggered_events = []
        now = datetime.now()
        if now.month in [6, 12] or (now.year == 2025 and now.month == 4):
            triggered_events.append("OPEC+ Meeting")
        if now.weekday() == 2 and now.hour == 10 and 25 <= now.minute <= 35:
            triggered_events.append("EIA Inventory Report")
        if now.weekday() == 1 and now.hour == 16 and 25 <= now.minute <= 35:
            triggered_events.append("API Inventory Report")
        if now.weekday() == 3 and now.day >= 25 and (now + timedelta(days=1)).month != now.month:
            triggered_events.append("Economic Data Release")
        if (now.month > 6 or (now.month == 6 and now.day >= 1)) and (now.month < 11 or (now.month == 11 and now.day <= 30)):
            triggered_events.append("Seasonal Weather Event")
        if now.weekday() == 0:
            triggered_events.append("Geopolitical Update")
        return triggered_events

    def run(self):
        logging.info(f"Trading Agent Started. Running every {TRADING_CHECK_INTERVAL} seconds...")
        while True:
            logging.info(f"Checking market conditions at {datetime.now()}")
            market_data = self.fetch_market_data()
            news_articles = self.fetch_news()
            sentiment_score = self.analyze_sentiment(news_articles)
            signal = self.generate_trading_signal(sentiment_score, market_data)

            if signal["Signal"] != "HOLD":
                trade_result = self.log_trade(signal)
                self.send_alert(trade_result, alert_category="trade", priority="high")
                trade_id = self.get_last_trade_id()
                self.check_telegram_reactions(trade_id)

            events = self.check_market_triggers()
            if events:
                self.send_alert("Triggered Events: " + ", ".join(events), alert_category="market", priority="high")

            next_check = datetime.now() + timedelta(seconds=TRADING_CHECK_INTERVAL)
            logging.info(f"Next check scheduled at {next_check}")
            time.sleep(TRADING_CHECK_INTERVAL)

    def get_last_trade_id(self):
        """Get the ID of the most recent trade."""
        conn = sqlite3.connect(MARKET_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT last_insert_rowid()")
        trade_id = cursor.fetchone()[0]
        conn.close()
        return trade_id

    
    def check_telegram_reactions(self, trade_id):
        """Poll Telegram for reactions (e.g., Thumbs Up) on the trade message."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        try:
            # Get updates since the last check, ensuring a valid offset
            offset = self.last_update_id + 1 if hasattr(self, 'last_update_id') and self.last_update_id is not None else -1
            response = requests.get(url, params={"offset": offset})
            updates = response.json().get("result", [])
            for update in updates:
                message = update.get("message")
                if message and message.get("message_id") and message.get("chat", {}).get("id") == int(TELEGRAM_CHAT_ID):
                    # Retrieve the message ID for the trade
                    conn = sqlite3.connect(MARKET_DB)
                    cursor = conn.cursor()
                    cursor.execute("SELECT telegram_message_id FROM trades WHERE id = ?", (trade_id,))
                    trade_message_id = cursor.fetchone()
                    conn.close()
                    if trade_message_id and message["message_id"] == trade_message_id[0]:
                        # Check reactions
                        reaction_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMessageReactionCount"
                        reaction_params = {"chat_id": TELEGRAM_CHAT_ID, "message_id": message["message_id"]}
                        reaction_response = requests.get(reaction_url, params=reaction_params)
                        reactions = reaction_response.json().get("result", [])
                        for reaction in reactions:
                            if reaction.get("type") == "reaction_type_thumbs_up" and reaction.get("total_count", 0) > 0:
                                self.confirm_trade_from_reaction(trade_id)
                                self.last_update_id = update.get("update_id", 0)  # Update last_update_id
                                return
            if updates:
                self.last_update_id = updates[-1].get("update_id", 0)  # Update last_update_id with the latest
        except Exception as e:
            logging.error(f"Error polling Telegram reactions: {e}")

    def confirm_trade_from_reaction(self, trade_id):
        """Confirm a trade based on a Telegram reaction (e.g., Thumbs Up)."""
        try:
            conn = sqlite3.connect(MARKET_DB)
            cursor = conn.cursor()
            cursor.execute("UPDATE trades SET confirmed = 1 WHERE id = ?", (trade_id,))
            # Update balance
            cursor.execute("SELECT signal, entry_price, size FROM trades WHERE id = ?", (trade_id,))
            trade = cursor.fetchone()
            if trade:
                signal, entry_price, size = trade
                balance = self.get_account_balance()
                cost = entry_price * size
                if signal == "BUY":
                    balance -= cost
                else:  # SELL
                    balance += cost
                cursor.execute("UPDATE account SET balance = ? WHERE id = 1", (balance,))
            conn.commit()
            logging.info(f"Trade ID {trade_id} confirmed via Telegram reaction. Updated balance: ${balance:.2f}")
            # Log to trade_history
            self.log_confirmed_trade_to_history(trade_id, signal, entry_price, size, cost)
        except Exception as e:
            logging.error(f"Error confirming trade from reaction: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def log_confirmed_trade_to_history(self, trade_id, signal, entry_price, size, cost):
        """Log confirmed trade to trade_history."""
        conn = sqlite3.connect(MARKET_DB)
        cursor = conn.cursor()
        execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO trade_history (execution_time, trade_type, executed_price, shares, cost, note)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (execution_time, signal, entry_price, size, cost, f"Confirmed AI trade ID {trade_id} via Telegram"))
        conn.commit()
        conn.close()

    def get_last_sent_message_id(self):
        """Retrieve the last sent Telegram message ID from the database."""
        try:
            conn = sqlite3.connect(MARKET_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT telegram_message_id FROM trades WHERE telegram_message_id IS NOT NULL ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
        except Exception as e:
            logging.error(f"Error retrieving last sent message ID: {e}")
            return None
        return result[0] if result else None

if __name__ == "__main__":
    agent = TradingAgent()
    agent.run()