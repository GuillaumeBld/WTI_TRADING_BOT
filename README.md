# AI-Powered WTI Crude Oil Trading System

## Overview
This project implements an **AI-powered trading system** for **WTI Crude Oil Futures**, leveraging **machine learning, reinforcement learning (FinRL), and sentiment analysis** to optimize trading decisions. The system features **real-time data processing, backtesting, automated trade execution alerts, and risk management**, with deployment on **cloud infrastructure** for scalability.

## Features
- **Technical Analysis-Based Trading**: Uses RSI, MACD, ADX, and Moving Averages.
- **Machine Learning for Prediction**: Implements LightGBM for price forecasting.
- **Reinforcement Learning Optimization**: FinRL-based dynamic trading strategies.
- **Sentiment Analysis**: NLP models analyze crude oil news and social media.
- **Backtesting Engine**: Historical market simulation using `Backtrader`.
- **Trade Order Delivery via Telegram**: Trade recommendations sent via Telegram instead of direct execution.
- **Risk Management**: Dynamic stop-loss, take-profit, and position sizing.
- **Alerts & Notifications**: Sends trade signals via Telegram/Slack.

## Project Structure
```bash
├── data/               # Market data storage (historical & real-time)
├── models/             # Trained machine learning & reinforcement learning models
├── scripts/            # Core trading system scripts
│   ├── data_fetch.py   # Retrieves market data
│   ├── indicators.py   # Computes technical indicators
│   ├── strategy.py     # Trading logic & AI models
│   ├── execute.py      # Generates trade orders and sends to Telegram
│   ├── backtest.py     # Backtesting engine
│   ├── alerts.py       # Telegram/Slack notifications
├── notebooks/          # Jupyter notebooks for development & testing
├── config/             # Configuration files
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation
```

## Setup Instructions
### **1. Install Dependencies**
Ensure Python 3.8+ is installed, then run:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Set Up Environment Variables**
Copy `.env.example` to `.env` and fill in API credentials:
```bash
cp .env.example .env
```
Edit the file and add API keys for **market data** and **Telegram bot API**.

### **3. Fetch Market Data**
```bash
python scripts/data_fetch.py
```

### **4. Train & Evaluate Machine Learning Model**
```bash
python scripts/strategy.py --train
```

### **5. Backtest Strategy on Historical Data**
```bash
python scripts/backtest.py
```

### **6. Generate Trade Orders and Deliver via Telegram**
```bash
python scripts/execute.py --telegram
```

### **7. Monitor Trades via Telegram/Slack**
Set up alerts:
```bash
python scripts/alerts.py --configure
```

## Deployment
### **Docker Deployment**
```bash
docker build -t ai-trading-bot .
docker run -d --env-file .env ai-trading-bot
```

### **Cloud Deployment**
Deploy to **AWS/GCP/DigitalOcean** using the `deploy.sh` script:
```bash
bash deploy.sh
```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'Added new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Submit a pull request

## License
MIT License. See `LICENSE` for details.
