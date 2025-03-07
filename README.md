<picture>
  <source media="(prefers-color-scheme: dark)" srcset="data/Botpic.jpeg">
  <source media="(prefers-color-scheme: light)" srcset="data/Botpic.jpeg">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

# Trading Bot - Detailed Overview

This project is a complete, AI-enhanced trading system designed for analyzing WTI crude oil. It integrates multiple data sources—including market data from Yahoo Finance, news from NewsAPI, and satellite-derived alternative data—with advanced technical analysis, sentiment analysis (using DistilBERT/FinBERT), and dynamic risk management. The system is built in a modular, agent-based structure that supports both automated and semi-manual trade execution, backtesting, and performance optimization.

---

## New Strategy & Architecture

We have restructured the code into distinct “agents,” each responsible for a specific task. This approach improves maintainability and facilitates the integration of advanced AI techniques for dynamic risk management and execution optimization, ultimately enhancing the risk-adjusted return.

### 1. Data Acquisition Agent
- **data_fetch.py**: Fetches historical and real-time market data for WTI crude oil (e.g., “CL=F”) from Yahoo Finance. Data is stored in CSV files and an SQLite database, with robust retry logic.

### 2. Indicator & Technical Analysis Agent
- **indicators.py**: Processes market data to calculate technical indicators (RSI, EMA, MACD, ADX, etc.) and updates the database with these metrics.
- **dataframe_wrapper.py**: Provides a convenient wrapper to access DataFrame columns using dot notation.

### 3. News, Satellite & Sentiment Analysis Agent
- **fetch_news.py**: Retrieves relevant news articles from NewsAPI.
- **satellite_data.py** (new): Integrates processed satellite data—such as oil storage levels and tanker counts—to derive alternative signals related to global oil inventory and supply trends.
- **sentiment_analysis.py / crude_oil_sentiment_bot.py**: Applies NLP using DistilBERT or FinBERT to analyze article sentiment (positive, negative, neutral), logging detailed results to both text logs and an SQLite database.
- **FinBERT_Sentiment_Classifier.py**: Provides specialized financial sentiment analysis using the FinBERT model (if required).

### 4. Trading Signal & Strategy Agent
- **strategy.py**: Combines technical indicators, sentiment data, and alternative signals from satellite data to generate buy/sell/hold signals. Contains placeholders for further machine learning integration.
- **strategy_manager.py**: Dynamically adjusts the trading strategy by switching modes (e.g., from Adaptive to Self-Optimizing) based on performance metrics such as Sharpe ratio, win rate, and drawdown. This agent fine-tunes indicator weightings—including those from satellite data—over time.

### 5. Execution & Order Management Agent
- **alerts.py**: Loads signals, calculates risk parameters (e.g., ATR-based stop-loss, take-profit levels), finalizes trade decisions, filters duplicate signals, and sends alerts via Telegram.
- **execute.py**: Retrieves filtered signals from the database, handles trade execution with risk-based position sizing (e.g., risking 5% of the account balance per trade), and logs all trades.
- **trade_execution.py**: Provides additional trade logging and management features, ensuring position sizing and duplicate trade prevention.

### 6. Backtesting & Performance Evaluation Agent
- **backtest.py**: Simulates trading using historical data, applies recorded signals, and calculates performance metrics (profit, drawdown, Sharpe ratio). Results are saved to both SQLite and CSV.
- **test_performance.py**: Measures system latency, validates signal accuracy, and evaluates processing speed and scalability through detailed performance reports and visualizations.

### 7. Investment & Portfolio Management Agent
- **investment_tracker.py**: Manages manual trade confirmations alongside AI-suggested trades, tracks account balance and open positions, and records detailed trade histories (including partial fills and user notes).

### 8. System Orchestration & Automation Agent
- **trading_agent.py**: Acts as the central orchestrator, periodically triggering data fetching, signal generation, and risk management, while enforcing cooldown periods to prevent duplicate alerts.
- **generate_walkthrough.py**: Captures system operation via screenshots and annotations, then stitches them into a video walkthrough for demonstration.
- **run_sentiment_bot.sh**: A shell script that starts the sentiment analysis process, typically scheduled via cron.

### 9. Infrastructure & Deployment
- **Dockerfile & docker-compose.yml**: Containerize the entire system for cloud deployment, ensuring scalability and ease of management.
- **.env / .env.example**: Store sensitive configuration details like API keys (NEWSAPI_KEY, TELEGRAM_BOT_TOKEN, etc.).

### 10. Requirements & Utilities
- **requirements.txt**: Lists all Python dependencies (e.g., `pandas`, `numpy`, `torch`, `transformers`, etc.), including those required for optional modules like screenshot capture.
- **chart_utils.py**: Provides visualization utilities (e.g., generating candlestick charts with highlighted support zones).

---

## Workflow Overview

1. **Data Acquisition & Indicator Calculation**:
   - Execute `data_fetch.py` to download historical WTI market data.
   - Run `indicators.py` to compute technical signals and update the SQLite database.

2. **News, Satellite & Sentiment Analysis**:
   - Fetch recent news articles with `fetch_news.py`.
   - In parallel, retrieve satellite data using `satellite_data.py` to capture alternative signals (e.g., oil storage estimates and tanker counts).
   - Analyze sentiment using `sentiment_analysis.py` (or `crude_oil_sentiment_bot.py`) and store results in the database.

3. **Signal Generation & Strategy Management**:
   - Combine technical indicators, sentiment scores, and satellite-derived insights in `strategy.py` to generate trading signals.
   - Use `strategy_manager.py` to dynamically adjust strategy parameters based on performance targets.

4. **Execution & Trade Management**:
   - Process finalized signals via `alerts.py` and `execute.py`, ensuring risk-based position sizing and duplicate prevention.
   - Record trade details and update the portfolio using `trade_execution.py` and `investment_tracker.py`.

5. **Backtesting & Performance Evaluation**:
   - Simulate historical performance using `backtest.py`.
   - Evaluate system performance, latency, and accuracy with `test_performance.py`.

6. **Automation & Deployment**:
   - Orchestrate regular trading cycles via `trading_agent.py`.
   - Generate operational walkthroughs using `generate_walkthrough.py`.
   - Deploy the system in Docker using `Dockerfile` and `docker-compose.yml`.
   - Schedule regular execution with `run_sentiment_bot.sh` (via cron or similar).

---

## Key Enhancements

- **Modular, Agent-Based Architecture**: Each component functions as an independent agent with a clearly defined role, simplifying maintenance and future expansion.
- **Incorporation of Satellite Data**: The new module, `satellite_data.py`, processes satellite-derived metrics (e.g., oil storage levels, tanker activity) to provide alternative fundamental signals that complement traditional technical and sentiment analysis.
- **Dynamic Risk Management**: `strategy_manager.py` adjusts strategy parameters—including weights for satellite data inputs—based on real-time performance metrics (Sharpe ratio, win rate, drawdown).
- **AI-Driven Enhancements**: Placeholder areas in the strategy module allow for future integration of reinforcement learning or hybrid AI models to further optimize risk-adjusted returns.
- **Optimized Execution & Automation**: Improved duplicate filtering, risk-based position sizing, and robust automated orchestration reduce operational risks and ensure efficient trade management.

---

## Notes & Recommendations

- **Database Consistency**: Ensure that all file paths and table structures in your SQLite database are maintained consistently across modules.
- **Environment Variables**: Confirm that the `.env` file is properly configured with all necessary API keys and settings.
- **FinBERT vs. DistilBERT**: Evaluate the performance of both models; use FinBERT for specialized financial sentiment if available.
- **Satellite Data Integration**: For satellite data, use processed metrics from your data provider rather than raw images. Integrate these signals as an additional layer of fundamental analysis in your strategy.
- **Monitoring & Logging**: Regularly review logs (located in the `logs/` directory) and database content to ensure smooth operation of backtesting, trade logging, and sentiment computations.
- **Backtesting**: Utilize the backtesting module to iteratively refine your strategy before deploying live.
- **Future Enhancements**: Consider integrating reinforcement learning or additional AI models into the strategy module to further improve risk-adjusted returns.

---

This updated structure and strategy now incorporate satellite data alongside traditional technical and sentiment analysis, creating a more holistic approach to predicting market movements and enhancing risk-adjusted returns for WTI crude oil trading.
