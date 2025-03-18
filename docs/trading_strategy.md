# WTI Crude Oil Trading Strategy Documentation

## Overview

This document provides a comprehensive overview of the trading strategy implemented in the WTI Crude Oil Trading System. It is intended for developers who will maintain and extend the system in the future.

## Strategy Components

The trading strategy consists of several key components:

1. **Technical Indicators**: RSI, MACD, ADX, and EMAs
2. **Machine Learning Model**: LightGBM classifier
3. **Risk Management**: Dynamic stop-loss and take-profit levels
4. **Signal Generation**: Combining technical and ML signals
5. **Execution Logic**: Telegram-based trade recommendations

## Technical Indicators

### Relative Strength Index (RSI)

- **Implementation**: Located in `scripts/indicators.py`
- **Parameters**: Default window size of 14 periods
- **Usage**: Identifies overbought (RSI > 70) and oversold (RSI < 30) conditions
- **Signal Logic**:
  - Buy signal when RSI crosses above 30 from below
  - Sell signal when RSI crosses below 70 from above

### Moving Average Convergence Divergence (MACD)

- **Implementation**: Located in `scripts/indicators.py`
- **Parameters**: 
  - Fast EMA: 12 periods
  - Slow EMA: 26 periods
  - Signal Line: 9 periods
- **Usage**: Identifies trend direction and momentum
- **Signal Logic**:
  - Buy signal when MACD line crosses above signal line
  - Sell signal when MACD line crosses below signal line

### Average Directional Index (ADX)

- **Implementation**: Located in `scripts/indicators.py`
- **Parameters**: Default period of 14
- **Usage**: Measures trend strength (not direction)
- **Signal Logic**:
  - Strong trend when ADX > 25
  - Weak trend when ADX < 20
  - Used as a filter for other signals

### Exponential Moving Averages (EMAs)

- **Implementation**: Located in `scripts/indicators.py`
- **Parameters**: 9, 21, 50, and 200 periods
- **Usage**: Identifies trend direction and support/resistance levels
- **Signal Logic**:
  - Bullish when shorter EMAs are above longer EMAs
  - Bearish when shorter EMAs are below longer EMAs
  - Buy signal on golden cross (50 EMA crosses above 200 EMA)
  - Sell signal on death cross (50 EMA crosses below 200 EMA)

## Machine Learning Model

### LightGBM Classifier

- **Implementation**: Located in `scripts/strategy.py`
- **Purpose**: Predict price movement direction (up/down)
- **Features**:
  - Technical indicators (RSI, MACD, ADX, EMAs)
  - Price-based features (returns, volatility)
  - Volume-based features
- **Training Process**:
  - Data split: 80% training, 20% testing
  - Target: Binary classification (1 for price increase, 0 for decrease)
  - Hyperparameters: See `scripts/strategy.py` for details
- **Performance Metrics**:
  - Accuracy: Target > 55%
  - Precision: Target > 60%
  - Recall: Target > 50%

### Feature Engineering

- **Implementation**: Located in `scripts/strategy.py`
- **Process**:
  1. Calculate technical indicators
  2. Add price-based features
  3. Add volume-based features
  4. Handle missing values
  5. Scale features

## Risk Management

### Dynamic Stop-Loss

- **Implementation**: Located in `scripts/execute.py`
- **Calculation**:
  - Long positions: Entry price * (1 - risk_percentage)
  - Short positions: Entry price * (1 + risk_percentage)
  - Default risk_percentage: 2%
- **Adjustment**: Stop-loss levels are adjusted based on volatility

### Take-Profit Levels

- **Implementation**: Located in `scripts/execute.py`
- **Calculation**:
  - Long positions: Entry price * (1 + reward_percentage)
  - Short positions: Entry price * (1 - reward_percentage)
  - Default reward_percentage: 5%
- **Risk-Reward Ratio**: Target minimum of 1:2.5

### Position Sizing

- **Implementation**: Located in `scripts/execute.py`
- **Calculation**: Based on account size, volatility, and risk per trade
- **Formula**: Position Size = (Account Size * Risk Per Trade) / (Entry Price - Stop Loss)

## Signal Generation

### Combined Signal Approach

- **Implementation**: Located in `scripts/strategy.py`
- **Process**:
  1. Generate signals from technical indicators
  2. Generate signals from ML model
  3. Combine signals using a weighted approach
  4. Apply filters (ADX, volatility)
  5. Generate final trade recommendation

### Signal Weights

- Technical Indicators: 40%
- ML Model: 60%

### Signal Thresholds

- Strong Buy: Combined score > 0.7
- Buy: Combined score > 0.55
- Neutral: Combined score between 0.45 and 0.55
- Sell: Combined score < 0.45
- Strong Sell: Combined score < 0.3

## Execution Logic

### Trade Signal Format

- **Implementation**: Located in `scripts/execute.py`
- **Components**:
  - Symbol: WTI Crude Oil Futures (CL=F)
  - Action: BUY/SELL
  - Price: Current market price
  - Confidence: Signal strength (0-1)
  - Stop-Loss: Calculated stop-loss level
  - Take-Profit: Calculated take-profit level
  - Indicators: Current indicator values

### Telegram Delivery

- **Implementation**: Located in `scripts/execute.py`
- **Process**:
  1. Format trade signal as a message
  2. Send message to configured Telegram chat
  3. Log trade signal for future reference

## Backtesting

### Backtesting Engine

- **Implementation**: Located in `scripts/backtest.py`
- **Features**:
  - Historical data simulation
  - Performance metrics calculation
  - Visualization of results

### Performance Metrics

- **Implementation**: Located in `scripts/backtest.py`
- **Metrics**:
  - Total Return
  - Annual Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate
  - Number of Trades

## Optimization

### Parameter Optimization

- **Implementation**: Located in `scripts/test_performance.py`
- **Parameters Optimized**:
  - Indicator parameters (windows, periods)
  - ML model hyperparameters
  - Signal thresholds
  - Risk management parameters

### Performance Optimization

- **Implementation**: Located in `scripts/test_performance.py`
- **Areas**:
  - Latency reduction
  - Batch processing
  - Caching strategies
  - Resource utilization

## Extending the Strategy

### Adding New Indicators

1. Implement the indicator calculation in `scripts/indicators.py`
2. Add the indicator to the `add_all_indicators` function
3. Update the feature engineering in `scripts/strategy.py`
4. Retrain the model with the new features

### Modifying Signal Logic

1. Update the signal generation logic in `scripts/strategy.py`
2. Adjust the weights and thresholds as needed
3. Backtest the changes to validate performance
4. Update the documentation

### Adding New Data Sources

1. Implement the data fetching in `scripts/data_fetch.py`
2. Update the feature engineering to include the new data
3. Retrain the model with the new features
4. Backtest the changes to validate performance

## Troubleshooting

### Common Issues

- **Data Availability**: Check if Yahoo Finance API is accessible
- **Model Performance**: Verify model accuracy and retrain if needed
- **Signal Quality**: Check indicator calculations and signal logic
- **Execution Issues**: Verify Telegram configuration and connectivity

### Debugging

- Check log files in the `logs/` directory
- Use the `--verbose` flag for detailed output
- Run backtests to validate changes
- Use the performance testing tools in `scripts/test_performance.py`

## Conclusion

This trading strategy combines traditional technical analysis with modern machine learning techniques to generate trade signals for WTI Crude Oil Futures. It is designed to be robust, adaptable, and maintainable.

For any questions or issues, please contact the development team.
