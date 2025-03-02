"""
Backtesting Module - Demo Version

This is a simplified version of the backtesting module that evaluates
trading strategies on historical data without external dependencies.
"""

import os
import csv
import math
from datetime import datetime

def load_price_data(filepath):
    """
    Load price data from a CSV file
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries with the data
    """
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                processed_row = {
                    'Date': row['Date'],
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': int(row['Volume'])
                }
                data.append(processed_row)
        print(f"Loaded {len(data)} price records from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading price data from {filepath}: {e}")
        return []

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

def run_backtest(price_data, signals, initial_capital=100000.0, commission=0.001):
    """
    Run a backtest of the trading strategy
    
    Args:
        price_data (list): List of dictionaries with price data
        signals (list): List of dictionaries with trading signals
        initial_capital (float): Initial capital for the backtest
        commission (float): Commission rate per trade
        
    Returns:
        list: List of dictionaries with backtest results
    """
    # Create a date-indexed dictionary of prices for quick lookup
    price_dict = {item['Date']: item for item in price_data}
    
    # Initialize backtest results
    results = []
    
    # Initialize portfolio state
    portfolio = {
        'cash': initial_capital,
        'position': 0,
        'holdings': 0.0,
        'portfolio_value': initial_capital,
        'trades': 0,
        'wins': 0,
        'losses': 0
    }
    
    # Previous portfolio value for calculating returns
    prev_portfolio_value = initial_capital
    
    # Process each signal
    for i, signal in enumerate(signals):
        date = signal['Date']
        signal_type = signal['Signal']  # 1: Buy, -1: Sell, 0: Hold
        
        # Get price data for this date
        if date not in price_dict:
            continue
        
        price_info = price_dict[date]
        close_price = price_info['Close']
        
        # Calculate position change
        position_change = 0
        if i > 0:
            position_change = signal_type - signals[i-1]['Signal']
        else:
            position_change = signal_type
        
        # Calculate trade cost (commission)
        trade_cost = 0.0
        if position_change != 0:
            trade_cost = abs(position_change) * close_price * commission
            portfolio['trades'] += 1
        
        # Update portfolio based on position change
        if position_change > 0:  # Buying
            shares_to_buy = (portfolio['cash'] * 0.95) / close_price  # Use 95% of cash
            cost = shares_to_buy * close_price
            portfolio['cash'] -= cost
            portfolio['position'] = 1
            portfolio['holdings'] = shares_to_buy * close_price
        elif position_change < 0:  # Selling
            if portfolio['position'] > 0:  # If we have a position
                portfolio['cash'] += portfolio['holdings']
                
                # Record win/loss
                if portfolio['holdings'] > portfolio['portfolio_value'] - portfolio['cash']:
                    portfolio['wins'] += 1
                else:
                    portfolio['losses'] += 1
                
                portfolio['position'] = 0
                portfolio['holdings'] = 0.0
        
        # Deduct commission
        portfolio['cash'] -= trade_cost
        
        # Update holdings value based on current price
        if portfolio['position'] > 0:
            portfolio['holdings'] = (portfolio['holdings'] / price_info['Close']) * close_price
        
        # Calculate portfolio value
        portfolio_value = portfolio['cash'] + portfolio['holdings']
        
        # Calculate return
        daily_return = (portfolio_value / prev_portfolio_value) - 1 if prev_portfolio_value > 0 else 0
        prev_portfolio_value = portfolio_value
        
        # Update portfolio value
        portfolio['portfolio_value'] = portfolio_value
        
        # Add to results
        results.append({
            'Date': date,
            'Price': close_price,
            'Signal': signal_type,
            'Position': portfolio['position'],
            'Cash': portfolio['cash'],
            'Holdings': portfolio['holdings'],
            'Portfolio_Value': portfolio['portfolio_value'],
            'Daily_Return': daily_return,
            'Trade_Cost': trade_cost
        })
    
    return results, portfolio

def calculate_metrics(results, portfolio, initial_capital):
    """
    Calculate performance metrics from backtest results
    
    Args:
        results (list): List of dictionaries with backtest results
        portfolio (dict): Final portfolio state
        initial_capital (float): Initial capital for the backtest
        
    Returns:
        dict: Dictionary with performance metrics
    """
    # Calculate total return
    final_value = results[-1]['Portfolio_Value']
    total_return = (final_value / initial_capital) - 1
    
    # Calculate annualized return (assuming 252 trading days per year)
    days = len(results)
    annual_return = total_return * (252 / days)
    
    # Calculate Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    daily_returns = [result['Daily_Return'] for result in results]
    daily_returns_std = calculate_std(daily_returns)
    sharpe_ratio = (annual_return - risk_free_rate) / (daily_returns_std * math.sqrt(252)) if daily_returns_std > 0 else 0
    
    # Calculate maximum drawdown
    max_drawdown = calculate_max_drawdown(results)
    
    # Calculate win rate
    win_rate = portfolio['wins'] / portfolio['trades'] if portfolio['trades'] > 0 else 0
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Number of Trades': portfolio['trades']
    }

def calculate_std(values):
    """
    Calculate standard deviation
    
    Args:
        values (list): List of values
        
    Returns:
        float: Standard deviation
    """
    if not values:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def calculate_max_drawdown(results):
    """
    Calculate maximum drawdown
    
    Args:
        results (list): List of dictionaries with backtest results
        
    Returns:
        float: Maximum drawdown
    """
    max_value = 0
    max_drawdown = 0
    
    for result in results:
        value = result['Portfolio_Value']
        if value > max_value:
            max_value = value
        
        drawdown = (max_value - value) / max_value if max_value > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def save_results_to_csv(results, filepath):
    """
    Save backtest results to a CSV file
    
    Args:
        results (list): List of dictionaries with backtest results
        filepath (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', newline='') as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        print(f"Backtest results saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")
        return False

def generate_text_chart(results, width=80, height=20):
    """
    Generate a simple ASCII chart of portfolio value
    
    Args:
        results (list): List of dictionaries with backtest results
        width (int): Width of the chart
        height (int): Height of the chart
        
    Returns:
        str: ASCII chart
    """
    # Extract portfolio values
    values = [result['Portfolio_Value'] for result in results]
    
    # Find min and max values
    min_value = min(values)
    max_value = max(values)
    value_range = max_value - min_value
    
    # Create chart
    chart = []
    
    # Add header
    chart.append("Portfolio Value Chart")
    chart.append(f"Min: ${min_value:.2f}, Max: ${max_value:.2f}, Range: ${value_range:.2f}")
    chart.append("-" * width)
    
    # Create chart rows
    for i in range(height):
        row = []
        level = max_value - (i * value_range / height)
        
        # Add y-axis label
        row.append(f"${level:.2f} |")
        
        # Add chart points
        for j in range(width - 10):  # Subtract 10 for the y-axis label
            if j < len(values):
                index = int(j * len(values) / (width - 10))
                value = values[index]
                
                if value >= level - (value_range / height / 2) and value <= level + (value_range / height / 2):
                    row.append("*")
                else:
                    row.append(" ")
        
        chart.append("".join(row))
    
    # Add x-axis
    chart.append("-" * width)
    
    # Add x-axis labels
    x_axis = "Time"
    x_axis = x_axis.center(width)
    chart.append(x_axis)
    
    return "\n".join(chart)

def main():
    """
    Main function to run a backtest
    """
    print("WTI Crude Oil Trading System - Backtesting")
    print("==========================================")
    
    # Load price data
    price_data_path = "../data/crude_oil_data.csv"
    price_data = load_price_data(price_data_path)
    
    if not price_data:
        print("No price data available. Please run data_fetch.py first.")
        return
    
    # Ask user which signals to use
    print("\nWhich trading signals would you like to backtest?")
    print("1. Rule-based signals (../data/trading_signals.csv)")
    print("2. Machine learning signals (../data/trading_signals_ml.csv)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        signals_path = "../data/trading_signals.csv"
        strategy_name = "Rule-based"
    elif choice == "2":
        signals_path = "../data/trading_signals_ml.csv"
        strategy_name = "Machine Learning"
    else:
        print("Invalid choice. Using ML signals by default.")
        signals_path = "../data/trading_signals_ml.csv"
        strategy_name = "Machine Learning"
    
    # Load trading signals
    signals = load_signals(signals_path)
    
    if not signals:
        print(f"No trading signals available at {signals_path}. Please run strategy.py first.")
        return
    
    # Set backtest parameters
    initial_capital = 100000.0
    commission = 0.001
    
    print(f"Running backtest for {strategy_name} strategy with initial capital: ${initial_capital:.2f} and commission rate: {commission:.3f}")
    
    # Run backtest
    results, portfolio = run_backtest(price_data, signals, initial_capital, commission)
    
    if not results:
        print("Failed to run backtest.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results, portfolio, initial_capital)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Strategy: {strategy_name}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Portfolio Value: ${results[-1]['Portfolio_Value']:.2f}")
    
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Generate and print chart
    chart = generate_text_chart(results)
    print("\n" + chart)
    
    # Save results
    os.makedirs("../results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"../results/backtest_{strategy_name.lower().replace(' ', '_')}_{timestamp}.csv"
    save_results_to_csv(results, results_path)
    
    print("\nBacktesting complete!")
    print("You can now proceed with trade execution.")

if __name__ == "__main__":
    main()
