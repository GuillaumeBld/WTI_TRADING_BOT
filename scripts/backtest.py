"""
Backtesting Module

This script implements a backtesting engine to evaluate trading strategies
on historical data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_fetch import fetch_historical_data
from strategy import TradingStrategy

class Backtest:
    """
    Backtesting class to evaluate trading strategies
    """
    
    def __init__(self, initial_capital=100000.0):
        """
        Initialize the backtester
        
        Args:
            initial_capital (float): Initial capital for the backtest
        """
        self.initial_capital = initial_capital
        self.results = None
    
    def run(self, data, signals, commission=0.001):
        """
        Run the backtest
        
        Args:
            data (pandas.DataFrame): DataFrame with price data
            signals (pandas.DataFrame): DataFrame with trading signals
            commission (float): Commission rate per trade
            
        Returns:
            pandas.DataFrame: DataFrame with backtest results
        """
        # Make sure data and signals have the same index
        common_index = data.index.intersection(signals.index)
        data = data.loc[common_index]
        signals = signals.loc[common_index]
        
        # Create a DataFrame for the backtest results
        results = pd.DataFrame(index=signals.index)
        results['Price'] = data['Close']
        results['Signal'] = signals['Signal']
        
        # Calculate positions (1 for long, -1 for short, 0 for no position)
        results['Position'] = results['Signal'].diff()
        
        # Calculate holdings and cash
        results['Holdings'] = results['Signal'] * results['Price']
        results['Cash'] = self.initial_capital - (results['Signal'].diff() * results['Price']).cumsum()
        
        # Calculate portfolio value
        results['Portfolio'] = results['Holdings'] + results['Cash']
        
        # Calculate returns
        results['Returns'] = results['Portfolio'].pct_change()
        
        # Calculate commissions
        results['Commission'] = np.abs(results['Position']) * results['Price'] * commission
        results['Portfolio'] = results['Portfolio'] - results['Commission'].cumsum()
        
        self.results = results
        
        return results
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            dict: Dictionary with performance metrics
        """
        if self.results is None:
            print("No backtest results available")
            return None
        
        # Calculate metrics
        total_return = (self.results['Portfolio'].iloc[-1] / self.initial_capital) - 1
        annual_return = total_return / (len(self.results) / 252)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / (self.results['Returns'].std() * np.sqrt(252))
        
        # Calculate maximum drawdown
        portfolio_values = self.results['Portfolio']
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        trades = self.results['Position'].dropna()
        trades = trades[trades != 0]
        
        if len(trades) > 0:
            win_count = sum(1 for i in range(len(trades)) if trades.iloc[i] > 0 and self.results['Returns'].iloc[i] > 0)
            win_rate = win_count / len(trades)
        else:
            win_rate = 0
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Number of Trades': len(trades)
        }
        
        return metrics
    
    def plot_results(self, save_path=None):
        """
        Plot backtest results
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.results is None:
            print("No backtest results available")
            return None
        
        # Create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot portfolio value
        ax1.plot(self.results.index, self.results['Portfolio'])
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Backtest Results')
        ax1.grid(True)
        
        # Plot price and signals
        ax2.plot(self.results.index, self.results['Price'])
        
        # Plot buy signals
        buy_signals = self.results[self.results['Position'] > 0]
        ax2.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='g', alpha=0.7)
        
        # Plot sell signals
        sell_signals = self.results[self.results['Position'] < 0]
        ax2.scatter(sell_signals.index, sell_signals['Price'], marker='v', color='r', alpha=0.7)
        
        ax2.set_ylabel('Price ($)')
        ax2.grid(True)
        
        # Plot drawdown
        portfolio_values = self.results['Portfolio']
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        
        ax3.fill_between(self.results.index, drawdown, 0, color='r', alpha=0.3)
        ax3.set_ylabel('Drawdown')
        ax3.set_xlabel('Date')
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        return fig

def main():
    """
    Main function to run a backtest
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a backtest')
    parser.add_argument('--period', type=str, default='1y', help='Period to backtest (e.g., 1y, 2y, 5y)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--save', action='store_true', help='Save results and plot')
    args = parser.parse_args()
    
    # Fetch data
    print(f"Fetching data for period: {args.period}")
    data = fetch_historical_data(period=args.period)
    
    if data is None:
        print("Failed to fetch data")
        return
    
    # Create strategy and generate signals
    print("Generating trading signals...")
    strategy = TradingStrategy()
    strategy.load_model()
    signals = strategy.generate_signals(data)
    
    if signals is None:
        print("Failed to generate signals")
        return
    
    # Run backtest
    print("Running backtest...")
    backtest = Backtest(initial_capital=args.capital)
    results = backtest.run(data, signals, commission=args.commission)
    
    # Calculate and print metrics
    metrics = backtest.calculate_metrics()
    
    if metrics:
        print("\nBacktest Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    # Plot results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"../results/backtest_results_{timestamp}.csv"
        plot_path = f"../results/backtest_plot_{timestamp}.png"
        
        # Save results
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results.to_csv(results_path)
        print(f"Results saved to {results_path}")
        
        # Save plot
        backtest.plot_results(save_path=plot_path)
    else:
        backtest.plot_results()
        plt.show()

if __name__ == "__main__":
    main()
