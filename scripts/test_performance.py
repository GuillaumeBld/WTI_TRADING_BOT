"""
Performance Testing Module

This script tests the performance of the trading system in a simulated environment.
"""

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from data_fetch import fetch_historical_data
from strategy import TradingStrategy
from backtest import Backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/performance_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PerformanceTester:
    """
    Class to test the performance of the trading system
    """
    
    def __init__(self):
        """
        Initialize the performance tester
        """
        self.strategy = TradingStrategy()
        self.backtest = Backtest()
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs("../logs", exist_ok=True)
        os.makedirs("../results", exist_ok=True)
    
    def test_latency(self, iterations=100):
        """
        Test the latency of the trading system
        
        Args:
            iterations (int): Number of iterations to run
            
        Returns:
            dict: Dictionary with latency statistics
        """
        logger.info(f"Testing latency with {iterations} iterations")
        
        # Fetch data once
        data = fetch_historical_data(period='1mo')
        
        if data is None:
            logger.error("Failed to fetch data")
            return None
        
        # Load model
        self.strategy.load_model()
        
        # Measure signal generation latency
        signal_times = []
        
        for i in range(iterations):
            start_time = time.time()
            signals = self.strategy.generate_signals(data)
            end_time = time.time()
            
            if signals is not None:
                signal_times.append(end_time - start_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{iterations} iterations")
        
        # Calculate statistics
        signal_times = np.array(signal_times)
        
        stats = {
            'min_latency': signal_times.min(),
            'max_latency': signal_times.max(),
            'mean_latency': signal_times.mean(),
            'median_latency': np.median(signal_times),
            'p95_latency': np.percentile(signal_times, 95),
            'p99_latency': np.percentile(signal_times, 99),
            'std_latency': signal_times.std()
        }
        
        # Log statistics
        logger.info("Latency statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value:.6f} seconds")
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(signal_times, bins=20, alpha=0.7)
        plt.axvline(stats['mean_latency'], color='r', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean_latency']:.6f}s")
        plt.axvline(stats['p95_latency'], color='g', linestyle='dashed', linewidth=2, label=f"95th Percentile: {stats['p95_latency']:.6f}s")
        plt.axvline(stats['p99_latency'], color='b', linestyle='dashed', linewidth=2, label=f"99th Percentile: {stats['p99_latency']:.6f}s")
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.title('Signal Generation Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"../results/latency_histogram_{timestamp}.png")
        logger.info(f"Latency histogram saved to ../results/latency_histogram_{timestamp}.png")
        
        return stats
    
    def test_accuracy(self, test_period='6mo'):
        """
        Test the accuracy of the trading system
        
        Args:
            test_period (str): Period to test
            
        Returns:
            dict: Dictionary with accuracy statistics
        """
        logger.info(f"Testing accuracy with test period: {test_period}")
        
        # Fetch data
        data = fetch_historical_data(period=test_period)
        
        if data is None:
            logger.error("Failed to fetch data")
            return None
        
        # Load model
        self.strategy.load_model()
        
        # Generate signals
        signals = self.strategy.generate_signals(data)
        
        if signals is None:
            logger.error("Failed to generate signals")
            return None
        
        # Run backtest
        results = self.backtest.run(data, signals)
        
        if results is None:
            logger.error("Failed to run backtest")
            return None
        
        # Calculate metrics
        metrics = self.backtest.calculate_metrics()
        
        if metrics is None:
            logger.error("Failed to calculate metrics")
            return None
        
        # Log metrics
        logger.info("Accuracy metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Plot results
        fig = self.backtest.plot_results()
        
        if fig is not None:
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"../results/accuracy_test_{timestamp}.png")
            logger.info(f"Accuracy test plot saved to ../results/accuracy_test_{timestamp}.png")
        
        return metrics
    
    def test_optimization(self):
        """
        Test optimization techniques to improve performance
        
        Returns:
            dict: Dictionary with optimization results
        """
        logger.info("Testing optimization techniques")
        
        # Fetch data
        data = fetch_historical_data(period='1y')
        
        if data is None:
            logger.error("Failed to fetch data")
            return None
        
        # Load model
        self.strategy.load_model()
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 50, 100]
        batch_times = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Split data into batches
            n_samples = len(data)
            n_batches = n_samples // batch_size
            
            start_time = time.time()
            
            for i in range(n_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_samples)
                batch_data = data.iloc[batch_start:batch_end]
                
                # Generate signals for batch
                signals = self.strategy.generate_signals(batch_data)
            
            end_time = time.time()
            batch_times.append(end_time - start_time)
            
            logger.info(f"Batch size {batch_size} took {batch_times[-1]:.6f} seconds")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, batch_times, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Batch Size vs. Processing Time')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"../results/optimization_test_{timestamp}.png")
        logger.info(f"Optimization test plot saved to ../results/optimization_test_{timestamp}.png")
        
        # Find optimal batch size
        optimal_index = np.argmin(batch_times)
        optimal_batch_size = batch_sizes[optimal_index]
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        return {
            'batch_sizes': batch_sizes,
            'batch_times': batch_times,
            'optimal_batch_size': optimal_batch_size
        }

def main():
    """
    Main function to run performance tests
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the performance of the trading system')
    parser.add_argument('--latency', action='store_true', help='Test latency')
    parser.add_argument('--accuracy', action='store_true', help='Test accuracy')
    parser.add_argument('--optimization', action='store_true', help='Test optimization')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations for latency test')
    parser.add_argument('--period', type=str, default='6mo', help='Period for accuracy test')
    args = parser.parse_args()
    
    tester = PerformanceTester()
    
    if args.all or args.latency:
        logger.info("Running latency test")
        tester.test_latency(iterations=args.iterations)
    
    if args.all or args.accuracy:
        logger.info("Running accuracy test")
        tester.test_accuracy(test_period=args.period)
    
    if args.all or args.optimization:
        logger.info("Running optimization test")
        tester.test_optimization()
    
    if not (args.latency or args.accuracy or args.optimization or args.all):
        parser.print_help()

if __name__ == "__main__":
    main()
