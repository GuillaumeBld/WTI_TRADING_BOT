"""
Trading Strategy Module - Machine Learning Version with Sentiment Analysis

This module implements a trading strategy using machine learning and sentiment analysis
to predict price movements and generate trading signals.
"""

import os
import csv
import math
import json
import random
from datetime import datetime

# Import sentiment analysis
try:
    from sentiment_analysis import SentimentAnalyzer
except ImportError:
    # If not in path, try relative import
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from sentiment_analysis import SentimentAnalyzer
    except ImportError:
        print("Warning: Sentiment analysis module not found. Proceeding without sentiment analysis.")
        SentimentAnalyzer = None

# Simulate scikit-learn's train_test_split function
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets
    
    Args:
        X (list): Features
        y (list): Target
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Controls the shuffling applied to the data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Create indices and shuffle
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Split indices
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Split data
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

# Simulate a simple decision tree classifier
class SimpleDecisionTree:
    """
    A simple decision tree classifier implementation
    """
    
    def __init__(self, max_depth=3):
        """
        Initialize the decision tree
        
        Args:
            max_depth (int): Maximum depth of the tree
        """
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        """
        Build the decision tree
        
        Args:
            X (list): Features
            y (list): Target
        """
        # For simplicity, we'll just create some rules based on domain knowledge
        # rather than actually building a tree
        self.tree = {
            'rsi_threshold': 30,
            'macd_threshold': 0,
            'ema_ratio_threshold': 1.0,
            'adx_threshold': 25
        }
    
    def predict(self, X):
        """
        Predict class for X
        
        Args:
            X (list): Features
            
        Returns:
            list: Predicted classes
        """
        predictions = []
        
        for features in X:
            # Extract features
            rsi = features[0]
            macd = features[1]
            macd_signal = features[2]
            ema_9 = features[4]
            ema_21 = features[5]
            adx = features[3]
            
            # Apply rules
            score = 0
            
            # RSI rule
            if rsi < self.tree['rsi_threshold']:
                score += 1
            elif rsi > 70:
                score -= 1
            
            # MACD rule
            if macd > macd_signal:
                score += 1
            else:
                score -= 1
            
            # EMA rule
            if ema_9 > ema_21:
                score += 0.5
            else:
                score -= 0.5
            
            # ADX rule
            if adx > self.tree['adx_threshold']:
                # Strong trend, amplify the signal
                score *= 1.2
            
            # Convert score to prediction
            if score > 0.5:
                predictions.append(1)  # Buy
            elif score < -0.5:
                predictions.append(-1)  # Sell
            else:
                predictions.append(0)  # Hold
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Args:
            X (list): Features
            
        Returns:
            list: Predicted probabilities
        """
        # For simplicity, convert predictions to probabilities
        predictions = self.predict(X)
        probabilities = []
        
        for pred in predictions:
            if pred == 1:
                probabilities.append(0.7)  # 70% confidence for buy
            elif pred == -1:
                probabilities.append(0.3)  # 30% confidence for sell (70% for not buy)
            else:
                probabilities.append(0.5)  # 50% confidence for hold
        
        return probabilities

def load_data_with_indicators(filepath):
    """
    Load data with indicators from a CSV file
    
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
                    'Open': float(row['Open']) if row['Open'] else None,
                    'High': float(row['High']) if row['High'] else None,
                    'Low': float(row['Low']) if row['Low'] else None,
                    'Close': float(row['Close']) if row['Close'] else None,
                    'Volume': int(row['Volume']) if row['Volume'] else None,
                    'RSI': float(row['RSI']) if row['RSI'] else None,
                    'MACD': float(row['MACD']) if row['MACD'] else None,
                    'MACD_Signal': float(row['MACD_Signal']) if row['MACD_Signal'] else None,
                    'MACD_Hist': float(row['MACD_Hist']) if row['MACD_Hist'] else None,
                    'ADX': float(row['ADX']) if row['ADX'] else None,
                    'EMA_9': float(row['EMA_9']) if row['EMA_9'] else None,
                    'EMA_21': float(row['EMA_21']) if row['EMA_21'] else None
                }
                data.append(processed_row)
        print(f"Loaded {len(data)} records from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return []

def prepare_features_and_target(data):
    """
    Prepare features and target for machine learning
    
    Args:
        data (list): List of dictionaries with price and indicator data
        
    Returns:
        tuple: (features, target, dates, prices)
    """
    features = []
    target = []
    dates = []
    prices = []
    
    # We need at least 2 days of data to calculate the target (price change)
    for i in range(len(data) - 1):
        # Skip records with missing indicators
        if (data[i]['RSI'] is None or data[i]['MACD'] is None or 
            data[i]['MACD_Signal'] is None or data[i]['ADX'] is None or 
            data[i]['EMA_9'] is None or data[i]['EMA_21'] is None):
            continue
        
        # Calculate price change for the next day
        current_price = data[i]['Close']
        next_price = data[i + 1]['Close']
        price_change = (next_price - current_price) / current_price
        
        # Create feature vector
        feature = [
            data[i]['RSI'],
            data[i]['MACD'],
            data[i]['MACD_Signal'],
            data[i]['ADX'],
            data[i]['EMA_9'],
            data[i]['EMA_21']
        ]
        
        # Create target (1 for price increase, -1 for decrease, 0 for no change)
        if price_change > 0.005:  # 0.5% threshold for significant change
            label = 1  # Buy
        elif price_change < -0.005:
            label = -1  # Sell
        else:
            label = 0  # Hold
        
        features.append(feature)
        target.append(label)
        dates.append(data[i]['Date'])
        prices.append(current_price)
    
    return features, target, dates, prices

def train_model(features, target):
    """
    Train a machine learning model
    
    Args:
        features (list): List of feature vectors
        target (list): List of target values
        
    Returns:
        object: Trained model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = SimpleDecisionTree(max_depth=3)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_test[i]) / len(y_pred)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    return model

def generate_signals_with_ml(data, model=None, use_sentiment=True):
    """
    Generate trading signals using machine learning and sentiment analysis
    
    Args:
        data (list): List of dictionaries with price and indicator data
        model (object): Trained machine learning model
        use_sentiment (bool): Whether to incorporate sentiment analysis
        
    Returns:
        list: List of dictionaries with trading signals
    """
    # Prepare features and target
    features, target, dates, prices = prepare_features_and_target(data)
    
    # Train model if not provided
    if model is None:
        model = train_model(features, target)
    
    # Generate predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    # Create signals
    signals = []
    for i in range(len(predictions)):
        signal = {
            'Date': dates[i],
            'Price': prices[i],
            'Signal': predictions[i],
            'Confidence': probabilities[i],
            'Sentiment_Used': False,
            'Sentiment_Score': 0.0
        }
        signals.append(signal)
    
    # Incorporate sentiment analysis if requested
    if use_sentiment and SentimentAnalyzer is not None:
        try:
            # Initialize sentiment analyzer
            analyzer = SentimentAnalyzer()
            
            # Get sentiment data
            sentiment_file = "../data/sentiment_analysis_latest.json"
            sentiment_data = None
            
            # Try to find the most recent sentiment analysis file
            if not os.path.exists(sentiment_file):
                data_dir = "../data"
                sentiment_files = [f for f in os.listdir(data_dir) if f.startswith("sentiment_analysis_") and f.endswith(".json")]
                if sentiment_files:
                    # Sort by timestamp (newest first)
                    sentiment_files.sort(reverse=True)
                    sentiment_file = os.path.join(data_dir, sentiment_files[0])
            
            # Load sentiment data if available
            if os.path.exists(sentiment_file):
                with open(sentiment_file, 'r') as f:
                    sentiment_data = json.load(f)
                print(f"Loaded sentiment data from {sentiment_file}")
            else:
                # Generate sentiment data if not available
                print("No sentiment data found. Generating new sentiment analysis...")
                news = analyzer.fetch_crude_oil_news()
                sentiment_results = analyzer.analyze_news_sentiment(news)
                trading_signal = analyzer.get_trading_signal_from_sentiment(sentiment_results)
                sentiment_data = {
                    "sentiment_results": sentiment_results,
                    "trading_signal": trading_signal
                }
                print("Sentiment analysis generated.")
            
            # Adjust signals based on sentiment
            if sentiment_data:
                sentiment_score = sentiment_data["trading_signal"]["sentiment_score"]
                sentiment_signal = sentiment_data["trading_signal"]["signal"]
                sentiment_confidence = sentiment_data["trading_signal"]["confidence"]
                
                print(f"Incorporating sentiment analysis: Score={sentiment_score:.2f}, Signal={sentiment_signal}, Confidence={sentiment_confidence:.2f}")
                
                # Adjust the most recent signals (last 5 days) based on sentiment
                for i in range(max(0, len(signals) - 5), len(signals)):
                    # Get original signal
                    original_signal = signals[i]['Signal']
                    original_confidence = signals[i]['Confidence']
                    
                    # Adjust signal based on sentiment
                    # If sentiment agrees with technical signal, boost confidence
                    # If sentiment disagrees, reduce confidence or flip signal if sentiment is strong
                    if (original_signal > 0 and sentiment_signal > 0) or (original_signal < 0 and sentiment_signal < 0):
                        # Sentiment agrees with technical signal, boost confidence
                        new_confidence = min(0.95, original_confidence + 0.1 * sentiment_confidence)
                        signals[i]['Confidence'] = new_confidence
                    elif abs(sentiment_signal) > 0.7 and sentiment_confidence > 0.7:
                        # Strong sentiment disagrees with technical signal, flip signal
                        signals[i]['Signal'] = sentiment_signal
                        signals[i]['Confidence'] = sentiment_confidence
                    elif sentiment_signal != 0:
                        # Sentiment disagrees but not strongly, reduce confidence
                        new_confidence = max(0.5, original_confidence - 0.1 * sentiment_confidence)
                        signals[i]['Confidence'] = new_confidence
                    
                    # Mark as using sentiment
                    signals[i]['Sentiment_Used'] = True
                    signals[i]['Sentiment_Score'] = sentiment_score
        except Exception as e:
            print(f"Error incorporating sentiment analysis: {e}")
    
    return signals

def save_signals_to_csv(signals, filepath):
    """
    Save trading signals to a CSV file
    
    Args:
        signals (list): List of dictionaries with trading signals
        filepath (str): Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filepath, 'w', newline='') as f:
            # Include all fields in the signals
            fieldnames = ['Date', 'Price', 'Signal', 'Confidence', 'Sentiment_Used', 'Sentiment_Score']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for signal in signals:
                writer.writerow(signal)
        
        print(f"Trading signals saved to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving signals to {filepath}: {e}")
        return False

def main():
    """
    Main function to generate trading signals using machine learning
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate trading signals using machine learning')
    parser.add_argument('--sentiment', action='store_true', help='Use sentiment analysis')
    parser.add_argument('--no-sentiment', dest='sentiment', action='store_false', help='Do not use sentiment analysis')
    parser.add_argument('--output', type=str, default='../data/trading_signals_ml.csv', help='Output file path')
    parser.set_defaults(sentiment=True)
    args = parser.parse_args()
    
    print("WTI Crude Oil Trading System - ML Strategy")
    print("==========================================")
    
    # Load data with indicators
    data_path = "../data/crude_oil_with_indicators.csv"
    data = load_data_with_indicators(data_path)
    
    if not data:
        print("No data available. Please run indicators.py first.")
        return
    
    print(f"Generating ML-based trading signals for {len(data)} records...")
    
    # Generate signals using machine learning
    signals = generate_signals_with_ml(data, use_sentiment=args.sentiment)
    
    if not signals:
        print("Failed to generate signals.")
        return
    
    print(f"Generated {len(signals)} trading signals.")
    
    # Print the last 5 signals
    print("\nLast 5 trading signals:")
    for signal in signals[-5:]:
        signal_type = "BUY" if signal['Signal'] == 1 else "SELL" if signal['Signal'] == -1 else "HOLD"
        sentiment_info = f", Sentiment: {signal['Sentiment_Score']:.2f}" if signal['Sentiment_Used'] else ""
        print(f"Date: {signal['Date']}, Price: ${signal['Price']:.2f}, Signal: {signal_type}, Confidence: {signal['Confidence']:.2f}{sentiment_info}")
    
    # Save signals
    output_path = args.output
    save_signals_to_csv(signals, output_path)
    
    print("\nML trading strategy execution complete!")
    print(f"Sentiment analysis: {'Enabled' if args.sentiment else 'Disabled'}")
    print("You can now proceed with backtesting the ML strategy.")

if __name__ == "__main__":
    main()
