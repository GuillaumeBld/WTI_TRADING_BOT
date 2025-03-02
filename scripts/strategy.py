"""
Trading Strategy Module

This script implements the trading strategy using machine learning models
and technical indicators.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from indicators import add_all_indicators

class TradingStrategy:
    """
    Trading strategy class that uses machine learning to generate trading signals
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the trading strategy
        
        Args:
            model_path (str): Path to a pre-trained model
        """
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_features(self, data):
        """
        Prepare features for the model
        
        Args:
            data (pandas.DataFrame): DataFrame with price and indicator data
            
        Returns:
            pandas.DataFrame: DataFrame with features
        """
        # Add technical indicators
        df = add_all_indicators(data)
        
        # Create features
        features = df[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ADX', 
                       'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200']].copy()
        
        # Add price-based features
        features['Price_Change'] = df['Close'].pct_change()
        features['Volatility'] = df['Close'].rolling(window=20).std()
        features['Volume_Change'] = df['Volume'].pct_change()
        
        # Add target (next day's price movement)
        features['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def train_model(self, data, test_size=0.2, random_state=42):
        """
        Train the machine learning model
        
        Args:
            data (pandas.DataFrame): DataFrame with price data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            float: Model accuracy on test set
        """
        # Prepare features
        features = self.prepare_features(data)
        
        # Split features and target
        X = features.drop('Target', axis=1)
        y = features['Target']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train LightGBM model
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_binary = np.where(y_pred > 0.5, 1, 0)
        accuracy = np.mean(y_pred_binary == y_test)
        
        print(f"Model accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def save_model(self, model_path='../models/trading_model.pkl', scaler_path='../models/scaler.pkl'):
        """
        Save the trained model and scaler
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            print("No model to save")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        return True
    
    def load_model(self, model_path='../models/trading_model.pkl', scaler_path='../models/scaler.pkl'):
        """
        Load a trained model and scaler
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_signals(self, data):
        """
        Generate trading signals based on model predictions
        
        Args:
            data (pandas.DataFrame): DataFrame with price data
            
        Returns:
            pandas.DataFrame: DataFrame with trading signals
        """
        if self.model is None:
            print("No model loaded")
            return None
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Get features without target
        X = features.drop('Target', axis=1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        predictions = self.model.predict(X_scaled)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=X.index)
        signals['Prediction'] = predictions
        signals['Signal'] = np.where(predictions > 0.5, 1, -1)
        
        return signals

def main():
    """
    Main function to train and test the trading strategy
    """
    import argparse
    import pandas as pd
    from data_fetch import fetch_historical_data
    
    parser = argparse.ArgumentParser(description='Train and test trading strategy')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()
    
    # Fetch data
    data = fetch_historical_data(period='5y')
    
    if data is None:
        print("Failed to fetch data")
        return
    
    # Create strategy
    strategy = TradingStrategy()
    
    if args.train:
        print("Training model...")
        strategy.train_model(data)
        strategy.save_model()
    
    if args.test:
        print("Testing model...")
        strategy.load_model()
        signals = strategy.generate_signals(data)
        
        if signals is not None:
            print(f"Generated {len(signals)} trading signals")
            print(signals.tail())

if __name__ == "__main__":
    main()
