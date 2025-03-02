"""
Sentiment Analysis Module - Simulated Version

This module simulates sentiment analysis for crude oil news.
In a real implementation, this would use Perplexica with Llama 3.2 (3B)
running locally through Ollama.
"""

import os
import json
import random
from datetime import datetime, timedelta

class SentimentAnalyzer:
    """
    Class for analyzing sentiment of crude oil news (simulated)
    """
    
    def __init__(self, model_name="llama3.2:3b"):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name (str): Name of the Ollama model to use (for simulation only)
        """
        self.model_name = model_name
        print(f"Sentiment Analyzer initialized with model: {model_name}")
        print("NOTE: This is a simulated version. In a real implementation, this would use Perplexica with Ollama.")
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a text (simulated)
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        # Simple keyword-based sentiment analysis
        positive_keywords = ["rise", "increase", "gain", "growth", "positive", "bullish", "up", "higher", "surge"]
        negative_keywords = ["fall", "decrease", "decline", "drop", "negative", "bearish", "down", "lower", "plunge"]
        
        # Count occurrences of positive and negative keywords
        positive_count = sum(1 for keyword in positive_keywords if keyword in text.lower())
        negative_count = sum(1 for keyword in negative_keywords if keyword in text.lower())
        
        # Determine sentiment based on keyword counts
        if positive_count > negative_count:
            sentiment = "positive"
            # Score between 0.3 and 0.8
            score = 0.3 + (0.5 * (positive_count / (positive_count + negative_count + 0.1)))
            reasoning = f"Found {positive_count} positive keywords and {negative_count} negative keywords."
        elif negative_count > positive_count:
            sentiment = "negative"
            # Score between -0.3 and -0.8
            score = -0.3 - (0.5 * (negative_count / (positive_count + negative_count + 0.1)))
            reasoning = f"Found {negative_count} negative keywords and {positive_count} positive keywords."
        else:
            sentiment = "neutral"
            # Score between -0.2 and 0.2
            score = random.uniform(-0.2, 0.2)
            reasoning = "No clear sentiment detected."
        
        return {
            "sentiment": sentiment,
            "score": score,
            "reasoning": reasoning
        }
    
    def fetch_crude_oil_news(self, days=1, api_key=None):
        """
        Fetch recent news about crude oil
        
        Args:
            days (int): Number of days to look back
            api_key (str): API key for news API (optional)
            
        Returns:
            list: List of news articles
        """
        # For demo purposes, return some sample news
        # In a real implementation, this would use a news API
        
        sample_news = [
            {
                "title": "Crude Oil Prices Rise as OPEC+ Considers Production Cuts",
                "description": "Crude oil prices increased by 2% today as OPEC+ members discuss potential production cuts to stabilize the market amid growing concerns about global demand.",
                "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "source": "Sample News Source"
            },
            {
                "title": "US Crude Inventories Show Unexpected Decline",
                "description": "The Energy Information Administration reported a 3.2 million barrel decrease in US crude inventories last week, contrary to analysts' expectations of a 1.5 million barrel increase.",
                "published_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                "source": "Sample News Source"
            },
            {
                "title": "Global Economic Concerns Weigh on Oil Markets",
                "description": "Crude oil futures fell today as investors worry about slowing economic growth in major economies, potentially reducing demand for oil in the coming months.",
                "published_at": (datetime.now() - timedelta(hours=18)).isoformat(),
                "source": "Sample News Source"
            }
        ]
        
        return sample_news
    
    def analyze_news_sentiment(self, news_articles):
        """
        Analyze sentiment of multiple news articles
        
        Args:
            news_articles (list): List of news articles
            
        Returns:
            dict: Aggregated sentiment analysis results
        """
        if not news_articles:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "articles": []
            }
        
        results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_score = 0.0
        
        for article in news_articles:
            # Combine title and description for analysis
            text = f"{article['title']}. {article['description']}"
            
            # Analyze sentiment
            sentiment_result = self.analyze_text(text)
            
            # Count sentiments
            if sentiment_result["sentiment"] == "positive":
                positive_count += 1
            elif sentiment_result["sentiment"] == "negative":
                negative_count += 1
            else:
                neutral_count += 1
            
            # Add to total score
            total_score += sentiment_result["score"]
            
            # Add result to list
            results.append({
                "title": article["title"],
                "published_at": article["published_at"],
                "source": article["source"],
                "sentiment": sentiment_result["sentiment"],
                "score": sentiment_result["score"],
                "reasoning": sentiment_result.get("reasoning", "")
            })
        
        # Calculate average score
        average_score = total_score / len(news_articles)
        
        # Determine overall sentiment
        if average_score > 0.2:
            overall_sentiment = "positive"
        elif average_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": average_score,
            "article_count": len(news_articles),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "articles": results
        }
    
    def get_trading_signal_from_sentiment(self, sentiment_results):
        """
        Generate a trading signal based on sentiment analysis
        
        Args:
            sentiment_results (dict): Sentiment analysis results
            
        Returns:
            dict: Trading signal
        """
        average_score = sentiment_results["average_score"]
        
        # Generate signal based on sentiment score
        if average_score > 0.5:
            signal = 1  # Strong buy
            confidence = min(0.5 + abs(average_score) / 2, 0.95)
        elif average_score > 0.2:
            signal = 0.5  # Weak buy
            confidence = 0.5 + abs(average_score) / 2
        elif average_score < -0.5:
            signal = -1  # Strong sell
            confidence = min(0.5 + abs(average_score) / 2, 0.95)
        elif average_score < -0.2:
            signal = -0.5  # Weak sell
            confidence = 0.5 + abs(average_score) / 2
        else:
            signal = 0  # Hold
            confidence = 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "sentiment_score": average_score,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """
    Main function to analyze crude oil news sentiment
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze crude oil news sentiment')
    parser.add_argument('--days', type=int, default=1, help='Number of days to look back for news')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--model', type=str, default='llama3.2:3b', help='Ollama model to use')
    args = parser.parse_args()
    
    print("WTI Crude Oil Trading System - Sentiment Analysis")
    print("================================================")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(model_name=args.model)
    
    # Fetch news
    print(f"\nFetching crude oil news for the past {args.days} day(s)...")
    news = analyzer.fetch_crude_oil_news(days=args.days)
    
    if not news:
        print("No news articles found.")
        return
    
    print(f"Found {len(news)} news articles.")
    
    # Analyze sentiment
    print("\nAnalyzing sentiment...")
    sentiment_results = analyzer.analyze_news_sentiment(news)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print(f"Overall Sentiment: {sentiment_results['overall_sentiment']}")
    print(f"Average Score: {sentiment_results['average_score']:.2f}")
    print(f"Articles: {sentiment_results['article_count']} (Positive: {sentiment_results['positive_count']}, Negative: {sentiment_results['negative_count']}, Neutral: {sentiment_results['neutral_count']})")
    
    # Generate trading signal
    trading_signal = analyzer.get_trading_signal_from_sentiment(sentiment_results)
    
    # Print trading signal
    print("\nTrading Signal:")
    signal_type = "BUY" if trading_signal['signal'] > 0 else "SELL" if trading_signal['signal'] < 0 else "HOLD"
    signal_strength = "STRONG" if abs(trading_signal['signal']) == 1 else "WEAK" if abs(trading_signal['signal']) == 0.5 else ""
    print(f"Signal: {signal_strength} {signal_type}")
    print(f"Confidence: {trading_signal['confidence']:.2f}")
    
    # Save results if requested
    if args.save:
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/sentiment_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "sentiment_results": sentiment_results,
                "trading_signal": trading_signal
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    print("\nSentiment analysis complete!")

if __name__ == "__main__":
    main()
