"""
Crude Oil Sentiment Analysis Bot

This bot analyzes sentiment from news articles related to WTI crude oil
and generates trading signals based on the sentiment analysis.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from newsapi import NewsApiClient
from transformers import pipeline
import sqlite3
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrudeOilSentimentBot:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY'))
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.db_path = "data/market_data.db"
        
    def fetch_news(self, query: str = "crude oil WTI", days: int = 7) -> List[Dict]:
        """Fetch news articles related to crude oil"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            articles = self.newsapi.get_everything(
                q=query,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            return articles.get('articles', [])
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a text using DistilBERT"""
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

    def save_results(self, results: List[Dict]):
        """Save sentiment analysis results to SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        source TEXT,
                        title TEXT,
                        sentiment_label TEXT,
                        sentiment_score REAL
                    )
                ''')
                
                for result in results:
                    cursor.execute('''
                        INSERT INTO sentiment_results 
                        (date, source, title, sentiment_label, sentiment_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        result['date'],
                        result['source'],
                        result['title'],
                        result['sentiment']['label'],
                        result['sentiment']['score']
                    ))
                conn.commit()
            logger.info(f"Saved {len(results)} sentiment results to database")
        except Exception as e:
            logger.error(f"Error saving results to database: {e}")

    def run(self):
        """Main execution method"""
        logger.info("Starting sentiment analysis")
        
        # Fetch news articles
        articles = self.fetch_news()
        if not articles:
            logger.warning("No articles found")
            return

        # Analyze sentiment for each article
        results = []
        for article in articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.analyze_sentiment(text)
            results.append({
                'date': article['publishedAt'],
                'source': article['source']['name'],
                'title': article['title'],
                'sentiment': sentiment
            })

        # Save results
        self.save_results(results)
        logger.info("Sentiment analysis complete")

if __name__ == "__main__":
    bot = CrudeOilSentimentBot()
    bot.run()
