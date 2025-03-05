"""
Enhanced Sentiment Analysis Module for WTI Crude Oil Trading

This module analyzes sentiment from news articles related to WTI crude oil
using DistilBERT and generates trading signals based on the sentiment analysis.
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

class SentimentAnalyzer:
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

    def generate_trading_signal(self, sentiment_score: float) -> Dict:
        """Generate trading signal based on sentiment score"""
        if sentiment_score > 0.7:
            return {'signal': 'BUY', 'confidence': sentiment_score}
        elif sentiment_score < 0.3:
            return {'signal': 'SELL', 'confidence': 1 - sentiment_score}
        else:
            return {'signal': 'HOLD', 'confidence': 0.5}

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
    bot = SentimentAnalyzer()
    bot.run()

#!/usr/bin/env python
"""
Crude Oil Sentiment Analysis Bot with DistilBERT, News Archive, Knowledge Base, and Enhanced Features

Fetches WTI crude oil news from NewsAPI (past 24 hours) or via command-line arguments,
uses DistilBERT for sentiment analysis, extracts key events with refined priorities, and suggests trading
decisions with context from previous news and a knowledge base (energy_knowledge.json in trading_bot/data/).
Logs results to sentiment_log.txt in the trading_bot root directory with detailed per-article data.
Stores sentiment data in an SQLite database (market_data.db in trading_bot/data/).
"""

import os
import sys
import json
import argparse
import logging
import time
import re
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
from newsapi import NewsApiClient
from transformers import pipeline
from typing import List, Dict, Optional

# Set up logging for debugging, including cron-specific logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify virtual environment and dependencies
try:
    import pandas
    import newsapi
    import torch
    import transformers
    import sqlite3
    logger.info("All required dependencies are installed and accessible.")
except ImportError as e:
    logger.error(f"Missing required dependency: {e}. Please install all required packages.")
    sys.exit(1)

# Simple text preprocessing function
def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load knowledge base from JSON in the data directory once, with default content if missing
def load_knowledge_base(filename="energy_knowledge.json") -> Dict:
    data_path = os.path.join("..", "data", filename)
    try:
        with open(data_path, "r") as f:
            knowledge = json.load(f)
        if not knowledge or not knowledge.get("wti_crude_oil") or not knowledge.get("energy_market"):
            logger.warning(f"Knowledge base in {data_path} is incomplete. Populating with default content.")
            default_kb = {
                "wti_crude_oil": {
                    "definition": "West Texas Intermediate (WTI) is a grade of crude oil used as a benchmark in oil pricing.",
                    "characteristics": {
                        "type": "Light sweet crude oil",
                        "api_gravity": "Approximately 39.6 degrees",
                        "sulfur_content": "Approximately 0.24%",
                        "refining": "Easier and less costly to refine",
                        "delivery_point": "Cushing, Oklahoma, USA"
                    },
                    "production": {
                        "2025_production_estimate": "Approximately 13.4 million barrels per day (EIA projection)"
                    }
                },
                "energy_market": {
                    "crude_oil_segment": {
                        "global_demand": "Approximately 102.3 million barrels per day (IEA projection)"
                    },
                    "price_influencing_factors": {
                        "supply": [
                            "OPEC+ production quotas",
                            "U.S. shale oil output",
                            "Geopolitical disruptions",
                            "Natural disasters affecting production"
                        ],
                        "demand": [
                            "Global economic growth",
                            "Seasonal variations",
                            "Industrial activity",
                            "Energy transition policies"
                        ]
                    }
                }
            }
            with open(data_path, "w") as f:
                json.dump(default_kb, f, indent=2)
            return default_kb
        logger.info(f"Loaded knowledge base from {data_path}.")
        return knowledge
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}. Using default knowledge base.")
        default_kb = {"wti_crude_oil": {}, "energy_market": {}}
        with open(data_path, "w") as f:
            json.dump(default_kb, f, indent=2)
        return default_kb

# Initialize DistilBERT sentiment classifier
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased",
    framework="pt"
)

# Custom scoring function for article relevance
def score_article(article: Dict, kb: Dict) -> int:
    if not kb or not kb.get("energy_market", {}).get("price_influencing_factors", {}):
        logger.warning("Knowledge base is empty or missing; using default scoring (0).")
        return 0
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    score = 0
    supply_factors = kb.get("energy_market", {}).get("price_influencing_factors", {}).get("supply", [])
    demand_factors = kb.get("energy_market", {}).get("price_influencing_factors", {}).get("demand", [])
    for factor in supply_factors + demand_factors:
        if factor.lower() in text:
            score += 1
    return score

# News fetching function with dynamic query expansion and exponential backoff
def fetch_crude_oil_news(query: str = "crude oil WTI prices market trends news",
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None,
                         days_back: int = 1) -> List[Dict]:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.error("NEWSAPI_KEY environment variable is not set.")
        sys.exit(1)

    # Validate API key using a test request
    try:
        newsapi = NewsApiClient(api_key=api_key)
        test_response = newsapi.get_sources()
        if test_response.get('status') == 'error':
            logger.error(f"API key validation failed: {test_response.get('message')}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to validate API key: {e}")
        sys.exit(1)

    logger.info(f"🔍 Fetching news with query: {query}, Start Date: {start_date}, End Date: {end_date}, API Key: {api_key[:4]}...")

    if start_date and end_date:
        start = start_date
        end = end_date
    else:
        end = datetime.now()
        start = end - timedelta(days=days_back)

    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []
    page = 1
    total_results = float('inf')
    max_retries = 3
    retry_delay = 10  # initial delay in seconds

    while page * 100 <= total_results:
        try:
            response = newsapi.get_everything(
                q=query,
                from_param=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                language="en",
                sort_by="relevancy",
                page_size=100,
                page=page
            )

            logger.debug(f"🚀 Raw API Response for page {page}:\n{json.dumps(response, indent=2)}")

            articles = response.get("articles", [])
            total_results = response.get("totalResults", 0)
            if not articles:
                break

            for article in articles:
                article["fetch_timestamp"] = datetime.now().isoformat()
            all_articles.extend(articles)

            if len(articles) < 100:
                break
            page += 1

        except Exception as e:
            logger.error(f"News fetch error on page {page}: {e}")
            # If rate limit error is detected, backoff and retry
            if "rateLimited" in str(e) or "429" in str(e):
                if max_retries > 0:
                    logger.info(f"Rate limit reached. Waiting for {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    max_retries -= 1
                    continue
                else:
                    logger.error("Exceeded maximum retries due to rate limiting.")
                    break
            else:
                break

    # If no articles found, try a broader query once
    if not all_articles:
        broader_query = "crude oil WTI OR Brent OR oil market"
        logger.warning(f"No articles found for query '{query}'. Retrying with broader query: '{broader_query}'")
        return fetch_crude_oil_news(query=broader_query, start_date=start_date, end_date=end_date, days_back=days_back)

    # Process articles with pandas to validate timestamps
    df = pd.DataFrame(all_articles)
    if not df.empty and "publishedAt" in df.columns:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors='coerce')
        articles = df[["title", "description", "publishedAt", "source", "fetch_timestamp"]].to_dict(orient="records")
        for article in articles:
            article["publishedAt"] = article["publishedAt"].isoformat() if pd.notna(article["publishedAt"]) else datetime.now().isoformat()
    else:
        logger.warning("No valid articles found or DataFrame is empty.")
        return []

    kb = load_knowledge_base()
    articles = sorted(articles, key=lambda x: score_article(x, kb), reverse=True)
    logger.info(f"✅ Fetched {len(articles)} articles; returning top 20 most relevant.")
    if len(articles) < 10:
        logger.warning(f"Low article count ({len(articles)}). Consider broadening the query or checking NewsAPI limits.")
    return articles[:20]

# Save news to JSON archive in the data directory
def save_news_to_json(articles: List[Dict], filename="news_archive.json"):
    data_path = os.path.join("..", "data", filename)
    try:
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                archive = json.load(f)
        else:
            archive = []

        seen = set()
        unique_articles = []
        for article in articles:
            key = (article.get("title", ""), article.get("publishedAt", ""), article.get("fetch_timestamp", ""))
            if key not in seen:
                seen.add(key)
                unique_articles.append(article)

        # Ensure unique fetch_timestamp by adding a slight offset if necessary
        for i, article in enumerate(unique_articles):
            base_time = datetime.fromisoformat(article["fetch_timestamp"])
            article["fetch_timestamp"] = (base_time + timedelta(milliseconds=i)).isoformat()

        archive.extend(unique_articles)
        with open(data_path, "w") as f:
            json.dump(archive, f, indent=2)
        logger.info(f"Saved {len(unique_articles)} unique articles to {data_path}. Total archive size: {len(archive)}")
    except Exception as e:
        logger.error(f"Error saving news to JSON: {e}")

# Load previous news from JSON archive in the data directory
def load_previous_news(filename="news_archive.json", lookback_days=7) -> List[Dict]:
    data_path = os.path.join("..", "data", filename)
    try:
        if not os.path.exists(data_path):
            logger.warning("No previous news archive found.")
            return []
        with open(data_path, "r") as f:
            archive = json.load(f)
        if not archive:
            logger.warning(f"News archive at {data_path} is empty.")
            return []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        lookback_1day = datetime.now() - timedelta(hours=24)
        previous_articles = []
        skipped_articles = 0
        for article in archive:
            try:
                fetch_time = datetime.fromisoformat(article.get("fetch_timestamp", datetime.now().isoformat()))
                pub_time = datetime.fromisoformat(article.get("publishedAt", fetch_time.isoformat()))
                if lookback_days == 1 and fetch_time > lookback_1day and fetch_time < datetime.now():
                    previous_articles.append(article)
                elif fetch_time > cutoff_date and fetch_time < datetime.now() - timedelta(hours=24):
                    previous_articles.append(article)
            except Exception as e:
                logger.warning(f"Skipping invalid article in archive: {article}. Error: {e}")
                skipped_articles += 1
                continue
        logger.info(f"Loaded {len(previous_articles)} previous articles from the past {lookback_days} days. Skipped articles: {skipped_articles}")
        return previous_articles
    except json.JSONDecodeError:
        logger.error(f"Error: {data_path} is corrupt or invalid. Starting fresh.")
        return []
    except Exception as e:
        logger.error(f"Error loading previous news: {e}")
        return []

# Sentiment analysis with preprocessing
def analyze_sentiment(text: str) -> str:
    if not text or pd.isna(text):
        logger.warning("No text available for sentiment analysis, returning Neutral.")
        return "Neutral (score: 0.0) - No text available."
    try:
        preprocessed_text = preprocess_text(text)
        result = sentiment_classifier(preprocessed_text[:512])[0]
        label = result["label"]
        score = result["score"]
        if label == "POSITIVE":
            sentiment_score = score if score > 0.6 else score * 0.5
            sentiment = "Positive" if score > 0.6 else "Neutral"
        else:
            sentiment_score = -score if score > 0.6 else -score * 0.5
            sentiment = "Negative" if score > 0.6 else "Neutral"
        logger.debug(f"Analyzed sentiment for text: {preprocessed_text[:50]}... -> {sentiment} (score: {sentiment_score:.2f})")
        return f"{sentiment} (score: {sentiment_score:.2f}) - {'Optimism' if sentiment == 'Positive' else 'Pessimism' if sentiment == 'Negative' else 'Balanced'} detected."
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return "Neutral (score: 0.0) - Analysis failed."

# Enhanced event extraction with knowledge base integration
def extract_events(text: str, knowledge_base: Dict) -> Dict:
    if not text or pd.isna(text):
        return {"event": "No significant events detected.", "priority": 0}
    text = text.lower()
    if "opec" in text:
        return {"event": "Key event: OPEC-related announcement detected.", "priority": 3}
    elif any(word in text for word in ["geopolitical", "tension", "conflict", "sanctions"]):
        return {"event": "Key event: Geopolitical tensions reported.", "priority": 3}
    elif "inventory" in text or "cushing" in text or "stock" in text:
        return {"event": "Key event: Inventory change reported at Cushing or elsewhere.", "priority": 2}
    elif "production" in text and any(word in text for word in ["cut", "increase", "quota"]):
        return {"event": "Key event: Production change reported.", "priority": 2}
    elif "demand" in text and any(word in text for word in ["rise", "fall", "growth"]):
        return {"event": "Key event: Demand shift reported.", "priority": 1}
    elif "supply" in text and any(word in text for word in ["disrupt", "increase", "shale"]):
        return {"event": "Key event: Supply change reported.", "priority": 1}
    elif "price" in text and any(word in text for word in ["crash", "surge", "drop"]):
        return {"event": "Key event: Price movement reported.", "priority": 1}
    else:
        return {"event": "No significant events detected.", "priority": 0}

# Store sentiment data in SQLite database with validation and confirmation of inserted rows
def store_sentiment_in_db(sentiment_data: List[Dict], db_path: str = "market_data.db"):
    data_path = os.path.join("..", "data", db_path)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    try:
        with sqlite3.connect(data_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS sentiment_data (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                date TIMESTAMP,
                                source TEXT,
                                headline TEXT,
                                sentiment_label TEXT,
                                sentiment_score REAL,
                                event TEXT,
                                relevance_score INTEGER
                            )''')
            validated_data = []
            for data in sentiment_data:
                try:
                    dt = datetime.fromisoformat(data["date"].replace("Z", "+00:00")) if data["date"] else datetime.now()
                    source = data["source"][:255] if data["source"] else "Unknown"
                    headline = data["headline"][:1024] if data["headline"] else "No Title"
                    sentiment_label = data["sentiment_label"][:50] if data["sentiment_label"] else "Neutral"
                    sentiment_score = float(data["sentiment_score"]) if data["sentiment_score"] is not None else 0.0
                    event = data["event"][:255] if data["event"] else "None"
                    relevance_score = int(data["relevance_score"]) if data["relevance_score"] is not None else 0
                    validated_data.append((
                        dt.isoformat(),
                        source,
                        headline,
                        sentiment_label,
                        sentiment_score,
                        event,
                        relevance_score
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid data entry: {data}. Error: {e}")
                    continue

            if validated_data:
                logger.debug(f"📊 Data being inserted -> {validated_data[:2]}... (showing first 2 for brevity)")
                cur = conn.executemany('''INSERT INTO sentiment_data 
                                  (date, source, headline, sentiment_label, sentiment_score, event, relevance_score)
                                  VALUES (?, ?, ?, ?, ?, ?, ?)''', validated_data)
                conn.commit()
                inserted = conn.total_changes
                logger.info(f"Inserted {inserted} records into {data_path}.")
            else:
                logger.warning("No valid data to insert into the database.")
    except Exception as e:
        logger.error(f"Database error: {e}")

# Main agent class
class CrudeOilSentimentAgent:
    def __init__(self, knowledge_base: Dict):
        self.knowledge_base = knowledge_base
        self.fetch_news = fetch_crude_oil_news
        self.analyze_sentiment = analyze_sentiment
        self.extract_events = lambda text: extract_events(text, self.knowledge_base)

    def run_analysis(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, 
                     query: str = "crude oil WTI prices market trends news", save: bool = False):
        new_articles = self.fetch_news(query=query, start_date=start_date, end_date=end_date, 
                                       days_back=1 if not (start_date and end_date) else 0)
        if not new_articles:
            logger.warning("No new news available. Attempting to analyze previous articles.")
            previous_articles = load_previous_news(lookback_days=7)
            if not previous_articles:
                logger.warning("No articles available to analyze. Exiting.")
                sys.exit(1)
            return self.analyze_previous_articles(previous_articles, save)

        save_news_to_json(new_articles)
        previous_articles = load_previous_news(lookback_days=7)
        all_articles = new_articles + previous_articles

        sentiments = []
        events = []
        relevance_scores = []
        for article in all_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            if article in new_articles:
                logger.info(f"New Article: {article.get('title', '')}")
            sentiment_result = self.analyze_sentiment(text)
            event_result = self.extract_events(text)
            relevance = score_article(article, self.knowledge_base)
            sentiments.append(sentiment_result)
            events.append(event_result)
            relevance_scores.append(relevance)
        
        combined = sorted(zip(all_articles, sentiments, events, relevance_scores), 
                          key=lambda x: (x[2]["priority"], x[3]), reverse=True)
        if combined:
            sorted_articles, sorted_sentiments, sorted_events, sorted_relevance = zip(*combined)
        else:
            sorted_articles, sorted_sentiments, sorted_events, sorted_relevance = [], [], [], []

        try:
            scores = [float(s.split("score: ")[1].split(")")[0]) for s in sorted_sentiments]
        except Exception as e:
            logger.error(f"Error parsing sentiment scores: {e}")
            scores = []
        avg_score = sum(scores) / len(scores) if scores else 0.0
        overall_sentiment = "Positive" if avg_score > 0.25 else "Negative" if avg_score < -0.25 else "Neutral"
        
        decision = "Hold"
        if avg_score > 0.25:
            decision = "Buy"
        elif avg_score < -0.25:
            decision = "Sell"
        all_text = " ".join([a.get('description', '') for a in sorted_articles]).lower()
        events_str = " ".join([e["event"] for e in sorted_events])
        if "opec" in events_str and "cut" in all_text:
            decision = "Buy"
        elif "geopolitical" in events_str and "tension" in all_text:
            decision = "Sell"
        elif "inventory" in events_str and "build" in all_text:
            decision = "Sell"
        elif "inventory" in events_str and "draw" in all_text:
            decision = "Buy"

        wti_info = self.knowledge_base.get("wti_crude_oil", {})
        market_info = self.knowledge_base.get("energy_market", {})
        context = (
            f"Context from Knowledge Base:\n"
            f"- WTI is {wti_info.get('characteristics', {}).get('type', 'Light sweet crude oil')} "
            f"with delivery at {wti_info.get('characteristics', {}).get('delivery_point', 'Cushing, Oklahoma, USA')}.\n"
            f"- 2025 production estimate: {wti_info.get('production', {}).get('2025_production_estimate', 'Approximately 13.4 million barrels per day (EIA projection)')}.\n"
            f"- Global demand: {market_info.get('crude_oil_segment', {}).get('global_demand', 'Approximately 102.3 million barrels per day (IEA projection)')}.\n"
            f"- Influenced by supply factors: {', '.join(market_info.get('price_influencing_factors', {}).get('supply', []))}\n"
            f"- Influenced by demand factors: {', '.join(market_info.get('price_influencing_factors', {}).get('demand', []))}"
        )

        sentiment_data = []
        for i, article in enumerate(sorted_articles):
            dt = article.get("publishedAt", datetime.now().isoformat())
            sentiment_parts = sorted_sentiments[i].split(" - ")
            sentiment_label = sentiment_parts[0].split(" (")[0]
            sentiment_score = float(sentiment_parts[0].split("score: ")[1].split(")")[0])
            sentiment_data.append({
                "date": dt,
                "source": article.get("source", {}).get("name", "Unknown"),
                "headline": article.get("title", "No Title"),
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "event": sorted_events[i]["event"],
                "relevance_score": sorted_relevance[i]
            })

        summary = (
            f"Summary - Analyzed {len(new_articles)} new articles and {len(previous_articles)} previous articles:\n"
            f"Total Articles: {len(sorted_articles)}\n"
            f"Overall Sentiment: {overall_sentiment}\n"
            f"Average Sentiment Score: {avg_score:.2f}\n"
            f"Trading Decision: {decision}\n"
            f"{context}"
        )
        
        detailed_log = "Detailed Log - Date,Headline,Sentiment Label,Sentiment Score,Event,Relevance Score\n"
        for data in sentiment_data:
            detailed_log += f"{data['date']},{data['headline']},{data['sentiment_label']},{data['sentiment_score']},{data['event']},{data['relevance_score']}\n"
        log_content = f"{summary}\n{detailed_log}\n{'-'*50}\n"
        
        log_path = os.path.join("..", "sentiment_log.txt")
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now()}:\n{log_content}")
            logger.info("Logged analysis results to sentiment_log.txt.")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
        
        if save:
            store_sentiment_in_db(sentiment_data, "market_data.db")
            logger.info("Stored sentiment data in SQLite database.")
        return summary

    def analyze_previous_articles(self, previous_articles: List[Dict], save: bool = False) -> str:
        if not previous_articles:
            logger.warning("No previous news available to analyze. Exiting.")
            sys.exit(1)
        sentiments = []
        events = []
        relevance_scores = []
        for article in previous_articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment_result = self.analyze_sentiment(text)
            event_result = self.extract_events(text)
            relevance = score_article(article, self.knowledge_base)
            sentiments.append(sentiment_result)
            events.append(event_result)
            relevance_scores.append(relevance)
        combined = sorted(zip(previous_articles, sentiments, events, relevance_scores), 
                          key=lambda x: (x[2]["priority"], x[3]), reverse=True)
        if combined:
            sorted_articles, sorted_sentiments, sorted_events, sorted_relevance = zip(*combined)
        else:
            sorted_articles, sorted_sentiments, sorted_events, sorted_relevance = [], [], [], []
        try:
            scores = [float(s.split("score: ")[1].split(")")[0]) for s in sorted_sentiments]
        except Exception as e:
            logger.error(f"Error parsing sentiment scores for previous articles: {e}")
            scores = []
        avg_score = sum(scores) / len(scores) if scores else 0.0
        overall_sentiment = "Positive" if avg_score > 0.25 else "Negative" if avg_score < -0.25 else "Neutral"
        decision = "Hold"
        if avg_score > 0.25:
            decision = "Buy"
        elif avg_score < -0.25:
            decision = "Sell"
        all_text = " ".join([a.get('description', '') for a in sorted_articles]).lower()
        events_str = " ".join([e["event"] for e in sorted_events])
        if "opec" in events_str and "cut" in all_text:
            decision = "Buy"
        elif "geopolitical" in events_str and "tension" in all_text:
            decision = "Sell"
        elif "inventory" in events_str and "build" in all_text:
            decision = "Sell"
        elif "inventory" in events_str and "draw" in all_text:
            decision = "Buy"

        wti_info = self.knowledge_base.get("wti_crude_oil", {})
        market_info = self.knowledge_base.get("energy_market", {})
        context = (
            f"Context from Knowledge Base:\n"
            f"- WTI is {wti_info.get('characteristics', {}).get('type', 'Light sweet crude oil')} "
            f"with delivery at {wti_info.get('characteristics', {}).get('delivery_point', 'Cushing, Oklahoma, USA')}.\n"
            f"- 2025 production estimate: {wti_info.get('production', {}).get('2025_production_estimate', 'Approximately 13.4 million barrels per day (EIA projection)')}.\n"
            f"- Global demand: {market_info.get('crude_oil_segment', {}).get('global_demand', 'Approximately 102.3 million barrels per day (IEA projection)')}.\n"
            f"- Influenced by supply factors: {', '.join(market_info.get('price_influencing_factors', {}).get('supply', []))}\n"
            f"- Influenced by demand factors: {', '.join(market_info.get('price_influencing_factors', {}).get('demand', []))}"
        )

        sentiment_data = []
        for i, article in enumerate(sorted_articles):
            dt = article.get("publishedAt", datetime.now().isoformat())
            sentiment_parts = sorted_sentiments[i].split(" - ")
            sentiment_label = sentiment_parts[0].split(" (")[0]
            sentiment_score = float(sentiment_parts[0].split("score: ")[1].split(")")[0])
            sentiment_data.append({
                "date": dt,
                "source": article.get("source", {}).get("name", "Unknown"),
                "headline": article.get("title", "No Title"),
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "event": sorted_events[i]["event"],
                "relevance_score": sorted_relevance[i]
            })

        summary = (
            f"Summary - Analyzed 0 new articles and {len(previous_articles)} previous articles:\n"
            f"Total Articles: {len(sorted_articles)}\n"
            f"Overall Sentiment: {overall_sentiment}\n"
            f"Average Sentiment Score: {avg_score:.2f}\n"
            f"Trading Decision: {decision}\n"
            f"{context}"
        )
        
        detailed_log = "Detailed Log - Date,Headline,Sentiment Label,Sentiment Score,Event,Relevance Score\n"
        for data in sentiment_data:
            detailed_log += f"{data['date']},{data['headline']},{data['sentiment_label']},{data['sentiment_score']},{data['event']},{data['relevance_score']}\n"
        log_content = f"{summary}\n{detailed_log}\n{'-'*50}\n"
        
        log_path = os.path.join("..", "sentiment_log.txt")
        try:
            with open(log_path, "a") as f:
                f.write(f"{datetime.now()}:\n{log_content}")
            logger.info("Logged analysis results of previous articles to sentiment_log.txt.")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
        
        if save:
            store_sentiment_in_db(sentiment_data, "market_data.db")
            logger.info("Stored sentiment data of previous articles in SQLite database.")
        return summary

def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError("Date must be in YYYY-MM-DD format")

def main():
    parser = argparse.ArgumentParser(
        description='Crude Oil Sentiment Analysis Bot with Enhanced Features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start_date', type=parse_date, help='Start date for news analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=parse_date, help='End date for news analysis (YYYY-MM-DD)')
    parser.add_argument('--query', default="crude oil WTI prices market trends news", help='Search query for news')
    parser.add_argument('--save', action='store_true', help='Store sentiment data in SQLite database')
    args = parser.parse_args()

    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.error("NEWSAPI_KEY environment variable is not set. Exiting.")
        sys.exit(1)
    logger.info(f"API key loaded: {api_key[:4]}... (partial for security)")

    knowledge_base = load_knowledge_base()
    agent = CrudeOilSentimentAgent(knowledge_base)
    result = agent.run_analysis(
        start_date=args.start_date,
        end_date=args.end_date,
        query=args.query,
        save=args.save
    )
    logger.info(result)
    sys.exit(0)

if __name__ == "__main__":
    main()
