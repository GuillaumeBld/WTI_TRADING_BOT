#!/usr/bin/env python
"""
Script to fetch news articles from NewsAPI for WTI crude oil prices.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from newsapi import NewsApiClient
from typing import List, Dict
import json

def fetch_crude_oil_news(query="WTI crude oil prices", days_back=1) -> List[Dict]:
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        print("Error: NEWSAPI_KEY not set in .env.")
        sys.exit(1)

    newsapi = NewsApiClient(api_key=api_key)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    all_articles = []
    page = 1

    while True:
        try:
            response = newsapi.get_everything(
                q=query,
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                language="en",
                sort_by="relevancy",  # Prioritize relevance
                page_size=100,       # Max for free tier
                page=page
            )
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

        articles = response.get("articles", [])
        if not articles:
            break

        all_articles.extend(articles)
        page += 1

        # Ensure totalResults exists before attempting division
        total_results = response.get("totalResults")
        if total_results is not None and page > (total_results // 100) + 1:
            break

    if not all_articles:
        print(f"No articles found for '{query}' from {start_date} to {end_date}.")
        return []

    for article in all_articles:
        article["fetch_timestamp"] = datetime.now().isoformat()

    df = pd.DataFrame(all_articles)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df.dropna(subset=["publishedAt"], inplace=True)  # Ensure valid dates only
    articles = df[["title", "description", "publishedAt", "source", "fetch_timestamp"]].to_dict(orient="records")
    
    for article in articles:
        article["publishedAt"] = article["publishedAt"].isoformat()

    print(f"Fetched {len(articles)} articles.")
    return articles[:20]  # Return top 20 by relevance

if __name__ == "__main__":
    articles = fetch_crude_oil_news()
    if articles:
        with open("news_archive.json", "w") as f:
            json.dump(articles, f, indent=2)