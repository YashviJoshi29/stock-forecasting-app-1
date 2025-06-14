# Test script: Fetch news, clean headlines, run FinBERT sentiment, print results
import os
from dotenv import load_dotenv
from src.data.fetch_news import fetch_news_headlines
from src.sentiment.utils import clean_headlines
from src.sentiment.analyzer import SentimentAnalyzer, FinBERTSentimentModel

# Load API key from .env
load_dotenv()
api_key = os.getenv("NEWSAPI_KEY")

# Fetch news headlines
today = "2025-06-12"
from_date = "2025-06-10"
to_date = today
query = "Apple"
headlines = fetch_news_headlines(api_key, query, from_date, to_date)
print(f"Fetched {len(headlines)} headlines for '{query}' from {from_date} to {to_date}.")

# Clean headlines
cleaned = clean_headlines(headlines)

# Run FinBERT sentiment analysis
finbert_model = FinBERTSentimentModel()
analyzer = SentimentAnalyzer(finbert_model)
scores = analyzer.analyze_sentiment(cleaned)

# Print results
for h, s in zip(headlines, scores):
    print(f"{s:+.2f} | {h}")

print(f"\nAverage sentiment: {analyzer.aggregate_sentiment(scores):+.2f}")
