# Test script: Feature engineering and merging sentiment with stock data
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
import pandas as pd
from src.data.fetch_stock_data import fetch_stock_prices
from src.data.fetch_news import fetch_news_headlines
from src.sentiment.utils import clean_headlines
from src.sentiment.analyzer import SentimentAnalyzer, FinBERTSentimentModel
from src.forecasting.utils import create_time_series_features, aggregate_sentiment_scores

# Load API key from .env
load_dotenv()
api_key = os.getenv("NEWSAPI_KEY")

# Parameters
stock_ticker = "AAPL"
from_date = "2025-05-12"  # Use earliest allowed date for your NewsAPI plan
to_date = "2025-06-12"

# Fetch stock data
stock_df = fetch_stock_prices(stock_ticker, from_date, to_date)

# Fetch news headlines with dates
topic = "Apple"
news_items = fetch_news_headlines(api_key, topic, from_date, to_date, with_dates=True)
if news_items:
    headlines, news_dates = zip(*news_items)
else:
    headlines, news_dates = [], []

# Clean headlines
cleaned = clean_headlines(headlines)

# Run FinBERT sentiment analysis
finbert_model = FinBERTSentimentModel()
analyzer = SentimentAnalyzer(finbert_model)
scores = analyzer.analyze_sentiment(cleaned)

# Create sentiment DataFrame with actual news dates
sentiment_df = pd.DataFrame({
    'date': news_dates,
    'score': scores
})

# Aggregate sentiment by date
agg_sentiment = aggregate_sentiment_scores(sentiment_df)

# Reset index to avoid merge errors
stock_df = stock_df.reset_index(drop=True)
# Flatten MultiIndex columns if present
if isinstance(stock_df.columns, pd.MultiIndex):
    stock_df.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in stock_df.columns.values]
print("stock_df columns after flatten:", stock_df.columns)
# Ensure the date column is named 'Date'
if 'Date' not in stock_df.columns:
    stock_df = stock_df.rename(columns={stock_df.columns[0]: 'Date'})
# Rename columns to remove ticker suffix for feature engineering
for col in stock_df.columns:
    if col.endswith(f'_{stock_ticker}'):
        stock_df = stock_df.rename(columns={col: col.replace(f'_{stock_ticker}', '')})
print("stock_df columns after rename:", stock_df.columns)
# Merge with stock data
stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')
merged = pd.merge(stock_df, agg_sentiment, on='date', how='left')

# Create time series features
final_df = create_time_series_features(merged)

print(f"final_df shape after feature engineering: {final_df.shape}")
print(final_df.head())
print(f"\nColumns: {final_df.columns.tolist()}")
print(f"\nRows: {len(final_df)}")
