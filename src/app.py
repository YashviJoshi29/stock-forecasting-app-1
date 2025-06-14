import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from data.fetch_stock_data import fetch_stock_prices
from data.fetch_news import fetch_news_headlines
from sentiment.utils import clean_headlines
from sentiment.analyzer import SentimentAnalyzer, FinBERTSentimentModel
from forecasting.utils import create_time_series_features, aggregate_sentiment_scores
from forecasting.model import ForecastingModel

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

st.title("Stock Price Forecasting with News Sentiment")

# Sidebar inputs
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
news_topic = st.sidebar.text_input("News Topic", value="Apple")
from_date = st.sidebar.date_input("From Date", pd.to_datetime("2025-05-12"))
to_date = st.sidebar.date_input("To Date", pd.to_datetime("2025-06-12"))

if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching stock data..."):
        stock_df = fetch_stock_prices(ticker, str(from_date), str(to_date))
        stock_df = stock_df.reset_index(drop=True)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in stock_df.columns.values]
        if 'Date' not in stock_df.columns:
            stock_df = stock_df.rename(columns={stock_df.columns[0]: 'Date'})
        # Robustly remove ticker suffixes (including .NS, .BO, etc.) from all columns except 'Date'
        import re
        def strip_suffix(col):
            if col == 'Date':
                return col
            # Remove everything after the first underscore and any trailing dot-suffix (e.g., .NS, .BO)
            base = col.split('_')[0]
            base = re.sub(r'\.[A-Z]+$', '', base)
            return base
        stock_df.columns = [strip_suffix(col) for col in stock_df.columns]
        stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')

    with st.spinner("Fetching news headlines and running sentiment analysis..."):
        news_items = fetch_news_headlines(NEWSAPI_KEY, news_topic, str(from_date), str(to_date), with_dates=True)
        num_headlines = len(news_items) if news_items else 0
        st.info(f"Fetched {num_headlines} news headlines for topic '{news_topic}' from {from_date} to {to_date}.")
        if news_items:
            headlines, news_dates = zip(*news_items)
        else:
            headlines, news_dates = [], []
        # Show the actual headlines in an expander for debug/insight
        with st.expander("Show fetched news headlines"):
            if headlines:
                for h, d in zip(headlines, news_dates):
                    st.write(f"{d}: {h}")
            else:
                st.write("No headlines found for this range and topic.")
        # Filter out None headlines and their dates
        filtered = [(h, d) for h, d in zip(headlines, news_dates) if h is not None]
        if filtered:
            filtered_headlines, filtered_dates = zip(*filtered)
        else:
            filtered_headlines, filtered_dates = [], []
        cleaned = clean_headlines(filtered_headlines)
        st.info(f"Sample cleaned headline: {cleaned[0] if cleaned else 'None'}")
        finbert_model = FinBERTSentimentModel()
        analyzer = SentimentAnalyzer(finbert_model)
        scores = analyzer.analyze_sentiment(cleaned)
        st.info(f"Sample sentiment scores: {scores[:5] if scores else 'None'}")
        sentiment_df = pd.DataFrame({'date': filtered_dates, 'score': scores})
        agg_sentiment = aggregate_sentiment_scores(sentiment_df)

    with st.spinner("Merging data and creating features..."):
        merged = pd.merge(stock_df, agg_sentiment, on='date', how='left')
        # --- Robust date column handling ---
        if 'date' in merged.columns and 'Date' in merged.columns:
            merged['Date'] = merged['Date'].combine_first(merged['date'])
            merged = merged.drop(columns=['date'])
        elif 'date' in merged.columns:
            merged = merged.rename(columns={'date': 'Date'})
        # Ensure 'Close' column exists (remove ticker suffix if needed)
        if 'Close' not in merged.columns:
            for col in merged.columns:
                if col.startswith('Close'):
                    merged = merged.rename(columns={col: 'Close'})
        final_df = create_time_series_features(merged)

        # --- Robust missing data handling (same as test_modeling.py) ---
        sentiment_cols = [col for col in final_df.columns if 'sentiment' in col.lower() or 'score' in col.lower()]
        if len(final_df) < 100:
            for col in sentiment_cols:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna(0)
            final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        else:
            key_cols = sentiment_cols + [col for col in final_df.columns if col not in sentiment_cols]
            final_df = final_df.dropna(subset=key_cols)

        st.subheader("Feature Data Preview")
        st.dataframe(final_df.head())

    with st.spinner("Training and evaluating model..."):
        features = [col for col in final_df.columns if col not in ['Date', 'date', 'score', 'Return'] and final_df[col].dtype != 'O']
        target = 'Return'
        data = final_df.dropna(subset=features + [target])
        if len(data) < 10:
            st.error("Not enough data for modeling. Try a wider date range or different ticker/topic.")
        else:
            X = data[features]
            y = data[target]
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            model = ForecastingModel()
            model.train_model(X_train, y_train)
            results = model.evaluate_model(X_test, y_test)
            st.subheader("Model Evaluation")
            st.write(results)
            preds = model.predict(X_test)
            st.subheader("Sample Predictions vs Actual")
            st.dataframe(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).head())
            st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': preds}))

            # Add explanation
            st.markdown("""
**How to interpret the results:**
- The 'Actual' column shows the real stock returns for each period in your test set.
- The 'Predicted' column shows the model's forecasted returns for the same periods.
- The closer the predicted values are to the actual values, the better the model performance.
- The line chart visualizes how well the model tracks real returns.
- Model evaluation metrics (MSE, R²) above indicate overall accuracy.
""")

            # Optional: Feature importance for RandomForest
            if hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
                st.subheader("Feature Importance")
                st.bar_chart(feature_importance)
                st.markdown("**Top features contribute most to the model's predictions.**")

    # --- Trending topics section ---
    with st.sidebar.expander("Trending News Topics (last 7 days)"):
        import requests
        trending_url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={NEWSAPI_KEY}"
        try:
            resp = requests.get(trending_url)
            if resp.status_code == 200:
                articles = resp.json().get('articles', [])
                topics = pd.Series([a['title'] for a in articles]).str.extractall(r'([A-Z][a-zA-Z]+)')[0].value_counts().head(10)
                st.write("**Top Trending Topics:**")
                for topic, count in topics.items():
                    st.write(f"{topic}: {count} articles")
                # Add a button to use a trending topic as the news_topic
                selected = st.selectbox("Pick a trending topic to use:", topics.index.tolist())
                if st.button("Use this topic"):
                    st.session_state['news_topic'] = selected
            else:
                st.write("Could not fetch trending topics.")
        except Exception as e:
            st.write(f"Error fetching trending topics: {e}")

    st.success("Done!")
else:
    st.info("Set your parameters and click 'Run Forecast' to begin.")