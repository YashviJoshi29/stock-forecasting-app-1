import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from data.fetch_stock_data import fetch_stock_prices
from data.fetch_news import fetch_news_headlines
from sentiment.utils import clean_headlines
from sentiment.analyzer import SentimentAnalyzer, FinBERTSentimentModel
from forecasting.utils import create_time_series_features, aggregate_sentiment_scores
from forecasting.model import ForecastingModel
import requests

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

st.title("Stock Price Forecasting with News Sentiment")

# Sidebar inputs
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = 'AAPL'
if 'news_topic' not in st.session_state:
    st.session_state['news_topic'] = 'Apple'
ticker = st.sidebar.text_input("Stock Ticker", value=st.session_state['ticker'], key="main_ticker")
news_topic = st.sidebar.text_input("News Topic", value=st.session_state['news_topic'], key="main_news_topic")
from_date = st.sidebar.date_input("From Date", pd.to_datetime("2025-05-14"))
to_date = st.sidebar.date_input("To Date", pd.to_datetime("2025-06-12"))

# --- Suggested Stock Tickers and News Topics Section ---
with st.sidebar.expander("💡 Suggested Tickers & News Topics", expanded=False):
    region = st.radio("Select Ticker Region", ["US", "Indian"], key="ticker_region")
    us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "V"]
    india_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "LT.NS"]
    if region == "US":
        selected_ticker = st.selectbox("Pick a US Ticker", us_tickers, key="us_ticker")
    else:
        selected_ticker = st.selectbox("Pick an Indian Ticker", india_tickers, key="in_ticker")
    st.markdown("**Trending News Topics:**")
    topics = ["Earnings", "Mergers", "Layoffs", "AI", "Inflation", "Interest Rates", "Dividends", "Buyback", "IPO", "Regulation"]
    selected_topic = st.selectbox("Pick a News Topic", topics, key="topic")
    if st.button("Use Suggested Ticker/Topic"):
        st.session_state['ticker'] = selected_ticker
        st.session_state['news_topic'] = selected_topic
        st.success(f"Set ticker to {selected_ticker} and topic to {selected_topic}")

# --- Trending topics section ---
with st.sidebar.expander("Trending News Topics (last 7 days)"):
    trending_url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=100&apiKey={NEWSAPI_KEY}"
    try:
        if NEWSAPI_KEY:
            resp = requests.get(trending_url)
            if resp.status_code == 200:
                articles = resp.json().get('articles', [])
                # Extract words from titles and descriptions, filter out short/generic words
                all_text = ' '.join([
                    (a.get('title') or '') + ' ' + (a.get('description') or '')
                    for a in articles
                ])
                # Extract capitalized words (likely topics), min length 3
                import re
                words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', all_text)
                # Remove common stopwords and generic words
                stopwords = set(['The', 'This', 'That', 'With', 'From', 'Will', 'Have', 'Has', 'For', 'And', 'But', 'Are', 'Was', 'You', 'Your', 'More', 'Over', 'After', 'Into', 'Who', 'Why', 'How', 'Out', 'All', 'New', 'Top', 'One', 'Two', 'Day', 'Week', 'Year', 'News', 'Says', 'Said', 'On', 'In', 'At', 'By', 'Of', 'To', 'As', 'It', 'Be', 'Is', 'A', 'An'])
                topics = pd.Series([w for w in words if w not in stopwords]).value_counts().head(10)
                if not topics.empty:
                    st.write("**Top Trending Topics:**")
                    for topic, count in topics.items():
                        st.write(f"{topic}: {count} articles")
                    selected = st.selectbox("Pick a trending topic to use:", topics.index.tolist(), key="trending_topic")
                    if st.button("Use this topic", key="use_trending_topic"):
                        st.session_state['news_topic'] = selected
                        st.success(f"Set news topic to trending topic: {selected}")
                else:
                    st.write("No trending topics found in the last 7 days.")
            else:
                st.write(f"Could not fetch trending topics. Status code: {resp.status_code}")
        else:
            st.write("NEWSAPI_KEY not set.")
    except Exception as e:
        st.write(f"Error fetching trending topics: {e}")

if st.sidebar.button("Run Forecast"):
    with st.spinner("Fetching stock data..."):
        stock_df = fetch_stock_prices(ticker, str(from_date), str(to_date))
        stock_df = stock_df.reset_index(drop=True)
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in stock_df.columns.values]
        if 'Date' not in stock_df.columns:
            stock_df = stock_df.rename(columns={stock_df.columns[0]: 'Date'})
        for col in stock_df.columns:
            if col.endswith(f'_{ticker}'):
                stock_df = stock_df.rename(columns={col: col.replace(f'_{ticker}', '')})
        stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')

    with st.spinner("Fetching news headlines and running sentiment analysis..."):
        news_items = fetch_news_headlines(NEWSAPI_KEY, news_topic, str(from_date), str(to_date), with_dates=True)
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
        sentiment_df = pd.DataFrame({'date': filtered_dates, 'score': scores})
        agg_sentiment = aggregate_sentiment_scores(sentiment_df)

    with st.spinner("Merging data and creating features..."):
        merged = pd.merge(stock_df, agg_sentiment, on='date', how='left')
        # Ensure 'Close' column exists (remove ticker suffix if needed)
        if 'Close' not in merged.columns:
            for col in merged.columns:
                if col.startswith('Close'):
                    merged = merged.rename(columns={col: 'Close'})
        final_df = create_time_series_features(merged)
        # --- Robust missing data handling (adaptive) ---
        sentiment_cols = [col for col in final_df.columns if 'sentiment' in col.lower() or 'score' in col.lower()]
        # Merge duplicate date columns into a single 'Date' column for clarity
        if 'date' in final_df.columns and 'Date' in final_df.columns:
            final_df['Date'] = final_df['Date'].combine_first(final_df['date'])
            final_df = final_df.drop(columns=['date'])
        elif 'date' in final_df.columns:
            final_df = final_df.rename(columns={'date': 'Date'})
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
        # Only drop rows if essential price features or target are missing
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

    # --- Advanced Modeling & Comparison Section ---
    with st.expander("🔬 Advanced: AutoML, Walk-Forward, Baseline vs Sentiment", expanded=False):
        st.markdown("""
        This section compares a baseline (price-only) model to a sentiment-enhanced model using AutoML and walk-forward validation.
        """)
        from forecasting.automl_utils import automl_search, walk_forward_validation
        # Baseline: Only price features
        price_features = [col for col in final_df.columns if col not in ['Date', 'score', 'sentiment', 'Return'] and final_df[col].dtype != 'O']
        sentiment_features = price_features + [col for col in final_df.columns if 'score' in col.lower() or 'sentiment' in col.lower()]
        target = 'Return'
        data = final_df.dropna(subset=sentiment_features + [target])
        if len(data) >= 30:
            X_price = data[price_features]
            X_sent = data[sentiment_features]
            y = data[target]
            st.write("**AutoML: Searching for best baseline (price-only) model...")
            base_name, base_model, base_params, base_score = automl_search(X_price, y)
            st.write(f"Best Baseline Model: {base_name} {base_params}, MSE: {base_score:.4f}")
            st.write("**AutoML: Searching for best sentiment-enhanced model...")
            sent_name, sent_model, sent_params, sent_score = automl_search(X_sent, y)
            st.write(f"Best Sentiment Model: {sent_name} {sent_params}, MSE: {sent_score:.4f}")
            st.write("**Walk-forward validation (sentiment model):**")
            preds, actuals, mse, r2 = walk_forward_validation(sent_model, X_sent, y, window=20)
            st.write(f"Walk-forward MSE: {mse:.4f}, R²: {r2:.4f}")
            st.line_chart(pd.DataFrame({'Actual': actuals, 'Predicted': preds}))
            # Feature importance for sentiment model
            if hasattr(sent_model, 'feature_importances_'):
                importances = sent_model.feature_importances_
                feature_importance = pd.Series(importances, index=X_sent.columns).sort_values(ascending=False)
                st.subheader("Feature Importance (Sentiment Model)")
                st.bar_chart(feature_importance)
                st.markdown("**Top features contribute most to the model's predictions.**")
            # Baseline vs Sentiment comparison
            st.write(f"**Improvement from sentiment features:** Baseline MSE: {base_score:.4f} → Sentiment MSE: {sent_score:.4f}")
        else:
            st.info("Not enough data for advanced comparison (need at least 30 rows after NA handling). Try a wider date range.")
    st.success("Done!")
else:
    st.info("Set your parameters and click 'Run Forecast' to begin.")