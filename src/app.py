import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from src.data.fetch_stock_data import fetch_stock_prices
from src.data.fetch_news import fetch_news_headlines
from src.sentiment.utils import clean_headlines
from src.sentiment.analyzer import SentimentAnalyzer, FinBERTSentimentModel
from src.forecasting.utils import create_time_series_features, aggregate_sentiment_scores
from src.forecasting.model import ForecastingModel

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
        cleaned = clean_headlines(headlines)
        finbert_model = FinBERTSentimentModel()
        analyzer = SentimentAnalyzer(finbert_model)
        scores = analyzer.analyze_sentiment(cleaned)
        sentiment_df = pd.DataFrame({'date': news_dates, 'score': scores})
        agg_sentiment = aggregate_sentiment_scores(sentiment_df)

    with st.spinner("Merging data and creating features..."):
        merged = pd.merge(stock_df, agg_sentiment, on='date', how='left')
        # Ensure 'Close' column exists (remove ticker suffix if needed)
        if 'Close' not in merged.columns:
            for col in merged.columns:
                if col.startswith('Close'):
                    merged = merged.rename(columns={col: 'Close'})
        final_df = create_time_series_features(merged)
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

    st.success("Done!")
else:
    st.info("Set your parameters and click 'Run Forecast' to begin.")