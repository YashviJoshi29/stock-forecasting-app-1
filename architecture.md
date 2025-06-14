# Stock Price Forecasting System Architecture Document

## Overview
This document outlines the architecture of the Stock Price Forecasting System that integrates news sentiment analysis. The system is designed to predict stock prices based on historical data and sentiment derived from financial news headlines. The application is deployed using Streamlit, providing an interactive user interface.

## Data Sources
1. **Stock Price Data**: Historical stock prices are fetched using the yFinance API.
2. **News Headlines**: Recent financial news headlines are retrieved using the NewsAPI.

## Data Flow
1. **Data Fetching**:
   - The application fetches historical stock prices through `fetch_stock_data.py`.
   - It retrieves recent news headlines using `fetch_news.py`.

2. **Sentiment Analysis**:
   - News headlines are processed by the `SentimentAnalyzer` class in `analyzer.py`, which uses FinBERT to analyze sentiment and generate sentiment scores.

3. **Feature Engineering**:
   - The sentiment scores are aggregated and aligned with stock price data using utility functions in `utils.py` within the forecasting module.

4. **Model Training and Prediction**:
   - The `ForecastingModel` class in `model.py` implements an AutoML pipeline for model selection and training. It utilizes the processed data to train the model and make predictions.

5. **User Interface**:
   - The Streamlit app in `app.py` serves as the entry point, integrating all components and presenting the results to the user.

## Key Components
- **Streamlit App**: `src/app.py`
- **Forecasting Module**: 
  - Model: `src/forecasting/model.py`
  - Utilities: `src/forecasting/utils.py`
- **Sentiment Analysis Module**: 
  - Analyzer: `src/sentiment/analyzer.py`
  - Utilities: `src/sentiment/utils.py`
- **Data Fetching Module**: 
  - Stock Data: `src/data/fetch_stock_data.py`
  - News Data: `src/data/fetch_news.py`
- **Configuration**: `src/config.py`

## Deployment Plan
The application will be deployed using Streamlit, allowing users to interact with the forecasting model and view sentiment analysis results in real-time. The deployment will be hosted on a cloud platform that supports Streamlit applications.

## Selected Tools and APIs
- **yFinance**: For fetching historical stock prices.
- **NewsAPI**: For retrieving financial news headlines.
- **FinBERT**: For performing sentiment analysis on news headlines.
- **Auto-sklearn**: For automated machine learning model selection and training.
- **Streamlit**: For building the interactive web application.

## Conclusion
This architecture document provides a comprehensive overview of the Stock Price Forecasting System, detailing its components, data flow, and deployment strategy. The integration of news sentiment analysis enhances the predictive capabilities of the system, making it a valuable tool for investors and analysts.