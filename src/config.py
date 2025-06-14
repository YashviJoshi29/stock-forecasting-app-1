import os

API_KEY = "your_api_key_here"
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
STOCK_SYMBOL = "AAPL"  # Default stock symbol
FORECAST_DAYS = 30  # Number of days to forecast
SENTIMENT_THRESHOLD = 0.1  # Threshold for sentiment analysis to consider significant
DATA_FETCH_INTERVAL = "1d"  # Interval for fetching stock data
MODEL_SAVE_PATH = "models/forecasting_model.pkl"  # Path to save the trained model
LOGGING_LEVEL = "INFO"  # Logging level for the application