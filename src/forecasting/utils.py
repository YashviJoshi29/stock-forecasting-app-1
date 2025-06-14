def create_time_series_features(data):
    # Function to create time series features from the stock price data
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=21).std()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Momentum'] = data['Close'].diff(4)
    return data

def aggregate_sentiment_scores(sentiment_data):
    # Function to aggregate sentiment scores by date
    aggregated_scores = sentiment_data.groupby('date')['score'].mean().reset_index()
    return aggregated_scores