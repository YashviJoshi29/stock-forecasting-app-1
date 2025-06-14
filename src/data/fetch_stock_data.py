import yfinance as yf

def fetch_stock_prices(ticker, start_date, end_date):
    """
    Fetch historical stock prices for a given ticker symbol between specified dates.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple).
    start_date (str): The start date for fetching stock prices in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching stock prices in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical stock prices.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data.reset_index()

if __name__ == "__main__":
    import sys
    import pandas as pd
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    print(f"Fetching stock prices for {ticker} from {start_date} to {end_date}...")
    df = fetch_stock_prices(ticker, start_date, end_date)
    print(df.head())
    print(f"\nFetched {len(df)} rows.")