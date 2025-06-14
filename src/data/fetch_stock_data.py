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