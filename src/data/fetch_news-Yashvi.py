from datetime import datetime

def fetch_news_headlines(api_key, query, from_date, to_date, with_dates=False):
    import requests
    # NewsAPI free plan date limit (update as needed)
    MIN_DATE = "2025-05-14"  # Set to one day after the last allowed date in the error
    # Ensure from_date and to_date are not before MIN_DATE
    from_date = max(str(from_date), MIN_DATE)
    to_date = max(str(to_date), MIN_DATE)

    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        news_data = response.json()
        if with_dates:
            # Return list of (headline, publishedAt) tuples
            return [(article['title'], article['publishedAt'][:10]) for article in news_data['articles']]
        else:
            headlines = [article['title'] for article in news_data['articles']]
            return headlines
    else:
        raise Exception(f"Error fetching news: {response.status_code} - {response.text}")


if __name__ == "__main__":
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional for this test
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("NEWSAPI_KEY not found in environment. Please set it in a .env file or your environment variables.")
    else:
        query = "Apple"
        from_date = "2024-01-01"
        to_date = "2024-01-07"
        print(f"Fetching news headlines for '{query}' from {from_date} to {to_date}...")
        headlines = fetch_news_headlines(api_key, query, from_date, to_date)
        print(f"Fetched {len(headlines)} headlines. Sample:")
        for h in headlines[:5]:
            print("-", h)