def clean_headlines(headlines):
    # Function to clean news headlines
    cleaned_headlines = []
    for headline in headlines:
        if headline is None:
            continue  # Skip None headlines
        # Basic cleaning steps
        cleaned = headline.strip().lower()
        # Optionally, remove stopwords, punctuation, etc. for better results
        cleaned_headlines.append(cleaned)
    return cleaned_headlines

def align_dates(news_dates, stock_dates):
    # Function to align news dates with stock dates
    aligned_dates = []
    for date in stock_dates:
        if date in news_dates:
            aligned_dates.append(date)
    return aligned_dates