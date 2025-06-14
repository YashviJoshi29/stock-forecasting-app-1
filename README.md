# Stock Price Forecasting System with News Sentiment Analysis

## Overview
This project implements a stock price forecasting system that integrates news sentiment analysis to enhance prediction accuracy. The application is built using Streamlit for an interactive user interface, allowing users to visualize stock price trends and sentiment analysis results.

## Project Structure
```
stock-forecasting-app
├── src
│   ├── app.py
│   ├── forecasting
│   │   ├── model.py
│   │   └── utils.py
│   ├── sentiment
│   │   ├── analyzer.py
│   │   └── utils.py
│   ├── data
│   │   ├── fetch_stock_data.py
│   │   └── fetch_news.py
│   └── config.py
├── requirements.txt
├── README.md
└── architecture.md
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-forecasting-app.git
   cd stock-forecasting-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Features
- **Stock Price Forecasting**: Utilizes an AutoML pipeline to train models and predict future stock prices based on historical data.
- **News Sentiment Analysis**: Analyzes recent financial news headlines to gauge market sentiment, which is factored into the forecasting model.
- **Interactive Visualization**: Provides visual insights into stock price trends and sentiment scores.

## Key Components
- **Data Fetching**: Historical stock prices are fetched using the yFinance API, and recent news headlines are retrieved using NewsAPI.
- **Forecasting Model**: Implements a robust forecasting model that leverages historical stock data and sentiment scores for predictions.
- **Sentiment Analysis**: Utilizes FinBERT for analyzing the sentiment of news headlines, providing sentiment scores that influence stock price predictions.

## Deployment
The application is designed to be easily deployable on platforms that support Streamlit applications. Ensure that all environment variables and API keys are configured in the `src/config.py` file before deployment.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.