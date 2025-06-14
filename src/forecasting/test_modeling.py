# Test script: Train and evaluate forecasting model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from src.forecasting.model import ForecastingModel
from src.forecasting.test_feature_engineering import final_df

# --- Robust missing data handling (consistent with feature engineering pipeline) ---
# Merge duplicate date columns into a single 'Date' column for clarity
if 'date' in final_df.columns and 'Date' in final_df.columns:
    # Prefer 'Date' if both exist, else use whichever exists
    final_df['Date'] = final_df['Date'].combine_first(final_df['date'])
    final_df = final_df.drop(columns=['date'])
elif 'date' in final_df.columns:
    final_df = final_df.rename(columns={'date': 'Date'})

# Identify sentiment columns (commonly named 'sentiment', 'score', or similar)
sentiment_cols = [col for col in final_df.columns if 'sentiment' in col.lower() or 'score' in col.lower()]

# Handle missing values based on dataset size
if len(final_df) < 100:
    # Small dataset: fill missing sentiment as neutral (0)
    for col in sentiment_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)
    # Optionally, fill other missing values if needed (e.g., forward fill)
    final_df = final_df.fillna(method='ffill').fillna(method='bfill')
else:
    # Large dataset: drop rows with missing key features (sentiment, features, target)
    key_cols = sentiment_cols + [col for col in final_df.columns if col not in sentiment_cols]
    final_df = final_df.dropna(subset=key_cols)

# Prepare data: drop NA, select features and target
features = [col for col in final_df.columns if col not in ['Date', 'date', 'score', 'Return'] and final_df[col].dtype != 'O']
target = 'Return'
data = final_df.dropna(subset=features + [target])

X = data[features]
y = data[target]

# Simple train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train and evaluate model
try:
    model = ForecastingModel()
    model.train_model(X_train, y_train)
    results = model.evaluate_model(X_test, y_test)
    print("Model evaluation:", results)

    # Predict and show sample
    preds = model.predict(X_test)
    print("\nSample predictions:")
    print(pd.DataFrame({'Actual': y_test, 'Predicted': preds}).head())
except Exception as e:
    print(f"Error during modeling: {e}")
