# Test script: Train and evaluate forecasting model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from src.forecasting.model import ForecastingModel
from src.forecasting.test_feature_engineering import final_df

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
    sys.exit(1)
