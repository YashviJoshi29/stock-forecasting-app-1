import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def automl_search(X, y):
    """Try several models and hyperparameters, return the best."""
    models = {
        'RandomForest': (RandomForestRegressor(), {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}),
        'GradientBoosting': (GradientBoostingRegressor(), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}),
        'Ridge': (Ridge(), {'alpha': [0.1, 1, 10]}),
        'Lasso': (Lasso(), {'alpha': [0.1, 1, 10]})
    }
    best_score = float('inf')
    best_model = None
    best_name = None
    best_params = None
    tscv = TimeSeriesSplit(n_splits=3)
    for name, (model, params) in models.items():
        gscv = GridSearchCV(model, params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        gscv.fit(X, y)
        score = -gscv.best_score_
        if score < best_score:
            best_score = score
            best_model = gscv.best_estimator_
            best_name = name
            best_params = gscv.best_params_
    return best_name, best_model, best_params, best_score

def walk_forward_validation(model, X, y, window=20):
    """Walk-forward validation for time series."""
    preds = []
    actuals = []
    for i in range(window, len(X)):
        model.fit(X.iloc[:i], y.iloc[:i])
        pred = model.predict(X.iloc[[i]])[0]
        preds.append(pred)
        actuals.append(y.iloc[i])
    mse = mean_squared_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    return preds, actuals, mse, r2
