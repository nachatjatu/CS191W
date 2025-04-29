from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import numpy as np
import pandas as pd


def rolling_window_cv(train_days, gap_days, test_days, df, model, target_col='delta_pm25_t+1'):
    dates = df['date'].unique()
    train_start = 0
    num_days = len(dates)
    i = 1
    rmse_scores, mae_scores = [], []

    while True:
        # Compute start and end dates
        train_end = train_start + train_days
        test_start = train_end + gap_days
        test_end = test_start + test_days

        if test_end > num_days:
            break

        train_dates = dates[train_start: train_end]
        test_dates = dates[test_start: test_end]

        print(f'Fold {i}')
        print(train_dates.min(), train_dates.max())
        print(test_dates.min(), test_dates.max())

        # Generate train and val sets by dropping resp. columns
        train_data =  df[df['date'].isin(train_dates)]
        test_data = df[df['date'].isin(test_dates)]
        X_train, y_train = train_data.drop(['row', 'col', 'date', target_col], axis=1), train_data[target_col]
        X_val, y_val = test_data.drop(['row', 'col', 'date', target_col], axis=1), test_data[target_col]

        # Clone input model (fresh one) and fit to data, predict on validation set
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_preds = model_clone.predict(X_val)

        # Evaluate model performance w.r.t. evaluation metrics (RMSE, MAE)
        rmse = root_mean_squared_error(y_val, y_preds)
        mae = mean_absolute_error(y_val, y_preds)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        train_start += test_days
        i += 1

    return rmse_scores, mae_scores