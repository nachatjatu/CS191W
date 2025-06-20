from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.base import clone
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm.notebook import tqdm


def rolling_window_cv(tscv, unique_dates, df, features, model, target_col):
    """
    Performs rolling window cross-validation for time series data.

    Args:
        tscv (sklearn.model_selection.BaseCrossValidator): A time series cross-validator 
            (e.g., TimeSeriesSplit) that yields indices for rolling window splits.
        unique_dates (list): list of unique, sorted timestamps used 
            to split the dataset into training and validation windows.
        df (pd.DataFrame): Dataset containing `time`, feature, and target columns.
        features (list[str]): List of column names to be used as features.
        model (sklearn.base.BaseEstimator): A scikit-learn estimator.
        target_col (str): Name of the column to be predicted.

    Returns:
        tuple of lists:
            - rmse_scores (list[np.float]): RMSE scores for each validation window.
            - mae_scores (list[np.float]): MAE scores for each validation window.
    """
    rmse_scores = []
    mae_scores = []
    n_splits = tscv.get_n_splits(unique_dates)
    
    for train_idxs, val_idxs in tqdm(tscv.split(unique_dates), total=n_splits, leave=False):
        train_dates, val_dates = unique_dates[train_idxs], unique_dates[val_idxs]
        train_df, val_df = df[df["time"].isin(train_dates)], df[df["time"].isin(val_dates)]

        X_train, y_train = train_df[features], train_df[target_col]
        X_val, y_val = val_df[features], val_df[target_col]

        # create a fresh copy of the model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        y_preds = model_clone.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_preds)
        mae = mean_absolute_error(y_val, y_preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    return rmse_scores, mae_scores



def run_experiments(df, model, feature_experiments, train_days, gap_days, val_days):
    """Runs a series of feature experiments and reports results (RMSE, MAE)

    Args:
        df (pd.DataFrame): Pandas DataFrame containing training and test data
        model (sklearn.base.BaseEstimator): A scikit-learn estimator.
        feature_experiments (list[(str, list[str])]): list containing tuples 
            with name of feature experiment and a list of features to include in the experiment
        train_days (int): width of training window (how many days to train on)
        gap_days (int): width of gap (how many days to leave)
        val_days (int): width of validation window (how many days to evaluate on)

    Returns:
        pd.DataFrame: DataFrame of results for each experiment, with mean RMSE, MAE
        and window-specific RMSE scores.
    """
    unique_dates = df['time'].unique()
    
    tscv = TimeSeriesSplit(
        n_splits=10, 
        max_train_size=train_days,
        test_size=val_days, 
        gap=gap_days
    )

    results = []
    
    for experiment_name, features in tqdm(
        feature_experiments, total=len(feature_experiments), leave=False):
        print(f"Running experiment: {experiment_name}")

        rmse_scores, mae_scores = rolling_window_cv(
            tscv,
            unique_dates,
            df=df,
            features=features,
            model=model,
            target_col='delta_pm25_t+1'
        )

        mean_rmse = np.mean(rmse_scores)
        mean_mae = np.mean(mae_scores)

        exp_results = {
            'experiment': experiment_name,
            'n_features': len(features),
            'mean_rmse': mean_rmse,
            'mean_mae': mean_mae
        }

        for fold_idx, score in enumerate(rmse_scores, start=1):
            exp_results[f'rmse_fold_{fold_idx}'] = score

        results.append(exp_results)

    return pd.DataFrame(results)

    
