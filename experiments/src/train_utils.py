from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
import numpy as np
import pandas as pd


def add_moving_averages(df, windows, vars):
    """
    Adds moving average columns over time for each variable.

    Args:
        df (pd.DataFrame): Input DataFrame with 'row', 'col', and 'date' columns.
        windows (int or List[int]): Moving average window sizes (e.g., 3 for 3-day average).
        vars (str or List[str]): Variables to compute moving averages for.

    Returns:
        pd.DataFrame: DataFrame with new moving average columns added.
    """

    if isinstance(vars, str):
        vars = [vars]
    if isinstance(windows, int):
        windows = [windows]

    df_result = df.copy()
    df_result = df_result.sort_values(['row', 'col', 'date'])

    for var in vars:
        for window in windows:
            ma_col = f"{var}_ma{window}"
            df_result[ma_col] = (
                df_result
                .groupby(['row', 'col'])[var]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )

    return df_result


def add_time_lags(df, lags, vars):
    """
    Adds time-lagged variables as additional DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame with 'row', 'col', and 'date' columns.
        lags (int or List[int]): Time lags to apply (e.g., 1 for 1-step lag).
        vars (str or List[str]): Variables to generate lags for.

    Returns:
        pd.DataFrame: DataFrame with new time-lagged features.
    """

    if isinstance(vars, str):
        vars = [vars]
    if isinstance(lags, int):
        lags = [lags]

    df_result = df.copy()
    df_result = df_result.sort_values(['row', 'col', 'date'])

    for var in vars:
        for lag in lags:
            lagged_col = f"{var}-lag{lag}"
            df_result[lagged_col] = df_result.groupby(['row', 'col'])[var].shift(lag)

    return df_result


def add_spatial_lags(df, lags, vars, pool='mean'):
    """
    Adds spatially-lagged variables as additional DataFrame columns.

    Args:
        df (pd.DataFrame): input DataFrame with 'row', 'col', and 'date' columns.
        lags (int or List[int]): odd integers for spatial neighborhood sizes (e.g., 3 for 3x3).
        vars (str or List[str]): variable(s) to compute spatial lags for.
        pool (str): aggregation method - 'mean', 'max', or 'none'. Defaults to 'mean'.

    Returns:
        pd.DataFrame: DataFrame with new spatial lag features.
    """

    if isinstance(vars, str):
        vars = [vars]
    if isinstance(lags, int):
        lags = [lags]

    for lag in lags:
        assert lag % 2 == 1, f"Lag {lag} must be an odd integer"
    assert pool in ['mean', 'max', 'none'], f"Invalid pool method: {pool}"

    df_result = df.copy()

    for lag in lags:
        half = lag // 2
        shifts = [(dr, dc) for dr in range(-half, half + 1)
                            for dc in range(-half, half + 1)
                            if not (dr == 0 and dc == 0)]

        for var in vars:
            neighbor_cols = []

            for dr, dc in shifts:
                shifted = df[['row', 'col', 'date', var]].copy()
                shifted['row'] += dr
                shifted['col'] += dc
                col_name = f"{var}_r{-dr}_c{-dc}"  # direction toward center
                shifted.rename(columns={var: col_name}, inplace=True)

                df_result = df_result.merge(shifted, on=['row', 'col', 'date'], how='left')
                neighbor_cols.append(col_name)

            if pool == 'mean':
                pooled_col = f"{var}_mean_{lag}x{lag}"
                df_result[pooled_col] = df_result[neighbor_cols].mean(axis=1)
                df_result.drop(columns=neighbor_cols, inplace=True)

            elif pool == 'max':
                pooled_col = f"{var}_max_{lag}x{lag}"
                df_result[pooled_col] = df_result[neighbor_cols].max(axis=1)
                df_result.drop(columns=neighbor_cols, inplace=True)

            # If pool == 'none', retain raw neighbor columns

    return df_result


def cv_fold_helper(df, dates, window, model, target_col):
    """
    Evaluates model performance on a single cross-validation fold.

    Args:
        df (pd.DataFrame): DataFrame containing covariates, target, and date columns.
        dates (np.ndarray or List[str]): Ordered array or list of unique date identifiers.
        window (Dict[str, int]): Dictionary specifying index ranges for train/validation periods with keys 
            'train_start', 'train_end', 'val_start', and 'val_end'.
        model (sklearn.base.BaseEstimator): scikit-learn compatible model to be trained and evaluated.
        target_col (str, optional): Name of the column to predict. Defaults to 'delta_pm25_t+1'.

    Returns:
        Tuple[np.float64, np.float64]: 
            - Fold RMSE score.
            - Fold mean absolute change in PM2.5.
    """

    train_start, train_end = window['train_start'], window['train_end']
    val_start, val_end = window['val_start'], window['val_end']

    train_dates = dates[train_start: train_end]
    val_dates = dates[val_start: val_end]

    # generate train and val sets by dropping resp. columns
    train_data =  df[df['date'].isin(train_dates)]
    val_data = df[df['date'].isin(val_dates)]

    X_train, y_train = train_data.drop(['row', 'col', 'date', target_col], axis=1), train_data[target_col]
    X_val, y_val = val_data.drop(['row', 'col', 'date', target_col], axis=1), val_data[target_col]

    # clone input model (fresh one) and fit to data, predict on validation set
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)


    y_preds = model_clone.predict(X_val)

    # return RMSE
    return root_mean_squared_error(y_val, y_preds), np.mean(np.abs(y_val))


def rolling_window_cv(train_days, gap_days, val_days, df, model, target_col='delta_pm25_t+1'):
    """
    Performs rolling-window cross-validation for spatiotemporal time series data.

    Args:
        train_days (int): Number of consecutive days in the training window.
        gap_days (int): Number of days between the training and validation windows to reduce temporal autocorrelation.
        val_days (int): Number of consecutive days in the validation window.
        df (pd.DataFrame): DataFrame containing covariates, target, and date columns.
        model (sklearn.base.BaseEstimator): scikit-learn compatible model to be trained and evaluated.
        target_col (str, optional): Name of the column to predict. Defaults to 'delta_pm25_t+1'.

    Returns:
        Tuple[List[np.float64], List[np.float64]]: 
            - List of fold-wise RMSE scores.
            - List of fold-wise mean absolute PM2.5 changes.
    """

    dates = df['date'].unique()
    windows = []

    train_start = 0
    val_end = 0
    # compute train/val dates
    while val_end <= len(dates):
        train_end = train_start + train_days
        val_start = train_end + gap_days
        val_end = val_start + val_days

        if val_end > len(dates):
            break

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
        })

        train_start += val_days

    # perform cross-validation for each window
    rmse_scores, mean_abs_deltas = [], []
    for idx, window in enumerate(windows):
        print(f'Fold {idx}')

        rmse, mean_abs_delta = cv_fold_helper(
            df=df, 
            dates=dates, 
            window=window,
            model=model, 
            target_col=target_col, 
        )
        rmse_scores.append(rmse)
        mean_abs_deltas.append(mean_abs_delta)
        print(f"RMSE: {rmse}")

    return rmse_scores, mean_abs_deltas


def run_experiments(df, model, feature_experiments, train_days, gap_days, val_days):
    ):
    """
    Runs rolling-window cross-validation across multiple feature sets (experiments).

    For each experiment (defined by a subset of features), trains and evaluates the model 
    using rolling-window cross-validation and computes both unweighted and log-weighted 
    average RMSE scores.

    Args:
        df (pd.DataFrame): DataFrame containing all available covariates, the target column, and metadata (e.g., 'row', 'col', 'date').
        model (sklearn.base.BaseEstimator): scikit-learn compatible model to be trained and evaluated.
        feature_experiments (Iterable[Tuple[str, List[str]]]): A list of (experiment_name, feature_list) pairs,
            where each feature_list defines the covariates to use in that experiment.
        train_days (int): Number of consecutive days in the training window.
        gap_days (int): Number of days between the training and validation windows to reduce temporal autocorrelation.
        val_days (int): Number of consecutive days in the validation window.

    Returns:
        pd.DataFrame: DataFrame containing the results of each experiment, including the experiment name, 
        number of features, unweighted RMSE, and log-weighted RMSE.
    """

    results = []

    for experiment_name, features in feature_experiments:
        print(f"Running experiment: {experiment_name}")
        
        rmse_scores, mean_abs_deltas = rolling_window_cv(
            train_days=train_days,
            gap_days=gap_days,
            val_days=val_days,
            df=df[['row', 'col', 'date'] + features + ['delta_pm25_t+1']],
            model=model,
        )

        # compute log-weighted average on RMSE results
        log_weights = np.log1p(mean_abs_deltas) 
        soft_weights = log_weights / log_weights.sum()

        unweighted_rmse = np.mean(rmse_scores)
        weighted_rmse = np.dot(rmse_scores, soft_weights)
        
        results.append({
            'experiment': experiment_name,
            'n_features': len(features),
            'unweighted_rmse': unweighted_rmse,
            'weighted_rmse': weighted_rmse,
        })

    return pd.DataFrame(results)
