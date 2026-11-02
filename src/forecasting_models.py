"""
forecasting_models.py
Implementation of candidate machine learning and baseline forecasting models.
Includes recursive multi-step forecasting for short horizons.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class BaselinePersistenceModel:
    """
    Seasonal Naïve / Persistence Baseline.
    Predicts today's value based on the previous observation (y_{t-1})
    or same day last week (y_{t-7}) when available.
    """
    def __init__(self, lag_step: int = 1):
        self.lag_step = lag_step
        self.model_name = f"Seasonal Naïve (Lag {lag_step})"
        self.last_observed: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(y) > 0:
            self.last_observed = float(y.iloc[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # If target_lag_1 is present in X, use it directly
        col_name = f"target_lag_{self.lag_step}"
        if col_name in X.columns:
            return X[col_name].fillna(self.last_observed if self.last_observed is not None else 0.0).values
        elif "target_lag_1" in X.columns:
            return X["target_lag_1"].fillna(self.last_observed if self.last_observed is not None else 0.0).values
        else:
            return np.full(len(X), self.last_observed if self.last_observed is not None else 0.0)


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns initialized candidate forecasting models configured with conservative hyperparameters.
    """
    return {
        "Seasonal Naïve (Baseline)": BaselinePersistenceModel(lag_step=1),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=40,
            max_depth=2,
            learning_rate=0.05,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )
    }


def chronological_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    test_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict chronological train/test split.
    Drops any initial rows with NaN in features (e.g. initial lag rows).
    """
    valid_df = df.dropna(subset=feature_cols + [target_col]).copy()
    valid_df = valid_df.sort_values(by=date_col).reset_index(drop=True)

    total_valid = len(valid_df)
    if total_valid <= test_days:
        raise ValueError(
            f"Dataset has only {total_valid} valid rows after lag creation, "
            f"which is insufficient for a {test_days}-day test set. Reduce test_days."
        )

    split_idx = total_valid - test_days
    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    return train_df, test_df


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Fits all candidate models on train_df and computes out-of-sample predictions on test_df.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    models = get_candidate_models()
    trained_models: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            trained_models[name] = model
            predictions[name] = preds
        except Exception as e:
            # Model failed to train
            predictions[name] = np.full(len(X_test), np.nan)

    return trained_models, predictions


def generate_future_forecast(
    model: Any,
    historical_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    horizon_days: int = 7
) -> pd.DataFrame:
    """
    Generates experimental future forecasts recursively for a short horizon (e.g. 1 to 7 days).
    Does not use actual future data; updates lag and rolling statistics autoregressively.
    """
    last_date = pd.to_datetime(historical_df[date_col].max())
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

    # Maintain recent target history for recursive feature computation
    recent_targets = historical_df[target_col].tolist()

    future_records: List[Dict[str, Any]] = []

    for f_date in future_dates:
        row: Dict[str, Any] = {
            date_col: f_date,
            "day_of_week": f_date.day_of_week,
            "is_weekend": int(f_date.day_of_week in [5, 6]),
            "day_of_month": f_date.day
        }

        # Lag 1: most recent value (either historical or previously predicted)
        lag_1_val = recent_targets[-1]
        row["target_lag_1"] = lag_1_val

        # Rolling mean of previous 3 values
        prev_3 = recent_targets[-3:] if len(recent_targets) >= 3 else recent_targets
        row["target_rolling_mean_3"] = float(np.mean(prev_3))

        # Lag 7: value from 7 steps back if available
        if "target_lag_7" in feature_cols:
            if len(recent_targets) >= 7:
                row["target_lag_7"] = recent_targets[-7]
            else:
                row["target_lag_7"] = lag_1_val

        # Prepare feature vector matching training feature order
        feature_vector = pd.DataFrame([row])[feature_cols]

        # Predict next value
        predicted_val = float(model.predict(feature_vector)[0])
        # Campus energy cannot be negative
        predicted_val = max(0.0, predicted_val)

        row["predicted_" + target_col] = predicted_val
        future_records.append(row)

        # Append prediction to history for next recursive step
        recent_targets.append(predicted_val)

    return pd.DataFrame(future_records)
