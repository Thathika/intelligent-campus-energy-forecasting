"""
feature_engineering.py
Module for creating time-based and historical consumption features without target leakage.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np


FEATURE_DOCUMENTATION: Dict[str, str] = {
    "day_of_week": "Integer representing the day of the week (0=Monday, 6=Sunday). Captures weekly scheduling patterns.",
    "is_weekend": "Binary flag (1 for Saturday/Sunday, 0 for weekdays). Captures lower administrative and academic activity.",
    "day_of_month": "Calendar day (1-31). Captures subtle intra-month schedule changes without introducing excessive degrees of freedom.",
    "target_lag_1": "Target consumption from the immediately preceding day (t-1). Provides recent inertia and autoregressive baseline.",
    "target_lag_7": "Target consumption from exactly one week earlier (t-7). Captures weekly periodicity when available.",
    "target_rolling_mean_3": "3-day moving average of target consumption from days (t-3, t-2, t-1). Shifted by 1 to prevent target leakage."
}


def create_forecasting_features(
    df: pd.DataFrame,
    date_col: str = "_parsed_datetime",
    target_col: str = "Total",
    include_lag_7: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Constructs a conservative, leakage-free feature set suitable for small daily energy datasets.

    Parameters:
    - df: Preprocessed, chronologically sorted DataFrame
    - date_col: Datetime column name (defaults to '_parsed_datetime')
    - target_col: Name of the target variable to predict
    - include_lag_7: Whether to include 7-day lag (requires >= 8 observations)

    Returns:
    - feature_df: DataFrame with generated features
    - feature_cols: List of feature column names suitable for model training
    - metadata: Documentation explaining each feature and leakage mitigation
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Sort strictly by date to ensure proper shift calculations
    fdf = df.copy().sort_values(by=date_col).reset_index(drop=True)
    dt_series = pd.to_datetime(fdf[date_col])

    # 1. Calendar / Time Features
    fdf["day_of_week"] = dt_series.dt.day_of_week
    fdf["is_weekend"] = dt_series.dt.day_of_week.isin([5, 6]).astype(int)
    fdf["day_of_month"] = dt_series.dt.day

    feature_cols = ["day_of_week", "is_weekend", "day_of_month"]

    # 2. Autoregressive Lag Features (strictly from past days)
    # Lag 1: Yesterday's target
    fdf["target_lag_1"] = fdf[target_col].shift(1)
    feature_cols.append("target_lag_1")

    # Rolling mean of previous 3 days: shift(1) ensures target at day t is NEVER included
    fdf["target_rolling_mean_3"] = fdf[target_col].shift(1).rolling(window=3, min_periods=1).mean()
    feature_cols.append("target_rolling_mean_3")

    # Lag 7: Same day last week (optional, valid for len >= 8)
    if include_lag_7 and len(fdf) >= 14:
        fdf["target_lag_7"] = fdf[target_col].shift(7)
        feature_cols.append("target_lag_7")

    metadata: Dict[str, Any] = {
        "feature_columns": feature_cols,
        "feature_descriptions": {col: FEATURE_DOCUMENTATION.get(col, "Engineered feature") for col in feature_cols},
        "target_column": target_col,
        "total_rows_before_drop": len(fdf),
        "valid_rows_count": int(fdf[feature_cols + [target_col]].dropna().shape[0]),
        "leakage_prevention_notes": [
            "Component columns (TNEB, Solar, Diesel, Biogas, Load zones) are strictly excluded from features to prevent circular target leakage.",
            "All rolling statistics use .shift(1) so today's target is never part of its own predictor window.",
            "No future values or future calendar averages are referenced."
        ]
    }

    return fdf, feature_cols, metadata
