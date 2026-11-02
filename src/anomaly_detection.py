"""
anomaly_detection.py
Module for explainable anomaly identification in campus energy readings.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def detect_consumption_anomalies(
    df: pd.DataFrame,
    date_col: str = "_parsed_datetime",
    target_col: str = "Total",
    method: str = "zscore",
    threshold: float = 2.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies potentially unusual consumption readings using explainable statistical baselines.

    Methods:
    - 'zscore': Deviations from overall mean by more than threshold standard deviations (|Z| > threshold).
    - 'rolling': Deviations from a 7-day rolling median baseline by more than threshold standard deviations.
    - 'iqr': Outliers outside [Q1 - threshold*IQR, Q3 + threshold*IQR].

    Returns:
    - anomaly_df: Table of flagged records with observed, expected, deviation, and reason.
    - summary: Statistics on total observations analyzed and flagged counts.
    """
    if df is None or df.empty:
        raise ValueError("Cannot detect anomalies in empty DataFrame.")

    adf = df.copy().sort_values(by=date_col).reset_index(drop=True)
    target = adf[target_col]

    flags = pd.Series(False, index=adf.index)
    expected_vals = pd.Series(index=adf.index, dtype=float)
    reasons = pd.Series("", index=adf.index, dtype=str)

    if method == "zscore":
        mean_val = float(target.mean())
        std_val = float(target.std()) if target.std() > 0 else 1.0
        z_scores = (target - mean_val) / std_val
        expected_vals[:] = mean_val
        flags = z_scores.abs() > threshold
        for idx in adf[flags].index:
            z = z_scores.loc[idx]
            direction = "High surge" if z > 0 else "Low drop"
            reasons.loc[idx] = f"{direction}: {abs(z):.2f} std dev from campus mean ({mean_val:.0f})"

    elif method == "rolling":
        rolling_median = target.rolling(window=7, min_periods=3, center=False).median().bfill()
        rolling_std = target.rolling(window=7, min_periods=3, center=False).std().bfill()
        rolling_std = rolling_std.replace(0, 1.0)
        expected_vals = rolling_median
        devs = (target - rolling_median) / rolling_std
        flags = devs.abs() > threshold
        for idx in adf[flags].index:
            d = devs.loc[idx]
            direction = "Surge" if d > 0 else "Downturn"
            reasons.loc[idx] = f"{direction}: {abs(d):.2f} std dev from 7-day rolling baseline"

    elif method == "iqr":
        q1 = float(target.quantile(0.25))
        q3 = float(target.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        expected_vals[:] = float(target.median())
        flags = (target < lower_bound) | (target > upper_bound)
        for idx in adf[flags].index:
            val = target.loc[idx]
            if val > upper_bound:
                reasons.loc[idx] = f"Upper outlier: exceeds Q3 + {threshold}*IQR ({upper_bound:.0f})"
            else:
                reasons.loc[idx] = f"Lower outlier: falls below Q1 - {threshold}*IQR ({lower_bound:.0f})"

    # Format the flagged table
    flagged_indices = adf[flags].index
    flagged_rows: List[Dict[str, Any]] = []

    for idx in flagged_indices:
        obs_date = adf.loc[idx, date_col]
        obs_date_str = pd.to_datetime(obs_date).strftime("%Y-%m-%d") if pd.notnull(obs_date) else str(obs_date)
        obs_val = float(adf.loc[idx, target_col])
        exp_val = float(expected_vals.loc[idx])
        abs_diff = obs_val - exp_val
        pct_diff = (abs_diff / exp_val * 100.0) if exp_val != 0 else 0.0

        flagged_rows.append({
            "Date": obs_date_str,
            "Observed Value": round(obs_val, 1),
            "Expected / Reference": round(exp_val, 1),
            "Absolute Deviation": round(abs_diff, 1),
            "Percentage Deviation (%)": round(pct_diff, 1),
            "Flag Reason": reasons.loc[idx],
            "Investigation Guidance": "Review campus event logs, chiller status, or weekend shutdowns for this date."
        })

    anomaly_df = pd.DataFrame(flagged_rows)

    summary = {
        "method_used": method,
        "threshold": threshold,
        "total_observations": len(adf),
        "flagged_count": len(flagged_rows),
        "flagged_percentage": round((len(flagged_rows) / len(adf) * 100.0), 1) if len(adf) > 0 else 0.0,
        "disclaimer": (
            "Caution: Flagged readings indicate potentially unusual consumption requiring human investigation. "
            "They must not be automatically assumed to represent energy wastage, electrical faults, or meter malfunction."
        )
    }

    return anomaly_df, summary
