"""
evaluation.py
Module for computing evaluation metrics, model comparison tables, and prediction residuals.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Optional[float]]:
    """
    Computes MAE, RMSE, and MAPE metrics on held-out test observations.
    Handles zero division safely for MAPE.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    n = len(yt)
    if n == 0:
        return {"MAE": None, "RMSE": None, "MAPE_percent": None, "valid_count": 0}

    # Mean Absolute Error
    mae = float(np.mean(np.abs(yt - yp)))

    # Root Mean Squared Error
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # Mean Absolute Percentage Error (check for zero actuals)
    non_zero = yt != 0
    if np.any(non_zero):
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0)
    else:
        mape = None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "valid_count": n
    }


def compare_models(
    y_true: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    test_dates: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Produces a fair, side-by-side model comparison table using the exact same test observations.
    """
    results: List[Dict[str, Any]] = []

    y_true_arr = y_true.values

    date_str = ""
    if test_dates is not None and len(test_dates) > 0:
        d_min = pd.to_datetime(test_dates.min()).strftime("%Y-%m-%d")
        d_max = pd.to_datetime(test_dates.max()).strftime("%Y-%m-%d")
        date_str = f"{d_min} to {d_max}"

    for model_name, preds in predictions_dict.items():
        metrics = compute_metrics(y_true_arr, preds)
        results.append({
            "Model": model_name,
            "MAE": round(metrics["MAE"], 2) if metrics["MAE"] is not None else np.nan,
            "RMSE": round(metrics["RMSE"], 2) if metrics["RMSE"] is not None else np.nan,
            "MAPE (%)": round(metrics["MAPE_percent"], 2) if metrics["MAPE_percent"] is not None else np.nan,
            "Test Observations": metrics["valid_count"],
            "Evaluation Period": date_str
        })

    comp_df = pd.DataFrame(results)

    # Sort primarily by MAE (ascending)
    if not comp_df.empty and "MAE" in comp_df.columns:
        comp_df = comp_df.sort_values(by="MAE", na_position="last").reset_index(drop=True)
        comp_df["Rank"] = range(1, len(comp_df) + 1)
        # Move Rank to first column
        cols = ["Rank"] + [c for c in comp_df.columns if c != "Rank"]
        comp_df = comp_df[cols]

    return comp_df


def compute_residuals_df(
    test_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Constructs a detailed table of actual values, model predictions, and residuals (Actual - Predicted).
    """
    res_df = pd.DataFrame()
    res_df["Date"] = pd.to_datetime(test_df[date_col]).dt.strftime("%Y-%m-%d")
    res_df["Actual"] = test_df[target_col].values

    for model_name, preds in predictions_dict.items():
        res_df[f"{model_name} (Pred)"] = np.round(preds, 2)
        res_df[f"{model_name} (Residual)"] = np.round(test_df[target_col].values - preds, 2)

    return res_df

