"""
preprocessing.py
Module for cleaning, date parsing, chronological ordering, and data quality validation.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


def preprocess_energy_data(
    raw_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    date_format: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and prepares energy time-series data without lookahead bias.
    
    Parameters:
    - raw_df: Unprocessed pandas DataFrame (remains unchanged)
    - date_col: Name of the timestamp/date column
    - target_col: Primary forecasting target column
    - date_format: Explicit strftime format (e.g., '%d.%m.%Y'), or None for auto-detection

    Returns:
    - cleaned_df: Chronologically sorted and validated DataFrame
    - report: Comprehensive metadata and validation report dictionary
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Cannot preprocess an empty or None DataFrame.")

    if date_col not in raw_df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    if target_col not in raw_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Work strictly on a deep copy to preserve the original raw DataFrame
    df = raw_df.copy(deep=True)
    initial_rows = len(df)
    initial_cols = len(df.columns)

    report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "date_column": date_col,
        "target_column": target_col,
        "date_format_used": None,
        "date_min": None,
        "date_max": None,
        "duration_days": 0,
        "missing_dates_count": 0,
        "duplicates_found": 0,
        "null_counts": {},
        "negative_counts": {},
        "target_summary": {},
        "warnings": [],
        "info": []
    }

    # 1. Parse Date column
    parsed_date = None
    if date_format:
        try:
            parsed_date = pd.to_datetime(df[date_col], format=date_format)
            report["date_format_used"] = date_format
        except Exception as e:
            report["warnings"].append(f"Specified format '{date_format}' failed: {e}. Falling back to flexible parsing.")

    if parsed_date is None:
        # Try DD.MM.YYYY with dayfirst=True first (common European / Indian format)
        try:
            parsed_date = pd.to_datetime(df[date_col], dayfirst=True)
            report["date_format_used"] = "dayfirst=True (auto-detected DD.MM.YYYY / DD-MM-YYYY)"
        except Exception:
            try:
                parsed_date = pd.to_datetime(df[date_col])
                report["date_format_used"] = "standard ISO / auto-detected"
            except Exception as e:
                raise ValueError(f"Unable to parse date column '{date_col}'. Error: {e}")

    df["_parsed_datetime"] = parsed_date
    unparseable_count = int(df["_parsed_datetime"].isna().sum())
    if unparseable_count > 0:
        report["warnings"].append(f"Found {unparseable_count} unparseable date values.")

    # 2. Check for duplicate dates
    dup_mask = df.duplicated(subset=["_parsed_datetime"], keep=False)
    dup_count = int(dup_mask.sum())
    report["duplicates_found"] = dup_count
    if dup_count > 0:
        report["warnings"].append(f"Found {dup_count} duplicate timestamp records.")

    # 3. Sort chronologically
    df = df.sort_values(by="_parsed_datetime").reset_index(drop=True)

    # Date range metadata
    valid_dates = df["_parsed_datetime"].dropna()
    if not valid_dates.empty:
        d_min = valid_dates.min()
        d_max = valid_dates.max()
        report["date_min"] = d_min.strftime("%Y-%m-%d")
        report["date_max"] = d_max.strftime("%Y-%m-%d")
        report["duration_days"] = int((d_max - d_min).days) + 1

    # 4. Check for null values across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        null_c = int(df[col].isna().sum())
        if null_c > 0:
            report["null_counts"][col] = null_c
            report["warnings"].append(f"Column '{col}' contains {null_c} missing value(s).")

    # 5. Check for negative numbers or zeros in target and components
    for col in numeric_cols:
        neg_c = int((df[col] < 0).sum())
        if neg_c > 0:
            report["negative_counts"][col] = neg_c
            report["warnings"].append(f"Column '{col}' contains {neg_c} negative value(s).")

    # 6. Target column validation
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        try:
            df[target_col] = pd.to_numeric(df[target_col])
        except Exception as e:
            raise ValueError(f"Target column '{target_col}' cannot be converted to numeric: {e}")

    report["target_summary"] = {
        "count": int(df[target_col].count()),
        "mean": float(df[target_col].mean()),
        "std": float(df[target_col].std()) if len(df) > 1 else 0.0,
        "min": float(df[target_col].min()),
        "q25": float(df[target_col].quantile(0.25)),
        "median": float(df[target_col].median()),
        "q75": float(df[target_col].quantile(0.75)),
        "max": float(df[target_col].max())
    }

    # Small dataset warning
    if len(df) <= 45:
        report["warnings"].append(
            f"Small-sample notice: The dataset has {len(df)} daily observations. "
            "Model training and multi-step forecast stability are subject to significant small-sample uncertainty."
        )

    return df, report


def export_dataframe_to_csv(df: pd.DataFrame) -> str:
    """
    Exports DataFrame to a clean CSV string, excluding internal preprocessing helper columns.
    """
    export_df = df.copy()
    if "_parsed_datetime" in export_df.columns:
        export_df = export_df.drop(columns=["_parsed_datetime"])
    return export_df.to_csv(index=False)

