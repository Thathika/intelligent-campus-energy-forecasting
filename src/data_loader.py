"""
data_loader.py
Module for loading, auto-detecting, and validating energy datasets.
"""

import os
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preprocessed_energy.csv")

COMMON_DATE_NAMES = ["date", "datetime", "timestamp", "time", "day", "reading_date"]
COMMON_TARGET_NAMES = ["total", "energy", "energy_kwh", "kwh", "consumption", "power", "demand", "total_consumption"]

KNOWN_SUPPLY_COLS = ["tneb_campus", "tneb_new_stp", "solar", "diesel", "biogas"]
KNOWN_DEMAND_COLS = ["staff_quarters", "academic_blocks", "hostels", "chiller", "stp"]


def load_default_dataset() -> pd.DataFrame:
    """
    Load the default campus energy dataset from the data folder.
    Returns the raw pandas DataFrame.
    """
    if not os.path.exists(DEFAULT_DATA_PATH):
        raise FileNotFoundError(f"Default dataset not found at expected path: {DEFAULT_DATA_PATH}")
    return pd.read_csv(DEFAULT_DATA_PATH)


def load_csv_data(file_source) -> pd.DataFrame:
    """
    Load CSV data from a file path, file-like object, or Streamlit UploadedFile.
    """
    return pd.read_csv(file_source)


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect the most probable date and target columns from the DataFrame.
    """
    date_col = None
    target_col = None

    cols = list(df.columns)
    cols_lower = {col.lower().strip(): col for col in cols}

    # Detect Date column
    for candidate in COMMON_DATE_NAMES:
        if candidate in cols_lower:
            date_col = cols_lower[candidate]
            break

    # If not found by name, check for datetime-parseable or string format like DD.MM.YYYY
    if date_col is None:
        for col in cols:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                sample = df[col].dropna().head(5).astype(str)
                # Check for dots, hyphens, or slashes indicating dates
                if any('.' in s or '-' in s or '/' in s for s in sample):
                    try:
                        pd.to_datetime(sample, dayfirst=True)
                        date_col = col
                        break
                    except Exception:
                        pass

    # Detect Target column
    for candidate in COMMON_TARGET_NAMES:
        if candidate in cols_lower:
            target_col = cols_lower[candidate]
            break

    # Fallback for target: last numeric column
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]

    return date_col, target_col


def inspect_component_relationships(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Examines potential overlap or sum relationships between component columns and the target.
    Checks whether components sum to the target on the supply side and demand side.
    """
    results: Dict[str, Any] = {
        "supply_cols_detected": [],
        "demand_cols_detected": [],
        "supply_sum_matches_target": False,
        "demand_sum_matches_target": False,
        "supply_max_diff": None,
        "demand_max_diff": None,
        "notes": []
    }

    cols_lower = {col.lower().strip(): col for col in df.columns}
    detected_supply = [cols_lower[c] for c in KNOWN_SUPPLY_COLS if c in cols_lower]
    detected_demand = [cols_lower[c] for c in KNOWN_DEMAND_COLS if c in cols_lower]

    results["supply_cols_detected"] = detected_supply
    results["demand_cols_detected"] = detected_demand

    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        target_series = df[target_col]

        # Check supply sum
        if len(detected_supply) >= 2:
            supply_sum = df[detected_supply].sum(axis=1)
            diff = (supply_sum - target_series).abs().max()
            results["supply_max_diff"] = float(diff)
            if diff < 1e-4:
                results["supply_sum_matches_target"] = True
                results["notes"].append(
                    f"Supply components ({', '.join(detected_supply)}) mathematically sum exactly to '{target_col}' across all rows."
                )

        # Check demand sum
        if len(detected_demand) >= 2:
            demand_sum = df[detected_demand].sum(axis=1)
            diff = (demand_sum - target_series).abs().max()
            results["demand_max_diff"] = float(diff)
            if diff < 1e-4:
                results["demand_sum_matches_target"] = True
                results["notes"].append(
                    f"Demand load components ({', '.join(detected_demand)}) mathematically sum exactly to '{target_col}' across all rows."
                )

    return results

