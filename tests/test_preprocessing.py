"""
test_preprocessing.py
Unit tests for data loading, date parsing, chronological sorting, and data validation.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_default_dataset, auto_detect_columns, inspect_component_relationships
from src.preprocessing import preprocess_energy_data, export_dataframe_to_csv


def test_load_default_dataset():
    df = load_default_dataset()
    assert df is not None
    assert not df.empty
    assert len(df) == 31
    assert "Date" in df.columns
    assert "Total" in df.columns


def test_auto_detect_columns():
    df = pd.DataFrame({
        "Reading_Date": ["01.07.2026", "02.07.2026"],
        "Total_Consumption": [25000, 26000]
    })
    date_col, target_col = auto_detect_columns(df)
    assert date_col == "Reading_Date"
    assert target_col == "Total_Consumption"


def test_component_relationships():
    df = load_default_dataset()
    rel = inspect_component_relationships(df, target_col="Total")
    # In the actual dataset, supply components and demand components sum to Total
    assert rel["supply_sum_matches_target"] is True
    assert rel["demand_sum_matches_target"] is True
    assert rel["supply_max_diff"] == 0.0
    assert rel["demand_max_diff"] == 0.0


def test_preprocessing_chronological_order():
    df = load_default_dataset()
    # Raw dataset has dates in descending order (31.07 down to 01.07)
    cleaned_df, report = preprocess_energy_data(df, date_col="Date", target_col="Total")
    
    # Original should be untouched
    assert df["Date"].iloc[0] == "31.07.2026"

    # Cleaned must be strictly sorted ascending
    assert cleaned_df["_parsed_datetime"].is_monotonic_increasing
    assert cleaned_df["Date"].iloc[0] == "01.07.2026"
    assert cleaned_df["Date"].iloc[-1] == "31.07.2026"
    assert report["initial_rows"] == 31
    assert report["duration_days"] == 31
    assert report["duplicates_found"] == 0


def test_export_dataframe_to_csv():
    df = load_default_dataset()
    cleaned_df, _ = preprocess_energy_data(df, date_col="Date", target_col="Total")
    csv_str = export_dataframe_to_csv(cleaned_df)
    assert isinstance(csv_str, str)
    assert "Date,TNEB_Campus" in csv_str
    assert "_parsed_datetime" not in csv_str

