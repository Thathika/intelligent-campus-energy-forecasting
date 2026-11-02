"""
test_app_workflow.py
End-to-end integration test verifying that all modules, data loading, preprocessing,
feature engineering, forecasting, anomaly detection, sustainability, and page functions
execute without error.
"""

import os
import sys
import pandas as pd
import numpy as np

from src.data_loader import load_default_dataset, auto_detect_columns, inspect_component_relationships
from src.preprocessing import preprocess_energy_data, export_dataframe_to_csv
from src.feature_engineering import create_forecasting_features
from src.forecasting_models import (
    chronological_train_test_split,
    train_and_evaluate_models,
    generate_future_forecast
)
from src.evaluation import compare_models, compute_residuals_df
from src.anomaly_detection import detect_consumption_anomalies
from src.sustainability import calculate_cost_and_emissions
from utils.sample_data import generate_synthetic_campus_dataset


def test_full_pipeline_real_data():
    # 1. Load real dataset
    df = load_default_dataset()
    assert len(df) == 31
    date_col, target_col = auto_detect_columns(df)
    assert date_col == "Date"
    assert target_col == "Total"

    # 2. Preprocess
    cleaned_df, report = preprocess_energy_data(df, date_col, target_col)
    assert len(cleaned_df) == 31
    assert cleaned_df["_parsed_datetime"].is_monotonic_increasing

    # 3. Component balance
    comp_rel = inspect_component_relationships(cleaned_df, target_col)
    assert comp_rel["supply_sum_matches_target"] is True
    assert comp_rel["demand_sum_matches_target"] is True

    # 4. Feature engineering
    feat_df, feature_cols, feat_meta = create_forecasting_features(
        cleaned_df, date_col="_parsed_datetime", target_col=target_col, include_lag_7=True
    )
    assert "day_of_week" in feature_cols
    assert "target_lag_1" in feature_cols
    assert "target_rolling_mean_3" in feature_cols

    # 5. Chronological train/test split
    train_df, test_df = chronological_train_test_split(
        feat_df, date_col="_parsed_datetime", target_col=target_col, feature_cols=feature_cols, test_days=7
    )
    assert len(test_df) == 7
    assert len(train_df) == len(feat_df.dropna(subset=feature_cols + [target_col])) - 7

    # 6. Fit models and predict
    trained_models, predictions = train_and_evaluate_models(
        train_df, test_df, feature_cols, target_col
    )
    assert "Random Forest Regressor" in trained_models
    assert "Gradient Boosting Regressor" in trained_models
    assert "Linear Regression" in trained_models
    assert "Seasonal Naïve (Baseline)" in trained_models

    # 7. Model comparison & evaluation metrics
    comp_df = compare_models(test_df[target_col], predictions, test_df["_parsed_datetime"])
    assert not comp_df.empty
    assert "MAE" in comp_df.columns
    assert "RMSE" in comp_df.columns
    assert "MAPE (%)" in comp_df.columns
    assert comp_df["MAE"].notna().all()

    # 8. Residuals
    residuals_df = compute_residuals_df(test_df, "_parsed_datetime", target_col, predictions)
    assert not residuals_df.empty
    assert "Actual" in residuals_df.columns

    # 9. Recursive future forecast
    best_model_name = comp_df.iloc[0]["Model"]
    future_df = generate_future_forecast(
        model=trained_models[best_model_name],
        historical_df=cleaned_df,
        date_col="_parsed_datetime",
        target_col=target_col,
        feature_cols=feature_cols,
        horizon_days=7
    )
    assert len(future_df) == 7
    assert f"predicted_{target_col}" in future_df.columns
    assert (future_df[f"predicted_{target_col}"] >= 0).all()

    # 10. Anomaly detection (all 3 methods)
    for m in ["zscore", "rolling", "iqr"]:
        anom_df, summary = detect_consumption_anomalies(
            cleaned_df, date_col="_parsed_datetime", target_col=target_col, method=m, threshold=2.0
        )
        assert summary["total_observations"] == 31

    # 11. Sustainability calculation
    cost_s, emiss_s, sust_summary = calculate_cost_and_emissions(
        cleaned_df[target_col], unit_is_verified_kwh=True, tariff_inr_per_kwh=8.0, emissions_factor_kg_per_kwh=0.716
    )
    assert sust_summary["total_cost_inr"] > 0
    assert sust_summary["total_emissions_kg"] > 0


def test_full_pipeline_synthetic_data():
    syn_df = generate_synthetic_campus_dataset(num_days=90, seed=42)
    assert len(syn_df) == 90
    cleaned_df, _ = preprocess_energy_data(syn_df, "Date", "Total")
    assert len(cleaned_df) == 90

    comp_rel = inspect_component_relationships(cleaned_df, "Total")
    assert comp_rel["supply_sum_matches_target"] is True
    assert comp_rel["demand_sum_matches_target"] is True


if __name__ == "__main__":
    test_full_pipeline_real_data()
    test_full_pipeline_synthetic_data()
    print("ALL END-TO-END WORKFLOW TESTS PASSED SUCCESSFULLY!")

