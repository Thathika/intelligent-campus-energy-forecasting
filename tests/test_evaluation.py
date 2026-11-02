"""
test_evaluation.py
Unit tests for evaluation metrics (MAE, RMSE, MAPE) against exact theoretical values.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import compute_metrics, compare_models


def test_compute_metrics_known_values():
    # Exact scenario:
    # y_true = [100.0, 200.0, 300.0]
    # y_pred = [110.0, 190.0, 300.0]
    # errors = [-10.0, +10.0, 0.0]
    # abs_errors = [10.0, 10.0, 0.0] -> MAE = 20 / 3 = 6.6666667
    # squared_errors = [100.0, 100.0, 0.0] -> MSE = 200 / 3 = 66.6666667 -> RMSE = sqrt(66.6666667) = 8.1649658
    # pct_errors = [10/100, 10/200, 0/300] = [0.10, 0.05, 0.0] = 0.15 / 3 = 0.05 -> MAPE = 5.0%

    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = compute_metrics(y_true, y_pred)

    assert pytest.approx(metrics["MAE"], 0.001) == 6.6667
    assert pytest.approx(metrics["RMSE"], 0.001) == 8.1650
    assert pytest.approx(metrics["MAPE_percent"], 0.001) == 5.0
    assert metrics["valid_count"] == 3


def test_compute_metrics_zero_actual_handling():
    # If actual contains 0, MAPE should handle gracefully without ZeroDivisionError
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 95.0])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(5.0)
    assert metrics["MAPE_percent"] is not None  # Calculated on non-zero subsets safely


def test_compare_models_ranking():
    y_true = pd.Series([100.0, 150.0, 200.0])
    preds = {
        "Model_Good": np.array([102.0, 148.0, 201.0]),  # MAE = (2+2+1)/3 = 1.67
        "Model_Bad": np.array([120.0, 130.0, 180.0])    # MAE = (20+20+20)/3 = 20.0
    }

    comp_df = compare_models(y_true, preds)
    assert not comp_df.empty
    assert len(comp_df) == 2
    # Model_Good should be ranked 1 because of lower MAE
    assert comp_df["Model"].iloc[0] == "Model_Good"
    assert comp_df["Rank"].iloc[0] == 1
    assert comp_df["Model"].iloc[1] == "Model_Bad"
    assert comp_df["Rank"].iloc[1] == 2

