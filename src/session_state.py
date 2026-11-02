"""
session_state.py
Shared application state initialization and helper functions across all Streamlit pages.
Enterprise layout: deep forest green sidebar, bespoke data badges, and shared pipeline state.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_default_dataset, auto_detect_columns, inspect_component_relationships
from src.preprocessing import preprocess_energy_data
from src.feature_engineering import create_forecasting_features
from src.forecasting_models import (
    chronological_train_test_split,
    train_and_evaluate_models,
    generate_future_forecast
)
from src.evaluation import compare_models, compute_residuals_df
from src.anomaly_detection import detect_consumption_anomalies
from src.sustainability import calculate_cost_and_emissions, DEFAULT_TARIFF_INR_PER_KWH, DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH
from utils.sample_data import generate_synthetic_campus_dataset


def load_custom_css():
    """Injects custom enterprise stylesheet into Streamlit page."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initializes default application state if not already set."""
    if "is_initialized" not in st.session_state:
        st.session_state["is_initialized"] = True
        st.session_state["dataset_type"] = "real"  # 'real', 'uploaded', or 'synthetic'
        st.session_state["dataset_name"] = "preprocessed_energy.csv (July 2026)"
        st.session_state["raw_df"] = None
        st.session_state["cleaned_df"] = None
        st.session_state["preproc_report"] = None
        st.session_state["date_col"] = "Date"
        st.session_state["target_col"] = "Total"
        st.session_state["component_analysis"] = None
        st.session_state["unit_is_verified"] = False
        st.session_state["tariff_rate"] = DEFAULT_TARIFF_INR_PER_KWH
        st.session_state["emissions_factor"] = DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH

        # ML & Evaluation State
        st.session_state["test_days"] = 7
        st.session_state["train_df"] = None
        st.session_state["test_df"] = None
        st.session_state["feature_cols"] = []
        st.session_state["feature_metadata"] = None
        st.session_state["trained_models"] = {}
        st.session_state["predictions"] = {}
        st.session_state["comp_df"] = None
        st.session_state["residuals_df"] = None
        st.session_state["selected_model"] = "Random Forest Regressor"
        st.session_state["future_df"] = None
        st.session_state["forecast_horizon"] = 7

        # Anomaly State
        st.session_state["anomaly_df"] = None
        st.session_state["anomaly_summary"] = None
        st.session_state["anomaly_method"] = "zscore"
        st.session_state["anomaly_threshold"] = 2.0

        # Sustainability State
        st.session_state["cost_series"] = None
        st.session_state["emissions_series"] = None
        st.session_state["sustainability_summary"] = None

        # Load default dataset immediately
        load_and_process_dataset(source="default")


def load_and_process_dataset(source: str = "default", uploaded_file=None):
    """Loads and runs standard pipeline on chosen dataset."""
    try:
        if source == "default":
            raw_df = load_default_dataset()
            st.session_state["dataset_type"] = "real"
            st.session_state["dataset_name"] = "preprocessed_energy.csv (July 2026)"
        elif source == "synthetic":
            raw_df = generate_synthetic_campus_dataset()
            st.session_state["dataset_type"] = "synthetic"
            st.session_state["dataset_name"] = "Synthetic Simulation (120 Days)"
        elif source == "uploaded" and uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state["dataset_type"] = "uploaded"
            st.session_state["dataset_name"] = f"Custom: {getattr(uploaded_file, 'name', 'Uploaded File')}"
        else:
            return

        st.session_state["raw_df"] = raw_df

        # Detect columns
        date_col, target_col = auto_detect_columns(raw_df)
        st.session_state["date_col"] = date_col or "Date"
        st.session_state["target_col"] = target_col or "Total"

        # Preprocess
        cleaned_df, report = preprocess_energy_data(
            raw_df=raw_df,
            date_col=st.session_state["date_col"],
            target_col=st.session_state["target_col"]
        )
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["preproc_report"] = report

        # Inspect components
        comp_rel = inspect_component_relationships(cleaned_df, st.session_state["target_col"])
        st.session_state["component_analysis"] = comp_rel

        # Run feature engineering & ML pipeline
        run_full_ml_pipeline()

        # Run anomaly detection
        run_anomaly_pipeline()

        # Run sustainability estimation
        run_sustainability_pipeline()

    except Exception as e:
        st.error(f"Error loading and processing dataset: {e}")


def run_full_ml_pipeline():
    """Generates features, splits chronologically, fits models, evaluates, and forecasts."""
    cleaned_df = st.session_state.get("cleaned_df")
    target_col = st.session_state.get("target_col", "Total")
    if cleaned_df is None or cleaned_df.empty:
        return

    test_days = st.session_state.get("test_days", 7)
    if len(cleaned_df) <= test_days + 5:
        test_days = max(2, len(cleaned_df) // 4)
        st.session_state["test_days"] = test_days

    # 1. Feature Engineering
    feat_df, feature_cols, feat_meta = create_forecasting_features(
        df=cleaned_df,
        date_col="_parsed_datetime",
        target_col=target_col,
        include_lag_7=len(cleaned_df) >= 20
    )
    st.session_state["feature_cols"] = feature_cols
    st.session_state["feature_metadata"] = feat_meta

    # 2. Chronological Split
    try:
        train_df, test_df = chronological_train_test_split(
            df=feat_df,
            date_col="_parsed_datetime",
            target_col=target_col,
            feature_cols=feature_cols,
            test_days=test_days
        )
        st.session_state["train_df"] = train_df
        st.session_state["test_df"] = test_df

        # 3. Fit Models & Predict on Test
        trained_models, predictions = train_and_evaluate_models(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col
        )
        st.session_state["trained_models"] = trained_models
        st.session_state["predictions"] = predictions

        # 4. Compare Models
        comp_df = compare_models(
            y_true=test_df[target_col],
            predictions_dict=predictions,
            test_dates=test_df["_parsed_datetime"]
        )
        st.session_state["comp_df"] = comp_df

        # Default selected model to rank 1 model if available
        if not comp_df.empty and "Model" in comp_df.columns:
            best_model_name = comp_df.iloc[0]["Model"]
            if best_model_name in trained_models:
                st.session_state["selected_model"] = best_model_name

        # 5. Compute Residuals
        residuals_df = compute_residuals_df(
            test_df=test_df,
            date_col="_parsed_datetime",
            target_col=target_col,
            predictions_dict=predictions
        )
        st.session_state["residuals_df"] = residuals_df

        # 6. Generate Future Forecast
        sel_model_name = st.session_state.get("selected_model", "Random Forest Regressor")
        if sel_model_name in trained_models:
            future_df = generate_future_forecast(
                model=trained_models[sel_model_name],
                historical_df=cleaned_df,
                date_col="_parsed_datetime",
                target_col=target_col,
                feature_cols=feature_cols,
                horizon_days=st.session_state.get("forecast_horizon", 7)
            )
            st.session_state["future_df"] = future_df

    except Exception:
        st.session_state["comp_df"] = None
        st.session_state["future_df"] = None


def run_anomaly_pipeline():
    """Runs anomaly detection with current parameters."""
    cleaned_df = st.session_state.get("cleaned_df")
    target_col = st.session_state.get("target_col", "Total")
    if cleaned_df is None or cleaned_df.empty:
        return

    method = st.session_state.get("anomaly_method", "zscore")
    threshold = st.session_state.get("anomaly_threshold", 2.0)

    try:
        anomaly_df, summary = detect_consumption_anomalies(
            df=cleaned_df,
            date_col="_parsed_datetime",
            target_col=target_col,
            method=method,
            threshold=threshold
        )
        st.session_state["anomaly_df"] = anomaly_df
        st.session_state["anomaly_summary"] = summary
    except Exception:
        st.session_state["anomaly_df"] = pd.DataFrame()
        st.session_state["anomaly_summary"] = None


def run_sustainability_pipeline():
    """Calculates cost and emissions for historical and forecast data."""
    cleaned_df = st.session_state.get("cleaned_df")
    target_col = st.session_state.get("target_col", "Total")
    if cleaned_df is None or cleaned_df.empty:
        return

    tariff = st.session_state.get("tariff_rate", DEFAULT_TARIFF_INR_PER_KWH)
    factor = st.session_state.get("emissions_factor", DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH)
    verified = st.session_state.get("unit_is_verified", False)

    try:
        cost_s, emiss_s, summ = calculate_cost_and_emissions(
            energy_series=cleaned_df[target_col],
            unit_is_verified_kwh=verified,
            tariff_inr_per_kwh=tariff,
            emissions_factor_kg_per_kwh=factor
        )
        st.session_state["cost_series"] = cost_s
        st.session_state["emissions_series"] = emiss_s
        st.session_state["sustainability_summary"] = summ
    except Exception:
        pass


def render_sidebar():
    """Renders compact, functional enterprise deep forest green sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 12px; margin-bottom: 14px; border-bottom: 1px solid #1e3b2f;">
                <div style="font-size: 0.88rem; font-weight: 700; color: #ffffff; letter-spacing: 0.08em; text-transform: uppercase;">Campus Energy</div>
                <div style="font-size: 0.72rem; color: #8fa69a; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 2px;">Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        dtype = st.session_state.get("dataset_type", "real")
        dname = st.session_state.get("dataset_name", "Real Dataset")

        badge_class = "sidebar-badge-real" if dtype == "real" else "sidebar-badge-synthetic"
        badge_dot = "●" if dtype == "real" else "▲"
        st.markdown(f'<div class="sidebar-badge {badge_class}">{badge_dot} {dname}</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin: 14px 0; border-top: 1px solid #1e3b2f;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #8fa69a; margin-bottom: 6px;'>Data Source</div>", unsafe_allow_html=True)
        mode_choice = st.radio(
            "Select Data Mode:",
            options=["Historical Campus Data (July 2026)", "Synthetic Simulation (120 Days)"],
            index=0 if dtype == "real" else 1,
            label_visibility="collapsed"
        )

        if mode_choice == "Historical Campus Data (July 2026)" and dtype != "real":
            load_and_process_dataset(source="default")
            st.rerun()
        elif mode_choice == "Synthetic Simulation (120 Days)" and dtype != "synthetic":
            load_and_process_dataset(source="synthetic")
            st.rerun()

        st.markdown("<hr style='margin: 14px 0; border-top: 1px solid #1e3b2f;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #8fa69a; margin-bottom: 6px;'>Operational Factors</div>", unsafe_allow_html=True)

        new_tariff = st.number_input(
            "Electricity Tariff (₹/kWh):",
            min_value=1.0,
            max_value=30.0,
            value=float(st.session_state.get("tariff_rate", DEFAULT_TARIFF_INR_PER_KWH)),
            step=0.25,
            help="Energy charge rate based on applicable TNEB tariff."
        )
        if new_tariff != st.session_state.get("tariff_rate"):
            st.session_state["tariff_rate"] = new_tariff
            run_sustainability_pipeline()

        new_factor = st.number_input(
            "Grid Emission Factor (kg CO₂e/kWh):",
            min_value=0.1,
            max_value=2.0,
            value=float(st.session_state.get("emissions_factor", DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH)),
            step=0.01,
            help="CEA national grid baseline factor."
        )
        if new_factor != st.session_state.get("emissions_factor"):
            st.session_state["emissions_factor"] = new_factor
            run_sustainability_pipeline()

        unit_check = st.checkbox(
            "Target confirmed as kWh",
            value=st.session_state.get("unit_is_verified", False),
            help="Check if campus meters record explicitly in kilowatt-hours."
        )
        if unit_check != st.session_state.get("unit_is_verified"):
            st.session_state["unit_is_verified"] = unit_check
            run_sustainability_pipeline()

        st.markdown("<hr style='margin: 16px 0; border-top: 1px solid #1e3b2f;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #8fa69a; line-height: 1.5;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>Telemetry Status</span>
                    <span style="color:#86efac; font-weight:600;">ACTIVE</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>Aggregation</span>
                    <span style="color:#cbd5e1;">Daily Total</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span>Target Metric</span>
                    <span style="color:#cbd5e1;">Total Load</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
