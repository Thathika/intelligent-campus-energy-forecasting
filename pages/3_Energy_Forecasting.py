"""
3_Energy_Forecasting.py
Machine Learning Energy Demand Forecasting and Recursive Predictions
Editorial layout: chronological validation, candidate models, and multi-step projections.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar,
    run_full_ml_pipeline
)
from src.visualizations import plot_forecast_vs_actual
from src.forecasting_models import generate_future_forecast

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Machine Learning</div>
    <h1 style="margin:0;">Energy Demand Forecasting</h1>
    <div class="page-subtitle">Candidate machine learning regression models, chronological out-of-sample validation, and recursive future projections.</div>
    """,
    unsafe_allow_html=True
)

cleaned_df = st.session_state.get("cleaned_df")
target_col = st.session_state.get("target_col", "Total")
date_col = "_parsed_datetime"

if cleaned_df is None or cleaned_df.empty:
    st.warning("No preprocessed dataset available.")
    st.stop()

# 1. Chronological Split & Horizon Configuration
st.markdown("## Validation and Forecast Setup")
col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 2, 1])

with col_cfg1:
    max_test = min(10, max(2, len(cleaned_df) // 3))
    curr_test = st.session_state.get("test_days", min(7, max_test))
    test_days = st.slider(
        "Held-out test period (Days):",
        min_value=2,
        max_value=max_test,
        value=curr_test,
        help="Latest chronological records reserved exclusively for evaluation."
    )
    if test_days != st.session_state.get("test_days"):
        st.session_state["test_days"] = test_days
        run_full_ml_pipeline()

with col_cfg2:
    horizon = st.selectbox(
        "Forecast horizon (Days ahead):",
        options=[1, 3, 7],
        index=2 if st.session_state.get("forecast_horizon", 7) == 7 else 0,
        help="Number of days to forecast forward via recursive autoregression."
    )
    if horizon != st.session_state.get("forecast_horizon"):
        st.session_state["forecast_horizon"] = horizon
        run_full_ml_pipeline()

with col_cfg3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Fit & Forecast", use_container_width=True):
        with st.spinner("Training models chronologically..."):
            run_full_ml_pipeline()
        st.success("Models fit and forecast updated.")

train_df = st.session_state.get("train_df")
test_df = st.session_state.get("test_df")
trained_models = st.session_state.get("trained_models", {})
predictions = st.session_state.get("predictions", {})
comp_df = st.session_state.get("comp_df")

if train_df is None or test_df is None:
    st.error("Unable to execute chronological split. Insufficient observations.")
    st.stop()

train_start = pd.to_datetime(train_df[date_col].min()).strftime('%d %b %Y')
train_end = pd.to_datetime(train_df[date_col].max()).strftime('%d %b %Y')
test_start = pd.to_datetime(test_df[date_col].min()).strftime('%d %b %Y')
test_end = pd.to_datetime(test_df[date_col].max()).strftime('%d %b %Y')

st.markdown(
    f"""
    <div class="data-strip">
        <div class="data-strip-item"><b>Training Window:</b> {train_start} – {train_end} ({len(train_df)} days)</div>
        <div class="data-strip-sep">|</div>
        <div class="data-strip-item"><b>Held-Out Test Window:</b> {test_start} – {test_end} ({len(test_df)} days)</div>
        <div class="data-strip-sep">|</div>
        <div class="data-strip-item"><b>Validation Scheme:</b> Chronological (No Shuffle)</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Model Selection & Visual Comparison
st.markdown("## Forecast Visualization")
available_models = list(trained_models.keys())
default_model_idx = 0
best_model_name = comp_df.iloc[0]["Model"] if (comp_df is not None and not comp_df.empty) else available_models[0]
if best_model_name in available_models:
    default_model_idx = available_models.index(best_model_name)

selected_model = st.selectbox(
    "Active forecasting model for visualization:",
    options=available_models,
    index=default_model_idx
)

if selected_model != st.session_state.get("selected_model"):
    st.session_state["selected_model"] = selected_model
    if selected_model in trained_models:
        st.session_state["future_df"] = generate_future_forecast(
            model=trained_models[selected_model],
            historical_df=cleaned_df,
            date_col=date_col,
            target_col=target_col,
            feature_cols=st.session_state["feature_cols"],
            horizon_days=st.session_state.get("forecast_horizon", 7)
        )

future_df = st.session_state.get("future_df")

fig_forecast = plot_forecast_vs_actual(
    train_df=train_df,
    test_df=test_df,
    predictions_dict=predictions,
    future_df=future_df,
    date_col=date_col,
    target_col=target_col,
    selected_model=selected_model
)
st.plotly_chart(fig_forecast, use_container_width=True)

# 3. Future Predicted Demand
st.markdown(f"## Projected Future Demand ({selected_model})")

if future_df is not None and not future_df.empty:
    display_future = pd.DataFrame()
    display_future["Forecast Date"] = pd.to_datetime(future_df[date_col]).dt.strftime("%d %b %Y (%a)")
    display_future["Projected Value (Units)"] = future_df[f"predicted_{target_col}"].apply(lambda x: f"{x:,.0f}")
    display_future["Day Classification"] = np.where(future_df["is_weekend"] == 1, "Weekend", "Weekday")
    display_future["Lag-1 Recursive Feature"] = future_df["target_lag_1"].apply(lambda x: f"{x:,.0f}")
    display_future["Rolling-3 Mean Feature"] = future_df["target_rolling_mean_3"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(display_future, use_container_width=True)

# 4. Methodological Note
st.markdown(
    """
    <div class="notice-box">
        <b>Methodological Guarantees:</b><br>
        • <b>Target Leakage Prevention:</b> Component meters (TNEB, Solar, Diesel, Chiller, Hostels) are strictly excluded from prediction features. 
          Models only use strictly antecedent target lags (Lag-1) and shifted rolling statistics.<br>
        • <b>Chronological Honesty:</b> Data is partitioned strictly by time. Random k-fold cross-validation is not used because time series data exhibits temporal dependency.<br>
        • <b>Small-Sample Uncertainty:</b> With 31 daily observations, long-term seasonal patterns (annual academic calendar) cannot be confirmed mathematically. 
          Forecasts capture short-horizon autoregressive persistence and day-of-week load structure.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="academic-footnote">
        Multi-step ahead projections utilize recursive autoregression, updating the lag-1 and rolling features with previous model predictions.
    </div>
    """,
    unsafe_allow_html=True
)
