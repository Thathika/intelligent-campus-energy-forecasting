"""
4_Model_Evaluation.py
Forecasting Model Performance Evaluation and Residual Analysis
Editorial layout: model ranking table, error benchmarks, residual diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar
)
from src.visualizations import plot_model_comparison_bars, plot_residuals_scatter

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Validation</div>
    <h1 style="margin:0;">Model Performance & Benchmarking</h1>
    <div class="page-subtitle">Out-of-sample prediction error metrics (MAE, RMSE, MAPE) evaluated on identical held-out test observations.</div>
    """,
    unsafe_allow_html=True
)

comp_df = st.session_state.get("comp_df")
test_df = st.session_state.get("test_df")
target_col = st.session_state.get("target_col", "Total")
date_col = "_parsed_datetime"
predictions = st.session_state.get("predictions", {})

if comp_df is None or comp_df.empty or test_df is None:
    st.warning("Model evaluation data is not available. Please run forecasting first.")
    st.stop()

# 1. Performance Benchmark Table
st.markdown("## Out-of-Sample Performance Benchmark")
st.dataframe(
    comp_df.style.highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#dcfce7"),
    use_container_width=True
)

# 2. Visual Bar Comparison
st.markdown("## Error Comparison (MAE & RMSE)")
fig_bars = plot_model_comparison_bars(comp_df)
st.plotly_chart(fig_bars, use_container_width=True)

# 3. Test Predictions Table
st.markdown("## Held-Out Test Predictions vs. Actuals")

pred_display_df = pd.DataFrame()
pred_display_df["Evaluation Date"] = pd.to_datetime(test_df[date_col]).dt.strftime("%d %b %Y (%a)")
pred_display_df[f"Actual ({target_col})"] = test_df[target_col].values

for model_name, preds in predictions.items():
    pred_display_df[model_name] = np.round(preds, 1)

st.dataframe(
    pred_display_df.style.format({c: "{:,.1f}" for c in pred_display_df.columns if c != "Evaluation Date"}),
    use_container_width=True
)

# 4. Residuals Analysis
st.markdown("## Prediction Residuals Diagnostics")
col_res_select, col_res_stat = st.columns([2, 1])

with col_res_select:
    chosen_res_model = st.selectbox(
        "Select candidate model to inspect residuals:",
        options=list(predictions.keys()),
        index=0
    )

res_vector = test_df[target_col].values - predictions[chosen_res_model]

fig_res = plot_residuals_scatter(
    test_df=test_df,
    date_col=date_col,
    residuals=res_vector,
    model_name=chosen_res_model
)
st.plotly_chart(fig_res, use_container_width=True)

with col_res_stat:
    res_mean = float(np.mean(res_vector))
    res_std = float(np.std(res_vector))
    st.markdown("#### Residual Statistical Properties")
    st.markdown(
        f"""
        <div class="telemetry-card" style="padding:14px 16px;">
            <div class="telemetry-row">
                <span class="telemetry-label">Mean Bias Error (MBE)</span>
                <span class="telemetry-val">{res_mean:+,.1f}</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Standard Deviation</span>
                <span class="telemetry-val">{res_std:,.1f}</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Max Underprediction</span>
                <span class="telemetry-val">{float(np.max(res_vector)):+,.1f}</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Max Overprediction</span>
                <span class="telemetry-val">{float(np.min(res_vector)):+,.1f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 5. Metric Mathematical Reference
st.markdown("## Evaluation Metric Formulations")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        """
        **Mean Absolute Error (MAE)**  
        $$\\text{MAE} = \\frac{1}{n} \\sum |y_i - \\hat{y}_i|$$  
        Linear average of prediction error magnitudes. Directly expressed in recorded campus energy units.
        """
    )

with col_m2:
    st.markdown(
        """
        **Root Mean Squared Error (RMSE)**  
        $$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum (y_i - \\hat{y}_i)^2}$$  
        Quadratic penalty that emphasizes larger forecast misses. Crucial for peak utility demand sizing.
        """
    )

with col_m3:
    st.markdown(
        """
        **Mean Absolute Percentage Error (MAPE)**  
        $$\\text{MAPE} = \\frac{100\\%}{n} \\sum \\left|\\frac{y_i - \\hat{y}_i}{y_i}\\right|$$  
        Normalized percentage metric allowing comparison across different sub-meter scales.
        """
    )

st.markdown(
    """
    <div class="notice-box">
        <b>Sample Size Notice:</b> Evaluation metrics reflect performance over a 7-day chronological held-out test window. 
        While chronological validation eliminates target leakage, results cannot be assumed to generalize across different seasonal cycles or academic semesters.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="academic-footnote">
        Model ranking automatically highlights the lowest error values across the test period. Lower MAE and RMSE indicate superior predictive accuracy.
    </div>
    """,
    unsafe_allow_html=True
)
