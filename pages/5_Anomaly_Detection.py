"""
5_Anomaly_Detection.py
Statistical Outlier Identification and Inspection
Editorial layout: statistical deviation screening, outlier visualization, and component inspection.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar,
    run_anomaly_pipeline
)
from src.visualizations import plot_anomalies_scatter

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Surveillance</div>
    <h1 style="margin:0;">Consumption Anomaly Detection</h1>
    <div class="page-subtitle">Statistical deviation screening to identify atypical consumption readings requiring facility inspection.</div>
    """,
    unsafe_allow_html=True
)

cleaned_df = st.session_state.get("cleaned_df")
target_col = st.session_state.get("target_col", "Total")
date_col = "_parsed_datetime"

if cleaned_df is None or cleaned_df.empty:
    st.warning("No preprocessed dataset available.")
    st.stop()

# 1. Method & Sensitivity Configuration
st.markdown("## Screening Configuration")
col_meth, col_thresh = st.columns([2, 1])

with col_meth:
    method = st.selectbox(
        "Statistical Baseline Algorithm:",
        options=[
            ("zscore", "Z-Score (Standard deviation from global mean)"),
            ("rolling", "Rolling Baseline (Deviation from 7-day moving median)"),
            ("iqr", "Interquartile Range (IQR non-parametric boundary)")
        ],
        format_func=lambda x: x[1],
        index=0 if st.session_state.get("anomaly_method", "zscore") == "zscore" else 1
    )
    selected_method_key = method[0]
    if selected_method_key != st.session_state.get("anomaly_method"):
        st.session_state["anomaly_method"] = selected_method_key
        run_anomaly_pipeline()

with col_thresh:
    curr_thresh = float(st.session_state.get("anomaly_threshold", 2.0))
    threshold = st.slider(
        "Sensitivity Multiplier (Threshold):",
        min_value=1.0,
        max_value=3.5,
        value=curr_thresh,
        step=0.1
    )
    if threshold != st.session_state.get("anomaly_threshold"):
        st.session_state["anomaly_threshold"] = threshold
        run_anomaly_pipeline()

anomaly_df = st.session_state.get("anomaly_df", pd.DataFrame())
anomaly_summary = st.session_state.get("anomaly_summary", {})

# 2. KPI Summary Grid
flag_c = anomaly_summary.get('flagged_count', 0)
tot_obs = anomaly_summary.get('total_observations', len(cleaned_df))
flag_pct = anomaly_summary.get('flagged_percentage', 0.0)

st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Observations Analyzed</div>
            <div class="kpi-value">{tot_obs}</div>
            <div class="kpi-sub">Total daily telemetry records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Flagged Atypical Days</div>
            <div class="kpi-value">{flag_c}</div>
            <div class="kpi-sub">{flag_pct:.1f}% of total telemetry</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Active Statistical Filter</div>
            <div class="kpi-value" style="font-size:1.15rem; padding-top:4px;">{selected_method_key.upper()}</div>
            <div class="kpi-sub">{threshold}x standard deviation / IQR scale</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Anomaly Visualization
st.markdown("## Time-Series Anomaly Inspection")
fig_anomaly = plot_anomalies_scatter(cleaned_df, date_col, target_col, anomaly_df)
st.plotly_chart(fig_anomaly, use_container_width=True)

# 4. Audit Table
st.markdown("## Flagged Readings Audit")
if not anomaly_df.empty:
    st.dataframe(
        anomaly_df.style.format({
            "Observed Value": "{:,.0f}",
            "Expected / Reference": "{:,.0f}",
            "Absolute Deviation": "{:+,.0f}",
            "Percentage Deviation (%)": "{:+,.1f}%"
        }),
        use_container_width=True
    )

    st.markdown("### Component Breakdown for Flagged Date")
    flagged_dates = anomaly_df["Date"].tolist()
    chosen_date_str = st.selectbox("Select flagged date to inspect telemetry components:", options=flagged_dates)
    
    target_row = cleaned_df[pd.to_datetime(cleaned_df[date_col]).dt.strftime("%Y-%m-%d") == chosen_date_str]
    if not target_row.empty:
        st.dataframe(target_row.drop(columns=[date_col], errors="ignore"), use_container_width=True)
else:
    st.caption("No observations currently exceed the selected deviation threshold.")

# 5. Operational Notice
st.markdown(
    """
    <div class="notice-box">
        <b>Operational Context for Campus Facilities:</b><br>
        Flagged observations represent readings that deviate significantly from baseline expectations. 
        In university campus environments, common causes include:
        <ul style="margin: 6px 0 0 16px; padding: 0;">
            <li>Scheduled weekend chiller plant shutdowns (chiller load drops ~90%).</li>
            <li>Campus holidays or unscheduled power outages requiring diesel generator backup.</li>
            <li>Special academic symposiums, orientation events, or laboratory intensive experiments.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="academic-footnote">
        Anomaly screening is an advisory statistical filter and should be corroborated by substation logbooks before dispatching maintenance staff.
    </div>
    """,
    unsafe_allow_html=True
)
