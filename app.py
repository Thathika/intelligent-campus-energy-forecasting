"""
app.py
Campus Energy Overview
Editorial Energy Intelligence Dashboard for Educational Campus Electricity Monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Campus Energy Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar
)
from src.visualizations import (
    plot_historical_trend,
    plot_energy_sources,
    plot_campus_areas,
    plot_forecast_vs_actual
)

# Initialize application state & inject bespoke stylesheet
load_custom_css()
init_session_state()
render_sidebar()

cleaned_df = st.session_state.get("cleaned_df")
target_col = st.session_state.get("target_col", "Total")
date_col = "_parsed_datetime"
report = st.session_state.get("preproc_report", {})
summary = report.get("target_summary", {})
sust_summary = st.session_state.get("sustainability_summary", {})
anomaly_summary = st.session_state.get("anomaly_summary", {})
anomaly_df = st.session_state.get("anomaly_df", pd.DataFrame())
comp_df = st.session_state.get("comp_df")
future_df = st.session_state.get("future_df")
sel_model = st.session_state.get("selected_model", "Random Forest Regressor")

# 1. Editorial Header & Date Range
date_start = report.get("date_min", "2026-07-01")
date_end = report.get("date_max", "2026-07-31")
date_label = f"{pd.to_datetime(date_start).strftime('%d %b %Y')} – {pd.to_datetime(date_end).strftime('%d %b %Y')}" if date_start and date_end else "Observation Period"

st.markdown(
    f"""
    <div class="page-eyebrow">Campus Energy / Overview</div>
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 4px;">
        <h1 style="margin:0;">Energy Overview & Telemetry</h1>
        <div style="font-size:0.825rem; color:#4b5563; font-weight:600; background:#f1f0ec; padding:4px 10px; border-radius:4px; border:1px solid #e2dfd7;">
            {date_label}
        </div>
    </div>
    <div class="page-subtitle">Daily electricity consumption telemetry, on-site generation sources, load centers, and baseline forecasting.</div>
    """,
    unsafe_allow_html=True
)

if cleaned_df is None or cleaned_df.empty:
    st.warning("No dataset loaded. Please upload or reload data.")
    st.stop()

# 2. 4-Cell Analytical KPI Grid
tot_val = summary.get("count", 0) * summary.get("mean", 0.0)
mean_val = summary.get("mean", 0.0)
max_val = summary.get("max", 0.0)
flagged_c = anomaly_summary.get("flagged_count", 0) if anomaly_summary else 0
flagged_pct = anomaly_summary.get("flagged_percentage", 0.0) if anomaly_summary else 0.0

# Calculate peak date
peak_row = cleaned_df.loc[cleaned_df[target_col].idxmax()] if not cleaned_df.empty else None
peak_date_str = pd.to_datetime(peak_row[date_col]).strftime('%d %b') if peak_row is not None else "N/A"

st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Recorded Energy</div>
            <div class="kpi-value">{tot_val:,.0f}</div>
            <div class="kpi-sub">{len(cleaned_df)} observations recorded ({target_col})</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Daily Average Consumption</div>
            <div class="kpi-value">{mean_val:,.0f}</div>
            <div class="kpi-sub">Mean daily intake across campus</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Peak Recorded Reading</div>
            <div class="kpi-value">{max_val:,.0f}</div>
            <div class="kpi-sub">Highest daily demand on {peak_date_str}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Atypical Observations</div>
            <div class="kpi-value">{flagged_c}</div>
            <div class="kpi-sub">{flagged_pct:.1f}% flagged by statistical screening</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Operational context strip
verified = st.session_state.get("unit_is_verified", False)
cost_val = sust_summary.get("total_cost_inr", 0.0)
emiss_val = sust_summary.get("total_emissions_mt", 0.0)
tariff_val = st.session_state.get("tariff_rate", 8.0)

unit_badge = "Verified kWh" if verified else "Provisional units"
st.markdown(
    f"""
    <div class="data-strip">
        <div class="data-strip-item"><b>Estimated Expenditure:</b> ₹{cost_val:,.0f} (at ₹{tariff_val:.2f}/kWh, {unit_badge})</div>
        <div class="data-strip-sep">|</div>
        <div class="data-strip-item"><b>Grid Carbon Footprint:</b> {emiss_val:,.2f} MT CO₂e</div>
        <div class="data-strip-sep">|</div>
        <div class="data-strip-item"><b>Active Benchmark Model:</b> {sel_model}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Main Analytical Section: 68% / 32% Two-Column Split
col_chart, col_telemetry = st.columns([68, 32])

with col_chart:
    st.markdown("## Daily Energy Consumption")
    fig_trend = plot_historical_trend(cleaned_df, date_col, target_col)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_telemetry:
    st.markdown("## Telemetry Summary")
    
    # Calculate source & load metrics if available
    solar_sum = cleaned_df["Solar"].sum() if "Solar" in cleaned_df.columns else 0
    tneb_sum = cleaned_df["TNEB_Campus"].sum() if "TNEB_Campus" in cleaned_df.columns else 0
    total_gen = tot_val if tot_val > 0 else 1
    solar_share = (solar_sum / total_gen) * 100 if solar_sum > 0 else 0
    tneb_share = (tneb_sum / total_gen) * 100 if tneb_sum > 0 else 0
    
    diesel_active_days = int((cleaned_df["Diesel"] > 0).sum()) if "Diesel" in cleaned_df.columns else 0
    min_row = cleaned_df.loc[cleaned_df[target_col].idxmin()] if not cleaned_df.empty else None
    min_date_str = pd.to_datetime(min_row[date_col]).strftime('%d %b') if min_row is not None else "N/A"
    min_val = min_row[target_col] if min_row is not None else 0

    st.markdown(
        f"""
        <div class="telemetry-card">
            <div class="telemetry-header">Operational Generation Profile</div>
            <div class="telemetry-row">
                <span class="telemetry-label">Grid Supply Share (TNEB)</span>
                <span class="telemetry-val">{tneb_share:.1f}%</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">On-Site Solar Contribution</span>
                <span class="telemetry-val">{solar_share:.1f}%</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Diesel Dispatch Frequency</span>
                <span class="telemetry-val">{diesel_active_days} of {len(cleaned_df)} days</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Peak Demand Reading</span>
                <span class="telemetry-val">{max_val:,.0f} ({peak_date_str})</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Minimum Demand Reading</span>
                <span class="telemetry-val">{min_val:,.0f} ({min_date_str})</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Observation Period</span>
                <span class="telemetry-val">{len(cleaned_df)} consecutive days</span>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-label">Substation Feed</span>
                <span class="telemetry-val" style="color:#15803d; font-weight:600;">Standard HT-II</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 4. Secondary Analytical Row: Supply and Load Breakdown
st.markdown("## Supply Sources and Facility Load Distribution")
col_src, col_load = st.columns(2)

with col_src:
    fig_sources = plot_energy_sources(cleaned_df, date_col)
    if fig_sources:
        st.plotly_chart(fig_sources, use_container_width=True)
    else:
        st.info("Generation columns (Solar, Diesel, Biogas) not present in active dataset.")

with col_load:
    fig_areas = plot_campus_areas(cleaned_df, date_col)
    if fig_areas:
        st.plotly_chart(fig_areas, use_container_width=True)
    else:
        st.info("Facility load columns not present in active dataset.")

# 5. Out-of-Sample Demand Forecast
train_df = st.session_state.get("train_df")
test_df = st.session_state.get("test_df")
predictions = st.session_state.get("predictions", {})

if train_df is not None and test_df is not None:
    st.markdown("## Out-of-Sample Demand Forecast")
    fig_forecast = plot_forecast_vs_actual(
        train_df=train_df,
        test_df=test_df,
        predictions_dict=predictions,
        future_df=future_df,
        date_col=date_col,
        target_col=target_col,
        selected_model=sel_model
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# 6. Flagged Observations Table
st.markdown("## Flagged Atypical Observations")
if not anomaly_df.empty:
    display_cols = ["Date", "Observed Value", "Expected / Reference", "Absolute Deviation", "Percentage Deviation (%)", "Flag Reason"]
    existing_cols = [c for c in display_cols if c in anomaly_df.columns]
    st.dataframe(
        anomaly_df[existing_cols].style.format({
            "Observed Value": "{:,.0f}",
            "Expected / Reference": "{:,.0f}",
            "Absolute Deviation": "{:+,.0f}",
            "Percentage Deviation (%)": "{:+,.1f}%"
        }),
        use_container_width=True
    )
else:
    st.caption("No observations currently flagged under the active statistical threshold.")

# 7. Editorial Footnote
st.markdown(
    """
    <div class="academic-footnote">
        Data source: Historical campus meter records (July 2026). Analytical models use chronological splits to avoid lookahead bias.
        All forecasts and sustainability calculations represent decision-support estimates.
    </div>
    """,
    unsafe_allow_html=True
)
