"""
6_Cost_and_Carbon_Analysis.py
Electricity Cost and Grid Carbon Footprint Estimation
Editorial layout: utility billing modeling, carbon impact, and dual-axis telemetry.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar,
    run_sustainability_pipeline
)
from src.sustainability import calculate_cost_and_emissions
from src.visualizations import plot_cost_and_emissions_trend

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Sustainability</div>
    <h1 style="margin:0;">Cost & Carbon Footprint Modeling</h1>
    <div class="page-subtitle">Electricity expenditure and grid carbon footprint modeling based on configurable external utility factors.</div>
    """,
    unsafe_allow_html=True
)

cleaned_df = st.session_state.get("cleaned_df")
target_col = st.session_state.get("target_col", "Total")
date_col = "_parsed_datetime"
future_df = st.session_state.get("future_df")

if cleaned_df is None or cleaned_df.empty:
    st.warning("No preprocessed dataset available.")
    st.stop()

# 1. Operational Parameters & Verification
st.markdown("## Utility & Environmental Parameters")
col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1.2])

with col_p1:
    tariff_val = st.number_input(
        "Electricity Tariff Rate (₹/kWh):",
        min_value=1.0,
        max_value=30.0,
        value=float(st.session_state.get("tariff_rate", 8.0)),
        step=0.25,
        help="Applicable energy charge rate based on educational institution tariff."
    )
    if tariff_val != st.session_state.get("tariff_rate"):
        st.session_state["tariff_rate"] = tariff_val
        run_sustainability_pipeline()

with col_p2:
    factor_val = st.number_input(
        "Grid Emission Factor (kg CO₂e/kWh):",
        min_value=0.1,
        max_value=2.0,
        value=float(st.session_state.get("emissions_factor", 0.716)),
        step=0.01,
        help="Central Electricity Authority (CEA India) national grid baseline factor."
    )
    if factor_val != st.session_state.get("emissions_factor"):
        st.session_state["emissions_factor"] = factor_val
        run_sustainability_pipeline()

with col_p3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    verified = st.checkbox(
        "Unit confirmed as kWh",
        value=st.session_state.get("unit_is_verified", False)
    )
    if verified != st.session_state.get("unit_is_verified"):
        st.session_state["unit_is_verified"] = verified
        run_sustainability_pipeline()

if not verified:
    st.markdown(
        """
        <div class="notice-box-warning">
            <b>Provisional Engineering Units:</b> The source CSV column is labeled <code>Total</code> without stated units. 
            Financial and carbon metrics are calculated assuming 1 unit = 1 kWh. Check 'Unit confirmed as kWh' once verified against substation billing.
        </div>
        """,
        unsafe_allow_html=True
    )

# 2. Historical & Forecast Calculations
cost_series, emissions_series, hist_summary = calculate_cost_and_emissions(
    energy_series=cleaned_df[target_col],
    unit_is_verified_kwh=verified,
    tariff_inr_per_kwh=tariff_val,
    emissions_factor_kg_per_kwh=factor_val
)

has_forecast = (future_df is not None and not future_df.empty and f"predicted_{target_col}" in future_df.columns)
if has_forecast:
    f_cost_s, f_emiss_s, fut_summary = calculate_cost_and_emissions(
        energy_series=future_df[f"predicted_{target_col}"],
        unit_is_verified_kwh=verified,
        tariff_inr_per_kwh=tariff_val,
        emissions_factor_kg_per_kwh=factor_val
    )
else:
    fut_summary = None

# Metric Rows: Historical vs Forecasted
st.markdown("## Sustainability Summary Estimates")
col_h, col_f = st.columns(2)

with col_h:
    st.markdown(f"### Recorded Period ({len(cleaned_df)} Days)")
    st.markdown(
        f"""
        <div class="kpi-container" style="grid-template-columns: 1fr 1fr;">
            <div class="kpi-card">
                <div class="kpi-label">Estimated Expenditure</div>
                <div class="kpi-value">₹{hist_summary['total_cost_inr']:,.0f}</div>
                <div class="kpi-sub">Daily average: ₹{hist_summary['avg_daily_cost_inr']:,.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Carbon Footprint</div>
                <div class="kpi-value">{hist_summary['total_emissions_mt']:,.2f} MT</div>
                <div class="kpi-sub">{hist_summary['total_emissions_kg']:,.0f} kg CO₂e emitted</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_f:
    if has_forecast:
        st.markdown(f"### Forecasted Period ({len(future_df)} Days Ahead)")
        st.markdown(
            f"""
            <div class="kpi-container" style="grid-template-columns: 1fr 1fr;">
                <div class="kpi-card">
                    <div class="kpi-label">Projected Expenditure</div>
                    <div class="kpi-value">₹{fut_summary['total_cost_inr']:,.0f}</div>
                    <div class="kpi-sub">Daily average: ₹{fut_summary['avg_daily_cost_inr']:,.0f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Projected Carbon</div>
                    <div class="kpi-value">{fut_summary['total_emissions_mt']:,.2f} MT</div>
                    <div class="kpi-sub">{fut_summary['total_emissions_kg']:,.0f} kg CO₂e emitted</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("### Forecasted Period")
        st.caption("Execute forecasting to generate forward expenditure and carbon estimates.")

# 3. Dual-Axis Trend Visualization
st.markdown("## Daily Cost and Emissions Telemetry")
fig_sust = plot_cost_and_emissions_trend(
    dates=cleaned_df[date_col],
    cost_series=cost_series,
    emissions_series=emissions_series
)
st.plotly_chart(fig_sust, use_container_width=True)

# 4. Billing Disclaimer
st.markdown(
    """
    <div class="notice-box">
        <b>Tariff Accounting & Regulatory Disclaimers:</b><br>
        • <b>Single-Part Energy Model:</b> Calculations utilize a simplified volumetric formulation: <code>Cost = Energy × Tariff</code>.<br>
        • <b>Excluded Utility Elements:</b> Actual commercial utility billing (such as TANGEDCO HT Tariff II-B for educational institutions) 
          incorporates Maximum Demand (kVA) charges, power factor penalties/incentives, time-of-day (TOD) peak tariffs, and government electricity duty. 
          These components are excluded from this academic estimation.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="academic-footnote">
        Emission factor sourced from Central Electricity Authority (CEA) CO₂ Baseline Database for the Indian Power Sector (User Guide Version 19.0).
    </div>
    """,
    unsafe_allow_html=True
)
