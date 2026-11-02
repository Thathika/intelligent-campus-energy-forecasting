"""
1_Data_Preprocessing.py
Data Ingestion, Verification, and Preprocessing
Editorial layout: clean schema audit, balance check, and chronological validation.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar,
    load_and_process_dataset
)
from src.preprocessing import preprocess_energy_data, export_dataframe_to_csv
from src.data_loader import inspect_component_relationships

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Ingestion & Audit</div>
    <h1 style="margin:0;">Data Preprocessing & Validation</h1>
    <div class="page-subtitle">Upload CSV telemetry, map date and target fields, audit data types, and verify component balance and chronological order.</div>
    """,
    unsafe_allow_html=True
)

raw_df = st.session_state.get("raw_df")
cleaned_df = st.session_state.get("cleaned_df")
report = st.session_state.get("preproc_report", {})
active_name = st.session_state.get("dataset_name", "None")

# 1. Source Selection & Ingestion
st.markdown("## Telemetry Ingestion")
col_up1, col_up2 = st.columns([3, 1])

with col_up1:
    uploaded_file = st.file_uploader(
        "Upload CSV Telemetry File:",
        type=["csv"],
        help="Select a CSV file containing date and energy readings."
    )

with col_up2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("Restore Default Dataset", use_container_width=True):
        load_and_process_dataset(source="default")
        st.success("Default July 2026 dataset restored.")
        st.rerun()

if uploaded_file is not None:
    current_name = st.session_state.get("dataset_name", "")
    if uploaded_file.name not in current_name:
        load_and_process_dataset(source="uploaded", uploaded_file=uploaded_file)
        st.success(f"Loaded: {uploaded_file.name}")
        st.rerun()

if raw_df is None:
    st.warning("No dataset loaded.")
    st.stop()

# 2. Schema and Column Mapping
st.markdown("## Telemetry Mapping")
col_c1, col_c2 = st.columns(2)

all_cols = list(raw_df.columns)
with col_c1:
    default_date_idx = all_cols.index(st.session_state.get("date_col")) if st.session_state.get("date_col") in all_cols else 0
    selected_date_col = st.selectbox(
        "Observation Date Field:",
        options=all_cols,
        index=default_date_idx
    )

with col_c2:
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist() or all_cols
    default_target_idx = numeric_cols.index(st.session_state.get("target_col")) if st.session_state.get("target_col") in numeric_cols else (len(numeric_cols) - 1)
    selected_target_col = st.selectbox(
        "Primary Demand Metric (Total):",
        options=numeric_cols,
        index=default_target_idx
    )

if (selected_date_col != st.session_state.get("date_col") or selected_target_col != st.session_state.get("target_col")):
    st.session_state["date_col"] = selected_date_col
    st.session_state["target_col"] = selected_target_col
    cleaned_df, report = preprocess_energy_data(raw_df, selected_date_col, selected_target_col)
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["preproc_report"] = report
    st.session_state["component_analysis"] = inspect_component_relationships(cleaned_df, selected_target_col)
    st.rerun()

# 3. Data Integrity & Summary
st.markdown("## Data Integrity & Quality Audit")

tot_records = len(raw_df)
tot_nulls = int(raw_df.isna().sum().sum())
tot_dups = int(raw_df.duplicated().sum())
date_span = f"{report.get('date_min', 'N/A')} to {report.get('date_max', 'N/A')}"

st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Observations</div>
            <div class="kpi-value">{tot_records}</div>
            <div class="kpi-sub">{len(raw_df.columns)} active columns recorded</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Missing Values</div>
            <div class="kpi-value">{tot_nulls}</div>
            <div class="kpi-sub">Across all recorded fields</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Duplicate Rows</div>
            <div class="kpi-value">{tot_dups}</div>
            <div class="kpi-sub">Identical timestamp checks</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Chronological Duration</div>
            <div class="kpi-value">{report.get('duration_days', 0)} Days</div>
            <div class="kpi-sub">{date_span}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Component Balance Inspection Callout
comp_rel = st.session_state.get("component_analysis")
if comp_rel:
    s_match = comp_rel.get("supply_sum_matches_target", False)
    d_match = comp_rel.get("demand_sum_matches_target", False)
    s_status = "Exact Balance Verified" if s_match else "Supply Sum Variance"
    d_status = "Exact Balance Verified" if d_match else "Demand Sum Variance"
    
    st.markdown(
        f"""
        <div class="data-strip">
            <div class="data-strip-item"><b>Supply Balance (TNEB + Solar + Diesel):</b> {s_status}</div>
            <div class="data-strip-sep">|</div>
            <div class="data-strip-item"><b>Facility Load Balance (Academics + Hostels + Chiller + Staff + STP):</b> {d_status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if report.get("warnings"):
    for w in report["warnings"]:
        st.markdown(f"<div class='notice-box-warning'>{w}</div>", unsafe_allow_html=True)

# 4. Table Preview & Schema Tabs
st.markdown("## Record Preview & Schema Audit")
tab_prev, tab_stats, tab_schema = st.tabs(["Record Preview", "Descriptive Statistics", "Schema Audit"])

with tab_prev:
    st.dataframe(raw_df.head(15), use_container_width=True)

with tab_stats:
    st.dataframe(raw_df.describe().T.style.format("{:,.2f}"), use_container_width=True)

with tab_schema:
    audit_rows = []
    for col in raw_df.columns:
        audit_rows.append({
            "Column": col,
            "Data Type": str(raw_df[col].dtype),
            "Null Count": int(raw_df[col].isna().sum()),
            "Unique Count": int(raw_df[col].nunique()),
            "Sample Value": str(raw_df[col].dropna().iloc[0]) if not raw_df[col].dropna().empty else ""
        })
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

# 5. Export Action
st.markdown("## Export Cleaned Data")
csv_data = export_dataframe_to_csv(cleaned_df)
st.download_button(
    label="Download Cleaned CSV",
    data=csv_data,
    file_name=f"cleaned_campus_energy_{report.get('date_min', 'data')}.csv",
    mime="text/csv"
)

st.markdown(
    """
    <div class="academic-footnote">
        Data integrity check: Telemetry dates are converted to ISO 8601 timestamps and strictly sorted in chronological order.
    </div>
    """,
    unsafe_allow_html=True
)
