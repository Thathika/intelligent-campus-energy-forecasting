"""
7_Project_Information.py
Academic Project Documentation, Architecture, and Methodology
Editorial layout: objectives, architectural schematics, model formulations, and research scope.
"""

import streamlit as st

from src.session_state import (
    init_session_state,
    load_custom_css,
    render_sidebar
)

load_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    """
    <div class="page-eyebrow">Campus Energy / Methodology</div>
    <h1 style="margin:0;">Project Documentation & Architecture</h1>
    <div class="page-subtitle">Academic objectives, sequential pipeline architecture, machine learning model specifications, and research limitations.</div>
    """,
    unsafe_allow_html=True
)

# 1. Project Background & Problem Statement
st.markdown("## Background and Objectives")
st.markdown(
    """
    Higher education campuses operate complex electrical infrastructures that combine academic classrooms, research laboratories, 
    student residences, administrative facilities, and central utility plants (e.g., HVAC chillers and water treatment plants). 
    Power is supplied through utility grid connections supplemented by on-site generation sources such as solar photovoltaics and standby diesel units.

    **Core Academic Objectives:**
    1. **Data Preprocessing & Validation:** Clean, validate, and chronologically sort daily energy readings across supply and demand meters.
    2. **Exploratory Data Analysis:** Characterize daily demand curves, weekday versus weekend load variations, and chiller sensitivity.
    3. **Leakage-Free Feature Engineering:** Construct calendar and autoregressive lag/rolling features without lookahead bias.
    4. **Comparative Machine Learning Forecasting:** Fit and evaluate candidate models (Seasonal Naïve, Linear Regression, Random Forest, Gradient Boosting).
    5. **Out-of-Sample Chronological Validation:** Evaluate models exclusively on held-out test periods to reflect actual operational deployment.
    6. **Statistical Anomaly Identification:** Screen for atypical consumption readings using explainable deviation baselines (Z-score, rolling median, IQR).
    7. **Operational Cost and Carbon Estimation:** Provide decision-support indicators for electricity expenditure and emissions using configurable factors.
    """
)

# 2. Sequential System Workflow
st.markdown("## System Architecture Pipeline")
st.markdown(
    """
    The application follows an end-to-end sequential data and modeling pipeline:

    ```
    Raw CSV Telemetry
          │
          ▼
    1. Ingestion & Validation    ──> Date parsing (DD.MM.YYYY), sorting, component balance check
          │
          ▼
    2. Feature Engineering       ──> Lag-1, shifted rolling-3 mean, calendar indicators (no component leakage)
          │
          ▼
    3. Chronological Split       ──> Earliest days for training, latest days for held-out evaluation
          │
          ▼
    4. Model Benchmarking        ──> Fit candidate models, compute out-of-sample MAE, RMSE, MAPE
          │
          ▼
    5. Decision-Support Views    ──> Multi-step forecasting, anomaly review, cost & carbon estimation
    ```
    """
)

# 3. Model Specifications
st.markdown("## Forecasting Model Specifications")
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(
        """
        **Seasonal Naïve (Persistence Baseline)**  
        - *Formulation:* $\\hat{y}_t = y_{t-1}$  
        - *Role:* Computational reference. Any viable machine learning model must surpass this simple autoregressive baseline to justify complexity.

        **Linear Regression**  
        - *Formulation:* Ordinary least squares regression over temporal flags and recent lag variables:  
          $\\hat{y}_t = \\beta_0 + \\beta_1 y_{t-1} + \\beta_2 \\text{roll}_3(y) + \\sum \\gamma_j \\text{Day}_j$  
        - *Role:* Transparent, fully interpretable linear benchmark.
        """
    )

with col_s2:
    st.markdown(
        """
        **Random Forest Regressor**  
        - *Hyperparameters:* 50 estimators, maximum depth of 3, minimum split size of 3, minimum leaf of 2.  
        - *Role:* Conservative ensemble calibrated to prevent overfitting on small sample sizes while modeling non-linear interactions.

        **Gradient Boosting Regressor**  
        - *Hyperparameters:* 40 sequential estimators, maximum depth of 2, learning rate of 0.05.  
        - *Role:* Captures subtle interactions between weekday academic schedules and recent consumption inertia.
        """
    )

# 4. Prototype Limitations
st.markdown("## Scope and Research Limitations")
st.markdown(
    """
    <div class="notice-box">
        <b>Academic Scope & Data Limitations:</b><br>
        1. <b>Temporal Horizon (31 Observations):</b> The provided empirical campus dataset covers only 1 July 2026 through 31 July 2026. 
           Annual seasonal variations (monsoon, semester examinations, vacation periods) cannot be modeled mathematically from this single-month sample.<br>
        2. <b>Exogenous Variables:</b> Ambient temperature, relative humidity, solar irradiance, and student attendance schedules were not available in the telemetry.<br>
        3. <b>Decision-Support Role:</b> The system is an advisory analytical tool and does not actuate physical electrical equipment or guarantee financial savings.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="academic-footnote">
        Reference Project: "Intelligent Campus Energy Demand Forecasting Using Machine Learning". Developed for educational campus facility intelligence.
    </div>
    """,
    unsafe_allow_html=True
)
