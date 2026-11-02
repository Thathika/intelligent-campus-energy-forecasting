# Campus Energy Analytics & Forecasting System

An interactive decision-support application for educational campus electricity telemetry, machine learning demand forecasting, explainable anomaly detection, and operational sustainability modeling.

Designed as an enterprise-grade analytical tool for university campus energy planning.

---

## Project Overview

Educational institutions operate diverse electrical loads—including academic lecture halls, computational laboratories, student dormitories, dining halls, and central chiller utilities—supplied by utility grid power (TNEB), rooftop solar photovoltaics, diesel generators, and biogas units.

This system provides energy managers and facility teams with:
1. **Historical Telemetry Analytics:** Detailed breakdowns across grid intake, on-site generation sources, and facility consumption zones.
2. **Leakage-Free Feature Engineering:** Time-based and autoregressive lag indicators designed with strict chronological shifting.
3. **Machine Learning Forecasting:** Candidate models (Seasonal Naïve Baseline, Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor) tuned with conservative hyperparameters for small-sample stability.
4. **Chronological Validation:** Held-out testing that strictly avoids random data shuffling and lookahead bias.
5. **Explainable Anomaly Identification:** Transparent statistical baselines (Z-score, rolling median, IQR) to flag atypical readings for operational review.
6. **Cost & Carbon Sustainability Indicators:** Configurable electricity tariffs (₹/kWh) and grid emission factors (kg CO₂e/kWh) with explicit unit confirmation.
7. **Synthetic Demonstration Mode:** A reproducible 120-day simulated dataset to demonstrate multi-month seasonal patterns and model scaling.

> **Notice:** This software is an analytical decision-support and forecasting tool. It does not directly actuate electrical hardware, switch breakers, or claim unverified financial savings.

---

## Project Folder Structure

```
campus-energy-forecasting/
├── app.py                     # Main dashboard overview (Campus Energy Overview)
├── requirements.txt           # Python dependency specifications
├── README.md                  # System manual and Windows execution guide
├── .gitignore                 # Standard Python and environment ignore rules
├── .streamlit/
│   ├── config.toml            # Clean light enterprise theme and server settings
│   └── credentials.toml       # Streamlit telemetry configuration
│
├── data/
│   ├── preprocessed_energy.csv # Default empirical dataset (31 daily observations, July 2026)
│   └── preprocessed_energy (1).csv # Preserved uploaded dataset copy
│
├── docs/
│   └── final report(1).pdf    # Primary reference academic report
│
├── src/
│   ├── data_loader.py         # CSV loading, auto-column detection, and component sum auditing
│   ├── preprocessing.py       # Datetime parsing (DD.MM.YYYY), sorting, and null/duplicate audits
│   ├── feature_engineering.py # Leakage-free lag, rolling mean, and calendar feature creation
│   ├── forecasting_models.py  # Seasonal Naïve, Linear Regression, Random Forest, Gradient Boosting
│   ├── evaluation.py          # MAE, RMSE, safe MAPE, model ranking, and residual analysis
│   ├── anomaly_detection.py   # Explainable statistical outlier detection (Z-score, rolling, IQR)
│   ├── sustainability.py      # Configurable electricity cost and carbon emissions calculators
│   ├── visualizations.py      # Plotly interactive charts with restrained enterprise palette
│   └── session_state.py       # Centralized state management and shared sidebar controls
│
├── pages/
│   ├── 1_Data_Preprocessing.py        # CSV upload, column mapping, and data cleaning
│   ├── 2_Exploratory_Data_Analysis.py  # Historical trends, distributions, and correlation heatmap
│   ├── 3_Energy_Forecasting.py         # Chronological training and recursive future prediction
│   ├── 4_Model_Evaluation.py           # Multi-model error benchmarks and residual error analysis
│   ├── 5_Anomaly_Detection.py          # Flagged atypical readings and component deep-dive
│   ├── 6_Cost_and_Carbon_Analysis.py   # Tariff expenditure and carbon footprint estimation
│   └── 7_Project_Information.py        # Academic methodology, architecture, and limitations
│
├── utils/
│   └── sample_data.py         # Reproducible synthetic dataset generator (120 days)
│
├── assets/
│   └── style.css              # Custom CSS: clean light theme, compact cards, subtle borders
│
└── tests/
    ├── test_preprocessing.py  # Unit tests for loading, date parsing, and math sum balance
    ├── test_evaluation.py     # Unit tests for MAE, RMSE, and MAPE metrics
    ├── test_sustainability.py # Unit tests for cost and emissions calculations
    └── test_app_workflow.py   # End-to-end integration tests for real and synthetic pipelines
```

---

## Technology Stack

- **Language:** Python 3.10+ (tested on Python 3.14)
- **User Interface:** Streamlit (with Custom CSS styling)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Baselines:** Scikit-learn (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor)
- **Interactive Visualizations:** Plotly Express, Plotly Graph Objects
- **Automated Testing:** Pytest
- **Offline Capable:** Runs 100% locally without external AI APIs or paid cloud connections.

---

## Installation & Execution (Windows Instructions)

### 1. Open Terminal or PowerShell
Navigate to the project workspace directory:
```powershell
cd "c:\Users\Thathika\OneDrive\Desktop\ENERGY DEMAND FORECASTING\-ENERGY-DEMAND-FORECASTING"
```

### 2. Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Verify that all unit and integration tests pass:
```powershell
pytest tests
```

### 5. Launch Application
```powershell
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

---

## Dataset Description (July 2026 Campus Records)

The default dataset, `data/preprocessed_energy.csv`, contains 31 daily observations covering **1 July 2026 through 31 July 2026**.

### Column Definitions:
| Column Name | Category | Description |
|---|---|---|
| `Date` | Timestamp | Recorded observation date (format: `DD.MM.YYYY`) |
| `TNEB_Campus` | Supply (Grid) | Electricity imported from Tamil Nadu Electricity Board for main campus |
| `TNEB_New_STP` | Supply (Grid) | Grid electricity imported for the new Sewage Treatment Plant |
| `Solar` | Supply (Clean) | On-site solar photovoltaic rooftop generation |
| `Diesel` | Supply (Backup)| On-site diesel generator standby generation |
| `Biogas` | Supply (Clean) | On-site biomass/biogas energy output |
| `Staff_Quarters` | Demand (Load) | Residential faculty and staff quarters consumption |
| `Academic_Blocks` | Demand (Load) | Lecture halls, classrooms, laboratories, and administrative offices |
| `Hostels` | Demand (Load) | Student dormitory and residential blocks |
| `Chiller` | Demand (Load) | Central HVAC chiller plant for air conditioning |
| `STP` | Demand (Load) | Sewage Treatment Plant electrical load |
| `Total` | Target Variable | Aggregate campus energy reading |

### Verified Mathematical Accounting Balance:
Mathematical inspection reveals that for all 31 observations:
$$\text{Supply Sum} = \text{TNEB\_Campus} + \text{TNEB\_New\_STP} + \text{Solar} + \text{Diesel} + \text{Biogas} = \text{Total}$$
$$\text{Demand Sum} = \text{Staff\_Quarters} + \text{Academic\_Blocks} + \text{Hostels} + \text{Chiller} + \text{STP} = \text{Total}$$

> **Target Leakage Mitigation:** Component columns are mathematically identical to the target when summed. Therefore, **component sub-meters are strictly excluded from predictive features** to prevent circular data leakage.

---

## Forecasting Methodology & Small-Data Strategy

With 31 daily records, conventional random splits risk temporal leakage and overfitting. The system implements a small-data strategy:

1. **Strict Chronological Split:**
   - **Training Set:** Earlier observations (e.g., Days 1–24, July 1–24)
   - **Held-Out Test Set:** Latest observations (e.g., Days 25–31, July 25–31)
   - Zero random shuffling.
2. **Conservative Feature Set:**
   - `day_of_week`, `is_weekend`, `day_of_month`
   - `target_lag_1` (prior day target value)
   - `target_rolling_mean_3` (shifted by 1 day so today's target is never part of its own feature)
   - `target_lag_7` (7-day lag, included only when history permits)
3. **Conservative Hyperparameters:**
   - **Random Forest:** `n_estimators=50, max_depth=3, min_samples_split=3, min_samples_leaf=2`
   - **Gradient Boosting:** `n_estimators=40, max_depth=2, learning_rate=0.05, min_samples_split=3`
4. **Recursive Multi-Step Forecasting:**
   - For horizons of 1, 3, or 7 days ahead (e.g. into August 2026), predictions from step $t$ are fed autoregressively into lag features for step $t+1$. Actual future target values are never used.

---

## Evaluation Metrics

All models are evaluated on the exact same held-out test observations using:

- **Mean Absolute Error (MAE):** Average absolute deviation in physical units.
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE):** Penalizes large peak misses quadratically.
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Mean Absolute Percentage Error (MAPE):** Scale-independent percentage error with division-by-zero safeguards.
  $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

---

## Cost and Carbon Estimation

- **Simplified Energy Charge:**
  $$\text{Estimated Cost (₹)} = \text{Energy (kWh)} \times \text{Tariff (₹/kWh)}$$
  *(Default: ₹8.00/kWh based on TNEB High-Tension HT Tariff II-B).*
- **Grid Carbon Footprint:**
  $$\text{Estimated Carbon (kg CO}_2\text{e)} = \text{Energy (kWh)} \times \text{Emissions Factor (kg CO}_2\text{e/kWh)}$$
  *(Default: 0.716 kg CO₂e/kWh based on Central Electricity Authority (CEA) Baseline Database v19).*
- **Unit Verification:** The raw CSV lacks an explicit unit designation. The application clearly labels all calculations as **Provisional** until the user confirms the unit via the sidebar toggle.

---

## Synthetic Demonstration Mode

To demonstrate how the system handles larger multi-month timeframes, toggle to **Synthetic Simulation (120 Days)** in the sidebar.  
- Generated using a fixed random seed (`seed=42`) for 100% reproducibility.
- Features realistic weekday/weekend cycles, chiller fluctuations, and injected test anomalies.
- Clearly marked with prominent **Synthetic** status pills.
- Completely isolated from real data; never combined with empirical records.

---

## Academic Limitations & Disclaimers

1. **One-Month Horizon:** The empirical dataset covers only July 2026. Annual seasonal patterns (monsoon, semester breaks, exam periods) cannot be modeled mathematically from 31 observations.
2. **Missing Weather Variables:** Ambient temperature and relative humidity were not recorded in the telemetry.
3. **Decision-Support Tool:** The application provides advisory estimates and does not control electrical switchgear or guarantee specific cost reductions.
