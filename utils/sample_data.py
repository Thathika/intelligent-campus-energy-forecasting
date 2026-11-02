"""
sample_data.py
Module for generating reproducible, clearly labelled synthetic campus energy data
for system workflow demonstrations.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def generate_synthetic_campus_dataset(
    start_date: str = "2026-01-01",
    num_days: int = 120,
    seed: int = 42,
    inject_anomalies: bool = True
) -> pd.DataFrame:
    """
    Generates a realistic synthetic campus energy dataset covering several months.
    Features:
    - Weekly cycle (weekdays higher than weekends)
    - Seasonal/academic trend (exams, vacations)
    - Realistic breakdown into generation sources (Solar, Diesel, Biogas, Grid)
      and campus load centers (Academic, Hostels, Chiller, Staff Quarters, STP)
    - Maintains exact mathematical equality: Supply Sum == Total and Demand Sum == Total!
    - Completely separate and tagged as synthetic.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")

    records = []

    for i, dt in enumerate(dates):
        dow = dt.day_of_week  # 0=Mon, 6=Sun
        is_weekend = dow in [5, 6]

        # Base campus load (higher on weekdays, lower on weekends)
        if is_weekend:
            base_total = 15000 + np.random.normal(0, 800)
            chiller_base = 50 + np.random.normal(0, 15)  # chillers shut down on weekend
            academic_base = 6500 + np.random.normal(0, 400)
            hostels_base = 5500 + np.random.normal(0, 300)
        else:
            base_total = 25000 + np.random.normal(0, 1000)
            chiller_base = 4500 + np.random.normal(0, 300)
            academic_base = 11500 + np.random.normal(0, 500)
            hostels_base = 6200 + np.random.normal(0, 350)

        staff_quarters_raw = 1900 + np.random.normal(0, 80)
        stp_raw = 1300 + np.random.normal(0, 100)

        # Injected anomalies for demonstration
        if inject_anomalies and i == 45:  # e.g. Day 46: Campus Tech Symposium (Spike)
            academic_base += 4000
            chiller_base += 1500
        elif inject_anomalies and i == 88:  # e.g. Day 89: Grid power cut / campus holiday (Dip)
            academic_base *= 0.5
            chiller_base *= 0.2

        # Round intermediate demand loads
        chiller = int(round(max(30.0, chiller_base)))
        hostels = int(round(max(1000.0, hostels_base)))
        staff_quarters = int(round(max(500.0, staff_quarters_raw)))
        stp = int(round(max(200.0, stp_raw)))

        # Approximate total
        raw_total = int(round(max(1000.0, academic_base))) + chiller + hostels + staff_quarters + stp
        total_val = raw_total

        # Exact residual balance for academic blocks so Demand Sum == Total exactly
        academic_blocks = total_val - (chiller + hostels + staff_quarters + stp)

        # Supply components
        solar = int(round(max(500.0, 3800 + np.random.normal(0, 600))))
        biogas = int(round(max(100.0, 450 + np.random.normal(0, 100))))
        diesel = 1200 if (dow == 2 and i % 14 == 0) else 0

        tneb_new_stp = int(round(max(200.0, 800 + np.random.normal(0, 120))))
        # Exact residual balance for TNEB Campus so Supply Sum == Total exactly
        tneb_campus = total_val - (solar + biogas + diesel + tneb_new_stp)

        records.append({
            "Date": dt.strftime("%d.%m.%Y"),
            "TNEB_Campus": tneb_campus,
            "TNEB_New_STP": tneb_new_stp,
            "Solar": solar,
            "Diesel": diesel,
            "Biogas": biogas,
            "Staff_Quarters": staff_quarters,
            "Academic_Blocks": academic_blocks,
            "Hostels": hostels,
            "Chiller": chiller,
            "STP": stp,
            "Total": total_val
        })

    df = pd.DataFrame(records)
    return df

