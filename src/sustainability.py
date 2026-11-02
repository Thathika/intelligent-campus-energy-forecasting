"""
sustainability.py
Module for transparent electricity cost and carbon-emission estimation.
Includes strict unit verification flags and academic disclaimers.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


# Documented default benchmarks
DEFAULT_TARIFF_INR_PER_KWH = 8.00  # Typical Tamil Nadu TNEB HT educational/commercial tariff range (₹7.5 - ₹8.5/kWh)
DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH = 0.716  # CEA (Central Electricity Authority, India) CO2 Baseline Database for Indian Power Sector v19


def calculate_cost_and_emissions(
    energy_series: pd.Series,
    unit_is_verified_kwh: bool = False,
    tariff_inr_per_kwh: float = DEFAULT_TARIFF_INR_PER_KWH,
    emissions_factor_kg_per_kwh: float = DEFAULT_EMISSIONS_FACTOR_KG_CO2_PER_KWH
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Computes provisional or verified electricity costs and carbon emissions.

    Parameters:
    - energy_series: Numeric pandas Series representing energy readings
    - unit_is_verified_kwh: User confirmation flag that readings represent kWh
    - tariff_inr_per_kwh: Applicable tariff rate (INR / kWh)
    - emissions_factor_kg_per_kwh: Carbon intensity factor (kg CO2e / kWh)

    Returns:
    - cost_series: Series of estimated costs in INR
    - emissions_series_kg: Series of estimated carbon emissions in kg CO2e
    - summary: Aggregated metrics, unit status, and disclaimer notices
    """
    # Defensive check
    clean_energy = energy_series.fillna(0.0).clip(lower=0.0)

    # Simplified energy-charge cost: E * T
    cost_series = clean_energy * tariff_inr_per_kwh

    # Carbon emissions: E * F
    emissions_series_kg = clean_energy * emissions_factor_kg_per_kwh

    total_kwh = float(clean_energy.sum())
    total_cost_inr = float(cost_series.sum())
    total_emissions_kg = float(emissions_series_kg.sum())
    total_emissions_mt = total_emissions_kg / 1000.0  # Metric Tons

    avg_daily_kwh = float(clean_energy.mean()) if len(clean_energy) > 0 else 0.0
    avg_daily_cost = avg_daily_kwh * tariff_inr_per_kwh
    avg_daily_emissions_kg = avg_daily_kwh * emissions_factor_kg_per_kwh

    status = "Verified (kWh Confirmed)" if unit_is_verified_kwh else "Provisional (Unit Unverified)"

    summary: Dict[str, Any] = {
        "unit_status": status,
        "is_verified": unit_is_verified_kwh,
        "total_energy_units": round(total_kwh, 1),
        "tariff_rate_inr": tariff_inr_per_kwh,
        "emissions_factor": emissions_factor_kg_per_kwh,
        "total_cost_inr": round(total_cost_inr, 2),
        "total_emissions_kg": round(total_emissions_kg, 2),
        "total_emissions_mt": round(total_emissions_mt, 3),
        "avg_daily_cost_inr": round(avg_daily_cost, 2),
        "avg_daily_emissions_kg": round(avg_daily_emissions_kg, 2),
        "cost_disclaimer": (
            "The electricity cost calculation uses a simplified energy-charge formula (Energy × Tariff). "
            "It excludes two-part tariff components such as maximum demand (kVA) charges, power factor penalties/rebates, "
            "fixed monthly charges, fuel price adjustments, and electricity duty/taxes applicable under TNEB regulations."
        ),
        "carbon_disclaimer": (
            f"Emissions are estimated using {emissions_factor_kg_per_kwh} kg CO2e/kWh based on the Central Electricity Authority (CEA) "
            "Indian National Grid Baseline Database. This represents an academic estimate and does not constitute certified carbon credits or audited footprint reporting."
        ),
        "unit_warning": (
            "" if unit_is_verified_kwh else
            "Notice: Target measurement unit has not been explicitly verified as kWh in the source file. "
            "All monetary (₹) and carbon (kg CO2e) values are presented as provisional hypothetical estimates."
        )
    }

    return cost_series, emissions_series_kg, summary

