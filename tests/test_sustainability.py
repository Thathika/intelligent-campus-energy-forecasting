"""
test_sustainability.py
Unit tests for electricity cost estimation, carbon emissions estimation, and unit verification flags.
"""

import pytest
import pandas as pd
import numpy as np
from src.sustainability import calculate_cost_and_emissions


def test_sustainability_exact_math():
    # 100 kWh with tariff = 8 INR/kWh and emission = 0.716 kg/kWh
    energy = pd.Series([100.0, 200.0])
    cost, emissions, summary = calculate_cost_and_emissions(
        energy_series=energy,
        unit_is_verified_kwh=True,
        tariff_inr_per_kwh=8.0,
        emissions_factor_kg_per_kwh=0.716
    )

    # Cost: 100*8 = 800, 200*8 = 1600 -> Total = 2400
    assert cost.iloc[0] == 800.0
    assert cost.iloc[1] == 1600.0
    assert summary["total_cost_inr"] == 2400.0

    # Emissions: 100*0.716 = 71.6, 200*0.716 = 143.2 -> Total = 214.8 kg
    assert pytest.approx(emissions.iloc[0], 0.01) == 71.6
    assert pytest.approx(emissions.iloc[1], 0.01) == 143.2
    assert pytest.approx(summary["total_emissions_kg"], 0.01) == 214.8
    assert summary["is_verified"] is True
    assert "Verified" in summary["unit_status"]


def test_sustainability_unverified_flag():
    energy = pd.Series([500.0])
    _, _, summary = calculate_cost_and_emissions(
        energy_series=energy,
        unit_is_verified_kwh=False
    )
    assert summary["is_verified"] is False
    assert "Provisional" in summary["unit_status"]
    assert "Notice:" in summary["unit_warning"]

