import numpy as np
import pandas as pd
import pytest

from energy_dispatch.data.consumption import (
    RATIO_PICO_MEDIA,
    ZONA_HORARIA_ARUBA,
    estimate_island_consumption,
)


def _weather(start="2026-01-01 04:00", end="2026-03-01 04:00"):  # meses completos en hora local
    timestamps = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left")
    hour = timestamps.hour.to_numpy()
    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": 28.0 + 2.0 * np.sin((hour - 9) / 24 * 2 * np.pi),
        "humidity_pct": 74.0,
        "wind_speed_kmh": 22.0,
    })


MONTHLY_MWH = {(2026, 1): 74_400.0, (2026, 2): 67_200.0}  # 100 MW medios en ambos meses


def test_monthly_mean_matches_excel_level():
    consumption = estimate_island_consumption(_weather(), MONTHLY_MWH)
    by_month = consumption.groupby(consumption.index.tz_convert(ZONA_HORARIA_ARUBA).month).mean()
    assert by_month.loc[1] == pytest.approx(100_000.0, rel=1e-6)
    assert by_month.loc[2] == pytest.approx(100_000.0, rel=1e-6)


def test_peak_to_mean_ratio_is_calibrated():
    consumption = estimate_island_consumption(_weather(), MONTHLY_MWH)
    assert consumption.max() / consumption.mean() == pytest.approx(RATIO_PICO_MEDIA, rel=1e-3)


def test_evening_peak_and_night_valley():
    consumption = estimate_island_consumption(_weather(), MONTHLY_MWH)
    hourly = consumption.groupby(consumption.index.tz_convert(ZONA_HORARIA_ARUBA).hour).mean()
    assert hourly.idxmax() in (19, 20)
    assert hourly.idxmin() in (3, 4, 5)


def test_missing_month_raises():
    with pytest.raises(KeyError):
        estimate_island_consumption(_weather(), {(2026, 1): 74_400.0})
