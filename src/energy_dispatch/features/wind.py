"""Viento en el parque Vader Piet y su conversión a potencia.

Lo usan tanto el pipeline de datos (paso 4) como el entrenamiento del modelo, para que
la generación "real" con la que se evalúa sea exactamente la misma en ambos sitios.
"""
import numpy as np
import pandas as pd

# Parque Vader Piet: 10 turbinas Vestas V90-3.0 MW, 10 % de pérdidas.
N_TURBINES = 10
PARK_LOSSES = 0.10
CUT_IN_MS = 4.0
CUT_OUT_MS = 25.0

# Curva V90-3.0 MW simplificada (docs/umbrales-seguridad-aruba.md).
CURVE_MS = np.array([0, 4, 6, 8, 10, 12, 15, 25])
CURVE_KW = np.array([0, 97, 414, 944, 1668, 2447, 3000, 3000])


def idw_average(readings: pd.DataFrame, stations: pd.DataFrame, column: str) -> pd.Series:
    """Media de `column` por timestamp ponderada por el inverso de la distancia al parque.

    `stations` necesita `id` y `distance_m`; `readings` necesita `station_id` y `timestamp`.
    Si una estación no tiene dato en un instante, se usan solo las demás.
    """
    weights = 1 / stations.set_index("id")["distance_m"]
    df = readings[readings["station_id"].isin(weights.index)]
    weight = df["station_id"].map(weights).where(df[column].notna())
    weighted_sum = (df[column] * weight).groupby(df["timestamp"]).sum(min_count=1)
    return weighted_sum / weight.groupby(df["timestamp"]).sum()


def park_wind_kmh(readings: pd.DataFrame, stations: pd.DataFrame) -> pd.Series:
    """Velocidad del viento estimada en el parque (km/h), indexada por timestamp."""
    return idw_average(readings, stations, "wind_speed_kmh").rename("park_wind_kmh")


def wind_kmh_to_power_kw(wind_kmh) -> np.ndarray:
    """Potencia del parque (kW) para una velocidad de viento en km/h. NaN se propaga."""
    wind_ms = np.asarray(wind_kmh, dtype=float) / 3.6
    power_per_turbine = np.interp(wind_ms, CURVE_MS, CURVE_KW)
    power_per_turbine[(wind_ms < CUT_IN_MS) | (wind_ms >= CUT_OUT_MS)] = 0
    return power_per_turbine * N_TURBINES * (1 - PARK_LOSSES)
