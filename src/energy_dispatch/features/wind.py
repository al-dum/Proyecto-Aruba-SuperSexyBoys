"""Viento en el parque Vader Piet y su conversión a potencia.

Lo usan tanto el pipeline de datos (paso 4) como el entrenamiento del modelo, para que
la generación "real" con la que se evalúa sea exactamente la misma en ambos sitios.
"""
import numpy as np
import pandas as pd

# Parque Vader Piet: 10 turbinas Vestas V90-3.0 MW, 10 % de pérdidas.
N_TURBINES = 10
TURBINE_RATED_POWER_KW = 3000.0
PARK_LOSSES = 0.10
CUT_IN_MS = 4.0
CUT_OUT_MS = 25.0

# Curva V90-3.0 MW digitalizada del folleto (docs/ProductbrochureV90_3_0_UK.pdf), por turbina
# y a densidad estándar. Es la ÚNICA tabla del proyecto: generación, umbrales y pipeline la usan.
V90_WIND_MS = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
V90_POWER_KW = np.array([97, 160, 233, 317, 414, 516, 641, 780, 944, 1115, 1296, 1478,
                         1668, 1862, 2061, 2254, 2447, 2606, 2732, 2843, 2928, 2984, 3000])


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
    power_per_turbine = np.interp(wind_ms, V90_WIND_MS, V90_POWER_KW)
    # np.where (y no asignación) para que funcione también con un escalar.
    power_per_turbine = np.where((wind_ms < CUT_IN_MS) | (wind_ms >= CUT_OUT_MS), 0, power_per_turbine)
    return power_per_turbine * N_TURBINES * (1 - PARK_LOSSES)


def wind_kmh_for_power_share(share: float) -> float:
    """Velocidad del viento (km/h) a la que el parque da `share` de su potencia nominal."""
    per_turbine_kw = share * TURBINE_RATED_POWER_KW
    return float(np.interp(per_turbine_kw, V90_POWER_KW, V90_WIND_MS)) * 3.6


# Umbrales de alerta eólica (T2): 15 % y 40 % de la capacidad, y cut-out de la turbina.
WIND_CRITICAL_KMH = wind_kmh_for_power_share(0.15)
WIND_MODERATE_KMH = wind_kmh_for_power_share(0.40)
WIND_CUT_OUT_KMH = CUT_OUT_MS * 3.6
