import numpy as np
import pandas as pd

from  src.energy_dispatch.data.ingestion import (
    ALTURA_BUJE_M,
    ALTURA_MEDICION_M,
    VADER_PIET_LAT,
    VADER_PIET_LON,
    calcular_generacion_eolica,
    calcular_viento_parque,
    distancia_km,
    potencia_turbina_kw,
)

# Coordenadas de San Nicolaas (~4.6 km de Vader Piet) y Arikok (~7.4 km)
LAT_SNC, LON_SNC = 12.4364, -69.909
LAT_ARK, LON_ARK = 12.5009, -69.9538


def test_curva_potencia_puntos_y_limites():
    # Por debajo del cut-in (4 m/s) y por encima del cut-out (25 m/s) no se genera
    assert potencia_turbina_kw(3.9) == 0
    assert potencia_turbina_kw(25.1) == 0
    assert potencia_turbina_kw(4) == 77
    assert potencia_turbina_kw(10) == 1710
    assert potencia_turbina_kw(20) == 3000
    # Interpolación lineal entre puntos de la curva
    assert np.isclose(potencia_turbina_kw(9.5), (1273 + 1710) / 2)


def test_distancia_estaciones_a_vader_piet():
    assert np.isclose(distancia_km(LAT_SNC, LON_SNC, VADER_PIET_LAT, VADER_PIET_LON), 4.62, atol=0.01)
    assert np.isclose(distancia_km(LAT_ARK, LON_ARK, VADER_PIET_LAT, VADER_PIET_LON), 7.38, atol=0.01)


def _df_estaciones(lat_b=LAT_SNC, lon_b=LON_SNC):
    ts = pd.date_range("2025-09-01", periods=2, freq="5min")
    return pd.DataFrame({
        "station_id": ["A", "A", "B", "B"],
        "timestamp": list(ts) * 2,
        "latitude": [LAT_SNC, LAT_SNC, lat_b, lat_b],
        "longitude": [LON_SNC, LON_SNC, lon_b, lon_b],
        "wind_speed_kmh": [36.0, 36.0, 36.0, 0.0],  # 10, 10, 10 y 0 m/s
        "consumo_isla_kw": [20000.0, 19000.0, 20000.0, 19000.0],
    })


def test_viento_parque_ponderado_por_distancia_y_altura():
    df = _df_estaciones(lat_b=LAT_ARK, lon_b=LON_ARK)
    d_a = distancia_km(LAT_SNC, LON_SNC, VADER_PIET_LAT, VADER_PIET_LON)
    d_b = distancia_km(LAT_ARK, LON_ARK, VADER_PIET_LAT, VADER_PIET_LON)

    sin_altura = calcular_viento_parque(df, alfa=0.0)
    assert np.isclose(sin_altura.iloc[0], 10.0)
    # La estación más cercana (A, 10 m/s) pesa más que la lejana (B, 0 m/s)
    assert np.isclose(sin_altura.iloc[1], 10.0 / d_a / (1 / d_a + 1 / d_b))
    assert sin_altura.iloc[1] > 5.0

    con_altura = calcular_viento_parque(df, alfa=0.14)
    assert np.allclose(con_altura, sin_altura * (ALTURA_BUJE_M / ALTURA_MEDICION_M) ** 0.14)


def test_generacion_parque_unico_por_timestamp():
    # Misma ubicación -> media simple: 10 m/s y 5 m/s; 10 turbinas -> 17100 kW y 1900 kW
    out = calcular_generacion_eolica(_df_estaciones(), turbinas=10, alfa=0.0)
    assert out["generacion_eolica_kw"].tolist() == [17100, 1900, 17100, 1900]
    assert out["viento_buje_ms"].tolist() == [10, 5, 10, 5]
    assert out["deficit_kw"].tolist() == [2900, 17100, 2900, 17100]
