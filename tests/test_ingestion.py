import numpy as np
import pandas as pd

from  src.energy_dispatch.data.ingestion import (
    calcular_generacion_eolica,
    calibrar_turbinas_por_parque,
    potencia_turbina_kw,
)


def test_curva_potencia_puntos_y_limites():
    # Por debajo del cut-in (4 m/s) y por encima del cut-out (25 m/s) no se genera
    assert potencia_turbina_kw(3.9) == 0
    assert potencia_turbina_kw(25.1) == 0
    assert potencia_turbina_kw(4) == 77
    assert potencia_turbina_kw(10) == 1710
    assert potencia_turbina_kw(20) == 3000
    # Interpolación lineal entre puntos de la curva
    assert np.isclose(potencia_turbina_kw(9.5), (1273 + 1710) / 2)


def _df_parques():
    ts = pd.date_range("2025-09-01", periods=2, freq="5min")
    return pd.DataFrame({
        "station_id": ["A", "A", "B", "B"],
        "timestamp": list(ts) * 2,
        "wind_speed_kmh": [36.0, 0.0, 72.0, 36.0],  # 10, 0, 20 y 10 m/s
        "consumo_isla_kw": [10000.0, 9000.0, 10000.0, 9000.0],
    })


def test_generacion_total_por_timestamp():
    out = calcular_generacion_eolica(_df_parques(), turbinas_por_parque=2)
    assert out["generacion_eolica_kw"].tolist() == [3420, 0, 6000, 3420]
    assert out["generacion_eolica_total"].tolist() == [9420, 3420, 9420, 3420]
    assert out["deficit_kw"].tolist() == [580, 5580, 580, 5580]


def test_calibracion_turbinas():
    # Generación media con 1 turbina por parque = (4710 + 1710) / 2 = 3210 kW; consumo medio = 9500 kW
    # 100 % del consumo -> 9500 / 3210 ≈ 2.96 -> 3 turbinas por parque
    assert calibrar_turbinas_por_parque(_df_parques(), cuota_eolica=1.0) == 3
