import pandas as pd

from src.energy_dispatch.features.build import add_lag_features

# Datos de prueba: 2 estaciones (A y B) con 30 lecturas cada una cada 5 min.
# En A el generacion_eolica vale 0, 1, 2... 29 y en B vale 100, 101... 129,
# así es fácil saber qué valor debe tener cada lag.
def _df_prueba1():
    ts = pd.date_range("2025-09-01", periods=30, freq="5min")
    return  pd.DataFrame({
        "station_id":["A"] *30 + ["B"] *30,
        "timestamp":list(ts) *2,
        "generacion_eolica_kw": list(range(30)) + list(range(100,130)),
    })

# Se crean las columnas generacion_eolica_kw_lag_1 ... generacion_eolica_kw_lag_4 (t-5, t-10, t-15 y t-20 min)
def test_columnas():
    out = add_lag_features(_df_prueba1())
    for k in range(1, 5):
        assert f"generacion_eolica_kw_lag_{k}" in out.columns

# En la estación A, fila 10: lag_1 (hace 5 min) == 9 y lag_4 (hace 20 min) == 6
def test_valor_lag_correcto():
    out = add_lag_features(_df_prueba1())
    a = out[out["station_id"] == "A"].reset_index(drop=True)
    assert a.loc[10, "generacion_eolica_kw_lag_1"] == 9
    assert a.loc[10, "generacion_eolica_kw_lag_4"] == 6

# Las primeras 4 filas de A tienen lag_4 en NaN y la fila 4 ya no
def test_nan_inicio():
    out = add_lag_features(_df_prueba1())
    a = out[out["station_id"] == "A"].reset_index(drop=True)
    assert a.loc[0:3, "generacion_eolica_kw_lag_4"].isna().all()
    assert a.loc[4, "generacion_eolica_kw_lag_4"] == 0

#Las primeras 4 filas de B tienen lag_4 en NaN (no cogen datos de A)
def test_estaciones():
    out = add_lag_features(_df_prueba1())
    b = out[out["station_id"] == "B"].reset_index(drop=True)
    assert b.loc[0:3, "generacion_eolica_kw_lag_4"].isna().all()
    assert b.loc[4, "generacion_eolica_kw_lag_4"] == 100

#Guarda df.copy(), llama a la función y compara con pd.testing.assert_frame_equal
def test_copia():
    df = _df_prueba1()
    original = df.copy()
    add_lag_features(df)
    pd.testing.assert_frame_equal(df, original)

#Con dropna=True no queda ningún NaN en las columnas de lag
def test_dropna():
    out = add_lag_features(_df_prueba1(), dropna=True)
    columnas_lag = [c for c in out.columns if "_lag_" in c]
    assert not out[columnas_lag].isna().any().any()
    # Con lags hasta 4 se pierden 4 filas por estación: quedan 26 + 26
    assert len(out) == 52
