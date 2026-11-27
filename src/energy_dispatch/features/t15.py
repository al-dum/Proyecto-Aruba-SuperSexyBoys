"""Dataset de pronóstico a corto plazo (t+15 min) sin fuga de datos (T7 -> T9).

Cada fila es un instante de emisión t: el momento en que se pide el pronóstico. Las
variables usan solo datos de t o anteriores, más el calendario del instante objetivo
(que se conoce de antemano). El objetivo es la señal en t + horizonte.

Todo se construye sobre una rejilla regular de 5 min: así `shift(k)` significa
exactamente "hace k x 5 min" y un hueco queda como NaN en vez de juntar dos lecturas
separadas por horas (EntrenamientoML.pdf, §9.6 y §10.9).

Dos problemas, ambos a t+15 (notebooks/5_eda_notebook.ipynb):
- "wind": viento en el parque Vader Piet (km/h). La generación se obtiene después con la
  curva de potencia (`wind_kmh_to_power_kw`): se predice lo incierto (el viento) y no se
  obliga al modelo a reaprender una curva conocida (docs/MODELO_GRADIENT_BOOSTING.md).
- "demand": consumo estimado de la isla (kW). Es una serie MODELADA en el paso 5, no medida.
"""
import holidays
import numpy as np
import pandas as pd

from energy_dispatch.features.wind import idw_average, park_wind_kmh

FREQ = pd.Timedelta("5min")
HORIZON = pd.Timedelta("15min")
STEPS_PER_DAY = 288
TIMEZONE = "America/Aruba"  # UTC-4, sin horario de verano

SIGNALS = {"wind": "park_wind_kmh", "demand": "consumo_estimado_isla_kw"}
TARGET = "target"
TARGET_TIME = "target_time"
TARGET_HOUR = "target_hour_local"  # para el baseline por hora y el error por franja
ANCHOR = "now"  # valor de la señal en t: persistencia y ancla del modelo de incrementos

LAGS = {"5min": 1, "10min": 2, "15min": 3, "30min": 6, "1h": 12, "2h": 24}
ROLLING = {"30min": 6, "1h": 12, "3h": 36}
METEO = ["temperature_c", "humidity_pct", "pressure_hpa", "precipitation_mm"]
N_STATIONS = 3


def instant_table(readings: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Pasa de filas estación x instante (pasos 4/5) a una fila por instante.

    La meteorología se promedia con pesos por inverso de la distancia al parque (igual que
    el viento del parque) y el viento de cada estación se conserva como columna aparte,
    numeradas por cercanía (0 = la más cercana).
    """
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Hay lecturas duplicadas para la misma estación e instante.")

    grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=FREQ,
                         name="timestamp")
    out = pd.DataFrame(index=grid)
    out["park_wind_kmh"] = park_wind_kmh(df, stations).reindex(grid)
    for col in METEO:
        out[col] = idw_average(df, stations, col).reindex(grid)

    # La dirección es circular (359° y 1° están juntos): se promedian seno y coseno.
    rad = np.deg2rad(df["wind_direction_deg"])
    df = df.assign(wind_dir_sin=np.sin(rad), wind_dir_cos=np.cos(rad))
    for col in ("wind_dir_sin", "wind_dir_cos"):
        out[col] = idw_average(df, stations, col).reindex(grid)

    wide = df.pivot_table(index="timestamp", columns="station_id", values="wind_speed_kmh")
    wide = wide.reindex(grid)
    for i, sid in enumerate(stations.sort_values("distance_m")["id"]):
        out[f"wind_kmh_station_{i}"] = wide[sid] if sid in wide.columns else np.nan

    if "consumo_estimado_isla_kw" in df.columns:
        per_instant = df.groupby("timestamp")["consumo_estimado_isla_kw"].agg(["min", "max"])
        if not np.allclose(per_instant["min"], per_instant["max"], equal_nan=True):
            raise ValueError("El consumo de la isla debería ser igual en las tres estaciones.")
        out["consumo_estimado_isla_kw"] = per_instant["min"].reindex(grid)
    return out


def feature_columns(problem: str) -> list[str]:
    """Columnas de entrada del modelo. Nunca incluye el objetivo ni su instante."""
    cols = [ANCHOR, *(f"lag_{k}" for k in LAGS), "diff_15min", "diff_1h",
            *(f"{stat}_{k}" for k in ROLLING for stat in ("mean", "std")),
            "same_time_yesterday",
            *METEO, "wind_dir_sin", "wind_dir_cos", "temperature_ewm_2h",
            "temperature_diff_1h",
            "hour_sin", "hour_cos", "day_of_week", "day_off"]
    if problem == "wind":
        cols += [f"wind_kmh_station_{i}" for i in range(N_STATIONS)]
        cols += [f"wind_kmh_station_{i}_diff_15min" for i in range(N_STATIONS)]
    elif problem == "demand":
        cols += ["park_wind_kmh"]
    else:
        raise ValueError(f"Problema desconocido: {problem}")
    return cols


def build_dataset(instants: pd.DataFrame, problem: str,
                  horizon: pd.Timedelta = HORIZON, require_target: bool = True) -> pd.DataFrame:
    """Variables en t (solo pasado) y objetivo en t + horizonte, una fila por instante.

    `instants` debe venir de `instant_table` (rejilla regular). Se descartan las filas sin
    valor actual o sin objetivo; los NaN de los retardos se dejan como NaN (no se inventan
    datos) y los gestiona cada modelo. Con `require_target=False` (inferencia) no se exige
    objetivo, así que la última fila —sin futuro— también se conserva.
    """
    if not (instants.index.to_series().diff().dropna() == FREQ).all():
        raise ValueError("La tabla debe estar en una rejilla regular de 5 min (instant_table).")
    signal = SIGNALS[problem]
    if signal not in instants.columns:
        raise KeyError(f"Falta la columna {signal!r}: ejecuta el paso del pipeline que la crea.")
    steps = int(horizon / FREQ)
    if steps < 1 or horizon % FREQ != pd.Timedelta(0):
        raise ValueError("El horizonte debe ser un múltiplo positivo de 5 min.")

    s = instants[signal]
    X = pd.DataFrame(index=instants.index)
    X[ANCHOR] = s
    for name, k in LAGS.items():
        X[f"lag_{name}"] = s.shift(k)
    X["diff_15min"] = s - s.shift(3)
    X["diff_1h"] = s - s.shift(12)
    for name, window in ROLLING.items():
        rolling = s.rolling(window, min_periods=window // 2)  # hacia atrás: incluye t
        X[f"mean_{name}"] = rolling.mean()
        X[f"std_{name}"] = rolling.std()
    # Valor de ayer a la hora del objetivo (t + h - 24 h): ya se conoce en t.
    X["same_time_yesterday"] = s.shift(STEPS_PER_DAY - steps)

    for col in [*METEO, "wind_dir_sin", "wind_dir_cos"]:
        X[col] = instants[col]
    temperature = instants["temperature_c"]
    # Inercia térmica de los edificios: media exponencial hacia atrás (vida media 2 h).
    X["temperature_ewm_2h"] = temperature.ewm(halflife=24, ignore_na=True).mean()
    X["temperature_diff_1h"] = temperature - temperature.shift(12)

    if problem == "wind":
        for i in range(N_STATIONS):
            col = f"wind_kmh_station_{i}"
            X[col] = instants[col]
            X[f"{col}_diff_15min"] = instants[col] - instants[col].shift(3)
    else:
        X["park_wind_kmh"] = instants["park_wind_kmh"]

    # Calendario del instante objetivo, en hora local de Aruba.
    target_time = instants.index + horizon
    local = target_time.tz_convert(TIMEZONE)
    hour = local.hour + local.minute / 60
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["day_of_week"] = local.day_of_week
    aw_holidays = holidays.country_holidays("AW", years=sorted(set(local.year)))
    is_holiday = pd.Index(local.date).isin(list(aw_holidays.keys()))
    X["day_off"] = ((local.day_of_week >= 5) | is_holiday).astype(int)
    X[TARGET_HOUR] = local.hour

    X[TARGET] = s.shift(-steps)
    X[TARGET_TIME] = target_time

    keep = X[ANCHOR].notna()
    if require_target:
        keep &= X[TARGET].notna()
    return X[keep].copy()


def temporal_split(data: pd.DataFrame, test_fraction: float = 0.2):
    """Desarrollo = primer tramo; prueba = último `test_fraction` de los instantes.

    Purga: se quitan de desarrollo las filas cuyo objetivo (t + h) cae ya dentro del tramo
    de prueba. Si no, el final del entrenamiento "vería" valores del test (§9.6).
    """
    data = data.sort_index()
    cutoff = data.index[int(len(data) * (1 - test_fraction))]
    dev = data[(data.index < cutoff) & (data[TARGET_TIME] < cutoff)]
    test = data[data.index >= cutoff]
    return dev, test
