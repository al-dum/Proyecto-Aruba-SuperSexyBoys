"""Dataset de pronóstico de generación eólica SIN fuga de datos (data leakage).

Cada fila representa un "instante de emisión" t: el momento en que el operador pide el
pronóstico. Las variables explicativas usan solo información disponible en t o antes; el
objetivo es el viento (y la generación) en t + h, donde h = `horizon_steps` pasos de 5 min.

Fugas que tenía el planteamiento anterior y cómo se corrigen aquí
------------------------------------------------------------------
1. Objetivo contemporáneo. `generacion_eolica_kw` es una función determinista de
   `wind_speed_kmh` en el MISMO instante (curva de potencia). Usar el viento de t (o
   `wind_alert`, que también sale de él) para "predecir" la generación de t da un error
   casi nulo que no sirve de nada: en producción nadie conoce el viento futuro.
   -> Corrección: el objetivo se desplaza al futuro con `shift(-h)` y todas las variables
      se calculan con datos de t o anteriores.

2. Filas duplicadas por estación. La tabla del paso 4 tiene una fila por estación y
   timestamp, pero el objetivo (generación del parque) es el mismo para las tres. Con un
   split aleatorio, el mismo instante cae a la vez en train y en test.
   -> Corrección: una única fila por timestamp; cada estación pasa a ser una columna.

3. Ventanas que miran al futuro. Una media móvil centrada (`center=True`) o una
   interpolación entre el valor anterior y el siguiente usan datos posteriores a t.
   -> Corrección: solo `shift(k)` con k >= 0 y `rolling` hacia atrás (incluye t, que ya
      se conoce en el momento de emitir el pronóstico).

4. Retardos que cruzan huecos. `shift(k)` sobre filas desordenadas o con huecos no
   equivale a "k x 5 min antes".
   -> Corrección: se reindexa a una rejilla regular de 5 min; un hueco queda como NaN y
      no se rellena con un valor que no existía todavía.

Las fugas del split (aleatorio en vez de temporal, sin hueco de seguridad) y del
preprocesado ajustado con test se corrigen en `energy_dispatch.models.train`.
"""
import numpy as np
import pandas as pd

from energy_dispatch.features.wind import idw_average, park_wind_kmh, wind_kmh_to_power_kw

FREQ = "5min"
STEPS_PER_DAY = 288
TIMEZONE = "America/Aruba"  # UTC-4, sin horario de verano

TARGET_WIND = "target_wind_kmh"
TARGET_POWER = "target_generacion_eolica_kw"
TARGET_TIME = "target_time"
NON_FEATURE_COLUMNS = [TARGET_WIND, TARGET_POWER, TARGET_TIME]

WIND_LAGS = (1, 3, 6, 12, 24, 36, 72)          # 5 min ... 6 h
ROLLING_WINDOWS = {"30min": 6, "1h": 12, "3h": 36}


def _station_columns(readings: pd.DataFrame, stations: pd.DataFrame, column: str,
                     grid: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    """Pivota `column` a una columna por estación, numeradas por cercanía (0 = la más cercana)."""
    order = {sid: i for i, sid in enumerate(stations.sort_values("distance_m")["id"])}
    wide = readings.pivot_table(index="timestamp", columns="station_id", values=column)
    wide = wide.reindex(grid)[list(order)]
    wide.columns = [f"{prefix}_station_{order[sid]}" for sid in wide.columns]
    return wide


def build_forecast_dataset(readings: pd.DataFrame, stations: pd.DataFrame,
                           horizon_steps: int, freq: str = FREQ) -> pd.DataFrame:
    """Construye la tabla (features en t, objetivo en t + h) indexada por instante de emisión.

    `readings`: lecturas limpias de las estaciones cercanas (salida del paso 2).
    `stations`: estaciones con `id` y `distance_m` (salida del paso 1).
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps debe ser >= 1: con h = 0 el objetivo es el presente.")

    readings = readings.copy()
    readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True)
    readings = readings[readings["station_id"].isin(stations["id"])]

    # Fuga 4: rejilla regular -> shift(k) significa exactamente k x 5 min.
    grid = pd.date_range(readings["timestamp"].min(), readings["timestamp"].max(),
                         freq=freq, name="timestamp")
    wind = park_wind_kmh(readings, stations).reindex(grid)

    # Fuga 2: una fila por instante; las estaciones pasan a ser columnas.
    df = pd.concat(
        [wind, _station_columns(readings, stations, "wind_speed_kmh", grid, "wind_kmh")], axis=1
    )
    direction = np.deg2rad(readings["wind_direction_deg"])
    readings = readings.assign(dir_sin=np.sin(direction), dir_cos=np.cos(direction))
    for column in ("dir_sin", "dir_cos", "pressure_hpa", "temperature_c", "humidity_pct"):
        df[f"park_{column}"] = idw_average(readings, stations, column).reindex(grid)

    # Fuga 3: retardos y ventanas SOLO hacia atrás. shift(k) con k > 0 = valor de hace k pasos;
    # rolling(...) sin center=True termina en t, que ya es conocido al emitir el pronóstico.
    for lag in WIND_LAGS:
        df[f"park_wind_kmh_lag_{lag}"] = wind.shift(lag)
    for name, window in ROLLING_WINDOWS.items():
        rolling = wind.rolling(window, min_periods=window)
        df[f"park_wind_kmh_mean_{name}"] = rolling.mean()
        df[f"park_wind_kmh_std_{name}"] = rolling.std()
    df["park_wind_kmh_delta_1h"] = wind - wind.shift(12)
    df["park_pressure_tendency_1h"] = df["park_pressure_hpa"] - df["park_pressure_hpa"].shift(12)
    df["park_pressure_tendency_3h"] = df["park_pressure_hpa"] - df["park_pressure_hpa"].shift(36)

    # Mismo instante que el objetivo pero del último día ya observado: t + h - 288·n con
    # n >= 1 mínimo para que el desplazamiento sea >= 0 (es decir, un dato ya conocido en t).
    seasonal_shift = (-horizon_steps) % STEPS_PER_DAY
    df["park_wind_kmh_same_time_last_day"] = wind.shift(seasonal_shift)

    # Calendario de la hora objetivo. No es fuga: la hora futura se conoce de antemano.
    # No se usa el mes: con solo 3 meses de datos el modelo memorizaría el periodo.
    target_time = grid + horizon_steps * pd.Timedelta(freq)
    local_hour = target_time.tz_convert(TIMEZONE)
    hour_float = local_hour.hour + local_hour.minute / 60
    df["forecast_hour_local"] = local_hour.hour
    df["forecast_hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
    df["forecast_hour_cos"] = np.cos(2 * np.pi * hour_float / 24)

    # Fuga 1: el objetivo es el FUTURO (t + h). shift(-h) es el único desplazamiento negativo
    # del módulo y solo se aplica a columnas objetivo, nunca a features.
    # Se modela el viento y la generación sale de la curva de potencia conocida, en vez de
    # obligar al modelo a reaprender esa curva (no lineal y con saturación a 3 MW).
    df[TARGET_TIME] = target_time
    df[TARGET_WIND] = wind.shift(-horizon_steps)
    df[TARGET_POWER] = wind_kmh_to_power_kw(df[TARGET_WIND])

    return df.dropna(subset=["park_wind_kmh", TARGET_WIND])


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas que el modelo puede usar como entrada (todo menos objetivo y metadatos)."""
    return [column for column in df.columns if column not in NON_FEATURE_COLUMNS]
