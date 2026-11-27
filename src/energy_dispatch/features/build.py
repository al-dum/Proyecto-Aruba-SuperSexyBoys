import pandas as pd


def add_exact_time_features(
    df: pd.DataFrame,
    value_col: str,
    offsets: dict[str, str | pd.Timedelta],
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Busca valores a distancias temporales exactas en una serie única.

    Un offset positivo consulta el futuro (objetivo); uno negativo consulta el
    pasado (retardo). No interpola ni confunde filas con minutos. La entrada debe
    tener una sola fila por instante: agregar o separar estaciones previamente.
    """
    timestamps = pd.to_datetime(df[timestamp_col], utc=True)
    if timestamps.isna().any():
        raise ValueError("Los timestamps no pueden contener valores ausentes.")
    if timestamps.duplicated().any():
        raise ValueError("Se requiere una sola fila por timestamp.")
    if set(offsets).intersection(df.columns):
        raise ValueError("Las columnas temporales nuevas no pueden sobrescribir otras.")

    values = pd.Series(df[value_col].to_numpy(), index=pd.DatetimeIndex(timestamps))
    out = df.copy()
    for name, offset in offsets.items():
        requested_times = pd.DatetimeIndex(timestamps + pd.Timedelta(offset))
        out[name] = values.reindex(requested_times).to_numpy()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "generacion_eolica_kw",
    lags=range(1, 5),
    group_col: str = "station_id",
    timestamp_col: str = "timestamp",
    freq: str = "5min",
    dropna: bool = False,
) -> pd.DataFrame:
    """Añade retardos temporales de la generación eólica sin cruzar huecos."""
    out = df.sort_values([group_col, timestamp_col]).copy()
    timestamps = pd.to_datetime(out[timestamp_col])
    step = pd.Timedelta(freq)
    groups = out[group_col]

    lag_cols = {}
    for lag in lags:
        values = out.groupby(group_col)[target_col].shift(lag)
        previous_timestamps = timestamps.groupby(groups).shift(lag)
        lag_cols[f"{target_col}_lag_{lag}"] = values.where(
            timestamps - previous_timestamps == lag * step
        )

    out = out.assign(**lag_cols)
    if dropna:
        out = out.dropna(subset=list(lag_cols))
    return out
