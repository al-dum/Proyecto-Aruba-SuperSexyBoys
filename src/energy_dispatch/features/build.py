import pandas as pd

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = df[timestamp_col]
    return df.assign(hour=ts.dt.hour, dayofweek=ts.dt.dayofweek, month=ts.dt.month)

def add_lag_features(df: pd.DataFrame, target_col: str = "consumo_kw", lags=range(5, 21),
                     group_col: str = "station_id", timestamp_col: str = "timestamp",
                     freq: str = "5min", dropna: bool = False) -> pd.DataFrame:
    """Añade columnas `{target_col}_lag_{k}` con el valor de `target_col` k pasos atrás.

    - Un paso = `freq` (por defecto 5 min), así que lag_5 = 25 min y lag_20 = 100 min.
    - Los lags se calculan por estación (`group_col`) y solo miran al pasado (shift positivo),
      por lo que no hay fuga de información.
    - Si entre la fila y la fila k atrás no han pasado exactamente k * freq (hay un hueco
      en la serie), el lag queda NaN en lugar de tomar un dato de hace días.
    - Las primeras k filas de cada estación y las posteriores a un hueco quedan NaN.
      Con `dropna=True` se eliminan las filas con algún lag NaN.
    - Devuelve un DataFrame nuevo ordenado por (group_col, timestamp_col); no modifica `df`.
    """
    out = df.sort_values([group_col, timestamp_col]).copy()
    ts = pd.to_datetime(out[timestamp_col])
    step = pd.Timedelta(freq)
    by_group = out[group_col]

    lag_cols = {}
    for k in lags:
        lagged = out.groupby(group_col)[target_col].shift(k)
        prev_ts = ts.groupby(by_group).shift(k)
        lag_cols[f"{target_col}_lag_{k}"] = lagged.where(ts - prev_ts == k * step)

    out = out.assign(**lag_cols)
    if dropna:
        out = out.dropna(subset=list(lag_cols))
    return out

