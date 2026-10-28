import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = df[timestamp_col]
    return df.assign(hour=ts.dt.hour, dayofweek=ts.dt.dayofweek, month=ts.dt.month)


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
