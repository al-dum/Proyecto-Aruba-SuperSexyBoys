import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    ts = df[timestamp_col]
    return df.assign(hour=ts.dt.hour, dayofweek=ts.dt.dayofweek, month=ts.dt.month)
