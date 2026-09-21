import pandas as pd


def load_raw(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[timestamp_col]).sort_values(timestamp_col)
