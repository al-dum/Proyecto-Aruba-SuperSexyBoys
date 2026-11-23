import numpy as np
import pandas as pd


def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / y)) * 100)


def bias(y, yhat):
    """Error medio (pred - real). Positivo = el modelo sobrestima."""
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


def mae_by_group(y, yhat, groups) -> pd.Series:
    """MAE por grupo (p. ej. por hora del día, docs/METODOLOGIA.md §9)."""
    error = pd.Series(np.abs(np.asarray(y) - np.asarray(yhat)), index=np.asarray(groups))
    return error.groupby(level=0).mean()
