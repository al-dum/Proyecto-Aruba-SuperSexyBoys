import numpy as np

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / y)) * 100)


def bias(y, yhat):
    """Error medio (pred - real). Positivo = el modelo sobrestima."""
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))
