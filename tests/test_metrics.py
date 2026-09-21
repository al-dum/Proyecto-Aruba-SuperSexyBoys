import numpy as np

from energy_dispatch.evaluation.metrics import mae, rmse


def test_metrics_zero_error():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0 and rmse(y, y) == 0
