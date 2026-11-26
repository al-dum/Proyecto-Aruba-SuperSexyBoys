"""Comprobaciones de no-fuga del dataset y los modelos a t+15 (T9/T10)."""
import numpy as np
import pandas as pd
import pytest

from energy_dispatch.features.t15 import (
    ANCHOR, METEO, TARGET, TARGET_TIME, build_dataset, feature_columns, temporal_split,
)
from energy_dispatch.models.t15 import DeltaRegressor, HourlyMeanBaseline, wind_alert


def make_instants(n=600, seed=0):
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC", name="timestamp")
    df = pd.DataFrame(index=grid)
    df["park_wind_kmh"] = 25 + np.cumsum(rng.normal(0, 0.3, n))
    df["consumo_estimado_isla_kw"] = 120_000 + np.cumsum(rng.normal(0, 50, n))
    for col in METEO:
        df[col] = rng.normal(20, 1, n)
    df["wind_dir_sin"], df["wind_dir_cos"] = 0.5, 0.5
    for i in range(3):
        df[f"wind_kmh_station_{i}"] = df["park_wind_kmh"] + rng.normal(0, 1, n)
    return df


@pytest.mark.parametrize("problem", ["wind", "demand"])
def test_features_do_not_change_when_the_future_changes(problem):
    instants = make_instants()
    cut = instants.index[400]
    altered = instants.copy()
    altered.loc[altered.index > cut, :] = altered.loc[altered.index > cut, :] * 3 + 7

    a = build_dataset(instants, problem)
    b = build_dataset(altered, problem)
    features = feature_columns(problem)
    past = a.index <= cut
    pd.testing.assert_frame_equal(a.loc[past, features], b.loc[past, features])
    # El objetivo sí cambia, pero solo en las filas cuyo objetivo cae después del corte.
    known = past & (a[TARGET_TIME] <= cut)
    np.testing.assert_allclose(a.loc[known, TARGET], b.loc[known, TARGET])


def test_target_is_exactly_fifteen_minutes_ahead():
    instants = make_instants()
    data = build_dataset(instants, "wind")
    expected = instants["park_wind_kmh"].reindex(data.index + pd.Timedelta("15min")).to_numpy()
    np.testing.assert_allclose(data[TARGET].to_numpy(), expected)
    assert (data[TARGET_TIME] - data.index == pd.Timedelta("15min")).all()
    assert TARGET not in feature_columns("wind")


def test_gaps_are_not_bridged():
    instants = make_instants()
    instants.iloc[300:303] = np.nan  # 15 min sin datos
    data = build_dataset(instants, "wind")
    assert instants.index[300] not in data.index
    # La fila de justo después del hueco no "ve" como retardo de 5 min el último valor previo.
    assert np.isnan(data.loc[instants.index[303], "lag_5min"])


def test_split_purges_targets_that_reach_the_test_block():
    data = build_dataset(make_instants(), "wind")
    dev, test = temporal_split(data, 0.2)
    assert dev[TARGET_TIME].max() < test.index.min()
    assert dev.index.max() < test.index.min()


def test_hourly_baseline_learns_only_from_fit_data():
    X = pd.DataFrame({"target_hour_local": [0, 0, 1, 1]})
    model = HourlyMeanBaseline().fit(X.iloc[:2], [10.0, 20.0])
    np.testing.assert_allclose(model.predict(X), [15, 15, 15, 15])  # hora 1 -> media global


def test_delta_regressor_adds_predicted_change_to_current_value():
    from sklearn.dummy import DummyRegressor

    X = pd.DataFrame({ANCHOR: [10.0, 20.0, 30.0], "x": [1.0, 2.0, 3.0]})
    model = DeltaRegressor(DummyRegressor(), features=["x"]).fit(X, [12.0, 22.0, 32.0])
    np.testing.assert_allclose(model.predict(X), [12.0, 22.0, 32.0])


def test_wind_alert_thresholds():
    assert list(wind_alert([10, 25, 40, 95])) == ["CRITICAL", "MODERATE", "NORMAL", "CRITICAL"]
