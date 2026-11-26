import numpy as np
import pandas as pd
import pytest

from energy_dispatch.features.forecast import (
    TARGET_TIME, TARGET_WIND, build_forecast_dataset, feature_columns,
)
from energy_dispatch.features.wind import park_wind_kmh
from energy_dispatch.models.train import HourlyMeanBaseline, temporal_split

HORIZON = 12


def _inputs(n_steps: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-04-01", periods=n_steps, freq="5min", tz="UTC")
    stations = pd.DataFrame({"id": ["near", "far"], "distance_m": [1000.0, 3000.0]})
    frames = []
    for sid in stations["id"]:
        frames.append(pd.DataFrame({
            "station_id": sid,
            "timestamp": timestamps,
            "wind_speed_kmh": 24 + rng.normal(0, 2, n_steps),
            "wind_direction_deg": rng.uniform(0, 360, n_steps),
            "pressure_hpa": 1010 + rng.normal(0, 1, n_steps),
            "temperature_c": 29 + rng.normal(0, 1, n_steps),
            "humidity_pct": 75 + rng.normal(0, 3, n_steps),
        }))
    return pd.concat(frames, ignore_index=True), stations


def test_target_is_wind_h_steps_ahead():
    readings, stations = _inputs()
    dataset = build_forecast_dataset(readings, stations, HORIZON)
    wind = park_wind_kmh(readings, stations)

    t = dataset.index[100]
    expected_time = t + HORIZON * pd.Timedelta("5min")
    assert dataset.loc[t, TARGET_TIME] == expected_time
    assert dataset.loc[t, TARGET_WIND] == pytest.approx(wind.loc[expected_time])


def test_features_do_not_change_when_the_future_changes():
    readings, stations = _inputs()
    cutoff = pd.Timestamp("2026-04-02 00:00", tz="UTC")
    future_changed = readings.copy()
    after = future_changed["timestamp"] > cutoff
    future_changed.loc[after, ["wind_speed_kmh", "pressure_hpa"]] *= 3

    original = build_forecast_dataset(readings, stations, HORIZON)
    perturbed = build_forecast_dataset(future_changed, stations, HORIZON)
    features = feature_columns(original)

    past = original.index[original.index <= cutoff]
    pd.testing.assert_frame_equal(original.loc[past, features], perturbed.loc[past, features])
    # El objetivo sí debe cambiar: comprueba que la perturbación alcanza al futuro.
    assert not np.allclose(original.loc[past, TARGET_WIND].tail(HORIZON),
                           perturbed.loc[past, TARGET_WIND].tail(HORIZON))


def test_target_columns_are_not_features():
    readings, stations = _inputs()
    features = feature_columns(build_forecast_dataset(readings, stations, HORIZON))
    assert not any(col.startswith("target_") for col in features)


def test_temporal_split_leaves_an_embargo_of_one_horizon():
    readings, stations = _inputs()
    dataset = build_forecast_dataset(readings, stations, HORIZON)
    train_df, test_df = temporal_split(dataset, test_size=0.2)

    assert train_df.index.max() < test_df.index.min()
    assert train_df[TARGET_TIME].max() < test_df.index.min()


def test_hourly_mean_baseline_only_learns_from_fit_data():
    X = pd.DataFrame({"forecast_hour_local": [0, 0, 1, 1]})
    model = HourlyMeanBaseline().fit(X, [10.0, 20.0, 30.0, 50.0])
    pred = model.predict(pd.DataFrame({"forecast_hour_local": [0, 1, 7]}))
    np.testing.assert_allclose(pred, [15.0, 40.0, 27.5])
