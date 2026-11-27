import numpy as np
import pandas as pd
import pytest

from energy_dispatch.features.t15 import FREQ, TARGET, build_dataset, feature_columns, instant_table
from energy_dispatch.models.gradient_boosting import make_gradient_boosting
from energy_dispatch.predict import PredictionInputError, predict

STATIONS = pd.DataFrame({"id": ["a", "b", "c"], "distance_m": [100.0, 200.0, 300.0]})


def _readings(n_steps: int = 3 * 288, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-03-01", periods=n_steps, freq=FREQ, tz="UTC")
    frames = [pd.DataFrame({
        "station_id": sid, "timestamp": ts,
        "wind_speed_kmh": rng.uniform(0, 40, n_steps),
        "wind_direction_deg": rng.uniform(0, 360, n_steps),
        "temperature_c": rng.uniform(25, 32, n_steps),
        "humidity_pct": rng.uniform(50, 80, n_steps),
        "pressure_hpa": 1010 + rng.normal(0, 1, n_steps),
        "precipitation_mm": 0.0,
    }) for sid in STATIONS["id"]]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle():
    """Modelo pequeño entrenado en memoria con el mismo preprocesado que T9/T10."""
    instants = instant_table(_readings(), STATIONS)
    features = feature_columns("wind")
    data = build_dataset(instants, "wind")
    model = make_gradient_boosting(features, max_iter=20).fit(data, data[TARGET])
    return {"model": model, "features": features, "problem": "wind"}


def test_predict_returns_expected_keys_and_valid_alert(bundle):
    result = predict(_readings(), "wind", STATIONS, bundle)

    assert set(result) >= {"timestamp", "target_time", "prediction_kw", "wind_kmh", "alert"}
    assert result["alert"] in {"NORMAL", "MODERATE", "CRITICAL"}
    assert result["prediction_kw"] >= 0
    assert pd.Timestamp(result["target_time"]) - pd.Timestamp(result["timestamp"]) == pd.Timedelta("15min")


def test_last_row_for_inference_matches_training_row(bundle):
    """La fila de inferencia en t debe ser idéntica a la de entrenamiento en ese mismo t."""
    instants = instant_table(_readings(), STATIONS)
    features = feature_columns("wind")
    train = build_dataset(instants, "wind")
    infer = build_dataset(instants, "wind", require_target=False)

    t = train.index[-5]
    pd.testing.assert_series_equal(train.loc[t, features], infer.loc[t, features])


def test_missing_column_raises_clear_error(bundle):
    with pytest.raises(PredictionInputError, match="wind_speed_kmh"):
        predict(_readings().drop(columns=["wind_speed_kmh"]), "wind", STATIONS, bundle)


def test_insufficient_history_raises(bundle):
    readings = _readings()
    short = readings[readings["timestamp"] < readings["timestamp"].min() + pd.Timedelta("2h")]
    with pytest.raises(PredictionInputError, match="Historia insuficiente"):
        predict(short, "wind", STATIONS, bundle)
