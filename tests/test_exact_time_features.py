import pandas as pd
import pytest

from energy_dispatch.features.build import add_exact_time_features


def test_future_target_uses_exact_minutes_not_row_position():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 00:05",
            "2026-01-01 00:15", "2026-01-01 00:25",
        ], utc=True),
        "power": [10.0, 20.0, 30.0, 40.0],
    })

    result = add_exact_time_features(df, "power", {"target": "15min", "lag": "-5min"})

    assert result["target"].iloc[0] == 30.0
    assert result["target"].iloc[1:].isna().all()
    assert result["lag"].iloc[1] == 10.0
    assert result["lag"].iloc[[0, 2, 3]].isna().all()
    assert list(df.columns) == ["timestamp", "power"]


def test_unsorted_input_and_nondefault_index_keep_row_alignment():
    df = pd.DataFrame({
        "time": ["2026-01-01 00:15", "2026-01-01 00:00", "2026-01-01 00:05"],
        "power": [30.0, 10.0, 20.0],
    }, index=[7, 9, 12])

    result = add_exact_time_features(
        df, "power", {"target": pd.Timedelta("15min")}, timestamp_col="time",
    )

    assert result.index.tolist() == [7, 9, 12]
    assert result.loc[9, "target"] == 30.0
    assert pd.isna(result.loc[7, "target"])
    pd.testing.assert_series_equal(result["time"], df["time"])


def test_timestamps_with_different_timezones_are_aligned_in_utc():
    df = pd.DataFrame({
        "timestamp": ["2026-01-01 00:00-04:00", "2026-01-01 04:15+00:00"],
        "power": [10.0, 30.0],
    })
    result = add_exact_time_features(df, "power", {"target": "15min"})
    assert result["target"].iloc[0] == 30.0


@pytest.mark.parametrize("timestamps", [
    ["2026-01-01", "2026-01-01"],
    ["2026-01-01", None],
])
def test_invalid_or_duplicate_timestamps_are_rejected(timestamps):
    df = pd.DataFrame({"timestamp": timestamps, "power": [10.0, 20.0]})
    with pytest.raises(ValueError):
        add_exact_time_features(df, "power", {"target": "15min"})


def test_existing_columns_cannot_be_overwritten():
    df = pd.DataFrame({"timestamp": ["2026-01-01"], "power": [10.0]})
    with pytest.raises(ValueError, match="sobrescribir"):
        add_exact_time_features(df, "power", {"power": "15min"})


def test_existing_missing_values_are_not_imputed():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
        "power": [10.0, 20.0, 30.0, float("nan")],
    })
    result = add_exact_time_features(df, "power", {"target": "15min"})
    assert result["target"].isna().all()
