import pandas as pd

from energy_dispatch.features.build import add_lag_features


def test_add_lag_features_uses_only_contiguous_prior_observations():
    df = pd.DataFrame(
        {
            "station_id": ["a", "a", "a", "b", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:05",
                    "2026-01-01 00:15",
                    "2026-01-01 00:00",
                    "2026-01-01 00:05",
                ]
            ),
            "generacion_eolica_kw": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )

    result = add_lag_features(df, lags=[1])

    lags = result["generacion_eolica_kw_lag_1"].tolist()
    assert pd.isna(lags[0])
    assert lags[1] == 10.0
    assert pd.isna(lags[2])
    assert pd.isna(lags[3])
    assert lags[4] == 100.0
