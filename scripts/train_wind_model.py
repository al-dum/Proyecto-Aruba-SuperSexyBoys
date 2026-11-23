"""Entrena y evalúa el pronóstico de generación eólica del parque Vader Piet.

Compara el modelo principal (HistGradientBoostingRegressor) con los baselines definidos
en configs/config.yaml, sobre un split temporal sin fuga de datos.

Requiere haber ejecutado antes los pasos 1 y 2 de scripts/run_data_pipeline.py.

Uso:
    python scripts/train_wind_model.py                      # horizonte de config.yaml
    python scripts/train_wind_model.py --horizons 1 12 72 288
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# pylint: disable=wrong-import-position
from energy_dispatch.evaluation.metrics import bias, mae, mae_by_group, rmse  # noqa: E402
from energy_dispatch.features.forecast import (  # noqa: E402
    TARGET_POWER, TARGET_WIND, build_forecast_dataset, feature_columns,
)
from energy_dispatch.features.wind import wind_kmh_to_power_kw  # noqa: E402
from energy_dispatch.models.train import build_model, save, temporal_split  # noqa: E402
from energy_dispatch.utils.config import load_config  # noqa: E402
# pylint: enable=wrong-import-position


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las lecturas filtradas (paso 2) y las estaciones con su distancia (paso 1)."""
    readings_path = os.path.join(ROOT, cfg["data"]["readings"])
    stations_path = os.path.join(ROOT, cfg["data"]["stations"])
    for path in (readings_path, stations_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No existe {path}. Ejecuta antes los pasos 1 y 2 de "
                "scripts/run_data_pipeline.py."
            )
    readings = pd.read_csv(readings_path, usecols=[
        "station_id", "timestamp", "wind_speed_kmh", "wind_direction_deg",
        "pressure_hpa", "temperature_c", "humidity_pct",
    ])
    return readings, pd.read_json(stations_path)


def cross_validate(train_df: pd.DataFrame, features: list[str], cfg: dict,
                   horizon: int) -> float:
    """MAE medio (km/h) del modelo principal con validación cruzada temporal sobre train.

    TimeSeriesSplit entrena siempre con el pasado y valida con el futuro inmediato; el
    `gap = horizon` es el mismo embargo que en el split final (ver `temporal_split`).
    """
    splitter = TimeSeriesSplit(n_splits=cfg["model"]["cv_splits"], gap=horizon)
    scores = []
    for fit_idx, val_idx in splitter.split(train_df):
        fit_df, val_df = train_df.iloc[fit_idx], train_df.iloc[val_idx]
        model = build_model(cfg["model"]["name"], features, cfg["model"]["params"])
        model.fit(fit_df, fit_df[TARGET_WIND])
        scores.append(mae(val_df[TARGET_WIND].to_numpy(), model.predict(val_df)))
    return float(pd.Series(scores).mean())


def evaluate_horizon(readings, stations, cfg: dict, horizon: int):
    """Entrena todos los modelos para un horizonte y devuelve métricas y MAE por hora."""
    dataset = build_forecast_dataset(readings, stations, horizon, cfg["data"]["freq"])
    features = feature_columns(dataset)
    train_df, test_df = temporal_split(dataset, cfg["data"]["test_size"])
    print(f"\n=== Horizonte {horizon} pasos ({horizon * 5} min) ===")
    print(f"Train: {train_df.index.min()} -> {train_df.index.max()} ({len(train_df):,} filas)")
    print(f"Test:  {test_df.index.min()} -> {test_df.index.max()} ({len(test_df):,} filas)")
    print(f"Features: {len(features)}")

    main_name = cfg["model"]["name"]
    cv_mae = cross_validate(train_df, features, cfg, horizon)
    print(f"CV temporal ({cfg['model']['cv_splits']} folds) {main_name}: "
          f"MAE viento {cv_mae:.2f} km/h")

    rows, hourly = [], {}
    y_power = test_df[TARGET_POWER].to_numpy()
    for name in [*cfg["baselines"], main_name]:
        params = cfg["model"]["params"] if name == main_name else None
        model = build_model(name, features, params).fit(train_df, train_df[TARGET_WIND])
        wind_pred = model.predict(test_df)
        # Se predice viento y se convierte con la curva de potencia conocida.
        power_pred = wind_kmh_to_power_kw(wind_pred)
        rows.append({
            "horizon_steps": horizon,
            "model": name,
            "mae_wind_kmh": mae(test_df[TARGET_WIND].to_numpy(), wind_pred),
            "mae_kw": mae(y_power, power_pred),
            "rmse_kw": rmse(y_power, power_pred),
            "bias_kw": bias(y_power, power_pred),
        })
        hourly[name] = mae_by_group(y_power, power_pred, test_df["forecast_hour_local"])
        if name == main_name:
            model_dir = os.path.join(ROOT, cfg["paths"]["models"])
            os.makedirs(model_dir, exist_ok=True)
            save(model, os.path.join(model_dir, f"wind_{main_name}_h{horizon}.joblib"))

    metrics = pd.DataFrame(rows)
    persistence_mae = metrics.loc[metrics["model"] == "persistence", "mae_kw"]
    if not persistence_mae.empty:
        # Skill > 0: mejor que persistencia. Es la métrica estándar en pronóstico eólico.
        metrics["skill_vs_persistence"] = 1 - metrics["mae_kw"] / persistence_mae.iloc[0]
    print(metrics.drop(columns="horizon_steps").round(3).to_string(index=False))

    hourly_df = pd.DataFrame(hourly).rename_axis("forecast_hour_local")
    hourly_df.insert(0, "horizon_steps", horizon)
    return metrics, hourly_df


def main() -> None:
    """Entrena y evalúa todos los modelos para cada horizonte pedido y guarda las métricas."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/config.yaml"))
    parser.add_argument("--horizons", type=int, nargs="+",
                        help="Horizontes en pasos de 5 min (por defecto, el de config.yaml).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    horizons = args.horizons or [cfg["forecast"]["horizon_steps"]]
    readings, stations = load_inputs(cfg)

    results = [evaluate_horizon(readings, stations, cfg, h) for h in horizons]
    report_dir = os.path.join(ROOT, cfg["paths"]["reports"])
    os.makedirs(report_dir, exist_ok=True)
    metrics_path = os.path.join(report_dir, "wind_model_metrics.csv")
    hourly_path = os.path.join(report_dir, "wind_model_mae_by_hour.csv")
    pd.concat([m for m, _ in results]).to_csv(metrics_path, index=False)
    pd.concat([h for _, h in results]).to_csv(hourly_path)
    print(f"\nMétricas guardadas en {metrics_path}")
    print(f"MAE por hora guardado en {hourly_path}")


if __name__ == "__main__":
    main()
