"""T9 + T10: entrena, compara, evalúa y exporta los modelos a t+15 min.

    python scripts/train_t15.py                    # viento y demanda
    python scripts/train_t15.py --problems wind    # solo viento (no necesita el paso 5)

Flujo (EntrenamientoML.pdf, caps. 9-13):
1. Dataset sin fuga (energy_dispatch.features.t15) y split temporal 80/20 con purga.
2. T9: búsqueda de hiperparámetros de Gradient Boosting con TimeSeriesSplit (gap = horizonte)
   SOLO sobre desarrollo; luego comparación con baselines y Ridge en las mismas particiones.
3. T10: reentrenar con todo desarrollo, evaluar UNA vez en prueba (métricas, error por hora,
   por horizonte y cerca de los umbrales de seguridad) y exportar el modelo con versiones.

Salidas: reports/t15/*.csv|json, reports/figures/t15_*.png y models/gb_t15_<problema>.pkl.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# pylint: disable=wrong-import-position
from energy_dispatch.features.t15 import (  # noqa: E402
    FREQ, HORIZON, SIGNALS, TARGET, TARGET_TIME,
    build_dataset, feature_columns, instant_table, temporal_split,
)
from energy_dispatch.models.gradient_boosting import make_gradient_boosting  # noqa: E402
from energy_dispatch.models.t15 import (  # noqa: E402
    HOT_TEMPERATURE_C, WIND_CRITICAL_KMH, WIND_MODERATE_KMH, build_models, error_by_hour,
    regression_metrics, summarize_cv, temporal_cv, to_output, wind_alert,
)

STATIONS = os.path.join(ROOT, "data/processed/1_station_filtering/"
                              "aruba_stations_ordered_by_distance_m.json")
STEP4 = os.path.join(ROOT, "data/processed/4_eolic_lag_features/aruba_readings_with_eolic_lags.csv")
STEP5 = os.path.join(ROOT, "data/processed/5_island_consumption/"
                           "aruba_readings_with_consumption.csv")
REPORTS = os.path.join(ROOT, "reports/t15")
FIGURES = os.path.join(ROOT, "reports/figures")
MODELS = os.path.join(ROOT, "models")

# Rejilla de hiperparámetros (PDF §12.8). Árboles pequeños y hojas grandes: a 15 min el
# cambio de la señal es casi todo ruido y un modelo flexible lo sobreajusta.
GB_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_iter": [200, 500],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [50, 200, 500],
    "loss": ["squared_error", "absolute_error"],
}
EXTRA_HORIZONS = ["5min", "15min", "30min", "1h"]
UNITS = {"wind": "generación eólica (kW)", "demand": "consumo de la isla (kW)"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_instants(problems: list[str]) -> pd.DataFrame:
    stations = pd.read_json(STATIONS)
    if "demand" in problems:
        if not os.path.exists(STEP5):
            raise FileNotFoundError(
                f"Falta {STEP5}. El paso 5 necesita data/external/"
                "aruba_energy_consumption_2015_2024.xlsx; después: "
                "python scripts/run_data_pipeline.py (o --problems wind).")
        readings = pd.read_csv(STEP5)
    else:
        readings = pd.read_csv(STEP4)
    return instant_table(readings, stations)


def tune_gradient_boosting(dev, problem, features, n_splits, gap) -> pd.DataFrame:
    rows = []
    grid = list(ParameterGrid(GB_GRID))
    for i, params in enumerate(grid, 1):
        start = time.time()
        cv = temporal_cv({"gb": make_gradient_boosting(features, **params)}, dev, problem,
                         n_splits=n_splits, gap=gap)
        summary = summarize_cv(cv).iloc[0]
        rows.append({**params, **summary.to_dict(), "segundos": time.time() - start})
        if i % 12 == 0 or i == len(grid):
            log(f"  {i}/{len(grid)} configuraciones")
    return pd.DataFrame(rows).sort_values("mae_kw_mean").reset_index(drop=True)


def threshold_analysis(test, predictions, problem) -> pd.DataFrame:
    """Cómo se comportan los modelos cerca de los umbrales de seguridad (T2)."""
    rows = []
    y = test[TARGET].to_numpy()
    if problem == "wind":
        actual = wind_alert(y)
        near = (np.abs(y - WIND_CRITICAL_KMH) <= 2) | (np.abs(y - WIND_MODERATE_KMH) <= 2)
        for name, pred in predictions.items():
            alert = wind_alert(pred)
            crit = actual == "CRITICAL"
            rows.append({
                "modelo": name,
                "acierto_alerta": float(np.mean(alert == actual)),
                "recall_critico": float(np.mean(alert[crit] == "CRITICAL")) if crit.any() else np.nan,
                "falsas_alarmas_criticas": int(np.sum((alert == "CRITICAL") & ~crit)),
                "criticas_no_avisadas": int(np.sum((alert != "CRITICAL") & crit)),
                "mae_kw_cerca_umbral": regression_metrics(y[near], pred[near], problem)["mae_kw"],
                "n_cerca_umbral": int(near.sum()),
            })
    else:
        hot = test["temperature_c"].to_numpy() >= HOT_TEMPERATURE_C
        peak = y >= np.quantile(y, 0.9)
        for name, pred in predictions.items():
            row = {"modelo": name,
                   "mae_kw_pico_10pct": regression_metrics(y[peak], pred[peak], problem)["mae_kw"],
                   "n_pico": int(peak.sum()), "n_calor": int(hot.sum())}
            row["mae_kw_calor"] = (regression_metrics(y[hot], pred[hot], problem)["mae_kw"]
                                   if hot.any() else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def horizon_analysis(instants, problem, features, gb_params, test_fraction) -> pd.DataFrame:
    rows = []
    for horizon in EXTRA_HORIZONS:
        data = build_dataset(instants, problem, pd.Timedelta(horizon))
        dev, test = temporal_split(data, test_fraction)
        models = build_models(features, gb_params)
        for name in ("persistencia", "ridge", "gradient_boosting"):
            fitted = models[name].fit(dev, dev[TARGET])
            m = regression_metrics(test[TARGET], fitted.predict(test), problem)
            rows.append({"horizonte": horizon, "modelo": name, **m})
    return pd.DataFrame(rows)


def plot_results(problem, test, predictions, by_hour) -> list[str]:
    os.makedirs(FIGURES, exist_ok=True)
    paths = []
    yt = to_output(test[TARGET], problem)
    when = test[TARGET_TIME].dt.tz_convert("America/Aruba")

    week = (when >= when.iloc[0]) & (when < when.iloc[0] + pd.Timedelta("2D"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(when[week], yt[week.to_numpy()], label="real", color="black", lw=1.2)
    for name in ("persistencia", "gradient_boosting"):
        ax.plot(when[week], to_output(predictions[name], problem)[week.to_numpy()],
                label=name, lw=0.9, alpha=0.85)
    ax.set(title=f"t+15: {UNITS[problem]}, dos primeros días de prueba (hora de Aruba)",
           ylabel="kW")
    ax.legend()
    paths.append(os.path.join(FIGURES, f"t15_{problem}_real_vs_predicho.png"))
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    by_hour[["persistencia", "ridge", "gradient_boosting"]].plot(ax=ax, marker="o", ms=3)
    ax.set(title=f"t+15: MAE por hora local ({UNITS[problem]})", ylabel="MAE (kW)",
           xlabel="hora local del objetivo")
    paths.append(os.path.join(FIGURES, f"t15_{problem}_error_por_hora.png"))
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    residual = to_output(predictions["gradient_boosting"], problem) - yt
    ax.hist(residual, bins=80, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set(title="t+15: residuos de Gradient Boosting en prueba", xlabel="predicho - real (kW)")
    paths.append(os.path.join(FIGURES, f"t15_{problem}_residuos.png"))
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def export_model(problem, model, features, dev, test, gb_params, test_metrics) -> str:
    """T10: modelo + todo lo necesario para usarlo en predict.py (T12) y en Streamlit."""
    os.makedirs(MODELS, exist_ok=True)
    path = os.path.join(MODELS, f"gb_t15_{problem}.pkl")
    bundle = {
        "model": model,
        "problem": problem,
        "signal": SIGNALS[problem],
        "predicts": ("viento en el parque t+15 (km/h); generación = wind_kmh_to_power_kw(pred)"
                     if problem == "wind" else "consumo de la isla t+15 (kW)"),
        "features": features,
        "horizon": str(HORIZON),
        "freq": str(FREQ),
        "history_needed": "24 h de lecturas cada 5 min (retardos, medias móviles y ayer a la misma hora)",
        "build_features": "energy_dispatch.features.t15.instant_table + build_dataset",
        "gb_params": gb_params,
        "train_period": [str(dev.index.min()), str(dev.index.max())],
        "test_period": [str(test.index.min()), str(test.index.max())],
        "test_metrics": test_metrics,
        "wind_alert_thresholds_kmh": ({"critical_below": WIND_CRITICAL_KMH,
                                       "moderate_below": WIND_MODERATE_KMH}
                                      if problem == "wind" else None),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "joblib": joblib.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, path)

    # Comprobación de recarga (PDF §13.3): mismas predicciones que el objeto en memoria.
    reloaded = joblib.load(path)
    if not np.allclose(reloaded["model"].predict(test), model.predict(test)):
        raise RuntimeError("El modelo recargado no reproduce las predicciones.")
    return path


def run(problem, instants, n_splits, test_fraction) -> dict:
    log(f"===== {problem}: {UNITS[problem]} a t+15 =====")
    features = feature_columns(problem)
    data = build_dataset(instants, problem)
    dev, test = temporal_split(data, test_fraction)
    gap = int(HORIZON / FREQ)
    log(f"desarrollo {len(dev):,} filas ({dev.index.min():%Y-%m-%d} a {dev.index.max():%Y-%m-%d}),"
        f" prueba {len(test):,} ({test.index.min():%Y-%m-%d} a {test.index.max():%Y-%m-%d})")
    os.makedirs(REPORTS, exist_ok=True)

    # ---------- T9: ajuste de hiperparámetros con validación temporal ----------
    log(f"T9: búsqueda de hiperparámetros ({len(ParameterGrid(GB_GRID))} configuraciones x "
        f"{n_splits} folds)")
    experiments = tune_gradient_boosting(dev, problem, features, n_splits, gap)
    experiments.to_csv(os.path.join(REPORTS, f"{problem}_gb_experimentos.csv"), index=False)
    gb_params = {k: (v.item() if hasattr(v, "item") else v)
                 for k, v in experiments.iloc[0][list(GB_GRID)].items()}
    log(f"mejor GB: {gb_params} -> MAE CV {experiments.iloc[0]['mae_kw_mean']:.1f} kW")

    # ---------- T9: comparación de modelos en las mismas particiones ----------
    models = build_models(features, gb_params)
    cv = temporal_cv(models, dev, problem, n_splits=n_splits, gap=gap)
    cv.to_csv(os.path.join(REPORTS, f"{problem}_cv_por_fold.csv"), index=False)
    comparison = summarize_cv(cv)
    comparison.to_csv(os.path.join(REPORTS, f"{problem}_cv_comparacion.csv"))
    log("comparación CV (MAE kW):\n" + comparison.round(2).to_string())

    # ---------- T10: evaluación única en prueba ----------
    predictions, test_rows = {}, []
    for name, model in models.items():
        fitted = model.fit(dev, dev[TARGET])
        predictions[name] = fitted.predict(test)
        test_rows.append({"modelo": name,
                          **regression_metrics(test[TARGET], predictions[name], problem)})
    test_table = pd.DataFrame(test_rows).sort_values("mae_kw")
    test_table.to_csv(os.path.join(REPORTS, f"{problem}_test_metricas.csv"), index=False)
    log("prueba:\n" + test_table.round(3).to_string(index=False))

    by_hour = error_by_hour(test, predictions, problem)
    by_hour.to_csv(os.path.join(REPORTS, f"{problem}_test_error_por_hora.csv"))
    thresholds = threshold_analysis(test, predictions, problem)
    thresholds.to_csv(os.path.join(REPORTS, f"{problem}_test_umbrales.csv"), index=False)
    log("umbrales:\n" + thresholds.round(3).to_string(index=False))
    horizons = horizon_analysis(instants, problem, features, gb_params, test_fraction)
    horizons.to_csv(os.path.join(REPORTS, f"{problem}_test_horizontes.csv"), index=False)
    by_horizon = horizons.pivot(index="horizonte", columns="modelo", values="mae_kw")
    log("horizontes (MAE kW):\n" + by_horizon.reindex(EXTRA_HORIZONS).round(1).to_string())
    figures = plot_results(problem, test, predictions, by_hour)

    gb_test = test_table.set_index("modelo").loc["gradient_boosting"].dropna().to_dict()
    path = export_model(problem, models["gradient_boosting"], features, dev, test,
                        gb_params, gb_test)
    log(f"exportado y recarga comprobada: {os.path.relpath(path, ROOT)}")

    summary = {"problem": problem, "gb_params": gb_params, "n_dev": len(dev),
               "n_test": len(test), "model_path": os.path.relpath(path, ROOT),
               "figures": [os.path.relpath(p, ROOT) for p in figures],
               "test": test_table.set_index("modelo").round(4).to_dict(orient="index")}
    with open(os.path.join(REPORTS, f"{problem}_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", nargs="+", choices=list(SIGNALS), default=list(SIGNALS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    instants = load_instants(args.problems)
    for problem in args.problems:
        run(problem, instants, args.n_splits, args.test_fraction)
    log("listo")


if __name__ == "__main__":
    main()
