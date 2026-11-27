"""Catálogo de modelos a t+15 min (T9) y su evaluación (T10).

Dónde está cada modelo:
- `baselines.py`: persistencia, naive estacional, media de train y media por hora.
- `gradient_boosting.py`: el modelo principal (Gradient Boosting, con `DeltaRegressor`).
- este archivo: `build_models` (qué modelos se comparan), métricas, validación temporal y
  alertas.

Todos siguen la interfaz de scikit-learn y reciben la tabla completa de
`energy_dispatch.features.t15.build_dataset`; cada modelo elige sus columnas. Así se
comparan con las mismas particiones y las mismas métricas (EntrenamientoML.pdf, §12.5).

Orden (docs/METODOLOGIA.md §8): baselines -> Ridge -> Gradient Boosting. Si el modelo de
ML no gana a la persistencia, algo va mal.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_dispatch.evaluation.metrics import bias, mae, mape, rmse
from energy_dispatch.features.t15 import ANCHOR, TARGET, TARGET_HOUR
from energy_dispatch.features.wind import (
    WIND_CRITICAL_KMH, WIND_CUT_OUT_KMH, WIND_MODERATE_KMH, wind_kmh_to_power_kw,
)
from energy_dispatch.models.baselines import HourlyMeanBaseline, SignalBaseline, TrainMeanBaseline
from energy_dispatch.models.gradient_boosting import DeltaRegressor, make_gradient_boosting

# docs/umbrales-seguridad-aruba.md: > 30.3 °C dispara el aire acondicionado (alerta moderada).
HOT_TEMPERATURE_C = 30.3


def build_models(features: list[str], gb_params: dict | None = None) -> dict:
    """Modelos a comparar. Imputación y escalado van DENTRO del pipeline de Ridge, así se
    ajustan solo con los datos de cada `fit` (PDF §9.7 y §11.5)."""
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    return {
        "persistencia": SignalBaseline(ANCHOR),
        "naive_estacional": SignalBaseline("same_time_yesterday"),
        "media_train": TrainMeanBaseline(),
        "media_por_hora": HourlyMeanBaseline(),
        "ridge": DeltaRegressor(ridge, features),
        "gradient_boosting": make_gradient_boosting(features, **(gb_params or {})),
    }


def to_output(values, problem: str) -> np.ndarray:
    """Magnitud que se entrega: generación (kW) para el viento, consumo (kW) para la demanda."""
    values = np.asarray(values, dtype=float)
    return wind_kmh_to_power_kw(np.clip(values, 0, None)) if problem == "wind" else values


def regression_metrics(y_true, y_pred, problem: str) -> dict:
    """MAE, RMSE, R², sesgo y MAPE (PDF §12.6) sobre la magnitud que se entrega (kW).

    MAPE solo para la demanda: la generación eólica vale 0 a menudo y el MAPE explota.
    Para el viento se añade además el MAE en km/h (lo que predice el modelo).
    """
    yt, yp = to_output(y_true, problem), to_output(y_pred, problem)
    out = {
        "mae_kw": mae(yt, yp),
        "rmse_kw": rmse(yt, yp),
        "r2": r2_score(yt, yp),
        "sesgo_kw": bias(yt, yp),
    }
    if problem == "wind":
        out["mae_kmh"] = mae(np.asarray(y_true), np.asarray(y_pred))
    else:
        out["mape_pct"] = mape(yt, yp)
    return out


def temporal_cv(models: dict, dev: pd.DataFrame, problem: str, n_splits: int = 5,
                gap: int = 3) -> pd.DataFrame:
    """Validación cruzada temporal (TimeSeriesSplit, ventana expansiva, PDF §10.8-10.9).

    `gap` cuenta FILAS: con gap = 3 se descartan al menos 15 min entre train y validación,
    así ninguna fila de train tiene un objetivo dentro del bloque de validación (si faltan
    filas por huecos, la separación real solo puede ser mayor).
    """
    dev = dev.sort_index()
    rows = []
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for fold, (tr, va) in enumerate(splitter.split(dev)):
        train, valid = dev.iloc[tr], dev.iloc[va]
        for name, model in models.items():
            fitted = clone(model).fit(train, train[TARGET])
            metrics = regression_metrics(valid[TARGET], fitted.predict(valid), problem)
            rows.append({"modelo": name, "fold": fold, "n_train": len(train),
                         "n_valid": len(valid), "valid_desde": valid.index[0],
                         "valid_hasta": valid.index[-1], **metrics})
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame, primary: str = "mae_kw") -> pd.DataFrame:
    """Media ± desviación de cada métrica por modelo, ordenado por la métrica principal."""
    metrics = [c for c in cv.columns if c.endswith(("_kw", "_kmh", "_pct")) or c == "r2"]
    summary = cv.groupby("modelo")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    return summary.sort_values(f"{primary}_mean")


def wind_alert(wind_kmh) -> np.ndarray:
    """NORMAL / MODERATE / CRITICAL con los umbrales del paso 3 aplicados al viento del parque."""
    wind = np.asarray(wind_kmh, dtype=float)
    alert = np.where(wind < WIND_MODERATE_KMH, "MODERATE", "NORMAL").astype(object)
    alert[wind < WIND_CRITICAL_KMH] = "CRITICAL"
    alert[wind >= WIND_CUT_OUT_KMH] = "CRITICAL"
    return alert


def error_by_hour(test: pd.DataFrame, predictions: dict, problem: str) -> pd.DataFrame:
    """MAE (kW) por hora local del objetivo: las horas punta importan más al despacho."""
    yt = to_output(test[TARGET], problem)
    table = {name: pd.Series(np.abs(to_output(p, problem) - yt), index=test.index)
             .groupby(test[TARGET_HOUR].to_numpy()).mean()
             for name, p in predictions.items()}
    return pd.DataFrame(table).rename_axis("hora_local")
