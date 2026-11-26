"""Modelos a t+15 min (T9) y su evaluación (T10).

Todos siguen la interfaz de scikit-learn y reciben la tabla completa de
`energy_dispatch.features.t15.build_dataset`; cada modelo elige sus columnas. Así se
comparan con las mismas particiones y las mismas métricas (EntrenamientoML.pdf, §12.5).

Orden (docs/METODOLOGIA.md §8): baselines -> Ridge -> Gradient Boosting. Si el modelo de
ML no gana a la persistencia, algo va mal.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_dispatch.features.t15 import ANCHOR, TARGET, TARGET_HOUR
from energy_dispatch.features.wind import wind_kmh_to_power_kw

# Umbrales de alerta eólica del paso 3 (scripts/run_data_pipeline.py): el viento al que la
# curva V90 da el 15 % (crítico) y el 40 % (moderado) de la capacidad, y el cut-out.
WIND_CRITICAL_KMH = 22.24
WIND_MODERATE_KMH = 31.45
WIND_CUT_OUT_KMH = 90.0
# docs/umbrales-seguridad-aruba.md: > 30.3 °C dispara el aire acondicionado (alerta moderada).
HOT_TEMPERATURE_C = 30.3


class SignalBaseline(RegressorMixin, BaseEstimator):
    """Copia una columna ya conocida en t. No aprende nada.

    - Persistencia (`now`): "dentro de 15 min, lo mismo que ahora".
    - Naive estacional (`same_time_yesterday`): "lo mismo que ayer a esa hora". Si ayer no
      hay dato (primer día, huecos) cae a la persistencia.
    """

    def __init__(self, column: str = ANCHOR):
        self.column = column

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return X[self.column].fillna(X[ANCHOR]).to_numpy(dtype=float)


class TrainMeanBaseline(RegressorMixin, BaseEstimator):
    """Media del objetivo en entrenamiento (DummyRegressor del PDF, §12.2)."""

    def fit(self, X, y):  # pylint: disable=unused-argument
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class HourlyMeanBaseline(RegressorMixin, BaseEstimator):
    """Media del objetivo en entrenamiento para cada hora local del instante objetivo.

    Las medias salen SOLO de `fit`: calcularlas con toda la serie sería una fuga.
    """

    def __init__(self, hour_column: str = TARGET_HOUR):
        self.hour_column = hour_column

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.hourly_mean_ = y.groupby(X[self.hour_column]).mean()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X):
        return X[self.hour_column].map(self.hourly_mean_).fillna(self.global_mean_).to_numpy()


class DeltaRegressor(RegressorMixin, BaseEstimator):
    """Aprende el CAMBIO en 15 min (objetivo - valor actual) y lo suma al valor actual.

    A 15 min vista el valor actual explica casi todo; lo que hay que aprender es la
    variación. Además los árboles no extrapolan: si el consumo de test supera el máximo de
    train, un árbol que predice niveles se queda clavado en ese máximo, mientras que uno
    que predice incrementos no.
    """

    def __init__(self, estimator=None, features=None, anchor: str = ANCHOR):
        self.estimator = estimator
        self.features = features
        self.anchor = anchor

    def fit(self, X, y):
        delta = np.asarray(y, dtype=float) - X[self.anchor].to_numpy(dtype=float)
        self.estimator_ = clone(self.estimator).fit(X[list(self.features)], delta)
        return self

    def predict(self, X):
        delta = self.estimator_.predict(X[list(self.features)])
        return X[self.anchor].to_numpy(dtype=float) + delta


def make_gradient_boosting(features: list[str], **params) -> DeltaRegressor:
    """HistGradientBoostingRegressor (PDF §12.5): rápido y acepta NaN sin imputar.

    `early_stopping=False`: su parada temprana separa una validación ALEATORIA dentro de
    train, que en una serie temporal mezcla pasado y futuro.
    """
    defaults = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                "min_samples_leaf": 100, "loss": "squared_error"}
    gb = HistGradientBoostingRegressor(**{**defaults, **params}, early_stopping=False,
                                       random_state=42)
    return DeltaRegressor(gb, features)


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
        "mae_kw": mean_absolute_error(yt, yp),
        "rmse_kw": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": r2_score(yt, yp),
        "sesgo_kw": float(np.mean(yp - yt)),
    }
    if problem == "wind":
        out["mae_kmh"] = mean_absolute_error(np.asarray(y_true), np.asarray(y_pred))
    else:
        out["mape_pct"] = float(np.mean(np.abs((yt - yp) / yt)) * 100)
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
