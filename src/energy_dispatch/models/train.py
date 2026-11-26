"""Modelos de pronóstico: baselines, regresión lineal y Gradient Boosting.

Todos siguen la interfaz de scikit-learn (`fit(X, y)` / `predict(X)`) y reciben la tabla
completa de `energy_dispatch.features.forecast`; cada modelo elige sus columnas. Así se
comparan en igualdad de condiciones con el mismo split y las mismas métricas.

Orden recomendado (docs/METODOLOGIA.md §8): si el modelo de ML no gana a los baselines,
algo va mal.
"""
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ColumnBaseline(RegressorMixin, BaseEstimator):
    """Predice copiando una columna ya conocida en el instante de emisión.

    - Persistencia: `park_wind_kmh` -> "dentro de h pasos el viento será el de ahora".
    - Naive estacional: `park_wind_kmh_same_time_last_day` -> "el de ayer a la misma hora".
    """

    def __init__(self, column: str = "park_wind_kmh"):
        self.column = column

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """No aprende nada; existe para cumplir la interfaz de scikit-learn."""
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Devuelve la columna tal cual."""
        return X[self.column].to_numpy(dtype=float)


class HourlyMeanBaseline(RegressorMixin, BaseEstimator):
    """Climatología: media del objetivo en TRAIN para cada hora local del día objetivo.

    Las medias se calculan solo con los datos de `fit` (train). Calcularlas con toda la
    serie sería otra fuga: el baseline "conocería" el periodo de test.
    """

    def __init__(self, hour_column: str = "forecast_hour_local"):
        self.hour_column = hour_column

    def fit(self, X, y):
        """Guarda la media del objetivo por hora."""
        y = pd.Series(y, index=X.index)
        self.hourly_mean_ = y.groupby(X[self.hour_column]).mean()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X):
        """Media de la hora correspondiente (media global si esa hora no salió en train)."""
        return X[self.hour_column].map(self.hourly_mean_).fillna(self.global_mean_).to_numpy()


def _select(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("features", "passthrough", features)])


def build_model(name: str, features: list[str], params: dict | None = None):
    """Crea un modelo sin entrenar por nombre.

    El preprocesado (imputación, escalado) va DENTRO del pipeline, así se ajusta solo con
    train en cada `fit` y nunca ve test ni los folds de validación (otra fuente de fuga).
    """
    params = params or {}
    if name == "persistence":
        return ColumnBaseline("park_wind_kmh")
    if name == "seasonal_naive":
        return ColumnBaseline("park_wind_kmh_same_time_last_day")
    if name == "hourly_mean":
        return HourlyMeanBaseline()
    if name == "linear":
        return make_pipeline(_select(features), SimpleImputer(strategy="median"),
                             StandardScaler(), LinearRegression(**params))
    if name == "hist_gradient_boosting":
        # Gestiona los NaN de forma nativa (huecos de los lags): no hace falta imputar.
        return make_pipeline(_select(features), HistGradientBoostingRegressor(**params))
    raise ValueError(f"Modelo desconocido: {name}")


def train(X: pd.DataFrame, y, name: str = "hist_gradient_boosting",
          features: list[str] | None = None, params: dict | None = None):
    """Construye y entrena el modelo `name`."""
    return build_model(name, features or list(X.columns), params).fit(X, y)


def temporal_split(df: pd.DataFrame, test_size: float,
                   target_time_col: str = "target_time") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por fecha con hueco de seguridad (embargo). NUNCA aleatorio.

    - Test = la fracción `test_size` más reciente de la serie.
    - Train = filas cuyo OBJETIVO (t + h) es anterior al inicio de test. Sin este hueco, las
      últimas h filas de train tendrían como objetivo valores que están dentro de test: el
      modelo se entrenaría con respuestas del periodo que luego se usa para evaluarlo.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1.")
    df = df.sort_index()
    test_start = df.index[int(len(df) * (1 - test_size))]
    test = df[df.index >= test_start]
    train_df = df[df[target_time_col] < test_start]
    return train_df, test


def save(model, path: str) -> None:
    """Guarda el modelo entrenado con joblib."""
    joblib.dump(model, path)
