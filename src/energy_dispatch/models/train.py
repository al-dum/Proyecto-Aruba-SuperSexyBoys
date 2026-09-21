import joblib
from sklearn.ensemble import GradientBoostingRegressor


def train(X, y, params: dict | None = None):
    model = GradientBoostingRegressor(**(params or {}))
    return model.fit(X, y)


def save(model, path: str) -> None:
    joblib.dump(model, path)
