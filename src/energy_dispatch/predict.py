"""Predicción a t+15 min con el modelo exportado por `scripts/train_t15.py` (T11/T12).

Uso desde código (p. ej. el frontend, T13):

    from energy_dispatch.predict import predict
    result = predict(readings_df, problem="wind")   # dict con prediction_kw y alert

Uso desde la línea de comandos:

    python -m energy_dispatch.predict --readings lecturas.csv --problem wind

Entrada: lecturas en formato largo, una fila por estación e instante (5 min), con columnas
    station_id, timestamp, wind_speed_kmh, wind_direction_deg, temperature_c, humidity_pct,
    pressure_hpa, precipitation_mm
y, solo para `demand`, además `consumo_estimado_isla_kw` (serie modelada, no medida).
Hace falta al menos 24 h de historia hasta el instante t (la última lectura).

Salida: dict con `timestamp` (t), `target_time` (t + 15 min), `problem`, `prediction_kw`;
para `wind`, además `wind_kmh` (viento previsto) y `alert` (NORMAL / MODERATE / CRITICAL).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from energy_dispatch.features.t15 import (
    ANCHOR, HORIZON, METEO, SIGNALS, STEPS_PER_DAY, FREQ, build_dataset, instant_table,
)
from energy_dispatch.models.t15 import to_output, wind_alert

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
STATIONS_PATH = ROOT / "data/processed/1_station_filtering/aruba_stations_ordered_by_distance_m.json"

READING_COLUMNS = ["station_id", "timestamp", "wind_speed_kmh", "wind_direction_deg",
                   "temperature_c", "humidity_pct", "pressure_hpa", "precipitation_mm"]
MIN_HISTORY = pd.Timedelta("24h")


class PredictionInputError(ValueError):
    """Entrada no válida o insuficiente para predecir."""


def load_model(problem: str, path: str | os.PathLike | None = None) -> dict:
    """Carga el bundle (joblib) de `models/gb_t15_<problem>.pkl` y comprueba su contenido."""
    if problem not in SIGNALS:
        raise PredictionInputError(f"Problema desconocido: {problem!r}. Usa {list(SIGNALS)}.")
    path = Path(path) if path else MODELS_DIR / f"gb_t15_{problem}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta: python scripts/train_t15.py "
                                f"--problems {problem}")
    bundle = joblib.load(path)
    missing = {"model", "features", "problem"} - set(bundle)
    if missing or bundle["problem"] != problem:
        raise ValueError(f"{path} no es un modelo de {problem!r} válido (faltan {missing}).")
    return bundle


def _check_readings(readings: pd.DataFrame, problem: str, stations: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, fechas y estaciones. Devuelve una copia con `timestamp` en UTC."""
    required = set(READING_COLUMNS) | ({SIGNALS[problem]} if problem == "demand" else set())
    missing = sorted(required.difference(readings.columns))
    if missing:
        raise PredictionInputError(f"Faltan columnas en las lecturas: {missing}")

    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise PredictionInputError("Hay valores de `timestamp` que no son fechas válidas.")

    unknown = set(df["station_id"]).difference(stations["id"])
    if unknown:
        raise PredictionInputError(f"Estaciones no incluidas en la tabla de estaciones: "
                                   f"{sorted(unknown)}")
    return df


def predict(readings: pd.DataFrame, problem: str = "wind", stations: pd.DataFrame | None = None,
            bundle: dict | None = None) -> dict:
    """Predice el valor a t+15 min a partir de las lecturas hasta t (la última fecha)."""
    bundle = bundle or load_model(problem)
    stations = stations if stations is not None else pd.read_json(STATIONS_PATH)
    df = _check_readings(readings, problem, stations)

    instants = instant_table(df, stations)
    if instants.index[-1] - instants.index[0] < MIN_HISTORY:
        raise PredictionInputError(
            f"Historia insuficiente: {instants.index[-1] - instants.index[0]} de datos y "
            f"hacen falta al menos {MIN_HISTORY} (≥ {STEPS_PER_DAY} instantes de {FREQ}).")

    # Igual que en el entrenamiento, pero sin exigir objetivo: la última fila no tiene futuro.
    rows = build_dataset(instants, problem, require_target=False)
    last = rows.iloc[[-1]]
    if last.index[0] != instants.index[-1] or last[ANCHOR].isna().any():
        raise PredictionInputError(f"Falta el valor de {SIGNALS[problem]} en el último instante.")
    missing_meteo = [c for c in METEO if pd.isna(last[c].iloc[0])]
    if missing_meteo:
        raise PredictionInputError(f"Falta meteorología en el último instante: {missing_meteo}")

    raw = np.asarray(bundle["model"].predict(last[bundle["features"]]), dtype=float)  # shape (1,)
    t = last.index[0]
    result = {
        "timestamp": t.isoformat(),
        "target_time": (t + HORIZON).isoformat(),
        "problem": problem,
        "prediction_kw": float(to_output(raw, problem)[0]),
    }
    if problem == "wind":
        wind_kmh = max(float(raw[0]), 0.0)
        result["wind_kmh"] = wind_kmh
        result["alert"] = str(wind_alert(np.array([wind_kmh]))[0])
    return result


def read_readings(path: str) -> pd.DataFrame:
    """Lee lecturas desde CSV o JSON según la extensión del archivo."""
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise PredictionInputError("El archivo de lecturas debe ser .csv o .json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predice la generación o el consumo a t+15 min.")
    parser.add_argument("--readings", required=True, help="Lecturas (.csv o .json)")
    parser.add_argument("--problem", choices=list(SIGNALS), default="wind")
    parser.add_argument("--model", help="Ruta del .pkl (por defecto models/gb_t15_<problema>.pkl)")
    parser.add_argument("--stations", help="Tabla de estaciones (.json)")
    args = parser.parse_args(argv)
    try:
        bundle = load_model(args.problem, args.model)
        stations = pd.read_json(args.stations) if args.stations else pd.read_json(STATIONS_PATH)
        result = predict(read_readings(args.readings), args.problem, stations, bundle)
    except (PredictionInputError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
