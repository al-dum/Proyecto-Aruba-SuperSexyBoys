"""Pipeline de datos meteorológicos de Aruba (notebooks 0 a 4 y consumo insular estimado)"""
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from energy_dispatch.data.consumption import (  # noqa: E402  pylint: disable=wrong-import-position
    WEATHER_COLUMNS, estimate_island_consumption, project_monthly_consumption,
)
from energy_dispatch.features.build import add_lag_features  # noqa: E402  pylint: disable=wrong-import-position


RAW_DIR = os.path.join(ROOT, "data/raw")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")

READINGS_RAW_PATH = os.path.join(RAW_DIR, "readings/aruba_readings_20260924.json")
STATIONS_RAW_PATH = os.path.join(RAW_DIR, "aruba_stations.json")

CLEANING_DIR = os.path.join(PROCESSED_DIR, "0_cleaning")
CLEAN_CSV_PATH = os.path.join(CLEANING_DIR, "aruba_readings_cleaned.csv")
CLEAN_JSON_PATH = os.path.join(CLEANING_DIR, "aruba_readings_cleaned.json")
CLEAN_REPORT_PATH = os.path.join(CLEANING_DIR, "data_cleaning_report.json")

STATIONS_DIR = os.path.join(PROCESSED_DIR, "1_station_filtering")
STATIONS_PATH = os.path.join(STATIONS_DIR, "aruba_stations_ordered_by_distance_m.json")

VALIDATION_DIR = os.path.join(PROCESSED_DIR, "2_data_validation")
FILTERED_CSV_PATH = os.path.join(VALIDATION_DIR, "aruba_reading_filtered_by_station_distance.csv")
FILTERED_JSON_PATH = os.path.join(VALIDATION_DIR, "aruba_reading_filtered_by_station_distance.json")

THRESHOLDS_DIR = os.path.join(PROCESSED_DIR, "3_security_thresholds")
THRESHOLDS_CSV_PATH = os.path.join(THRESHOLDS_DIR, "aruba_readings_with_security_thresholds.csv")
THRESHOLDS_JSON_PATH = os.path.join(THRESHOLDS_DIR, "aruba_readings_with_security_thresholds.json")

LAGS_DIR = os.path.join(PROCESSED_DIR, "4_eolic_lag_features")
LAGS_CSV_PATH = os.path.join(LAGS_DIR, "aruba_readings_with_eolic_lags.csv")
LAGS_JSON_PATH = os.path.join(LAGS_DIR, "aruba_readings_with_eolic_lags.json")

CONSUMPTION_EXCEL_PATH = os.path.join(
    ROOT, "data/external/Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev.xlsx"
)
CONSUMPTION_DIR = os.path.join(PROCESSED_DIR, "5_island_consumption")
CONSUMPTION_CSV_PATH = os.path.join(CONSUMPTION_DIR, "aruba_readings_with_consumption.csv")
CONSUMPTION_JSON_PATH = os.path.join(CONSUMPTION_DIR, "aruba_readings_with_consumption.json")

# Paso 0: limpieza
ID_COLUMNS = ["id", "station_id", "timestamp"]
NUMERIC_COLUMNS = [
    "temperature_c", "humidity_pct", "wind_speed_kmh", "wind_direction_deg",
    "pressure_hpa", "precipitation_mm", "visibility_km", "uv_index",
]
REQUIRED_COLUMNS = set(ID_COLUMNS + NUMERIC_COLUMNS)
EXPECTED_INTERVAL = pd.Timedelta("5min")

# Límites conservadores: valores fuera de ellos se tratan como medidas inválidas.
VALID_RANGES = {
    "temperature_c": (10, 45),
    "humidity_pct": (0, 100),
    "wind_speed_kmh": (0, 180),
    "wind_direction_deg": (0, 360),
    "pressure_hpa": (850, 1100),
    "precipitation_mm": (0, 150),
    "visibility_km": (0, 100),
    "uv_index": (0, 15),
}

# Un salto mayor que estos valores en cinco minutos se conserva y se marca.
MAX_CHANGE_PER_INTERVAL = {
    "temperature_c": 5, "humidity_pct": 30, "wind_speed_kmh": 60,
    "wind_direction_deg": 180, "pressure_hpa": 15, "precipitation_mm": 30,
    "visibility_km": 10, "uv_index": 5,
}

# Paso 1: estaciones
TARGET_LAT = 12.473972584225828
TARGET_LON = -69.89153669806423
EARTH_RADIUS_M = 6371000.0
N_NEAREST_STATIONS = 3

# Pasos 3 y 4: parque Vader Piet (Vestas V90-3.0 MW)
N_TURBINES = 10
TURBINE_RATED_POWER_KW = 3000.0
WIND_CAPACITY_KW = N_TURBINES * TURBINE_RATED_POWER_KW
CUT_IN_MS = 4.0
RATED_WIND_MS = 15.0
CUT_OUT_MS = 25.0

# Curva V90-3.0 MW digitalizada del folleto, a densidad estándar.
V90_WIND_MS = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
                        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0])
V90_POWER_KW = np.array([97, 160, 233, 317, 414, 516, 641, 780, 944, 1115, 1296, 1478,
                         1668, 1862, 2061, 2254, 2447, 2606, 2732, 2843, 2928, 2984, 3000])

LAGS = range(1, 5)


def step_0_clean_readings() -> pd.DataFrame:
    """Deduplica, invalida fuera de rango, imputa huecos aislados y marca anomalías."""
    print("\n=== Paso 0: limpieza de lecturas ===")
    raw_df = pd.read_json(READINGS_RAW_PATH)
    missing_columns = REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing_columns:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    print(f"Filas originales: {len(df):,}")
    print(f'Fechas inválidas: {df["timestamp"].isna().sum():,}')

    # Duplicados exactos: todos los campos iguales salvo `id`; se conserva el primero.
    df = df.dropna(subset=["station_id", "timestamp"]).sort_values(["station_id", "timestamp", "id"])
    duplicate_columns = [column for column in df.columns if column != "id"]
    exact_duplicate = df.duplicated(subset=duplicate_columns, keep="first")
    duplicates_removed = int(exact_duplicate.sum())
    df = df.loc[~exact_duplicate].copy()
    print(f"Duplicados exactos eliminados: {duplicates_removed:,}")
    print(f"Filas tras deduplicar: {len(df):,}")

    # Rangos imposibles -> NaN.
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    invalid_range = pd.DataFrame(False, index=df.index, columns=NUMERIC_COLUMNS)
    for column, (lower, upper) in VALID_RANGES.items():
        invalid_range[column] = ~df[column].between(lower, upper) & df[column].notna()
        df.loc[invalid_range[column], column] = np.nan

    df["interval_since_previous"] = df.groupby("station_id")["timestamp"].diff()
    df["gap_after_previous"] = df["interval_since_previous"].gt(EXPECTED_INTERVAL)
    df["irregular_interval"] = (
        df["interval_since_previous"].notna()
        & df["interval_since_previous"].ne(EXPECTED_INTERVAL)
    )

    # Imputación solo de ausencias aisladas entre dos lecturas válidas a 5 min.
    imputed_values = 0
    imputed_cell = pd.DataFrame(False, index=df.index, columns=NUMERIC_COLUMNS)
    next_timestamp = df.groupby("station_id")["timestamp"].shift(-1)
    for column in NUMERIC_COLUMNS:
        previous_value = df.groupby("station_id")[column].shift()
        next_value = df.groupby("station_id")[column].shift(-1)
        isolated_missing = (
            df[column].isna()
            & previous_value.notna()
            & next_value.notna()
            & df["interval_since_previous"].eq(EXPECTED_INTERVAL)
            & next_timestamp.sub(df["timestamp"]).eq(EXPECTED_INTERVAL)
        )
        df.loc[isolated_missing, column] = (
            previous_value.loc[isolated_missing] + next_value.loc[isolated_missing]
        ) / 2
        imputed_cell.loc[isolated_missing, column] = True
        imputed_values += int(isolated_missing.sum())

    # Anomalías auditables: se marcan, no se eliminan.
    for column, max_change in MAX_CHANGE_PER_INTERVAL.items():
        change = df.groupby("station_id")[column].diff().abs()
        df[f"anomalous_jump_{column}"] = (
            df["interval_since_previous"].eq(EXPECTED_INTERVAL) & change.gt(max_change)
        )
    for column in ["temperature_c", "humidity_pct", "pressure_hpa", "visibility_km"]:
        df[f"suspicious_zero_{column}"] = df[column].eq(0)

    anomaly_columns = [c for c in df if c.startswith(("anomalous_jump_", "suspicious_zero_"))]
    df["has_anomaly_flag"] = df[anomaly_columns].any(axis=1)
    df["was_imputed"] = imputed_cell.any(axis=1)
    unresolved_missing = df[NUMERIC_COLUMNS].isna().any(axis=1)
    rows_removed_incomplete = int(unresolved_missing.sum())
    clean_df = df.loc[~unresolved_missing].copy()

    print(f"Mediciones fuera de rango: {int(invalid_range.to_numpy().sum()):,}")
    print(f"Valores imputados: {imputed_values:,}")
    print(f"Filas eliminadas por valores incompletos: {rows_removed_incomplete:,}")
    print(f'Filas con una anomalía marcada: {clean_df["has_anomaly_flag"].sum():,}')

    os.makedirs(CLEANING_DIR, exist_ok=True)
    clean_df.to_csv(CLEAN_CSV_PATH, index=False)
    clean_df.to_json(CLEAN_JSON_PATH, orient="records", date_format="iso")

    report = {
        "input_rows": int(len(raw_df)),
        "rows_without_valid_key": int(len(raw_df) - len(df) - duplicates_removed),
        "exact_duplicates_removed": duplicates_removed,
        "invalid_measurements": int(invalid_range.to_numpy().sum()),
        "isolated_values_imputed": imputed_values,
        "rows_removed_with_unresolved_missing_values": rows_removed_incomplete,
        "network_gaps_detected": int(df["gap_after_previous"].sum()),
        "rows_with_anomaly_flag": int(clean_df["has_anomaly_flag"].sum()),
        "output_rows": int(len(clean_df)),
    }
    with open(CLEAN_REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(json.dumps(report, indent=2))
    return clean_df


def step_1_filter_stations() -> pd.DataFrame:
    """Calcula la distancia Haversine al parque y guarda las estaciones más cercanas."""
    print("\n=== Paso 1: selección de estaciones ===")
    stations = pd.read_json(STATIONS_RAW_PATH).copy()

    lat_origin, lon_origin, lat_dest, lon_dest = map(
        np.radians, [stations["latitude"], stations["longitude"], TARGET_LAT, TARGET_LON]
    )
    half_chord_sq = (
        np.sin((lat_dest - lat_origin) / 2.0) ** 2
        + np.cos(lat_origin) * np.cos(lat_dest) * np.sin((lon_dest - lon_origin) / 2.0) ** 2
    )
    stations["distance_m"] = EARTH_RADIUS_M * 2 * np.arcsin(np.sqrt(half_chord_sq))

    nearest = (
        stations.sort_values(by="distance_m", ascending=True)
        .reset_index(drop=True)
        .head(N_NEAREST_STATIONS)
    )
    os.makedirs(STATIONS_DIR, exist_ok=True)
    nearest.to_json(STATIONS_PATH)
    print(nearest[["id", "distance_m"]].to_string(index=False))
    return nearest


def step_2_filter_readings() -> pd.DataFrame:
    """Conserva solo las lecturas limpias de las estaciones seleccionadas."""
    print("\n=== Paso 2: filtrado de lecturas por estación ===")
    readings = pd.read_json(CLEAN_JSON_PATH)
    stations = pd.read_json(STATIONS_PATH)

    filtered = (
        readings[readings["station_id"].isin(stations["id"])]
        .sort_values(by="timestamp", ascending=True)
        .reset_index(drop=True)
    )
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    filtered.to_csv(FILTERED_CSV_PATH, index=False)
    filtered.to_json(FILTERED_JSON_PATH)
    print(f"Lecturas filtradas: {len(filtered):,}")
    return filtered


def step_3_security_thresholds() -> pd.DataFrame:
    """Añade la etiqueta `wind_alert` (NORMAL / MODERATE / CRITICAL)."""
    print("\n=== Paso 3: umbrales de seguridad eólica ===")
    df = pd.read_csv(FILTERED_CSV_PATH).copy()

    threshold_40_pct = WIND_CAPACITY_KW * 0.40
    threshold_15_pct = WIND_CAPACITY_KW * 0.15
    critical_wind_ms = np.interp(threshold_15_pct / N_TURBINES, V90_POWER_KW, V90_WIND_MS)
    moderate_wind_ms = np.interp(threshold_40_pct / N_TURBINES, V90_POWER_KW, V90_WIND_MS)
    wind_critical_kmh = critical_wind_ms * 3.6
    wind_moderate_kmh = moderate_wind_ms * 3.6
    wind_cut_out_kmh = CUT_OUT_MS * 3.6

    print(f"15% de capacidad: {threshold_15_pct:,.0f} kW -> {critical_wind_ms:.2f} m/s "
          f"-> {wind_critical_kmh:.1f} km/h")
    print(f"40% de capacidad: {threshold_40_pct:,.0f} kW -> {moderate_wind_ms:.2f} m/s "
          f"-> {wind_moderate_kmh:.1f} km/h")
    print(f"Cut-out V90: {CUT_OUT_MS:.0f} m/s -> {wind_cut_out_kmh:.1f} km/h")

    if "wind_power_kw" in df.columns:
        is_critical_deficit = df["wind_power_kw"] < threshold_15_pct
        is_moderate_deficit = df["wind_power_kw"] < threshold_40_pct
    else:
        is_critical_deficit = df["wind_speed_kmh"] < wind_critical_kmh
        is_moderate_deficit = df["wind_speed_kmh"] < wind_moderate_kmh

    if "wind_speed_kmh" in df.columns:
        is_cut_out = df["wind_speed_kmh"] >= wind_cut_out_kmh
    else:
        is_cut_out = pd.Series(False, index=df.index)

    # De menos a más grave: la última asignación gana.
    df["wind_alert"] = "NORMAL"
    df.loc[is_moderate_deficit, "wind_alert"] = "MODERATE"
    df.loc[is_critical_deficit, "wind_alert"] = "CRITICAL"
    df.loc[is_cut_out, "wind_alert"] = "CRITICAL"
    print(df["wind_alert"].value_counts().to_string())

    os.makedirs(THRESHOLDS_DIR, exist_ok=True)
    df.to_csv(THRESHOLDS_CSV_PATH, index=False)
    df.to_json(THRESHOLDS_JSON_PATH, orient="records", indent=2, force_ascii=False)
    return df


def step_4_eolic_lag_features() -> pd.DataFrame:
    """Estima la generación del parque y crea sus retardos sin cruzar huecos."""
    print("\n=== Paso 4: generación eólica y variables lag ===")
    required_columns = {"station_id", "timestamp", "wind_speed_kmh", "wind_alert"}

    df = pd.read_csv(THRESHOLDS_CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    nearby_stations = pd.read_json(STATIONS_PATH)
    if set(df["station_id"]) != set(nearby_stations["id"]):
        raise ValueError("Las lecturas no corresponden a las estaciones seleccionadas. "
                         "Ejecuta de nuevo los pasos 2 y 3.")

    # Viento en el parque: media ponderada por inverso de la distancia.
    weights = 1 / nearby_stations.set_index("id")["distance_m"]
    weather = df[df["station_id"].isin(weights.index)].copy()
    weather["weight"] = weather["station_id"].map(weights)
    wind_at_park = (
        weather.assign(weighted_wind=weather["wind_speed_kmh"] * weather["weight"])
        .groupby("timestamp")[["weighted_wind", "weight"]].sum()
        .eval("weighted_wind / weight")
    )

    # Curva simplificada Vestas V90-3.0 MW: 10 turbinas, buje a 80 m y 10 % de pérdidas.
    wind_hub_ms = wind_at_park.div(3.6) * (80 / 10) ** 0.14
    curve_ms = np.array([0, 4, 6, 8, 10, 12, 15, 25])
    curve_kw = np.array([0, 97, 414, 944, 1668, 2447, 3000, 3000])
    power_per_turbine = np.interp(wind_hub_ms, curve_ms, curve_kw)
    power_per_turbine[(wind_hub_ms < CUT_IN_MS) | (wind_hub_ms >= CUT_OUT_MS)] = 0
    generation = pd.Series(power_per_turbine * N_TURBINES * 0.90,
                           index=wind_at_park.index, name="generacion_eolica_kw")
    df = df.merge(generation, left_on="timestamp", right_index=True,
                  how="left", validate="many_to_one")

    deltas = df.sort_values(["station_id", "timestamp"]).groupby("station_id")["timestamp"].diff()
    print(f"Lecturas con un hueco previo: {int((deltas > pd.Timedelta('5min')).sum()):,}")

    df_with_lags = add_lag_features(
        df,
        target_col="generacion_eolica_kw",
        lags=LAGS,
        group_col="station_id",
        timestamp_col="timestamp",
        freq="5min",
    )
    lag_columns = [f"generacion_eolica_kw_lag_{lag}" for lag in LAGS]
    print(df_with_lags[lag_columns].isna().sum().rename("missing_values").to_string())

    model_df = df_with_lags.dropna(subset=lag_columns).copy()
    print(f"Filas originales: {len(df_with_lags):,}")
    print(f"Filas con todos los lags: {len(model_df):,}")

    os.makedirs(LAGS_DIR, exist_ok=True)
    model_df.to_csv(LAGS_CSV_PATH, index=False)
    model_df.to_json(LAGS_JSON_PATH, orient="records", indent=2,
                     force_ascii=False, date_format="iso")
    return model_df


def step_5_island_consumption() -> pd.DataFrame:
    """Añade `consumo_estimado_isla_kw`: consumo total de la isla estimado (modelado, no medido)."""
    print("\n=== Paso 5: consumo eléctrico estimado de la isla ===")
    df = pd.read_csv(LAGS_CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Clima medio de la isla con las 12 estaciones limpias, no solo las del parque.
    readings = pd.read_csv(CLEAN_CSV_PATH, usecols=["timestamp", *WEATHER_COLUMNS])
    readings["timestamp"] = pd.to_datetime(readings["timestamp"], utc=True)
    island_weather = readings.groupby("timestamp", as_index=False)[WEATHER_COLUMNS].mean()

    years = sorted(island_weather["timestamp"].dt.tz_convert("America/Aruba").dt.year.unique())
    monthly_mwh = project_monthly_consumption(CONSUMPTION_EXCEL_PATH, years)
    consumption = estimate_island_consumption(island_weather, monthly_mwh)

    df = df.merge(consumption, left_on="timestamp", right_index=True,
                  how="left", validate="many_to_one")
    print(f'Filas sin consumo: {int(df["consumo_estimado_isla_kw"].isna().sum()):,}')
    print(f"Consumo isla -> media {consumption.mean() / 1000:,.1f} MW | "
          f"mín {consumption.min() / 1000:,.1f} MW | máx {consumption.max() / 1000:,.1f} MW | "
          f"pico/media {consumption.max() / consumption.mean():.3f}")
    share = (df["generacion_eolica_kw"] / df["consumo_estimado_isla_kw"]).mean()
    print(f"Cobertura eólica media del parque: {share:.1%}")

    os.makedirs(CONSUMPTION_DIR, exist_ok=True)
    df.to_csv(CONSUMPTION_CSV_PATH, index=False)
    df.to_json(CONSUMPTION_JSON_PATH, orient="records", indent=2,
               force_ascii=False, date_format="iso")
    return df


def main() -> None:
    step_0_clean_readings()
    step_1_filter_stations()
    step_2_filter_readings()
    step_3_security_thresholds()
    step_4_eolic_lag_features()
    step_5_island_consumption()
    print(f"\nPipeline completado. Tabla para el modelo: {CONSUMPTION_CSV_PATH}")


if __name__ == "__main__":
    main()
