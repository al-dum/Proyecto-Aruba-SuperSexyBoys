# Lógica modular del proyecto  para detección de  duplicados, aditoias de huecos e inyeción de consumos
# SOLO SON FUNCIONES SE LANZA DESDE SCRIPT
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)

# Cuota media del consumo de la isla cubierta por la eólica (14 %); sirve para calibrar las turbinas por parque
CUOTA_EOLICA = 0.14

# Curva de potencia de la turbina Vestas V90-3.0 MW (densidad del aire 1.225 kg/m3): viento (m/s) -> potencia (kW)
# Arranque (cut-in) a 4 m/s, potencia nominal 3000 kW desde 15 m/s y corte (cut-out) a 25 m/s
CURVA_POTENCIA_MS = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 25], dtype="float64")
CURVA_POTENCIA_KW = np.array(
    [77, 190, 353, 581, 886, 1273, 1710, 2145, 2544, 2837, 2965, 2995, 3000, 3000], dtype="float64"
)


def proyectar_historico_excel(excel_path: Path) -> dict:
    """Lee la hoja 3.8 del Excel histórico (2015-2024) y proyecta MWh mensuales para 2025 y 2026."""
    logging.info(f"Cargando registro histórico oficial desde: {excel_path.name}")
    xls = pd.ExcelFile(excel_path)
    df_full = pd.read_excel(excel_path, sheet_name=xls.sheet_names[0], header=None)

    all_data = []
    # Bloques de años post-pandemia (2024, 2023, 2022, 2021) para capturar la tendencia real
    for r in [0, 20, 40, 60]:
        year_str = str(df_full.iloc[r, 0]).split(",")[-1].strip()
        year = int(year_str)
        for m_idx in range(12):
            val = pd.to_numeric(df_full.iloc[r + 3 + m_idx, 8], errors="coerce")
            if pd.notna(val):
                all_data.append({"Year": year, "Month_Num": m_idx + 1, "Total_MWh": val})

    df_hist = pd.DataFrame(all_data)
    monthly_proj = {}
    for m in range(1, 13):
        m_data = df_hist[df_hist["Month_Num"] == m]
        slope, intercept, _, _, _ = stats.linregress(m_data["Year"], m_data["Total_MWh"])
        monthly_proj[(2025, m)] = slope * 2025 + intercept
        monthly_proj[(2026, m)] = slope * 2026 + intercept

    return monthly_proj


def cargar_y_unificar_origen(readings_path: Path, stations_path: Path) -> pd.DataFrame:
    """1. Lee los ficheros JSON de origen y unifica formatos, tipos de datos y zona horaria UTC."""
    logging.info(f"Leyendo mediciones desde: {readings_path.name}")
    with open(readings_path, "r", encoding="utf-8") as f:
        df_lecturas = pd.DataFrame(json.load(f))

    with open(stations_path, "r", encoding="utf-8") as f:
        df_stations = pd.DataFrame(json.load(f))

    # Unificar timestamp en UTC estricto (ISO 8601)
    df_lecturas["timestamp"] = pd.to_datetime(
        df_lecturas["timestamp"], utc=True, errors="coerce"
    )

    df = pd.merge(
        df_lecturas,
        df_stations,
        left_on="station_id",
        right_on="id",
        suffixes=("", "_station"),
    ).drop(columns=["id_station"], errors="ignore")

    cols_float = [
        "temperature_c", "humidity_pct", "wind_speed_kmh", "wind_direction_deg",
        "pressure_hpa", "precipitation_mm", "visibility_km",
        "latitude", "longitude", "elevation_m"
    ]
    for col in cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "uv_index" in df.columns:
        df["uv_index"] = (
            pd.to_numeric(df["uv_index"], errors="coerce")
            .fillna(0)
            .round()
            .astype("int64")
        )

    return df


def ordenar_y_limpiar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """2. Ordena cronológicamente y detecta/elimina duplicados o marcas de tiempo repetidas."""
    total_inicial = len(df)
    df = df.dropna(subset=["timestamp"])

    # Orden cronológico por central y marca de tiempo
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)

    duplicados = df.duplicated(subset=["station_id", "timestamp"], keep="first")
    num_dup = int(duplicados.sum())

    if num_dup > 0:
        logging.warning(f"Detectadas y eliminadas {num_dup} lecturas con timestamp repetido.")
        df = df[~duplicados].reset_index(drop=True)
    else:
        logging.info("Verificación completada: 0 marcas de tiempo duplicadas.")

    logging.info(f"Total registros válidos ordenados: {len(df):,} (de {total_inicial:,}).")
    return df


def auditar_frecuencia_y_huecos(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """3. Comprueba la frecuencia de muestreo (5min) y localiza huecos en la serie de cada central."""
    logging.info(f"Auditando frecuencia de muestreo ({freq}) en las {df['station_id'].nunique()} centrales...")
    delta_esperado = pd.Timedelta(freq)

    df["delta_t"] = df.groupby("station_id")["timestamp"].diff()
    huecos = df[df["delta_t"] > delta_esperado]

    if len(huecos) > 0:
        logging.warning(f"Se localizaron {len(huecos)} huecos temporales mayores a {freq}.")
        resumen = huecos.groupby("name")["delta_t"].agg(["count", "max"])
        logging.info(f"Resumen de huecos por central:\n{resumen}")
    else:
        logging.info(f"Frecuencia continua perfecta de {freq} en todas las centrales.")

    return df.drop(columns=["delta_t"])


def extrapolar_consumo_isla(df: pd.DataFrame, monthly_proj: dict) -> pd.DataFrame:
    """
    Extrapola el consumo eléctrico total de la isla (consumo_isla_kw) en cada marca de tiempo combinando:
    - La proyección mensual del Excel oficial (MWh -> kW medios).
    - La curva horaria de carga (hora UTC).
    - La media de las variables meteorológicas de las estaciones (temperatura, humedad, viento).
    Las estaciones son parques eólicos, así que el consumo es el mismo para todas en cada instante.
    """
    logging.info("Extrapolando consumo eléctrico de la isla a partir de histórico y clima...")
    isla = df.groupby("timestamp")[["temperature_c", "humidity_pct", "wind_speed_kmh"]].mean()
    ts = isla.index

    # 1. Potencia media total de la isla en ese mes (kW)
    island_mwh = np.array([monthly_proj.get((y, m), 85000.0) for y, m in zip(ts.year, ts.month)])
    island_avg_kw = (island_mwh * 1000.0) / (ts.days_in_month.to_numpy() * 24.0)

    # 2. Curva horaria normalizada (media diaria = 1.0 para conservar la energía mensual del Excel)
    h_grid = np.linspace(0, 24, 288, endpoint=False)
    raw_grid = 0.78 + 0.32 * np.sin((h_grid - 7) / 24 * 2 * np.pi) ** 2 + 0.12 * np.exp(-((h_grid - 14.5) ** 2) / 10)
    norm_factor = raw_grid.mean()

    horas = (ts.hour + ts.minute / 60.0).to_numpy()
    curva_diurna = (
        0.78
        + 0.32 * np.sin((horas - 7) / 24 * 2 * np.pi) ** 2
        + 0.12 * np.exp(-((horas - 14.5) ** 2) / 10)
    ) / norm_factor

    # 3. Modulación meteorológica con la media de las estaciones en cada instante
    temp = isla["temperature_c"].to_numpy()
    hum = isla["humidity_pct"].to_numpy()
    wind = isla["wind_speed_kmh"].to_numpy()

    factor_clima = (
        1.0
        + 0.035 * (temp - 28.5)
        + 0.002 * (hum - 74.0)
        - 0.003 * (wind - 22.0)
    )
    factor_clima = np.clip(factor_clima, 0.75, 1.35)

    # Pequeña fluctuación estocástica proporcional al consumo medio
    np.random.seed(42)
    ruido = np.random.normal(0, island_avg_kw * 0.008)

    # Consumo de la isla en kW, repetido en las filas de cada estación con la misma marca de tiempo
    consumo_isla = pd.Series(
        np.round((island_avg_kw * curva_diurna * factor_clima) + ruido, 2), index=ts
    )
    df["consumo_isla_kw"] = df["timestamp"].map(consumo_isla)

    logging.info(
        f"Consumo Isla (kW) -> Media: {consumo_isla.mean():,.2f} kW | "
        f"Mín: {consumo_isla.min():,.2f} kW | Máx: {consumo_isla.max():,.2f} kW"
    )
    return df


def potencia_turbina_kw(viento_ms) -> np.ndarray:
    """Interpola la curva de potencia de la turbina; fuera de [cut-in, cut-out] la potencia es 0 kW."""
    viento_ms = np.asarray(viento_ms, dtype="float64")
    potencia = np.interp(viento_ms, CURVA_POTENCIA_MS, CURVA_POTENCIA_KW)
    fuera_rango = (viento_ms < CURVA_POTENCIA_MS[0]) | (viento_ms > CURVA_POTENCIA_MS[-1])
    return np.where(fuera_rango, 0.0, potencia)


def calibrar_turbinas_por_parque(df: pd.DataFrame, cuota_eolica: float = CUOTA_EOLICA) -> int:
    """
    Calcula cuántas turbinas necesita cada parque para que la generación eólica total cubra
    de media `cuota_eolica` del consumo de la isla (mismo número de turbinas en todos los parques).
    """
    gen_una_turbina = potencia_turbina_kw(df["wind_speed_kmh"].to_numpy() / 3.6)
    gen_total_una_turbina = pd.Series(gen_una_turbina).groupby(df["timestamp"].to_numpy()).sum().mean()
    consumo_medio = df.drop_duplicates("timestamp")["consumo_isla_kw"].mean()

    turbinas = max(1, int(round(cuota_eolica * consumo_medio / gen_total_una_turbina)))
    logging.info(
        f"Calibración eólica ({cuota_eolica:.0%} del consumo): {turbinas} turbinas por parque "
        f"({turbinas * CURVA_POTENCIA_KW.max() / 1000:,.1f} MW por parque)"
    )
    return turbinas


def calcular_generacion_eolica(df: pd.DataFrame, turbinas_por_parque: int) -> pd.DataFrame:
    """
    Calcula la generación eólica (kW) de cada parque a partir de wind_speed_kmh y su número de turbinas,
    la generación eólica total de la isla en cada marca de tiempo y el déficit frente al consumo.
    """
    logging.info("Calculando generación eólica a partir de la curva de potencia de la turbina...")
    viento_ms = df["wind_speed_kmh"].to_numpy() / 3.6

    df["generacion_eolica_kw"] = np.round(potencia_turbina_kw(viento_ms) * turbinas_por_parque, 2)
    df["generacion_eolica_total"] = np.round(
        df.groupby("timestamp")["generacion_eolica_kw"].transform("sum"), 2
    )

    logging.info(
        f"Generación eólica por parque (kW) -> Media: {df['generacion_eolica_kw'].mean():,.2f} kW | "
        f"Mín: {df['generacion_eolica_kw'].min():,.2f} kW | Máx: {df['generacion_eolica_kw'].max():,.2f} kW"
    )
    logging.info(
        f"Generación eólica total -> Media: {df['generacion_eolica_total'].mean():,.2f} kW"
    )

    # Déficit (RF-03) = Consumo de la isla - Generación eólica disponible
    if "consumo_isla_kw" in df.columns:
        df["deficit_kw"] = np.round(df["consumo_isla_kw"] - df["generacion_eolica_total"], 2)
    return df
def ejecutar_pipeline_ingesta(project_root: Path):
  """Orquesta todo el pipeline de ingesta y guarda los ficheros en data/interim/."""
  # Normalizar la ruta raíz para evitar problemas de barras en Windows/Linux
  project_root = Path(project_root).resolve()

  posibles_lecturas = [
      project_root / "data" / "raw" / "aruba_readings_20260924.json",
      project_root / "aruba_lecturas" / "aruba_readings_20260924.json",
      project_root / "data" / "raw" / "aruba_lecturas.json",
      project_root / "aruba_lecturas" / "aruba_lecturas.json",
  ]
  readings_path = next(
      (p for p in posibles_lecturas if p.exists()), posibles_lecturas[0]
  )

  posibles_estaciones = [
      project_root / "data" / "raw" / "aruba_stations.json",
      project_root / "aruba_lecturas" / "aruba_stations.json",
      project_root / "aruba_stations.json",
  ]
  stations_path = next(
      (p for p in posibles_estaciones if p.exists()), posibles_estaciones[0]
  )

  posibles_excel = [
      project_root
      / "data"
      / "external"
      / "Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev_2.xlsx",
      project_root
      / "data"
      / "external"
      / "Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev.xlsx",
      project_root
      / "data"
      / "raw"
      / "Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev_2.xlsx",
      project_root
      / "Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev_2.xlsx",
  ]
  excel_path = next((p for p in posibles_excel if p.exists()), posibles_excel[0])

  # 1. Procesar y extrapolar los datos
  monthly_proj = proyectar_historico_excel(excel_path)
  df = cargar_y_unificar_origen(readings_path, stations_path)
  df = ordenar_y_limpiar_duplicados(df)
  df = auditar_frecuencia_y_huecos(df, freq="5min")
  df = extrapolar_consumo_isla(df, monthly_proj)
  df = calcular_generacion_eolica(df, calibrar_turbinas_por_parque(df))

  # 2. Crear la carpeta data/interim con ruta absoluta normalizada
  interim_dir = (project_root / "data" / "interim").resolve()
  interim_dir.mkdir(parents=True, exist_ok=True)

  # 3. Definir rutas y convertirlas a string estricto para Pandas
  csv_out = interim_dir / "aruba_base_limpio.csv"
  json_out = interim_dir / "aruba_readings_con_consumo.json"

  df.to_csv(str(csv_out), index=False)

  cols_json = [
      "id",
      "station_id",
      "timestamp",
      "temperature_c",
      "humidity_pct",
      "wind_speed_kmh",
      "wind_direction_deg",
      "pressure_hpa",
      "precipitation_mm",
      "visibility_km",
      "uv_index",
      "consumo_isla_kw",
      "generacion_eolica_kw",
      "generacion_eolica_total",
      "deficit_kw",
  ]
  df[cols_json].to_json(str(json_out), orient="records", date_format="iso", indent=2)

  logging.info(f"Dataset base CSV guardado en: {csv_out}")
  logging.info(f"JSON enriquecido guardado en: {json_out}")