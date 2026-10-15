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

# Cuota de consumo por central/estación basada en los grupos tarifarios del Excel (Suma = 1.0)
STATION_SHARES = {
    "d8a1ce0a-873e-436f-9c1d-a2179ee2601c": 0.160,  # Palm Beach (Zona Hotelera / Gran Comercial)
    "c3a0e9a3-36fe-4c50-8c3e-e100045b4a63": 0.145,  # Oranjestad Central (Capital Comercial/Residencial)
    "09ea9cd3-4d28-4049-9dfb-3c5bd5849e9c": 0.115,  # Noord (Residencial y Comercial)
    "de2ee07b-de63-49db-a669-64177a688bbf": 0.105,  # Queen Beatrix Airport (Aeropuerto y Servicios)
    "963fdb57-b257-472d-b56e-892b265c2524": 0.095,  # San Nicolaas (Urbano e Industrial Sur)
    "8f0185fa-a4c3-41c1-8f3b-d08f4f6aece0": 0.080,  # Malmok (Turismo Costero / Residencial)
    "c2d4a677-28eb-494b-ad99-d4dfc110e474": 0.075,  # Santa Cruz (Residencial Central)
    "5c5c4158-91cb-4073-ab4c-9bb10037dc16": 0.065,  # Paradera (Residencial)
    "82a31d3a-9895-42e5-8ea0-11e8cb15b964": 0.060,  # Savaneta (Residencial Costero)
    "4ca4c103-894d-4f59-a911-f3abb8077e3c": 0.045,  # Alto Vista (Norte Residencial)
    "f678bd98-07d3-459d-bd4d-7aaba7051847": 0.030,  # Hooiberg Summit (Baja densidad / Telecom)
    "acb2fa5c-518a-4a38-b68d-bf8e3b041ccf": 0.025,  # Arikok National Park (Parque / Rural)
}


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


def extrapolar_consumo_centrales(df: pd.DataFrame, monthly_proj: dict) -> pd.DataFrame:
    """
    Extrapola el consumo eléctrico (consumo_kw) de cada central combinando:
    - La proyección mensual del Excel oficial (MWh -> kW medios).
    - La cuota tarifaria de cada estación (STATION_SHARES).
    - Las variables meteorológicas reales del JSON (temperatura, humedad, viento) y hora UTC.
    """
    logging.info("Extrapolando consumo eléctrico por central a partir de histórico y clima...")
    year = df["timestamp"].dt.year
    month = df["timestamp"].dt.month
    days_in_month = df["timestamp"].dt.days_in_month

    # 1. Potencia media total de la isla en ese mes (kW)
    island_mwh = np.array([monthly_proj.get((y, m), 85000.0) for y, m in zip(year, month)])
    island_avg_kw = (island_mwh * 1000.0) / (days_in_month * 24.0)

    # 2. Asignar cuota base a cada central
    share_estacion = df["station_id"].map(STATION_SHARES).fillna(1.0 / 12.0).to_numpy()
    base_station_kw = island_avg_kw * share_estacion

    # 3. Curva horaria normalizada (media diaria = 1.0 para conservar la energía mensual del Excel)
    h_grid = np.linspace(0, 24, 288, endpoint=False)
    raw_grid = 0.78 + 0.32 * np.sin((h_grid - 7) / 24 * 2 * np.pi) ** 2 + 0.12 * np.exp(-((h_grid - 14.5) ** 2) / 10)
    norm_factor = raw_grid.mean()

    horas = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    curva_diurna = (
        0.78
        + 0.32 * np.sin((horas - 7) / 24 * 2 * np.pi) ** 2
        + 0.12 * np.exp(-((horas - 14.5) ** 2) / 10)
    ) / norm_factor

    # 4. Modulación meteorológica a partir de las mediciones reales del JSON
    temp = df["temperature_c"].to_numpy()
    hum = df["humidity_pct"].to_numpy()
    wind = df["wind_speed_kmh"].to_numpy()

    factor_clima = (
        1.0
        + 0.035 * (temp - 28.5)
        + 0.002 * (hum - 74.0)
        - 0.003 * (wind - 22.0)
    )
    factor_clima = np.clip(factor_clima, 0.75, 1.35)

    # Pequeña fluctuación estocástica proporcional al tamaño de la central
    np.random.seed(42)
    ruido = np.random.normal(0, base_station_kw * 0.008)

    # Consumo final en kW de la central y consumo total insular sincronizado
    df["consumo_kw"] = np.round((base_station_kw * curva_diurna * factor_clima) + ruido, 2)
    df["consumo_isla_total_kw"] = np.round(
        df.groupby("timestamp")["consumo_kw"].transform("sum"), 2
    )

    logging.info(
        f"Consumo por central (kW) -> Media: {df['consumo_kw'].mean():,.2f} kW | "
        f"Mín: {df['consumo_kw'].min():,.2f} kW | Máx: {df['consumo_kw'].max():,.2f} kW"
    )
    logging.info(
        f"Consumo Total Isla (suma 12 centrales) -> Media: {df['consumo_isla_total_kw'].mean():,.2f} kW"
    )
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
  df = extrapolar_consumo_centrales(df, monthly_proj)

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
      "consumo_kw",
  ]
  df[cols_json].to_json(str(json_out), orient="records", date_format="iso", indent=2)

  logging.info(f"Dataset base CSV guardado en: {csv_out}")
  logging.info(f"JSON enriquecido guardado en: {json_out}")