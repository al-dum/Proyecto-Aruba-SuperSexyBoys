"""Estimación del consumo eléctrico total de Aruba a resolución de 5 minutos.

La serie es MODELADA, no medida. Combina:
- Los totales mensuales del CBS/Elmar (Tabla 3.8, 2015-2024), proyectados por
  regresión lineal al año pedido: fijan la energía de cada mes.
- Un perfil diario en hora local (UTC-4), distinto en laborables y en fines de
  semana/festivos de Aruba.
- El clima medio de la isla: temperatura suavizada (inercia térmica de los
  edificios), humedad y viento.
- Ruido AR(1) con memoria, como los residuos de una carga real.
Por último se calibra la amplitud para que pico/media coincida con la publicada
por WEB Aruba (172/121 MW, 2026) sin alterar la energía de cada mes.

Basado en `extrapolar_consumo_centrales` de la rama
`features/igomez_consumo_lags_realistas`, simplificado a un único total insular.
"""
import holidays
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import lfilter

# Referencias de WEB Aruba (Power Production Figures, 2026): carga media 121 MW, pico 172 MW.
RATIO_PICO_MEDIA = 172.0 / 121.0
ZONA_HORARIA_ARUBA = "America/Aruba"  # UTC-4, sin horario de verano

# Perfil horario típico de isla caribeña dominada por aire acondicionado (hora LOCAL 0-23):
# valle de madrugada (4-5 h), meseta de tarde por el calor y pico nocturno (19-20 h).
PERFIL_LABORABLE = np.array([
    0.86, 0.82, 0.79, 0.77, 0.76, 0.77, 0.80, 0.86, 0.93, 0.99, 1.03, 1.06,
    1.08, 1.09, 1.10, 1.10, 1.09, 1.08, 1.10, 1.14, 1.13, 1.08, 1.01, 0.93,
])
# Fines de semana y festivos: arranque matinal más tardío y menos actividad comercial de día.
AJUSTE_FIN_SEMANA = np.array([
    1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.95, 0.92, 0.92, 0.93, 0.94, 0.94,
    0.94, 0.94, 0.94, 0.94, 0.95, 0.96, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00,
])
ENERGIA_DIA_LIBRE = 0.97  # un día libre consume ~3 % menos que uno laborable

# Sensibilidad al clima respecto a valores típicos de Aruba.
TEMP_REF_C, HUM_REF_PCT, WIND_REF_KMH = 28.5, 74.0, 22.0
COEF_TEMP, COEF_HUM, COEF_WIND = 0.030, 0.002, -0.003

# Bloques anuales de la hoja 3.8 (2024, 2023, 2022, 2021): tendencia post-pandemia.
EXCEL_BLOCK_ROWS = (0, 20, 40, 60)
EXCEL_TOTAL_COL = 8

WEATHER_COLUMNS = ["temperature_c", "humidity_pct", "wind_speed_kmh"]


def project_monthly_consumption(excel_path, years) -> dict:
    """Lee la hoja 3.8 del Excel del CBS y proyecta los MWh de cada mes de `years`."""
    sheet = pd.read_excel(excel_path, sheet_name=0, header=None)
    records = []
    for row in EXCEL_BLOCK_ROWS:
        year = int(str(sheet.iloc[row, 0]).rsplit(",", maxsplit=1)[-1].strip())
        for month in range(1, 13):
            value = pd.to_numeric(sheet.iloc[row + 2 + month, EXCEL_TOTAL_COL], errors="coerce")
            if pd.notna(value):
                records.append({"year": year, "month": month, "mwh": float(value)})
    history = pd.DataFrame(records)

    projection = {}
    for month, group in history.groupby("month"):
        fit = stats.linregress(group["year"], group["mwh"])
        for year in years:
            projection[(int(year), int(month))] = fit.slope * year + fit.intercept
    return projection


def _daily_curve(local_hour: np.ndarray, day_off: np.ndarray) -> np.ndarray:
    """Interpola el perfil horario y lo normaliza a media diaria 1.0 (conserva la energía)."""
    hours = np.arange(25)  # se cierra el ciclo (24 h == 0 h) para interpolar sin saltos
    workday = np.append(PERFIL_LABORABLE, PERFIL_LABORABLE[0])
    weekend_profile = PERFIL_LABORABLE * AJUSTE_FIN_SEMANA
    weekend = np.append(weekend_profile, weekend_profile[0])

    grid = np.linspace(0, 24, 288, endpoint=False)
    workday_curve = np.interp(local_hour, hours, workday) / np.interp(grid, hours, workday).mean()
    weekend_curve = (np.interp(local_hour, hours, weekend)
                     / np.interp(grid, hours, weekend).mean() * ENERGIA_DIA_LIBRE)
    return np.where(day_off, weekend_curve, workday_curve)


def _ar1_noise(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Serie AR(1) estacionaria e_t = phi * e_{t-1} + eps_t con desviación marginal `sigma`."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], eps)


def estimate_island_consumption(weather: pd.DataFrame, monthly_mwh: dict,
                                seed: int = 42) -> pd.Series:
    """Devuelve `consumo_estimado_isla_kw` indexado por timestamp (UTC).

    `weather` debe tener una fila por timestamp con `timestamp` (tz-aware) y las
    columnas de `WEATHER_COLUMNS`, típicamente la media de todas las estaciones.
    """
    weather = weather.sort_values("timestamp").reset_index(drop=True)
    local = weather["timestamp"].dt.tz_convert(ZONA_HORARIA_ARUBA)
    year, month = local.dt.year.to_numpy(), local.dt.month.to_numpy()

    missing = sorted({(y, m) for y, m in zip(year, month)} - set(monthly_mwh))
    if missing:
        raise KeyError(f"No hay proyección mensual para: {missing}")

    # 1. Potencia media de la isla en ese mes (MWh -> kW medios), en calendario local.
    mwh = np.array([monthly_mwh[(y, m)] for y, m in zip(year, month)])
    base_kw = mwh * 1000.0 / (local.dt.days_in_month.to_numpy() * 24.0)

    # 2. Perfil diario; los festivos de Aruba se tratan como domingo.
    festivos = holidays.country_holidays("AW", years=sorted(set(year.tolist())))
    day_off = (local.dt.dayofweek >= 5).to_numpy() | local.dt.date.isin(festivos).to_numpy()
    local_hour = (local.dt.hour + local.dt.minute / 60.0).to_numpy()
    curve = _daily_curve(local_hour, day_off)

    # 3. Clima: la temperatura se suaviza (vida media 2 h) porque los edificios tardan en
    #    calentarse y el aire acondicionado responde con retraso.
    smooth_temp = weather["temperature_c"].ewm(halflife="2h", times=weather["timestamp"]).mean()
    climate = np.clip(
        1.0
        + COEF_TEMP * (smooth_temp.to_numpy() - TEMP_REF_C)
        + COEF_HUM * (weather["humidity_pct"].to_numpy() - HUM_REF_PCT)
        + COEF_WIND * (weather["wind_speed_kmh"].to_numpy() - WIND_REF_KMH),
        0.75, 1.35,
    )

    # 4. Ruido AR(1) multiplicativo (phi=0.995 a 5 min, vida media ~2.3 h).
    noise = 1.0 + _ar1_noise(len(weather), 0.995, 0.03, np.random.default_rng(seed))
    consumption = pd.Series(base_kw * curve * climate * noise, index=weather["timestamp"])

    # 5. El nivel de cada mes lo fija el Excel: el clima y el ruido solo reparten la energía
    #    dentro del mes. Luego se estira la desviación respecto a esa media para que
    #    pico/media = RATIO_PICO_MEDIA, sin cambiar la media mensual.
    month_key = pd.Series(year * 100 + month, index=consumption.index)
    month_mean = pd.Series(base_kw, index=consumption.index)
    consumption = consumption / consumption.groupby(month_key).transform("mean") * month_mean
    deviation = consumption - month_mean
    target_peak = RATIO_PICO_MEDIA * consumption.mean()
    k = brentq(lambda k: (month_mean + k * deviation).max() - target_peak, 0.0, 20.0)
    calibrated = month_mean + k * deviation

    return calibrated.round(2).rename("consumo_estimado_isla_kw")
