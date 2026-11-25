# Sistema de Predicción y Despacho Energético mediante Machine Learning

Predice demanda y generación (p. ej. renovable) con modelos de ML y usa las
predicciones para optimizar el despacho de energía.

## Estructura

```
configs/                 Parámetros (YAML)
data/{raw,interim,processed,external}
notebooks/               Exploración y prototipos
models/                  Modelos entrenados (.joblib)
reports/figures/         Gráficas y resultados
scripts/                 Utilidades de línea de comandos
src/energy_dispatch/
  data/                  Carga y limpieza
  features/              Ingeniería de variables
  models/                Entrenamiento y predicción
  dispatch/              Optimización del despacho
  evaluation/            Métricas (MAE, RMSE, MAPE, coste)
  utils/                 Config, logging
tests/
main.py                  Punto de entrada (pipeline)
app/                     Interfaz Streamlit (streamlit run app/streamlit_app.py)
```

## Uso (*LINUX*/*MAC*)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python main.py --config configs/config.yaml
pytest
```
## Uso (*WINDOWS*)

```bash
python -m venv .venv.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python main.py --config configs/config.yaml
pytest
```

## Pipeline de datos

`scripts/run_data_pipeline.py` ejecuta en orden los pasos prototipados en los
notebooks `0` a `4` y añade el consumo estimado de la isla:

| Paso | Notebook | Salida en `data/processed/` |
|---|---|---|
| 0. Limpieza y auditoría de lecturas | `0_data_cleaning.ipynb` | `0_cleaning/` |
| 1. Selección de las 3 estaciones más cercanas | `1_filter_stations.ipynb` | `1_station_filtering/` |
| 2. Filtrado de lecturas por estación | `2_data_validation.ipynb` | `2_data_validation/` |
| 3. Umbrales de seguridad eólica (`wind_alert`) | `3_calculate_security_umbrals.ipynb` | `3_security_thresholds/` |
| 4. Generación eólica y variables lag | `4_eolic_generation_lag_features.ipynb` | `4_eolic_lag_features/` |
| 5. Consumo eléctrico estimado de la isla (`consumo_estimado_isla_kw`) | `src/energy_dispatch/data/consumption.py` | `5_island_consumption/` |

Requiere `data/raw/aruba_stations.json`,
`data/raw/readings/aruba_readings_20260924.json` y, para el paso 5, el Excel del
CBS `data/external/aruba_energy_consumption_2015_2024.xlsx` (tabla original del CBS:
`Table-3.8-Energy-consumption-per-month-in-KwH-by-tariff-group-2015-2024rev.xlsx`). Con el entorno activado, se
puede lanzar desde cualquier directorio:

```bash
python scripts/run_data_pipeline.py
```

Las carpetas de salida se crean si no existen y los ficheros previos se
sobrescriben. La tabla final para el modelo queda en
`data/processed/5_island_consumption/aruba_readings_with_consumption.csv`.

`consumo_estimado_isla_kw` es el consumo total de Aruba **modelado, no medido**: un valor
por timestamp (igual en las filas de las tres estaciones). Parte de los totales
mensuales del CBS proyectados al año de la lectura, y les aplica un perfil diario
en hora local (laborables y festivos), el clima medio de las 12 estaciones y
ruido AR(1). Después se calibra a la relación pico/media de WEB Aruba
(172/121 MW).

## Metodología

Ver [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md): flujo de trabajo desde datos
crudos (auditoría, limpieza, validación) hasta modelo en producción conectado al
despacho.
