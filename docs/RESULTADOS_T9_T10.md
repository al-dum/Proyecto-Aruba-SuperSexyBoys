# T9 y T10: modelos a t+15 min (Gradient Boosting)

Entrenamiento, comparación, evaluación y exportación de los modelos de pronóstico a
**15 minutos vista** para las dos señales del EDA (`notebooks/5_eda_notebook.ipynb`):

- **Viento → generación eólica** del parque Vader Piet (kW).
- **Consumo eléctrico de la isla** (kW). Es una serie modelada, no medida (ver Limitaciones).

Se siguen los capítulos 9–13 de `docs/EntrenamientoML.pdf` y `docs/METODOLOGIA.md` §7–9.

Todas las cifras de este documento salen de `reports/t15/` (última ejecución de
`scripts/train_t15.py`, con la curva V90 unificada en `features/wind.py`).

## Cómo reproducirlo

```bash
python scripts/run_data_pipeline.py          # o, si ya existen los pasos 0-2, solo 3-5
python scripts/train_t15.py                  # viento y demanda (~4 min cada uno)
python scripts/train_t15.py --problems wind  # solo viento: no necesita el Excel del CBS
pytest tests/test_t15.py                     # comprobaciones de no-fuga
```

| Fichero | Qué hace |
|---|---|
| `src/energy_dispatch/features/t15.py` | Dataset sin fuga: una fila por instante, rejilla regular de 5 min, variables solo con pasado, objetivo en t+15, split temporal con purga |
| `src/energy_dispatch/models/gradient_boosting.py` | El modelo principal: `DeltaRegressor` (aprende el cambio a 15 min) y `make_gradient_boosting` |
| `src/energy_dispatch/models/baselines.py` | Referencias: persistencia, naive estacional, media de entrenamiento y media por hora |
| `src/energy_dispatch/models/t15.py` | Catálogo `build_models` (qué se compara), métricas, validación cruzada temporal y alertas |
| `src/energy_dispatch/predict.py` | Predicción con el modelo exportado, desde Python o desde la terminal (ver README) |
| `scripts/train_t15.py` | T9 + T10 de principio a fin; escribe `reports/t15/`, `reports/figures/t15_*` y `models/gb_t15_*.pkl` |
| `tests/test_t15.py` | El objetivo está exactamente a +15 min; no se cruzan huecos; la purga funciona; la media por hora solo aprende de train |

## Planteamiento

- **Cada fila es un instante de emisión t.** Variables: valor actual, retardos (5 min–2 h),
  incrementos, medias y desviaciones móviles hacia atrás (30 min, 1 h, 3 h), valor de ayer a
  la hora del objetivo, meteorología actual (media ponderada por distancia), viento de cada
  estación y calendario del instante objetivo (hora en seno/coseno, día de la semana,
  festivos de Aruba).
- **Viento:** se predice el viento del parque (km/h) y la generación se obtiene con la curva
  V90 (`wind_kmh_to_power_kw`), como en `docs/MODELO_GRADIENT_BOOSTING.md`. Las métricas se
  dan en kW de generación, que es lo que se entrega al despacho.
- **Se predice el cambio, no el nivel** (`DeltaRegressor`): a 15 min el valor actual explica
  casi todo y lo que hay que aprender es la variación. Además, los árboles no extrapolan: un
  árbol que predice niveles se queda clavado en el máximo de entrenamiento.
- **Split temporal 80/20** sobre 109 988 instantes:
  - Desarrollo: 1-sep-2025 a 6-jul-2026 (84 512 instantes).
  - Prueba: 6-jul-2026 a 17-sep-2026 (21 129 instantes).
  - Se purgan de desarrollo las filas cuyo objetivo cae dentro de prueba.
- **Validación (T9):** `TimeSeriesSplit` de 5 folds con `gap = 3` filas (15 min, el
  horizonte) sobre desarrollo. Imputación y escalado van dentro del pipeline.
  `early_stopping` de Gradient Boosting desactivado: su validación interna es aleatoria.

## T9: entrenamiento y comparación

### Ajuste de hiperparámetros

72 configuraciones (`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `loss`)
× 5 folds por señal. Todas en `reports/t15/<señal>_gb_experimentos.csv`.

| | Viento | Demanda |
|---|---|---|
| `learning_rate` | 0.03 | 0.03 |
| `max_iter` | 200 | 200 |
| `max_leaf_nodes` | 31 | 31 |
| `min_samples_leaf` | 50 | 50 |
| `loss` | squared_error | absolute_error |

En viento, `squared_error` y `absolute_error` quedan prácticamente empatados (MAE de
validación 65.6 frente a 65.6 kW). En demanda gana `absolute_error`. Los dos casos usan la
configuración que da el informe (`reports/t15/<señal>_resumen.json`).

### Comparación en validación cruzada (MAE en kW, media ± desviación de 5 folds)

| Modelo | Viento → generación | Demanda |
|---|---:|---:|
| **Gradient Boosting** | **65.6 ± 3.6** | **514 ± 30** |
| Ridge | 74.5 ± 3.2 | 613 ± 30 |
| Persistencia | 102.1 ± 4.1 | 1 051 ± 62 |
| Media por hora | 416.6 ± 39.4 | 8 598 ± 2 696 |
| Naive estacional (ayer) | 575.8 ± 48.8 | 4 349 ± 206 |
| Media de entrenamiento | 1 187.5 ± 33.1 | 13 503 ± 1 289 |

Detalle por fold en `reports/t15/<señal>_cv_por_fold.csv`.

## T10: evaluación en prueba y modelo final

La prueba se usa **una sola vez**, con los modelos reentrenados en todo desarrollo.

### Métricas en prueba (6-jul a 17-sep de 2026)

| Modelo | Viento: MAE kW | RMSE kW | R² | Demanda: MAE kW | RMSE kW | MAPE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **64.8** | **101.7** | **0.996** | **575.7** | **730.1** | **0.46 %** | **0.998** |
| Ridge | 75.1 | 144.2 | 0.991 | 666.0 | 847.1 | 0.54 % | 0.997 |
| Persistencia | 106.4 | 181.6 | 0.986 | 1 182.6 | 1 461.3 | 0.96 % | 0.991 |
| Media por hora | 424.0 | 792.8 | 0.735 | 11 720.1 | 12 915.5 | 9.27 % | 0.313 |
| Naive estacional | 600.0 | 1 095.7 | 0.494 | 4 372.7 | 5 646.9 | 3.53 % | 0.869 |
| Media de entrenamiento | 1 244.1 | 1 543.5 | −0.004 | 16 310.3 | 19 495.6 | 12.50 % | −0.565 |

No se da MAPE para la generación: vale 0 a menudo y el MAPE no tiene sentido.

### Por horizonte (MAE en kW, prueba)

| Horizonte | Viento: GB | Ridge | Persistencia | Demanda: GB | Ridge | Persistencia |
|---|---:|---:|---:|---:|---:|---:|
| 5 min | **38** | 40 | 47 | **329** | 349 | 477 |
| **15 min** | **65** | 75 | 106 | **576** | 666 | 1 183 |
| 30 min | **93** | 117 | 194 | **819** | 1 064 | 2 213 |
| 1 h | **136** | 183 | 365 | **1 188** | 1 796 | 4 223 |

Gradient Boosting gana en todos los horizontes y en las dos señales.

### Cerca de los umbrales de seguridad (T2)

**Viento** (alertas del paso 3 sobre el viento del parque a t+15: crítico < 22.2 km/h,
moderado < 31.4 km/h). Hay 6 152 instantes con alerta crítica real en la prueba:

| Modelo | Acierto de alerta | Críticas detectadas | Falsas alarmas críticas | Críticas no avisadas | MAE kW a ±2 km/h de un umbral |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **98.8 %** | 98.1 % | **136** | 114 | **56.6** |
| Ridge | 98.2 % | **98.3 %** | 267 | **102** | 72.8 |
| Persistencia | 97.6 % | 96.0 % | 245 | 248 | 104.5 |

**Demanda** (horas punta = 10 % de mayor consumo, 2 113 instantes; calor = temperatura ≥ 30.3 °C,
umbral moderado del documento de umbrales, 8 014 instantes):

| Modelo | MAE kW en punta | MAE kW con calor |
|---|---:|---:|
| **Gradient Boosting** | **682** | **590** |
| Ridge | 857 | 698 |
| Persistencia | 1 377 | 1 576 |

Error por hora local en `reports/t15/<señal>_test_error_por_hora.csv` y
`reports/figures/t15_<señal>_error_por_hora.png`:

- **Viento:** Ridge y persistencia fallan más a las 18 h (216 y 226 kW). Gradient Boosting
  tiene su peor hora a las 3 h (92 kW) y su mejor a las 12 h (47 kW).
- **Demanda:** Gradient Boosting falla más a las 20 h (691 kW) y menos a las 2 h (463 kW).
  Su error es mayor de 9 h a 22 h, cuando el consumo es alto.

### Elección del modelo final

- **Demanda: Gradient Boosting**, sin discusión. Gana en validación, en prueba, en todos los
  horizontes y en las horas punta y de calor.
- **Viento: Gradient Boosting.** Gana en validación (65.6 frente a 74.5 kW de Ridge, una
  diferencia mayor que la desviación entre folds) y en prueba (64.8 frente a 75.1 kW). Además
  da menos falsas alarmas críticas (136 frente a 267) y un error mucho menor cerca de los
  umbrales (57 frente a 73 kW). A cambio, deja sin avisar 114 críticas frente a 102 de Ridge
  (1.9 % frente a 1.7 % de las críticas reales). Si en la interfaz no se quiere perder ninguna,
  conviene un pronóstico conservador (`loss="quantile"`, cuantil 0.3) o un margen sobre el
  umbral.

### Exportación

`models/gb_t15_wind.pkl` y `models/gb_t15_demand.pkl` (joblib). Cada fichero es un
diccionario con el modelo y lo necesario para usarlo en `predict.py`:

```python
import joblib
bundle = joblib.load("models/gb_t15_demand.pkl")
bundle["model"].predict(X)   # X = build_dataset(instant_table(lecturas, estaciones), "demand")
bundle["features"], bundle["horizon"], bundle["history_needed"], bundle["test_metrics"]
bundle["versions"]           # Python 3.14.8, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4
```

- Para el viento, `predict` devuelve km/h: la generación es
  `wind_kmh_to_power_kw(pred)` y la alerta `wind_alert(pred)`.
- Se necesitan **24 h de lecturas previas** cada 5 min (retardos, medias móviles y "ayer a la
  misma hora").
- Se entrenó con las mismas versiones que el `.venv` del proyecto y se comprobó que el
  modelo recargado reproduce las mismas predicciones (PDF §13.3). Cargarlo con otra versión de
  scikit-learn puede fallar o avisar.
- Para cargarlo hay que poder importar `energy_dispatch` (`pip install -e .` o
  `PYTHONPATH=src`).

## Limitaciones

- **El consumo es sintético** (paso 5). El modelo aprende a reproducir el generador de
  `consumption.py`, no la demanda real de Aruba. Además, ese generador fija el nivel con la
  media de cada mes completo, así que la serie ya "conoce" su media mensual. Los números de
  demanda miden la calidad sobre esa estimación, no en producción.
- **Un año de datos, pero una sola prueba corta.** Las lecturas cubren del 1-sep-2025 al
  17-sep-2026 y tienen 36 huecos de red. La prueba cubre 10 semanas (julio a septiembre), así
  que no valida el comportamiento en otras estaciones del año.
- **A 15 min el viento casi no cambia** (la persistencia falla 0.18 km/h de media). La ventaja
  de cualquier modelo es pequeña en términos absolutos; la mejora de Gradient Boosting sobre la
  persistencia es del 39 % en kW a 15 min y crece hasta el 63 % a 1 h.
- **Sin pronóstico meteorológico numérico (NWP).** Todas las variables salen de las estaciones
  y del pasado, así que a horizontes más largos el modelo no puede anticipar cambios de tiempo
  que no se ven en las estaciones.
