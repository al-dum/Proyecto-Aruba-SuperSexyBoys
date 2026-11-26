# T9 y T10: modelos a t+15 min (Gradient Boosting)

Entrenamiento, comparación, evaluación y exportación de los modelos de pronóstico a
**15 minutos vista** para las dos señales del EDA (`notebooks/5_eda_notebook.ipynb`):

- **Viento → generación eólica** del parque Vader Piet (kW).
- **Consumo eléctrico de la isla** (kW).

Se siguen los capítulos 9–13 de `docs/EntrenamientoML.pdf` y `docs/METODOLOGIA.md` §7–9.

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
| `src/energy_dispatch/models/t15.py` | Baselines, Ridge, Gradient Boosting, métricas, validación cruzada temporal, alertas |
| `scripts/train_t15.py` | T9 + T10 de principio a fin; escribe `reports/t15/`, `reports/figures/t15_*` y `models/gb_t15_*.pkl` |
| `tests/test_t15.py` | Si se cambian los datos posteriores a t, las variables de las filas ≤ t no cambian; el objetivo está exactamente a +15 min; no se cruzan huecos; la purga funciona |

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
- **Split temporal 80/20:** desarrollo del 31-mar al 12-jun de 2026 (20 957 instantes) y
  prueba del 12-jun al 30-jun (5 241). Se purgan de desarrollo las filas cuyo objetivo cae
  dentro de prueba.
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
| `max_iter` | 500 | 200 |
| `max_leaf_nodes` | 7 | 15 |
| `min_samples_leaf` | 50 | 200 |
| `loss` | absolute_error | absolute_error |

En las dos señales gana `absolute_error` con árboles pequeños. Es lo mismo que se vio a 1 h:
a corto plazo el cambio de la señal es casi todo ruido y un modelo flexible con pérdida
cuadrática lo persigue (peor configuración: 96 kW en viento y 716 kW en demanda).

### Comparación en validación cruzada (MAE en kW, media ± desviación de 5 folds)

| Modelo | Viento → generación | Demanda |
|---|---:|---:|
| **Gradient Boosting** | **79 ± 5** | **638 ± 13** |
| Ridge | 92 ± 16 | 741 ± 28 |
| Persistencia | 118 ± 6 | 1 280 ± 49 |
| Media por hora | 502 ± 62 | 7 269 ± 2 211 |
| Naive estacional (ayer) | 684 ± 118 | 5 796 ± 991 |
| Media de entrenamiento | 1 347 ± 34 | 15 425 ± 1 399 |

Detalle por fold en `reports/t15/<señal>_cv_por_fold.csv`.

## T10: evaluación en prueba y modelo final

La prueba se usa **una sola vez**, con los modelos reentrenados en todo desarrollo.

### Métricas en prueba (12–30 de junio)

| Modelo | Viento: MAE kW | RMSE kW | R² | Demanda: MAE kW | RMSE kW | MAPE | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | 79.0 | 129.8 | 0.992 | **625** | **788** | **0.52 %** | 0.998 |
| Ridge | **76.6** | **128.8** | 0.992 | 717 | 909 | 0.59 % | 0.997 |
| Persistencia | 104.3 | 157.2 | 0.988 | 1 299 | 1 622 | 1.08 % | 0.991 |

No se da MAPE para la generación: vale 0 a menudo y el MAPE no tiene sentido.

### Por horizonte (MAE en kW, prueba)

| Horizonte | Viento: GB | Ridge | Persistencia | Demanda: GB | Ridge | Persistencia |
|---|---:|---:|---:|---:|---:|---:|
| 5 min | 43 | 41 | 47 | 357 | 377 | 523 |
| **15 min** | **79** | 77 | 104 | **625** | 717 | 1 299 |
| 30 min | 124 | 122 | 190 | 902 | 1 142 | 2 441 |
| 1 h | 151 | 195 | 356 | 1 295 | 1 935 | 4 697 |

### Cerca de los umbrales de seguridad (T2)

**Viento** (alertas del paso 3 sobre el viento del parque a t+15: crítico < 22.2 km/h,
moderado < 31.4 km/h):

| Modelo | Acierto de alerta | Críticas detectadas | Falsas alarmas críticas | Críticas no avisadas | MAE kW a ±2 km/h de un umbral |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | **98.0 %** | 96.8 % | **60** | 47 | **83** |
| Ridge | 97.7 % | **98.3 %** | 93 | **25** | 85 |
| Persistencia | 97.5 % | 95.7 % | 66 | 64 | 112 |

**Demanda** (horas punta = 10 % de mayor consumo; calor = temperatura ≥ 30.3 °C, umbral
moderado del documento de umbrales):

| Modelo | MAE kW en punta | MAE kW con calor |
|---|---:|---:|
| **Gradient Boosting** | **760** | **647** |
| Ridge | 982 | 771 |
| Persistencia | 1 611 | 1 684 |

Error por hora local en `reports/t15/<señal>_test_error_por_hora.csv` y
`reports/figures/t15_<señal>_error_por_hora.png`. En viento todos los modelos fallan más a
las 18 h (el cambio diario brusco ya visto a 1 h); Gradient Boosting es el que menos.

### Elección del modelo final

- **Demanda: Gradient Boosting**, sin discusión. Es el mejor en validación y en prueba, reduce
  el error de la persistencia a la mitad y es el mejor justo cuando más importa al despacho
  (puntas y calor).
- **Viento: Gradient Boosting**, con matices. Se elige por validación cruzada (79 frente a
  92 kW de Ridge y con la mitad de variabilidad entre folds). En prueba Ridge queda 2.4 kW por
  delante, una diferencia menor que la desviación entre folds: a 15 min están empatados y los
  dos mejoran la persistencia un 24–27 %. La balanza se inclina hacia Gradient Boosting porque
  es mucho mejor desde 1 h (151 frente a 195 kW), da menos falsas alarmas críticas y falla menos
  en la hora punta de error (18 h). A cambio, deja sin avisar más alertas críticas que Ridge
  (47 frente a 25): si en la interfaz se prefiere no perder ninguna, conviene un pronóstico
  conservador (`loss="quantile"`, cuantil 0.3) o un margen sobre el umbral.

### Exportación

`models/gb_t15_wind.pkl` y `models/gb_t15_demand.pkl` (joblib). Cada fichero es un
diccionario con el modelo y lo necesario para usarlo en `predict.py` (T12):

```python
import joblib
bundle = joblib.load("models/gb_t15_demand.pkl")
bundle["model"].predict(X)   # X = build_dataset(instant_table(lecturas, estaciones), "demand")
bundle["features"], bundle["horizon"], bundle["history_needed"], bundle["test_metrics"]
bundle["versions"]           # Python 3.14, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4
```

- Para el viento, `predict` devuelve km/h: la generación es
  `wind_kmh_to_power_kw(pred)` y la alerta `wind_alert(pred)`.
- Se necesitan **24 h de lecturas previas** cada 5 min (retardos, medias móviles y "ayer a la
  misma hora").
- Se entrenó con las mismas versiones que el `.venv` del proyecto y se comprobó que el
  modelo recargado reproduce las mismas predicciones (PDF §13.3). Cargarlo con otra versión de
  scikit-learn puede fallar o avisar.
- Para cargarlo hay que poder importar `energy_dispatch` (`pip install -e .`).

## Limitaciones

- **El consumo es sintético** (paso 5). El modelo aprende a reproducir el generador de
  `consumption.py`, no la demanda real de Aruba. Además, ese generador fija el nivel con la
  media de cada mes completo, así que la serie ya "conoce" su media mensual. Los números de
  demanda miden la calidad sobre esa estimación, no en producción.
- **Solo tres meses** (abril–junio): no hay estacionalidad anual. La prueba cubre 19 días de
  junio.
- **A 15 min el viento casi no cambia** (la persistencia falla 0.17 km/h de media). La ventaja
  de cualquier modelo es pequeña en términos absolutos; donde Gradient Boosting aporta de
  verdad es a partir de 30 min–1 h.
