# Por qué Gradient Boosting para el pronóstico eólico

Este documento justifica la elección de **Gradient Boosting**
(`HistGradientBoostingRegressor` de scikit-learn) como modelo principal para pronosticar la
generación del parque eólico Vader Piet. También explica cómo se eliminó la fuga de datos
(*data leakage*) y qué resultados da frente a los modelos de referencia.

- Código: `src/energy_dispatch/features/forecast.py`, `src/energy_dispatch/models/train.py`
- Entrenamiento: `python scripts/train_wind_model.py --horizons 1 12 72 288`
- Tests de no-fuga: `tests/test_forecast.py`

## 1. El problema

| | |
|---|---|
| **Qué se predice** | Velocidad del viento en el parque en *t + h*; la generación (kW) se obtiene con la curva de potencia V90-3.0 MW |
| **Granularidad** | 5 minutos |
| **Horizontes evaluados** | 5 min, 1 h, 6 h y 24 h (12, 72 y 288 pasos) |
| **Datos** | 3 estaciones cercanas al parque, 31-mar a 30-jun de 2026 (~26 000 instantes) |
| **Uso** | Alimentar el despacho (`dispatch.horizon_hours: 24`) |

Rasgos de la serie que condicionan el modelo:

- **Ciclo diario muy marcado**: el viento sube de noche (~27 km/h) y baja a mediodía
  (~21 km/h). La autocorrelación a 24 h es 0,79 y a 12 h es −0,73.
- **Ráfagas aisladas**: 61 instantes superan 35 km/h con una media de 24 km/h.
- **Relación viento → potencia muy no lineal**: cero por debajo de 4 m/s, crecimiento casi
  cúbico y saturación a 3 MW por turbina.
- **Solo 3 meses**: no hay estacionalidad anual que aprender.

## 2. Por qué Gradient Boosting

| Modelo | A favor | En contra para este proyecto |
|---|---|---|
| Persistencia / naive estacional | Imbatibles como referencia, coste cero | No aprenden nada; fallan cuando el ciclo diario cambia de tendencia |
| Media por hora (climatología) | Captura el ciclo diario | Ignora el estado actual del viento |
| Regresión lineal | Simple, interpretable, buena a muy corto plazo | No capta no linealidades ni interacciones (hora × nivel de viento) |
| Random Forest | No lineal, robusto | Más lento, no gestiona NaN tan bien y no tiene pérdida cuantílica nativa en scikit-learn |
| ARIMA / SARIMA | Clásico en series temporales | Una sola serie, mal con variables exógenas y con estacionalidad de 288 pasos |
| LSTM / redes neuronales | Potentes con mucha historia | 3 meses y 3 estaciones son pocos datos; difícil de validar y mantener |
| **Gradient Boosting** | **No lineal, combina lags + calendario + meteorología, rápido, maneja NaN** | Puede sobreajustar ráfagas si no se regulariza (ver §4) |

Motivos concretos:

1. **Mezcla de variables heterogéneas sin preparación**: retardos del viento, medias
   móviles, presión, dirección (seno/coseno) y hora del día. Los árboles encuentran
   interacciones como "a las 18 h con viento alto, el viento cae" sin que haya que
   escribirlas a mano.
2. **No linealidad**: el efecto de la hora y del nivel de viento no es aditivo, y la
   regresión lineal no puede representarlo.
3. **NaN nativos**: los retardos quedan vacíos tras un hueco en los datos (decisión
   explícita para no inventar valores). `HistGradientBoostingRegressor` los trata sin
   imputar.
4. **Pérdidas robustas y cuantílicas**: `absolute_error` es robusta a ráfagas, y
   `quantile` permite un pronóstico conservador para el despacho (sobrestimar el viento
   es el error caro).
5. **Rápido y sin dependencias nuevas**: entrena en segundos con ~21 000 filas y viene
   en scikit-learn. LightGBM o XGBoost darían resultados similares.

### Por qué se predice viento y no potencia

La generación es una función **conocida y determinista** del viento (la curva de
potencia). Si se predice la potencia directamente, el modelo tiene que reaprender esa
curva, con sus saturaciones, a partir de los datos. Es mejor predecir el viento, que es
lo realmente incierto, y aplicar la curva después (`wind_kmh_to_power_kw`).

## 3. Fuga de datos: qué había y cómo se corrigió

La tabla del paso 4 (`aruba_readings_with_eolic_lags.csv`) **no sirve tal cual para
entrenar**:

| # | Fuga | Por qué es un problema | Corrección |
|---|---|---|---|
| 1 | **Objetivo contemporáneo**: `generacion_eolica_kw(t)` en la misma fila que `wind_speed_kmh(t)` y `wind_alert(t)` | La generación *es* una función del viento en *t*: el modelo "acierta" al 100 % y en producción no sirve, porque el viento futuro no se conoce | El objetivo se desplaza al futuro con `shift(-h)`; todas las variables usan datos de *t* o anteriores |
| 2 | **Filas duplicadas por estación**: 3 filas por instante con el mismo objetivo | Con un split aleatorio el mismo instante aparece en train y en test | Una fila por instante; cada estación es una columna |
| 3 | **Split aleatorio** (`test_size: 0.2` sin orden) | Entrena con el futuro y evalúa con el pasado | Split temporal: test = el último 20 % de la serie |
| 4 | **Sin embargo entre train y test** | Las últimas *h* filas de train tienen como objetivo valores que están en test | Train solo incluye filas cuyo objetivo (*t + h*) es anterior al inicio de test |
| 5 | **Ventanas que miran al futuro** (medias centradas, interpolaciones) | Usan valores posteriores a *t* | Solo `shift(k ≥ 0)` y `rolling` hacia atrás |
| 6 | **Preprocesado ajustado con todos los datos** (imputación, escalado, media por hora) | Las estadísticas de test se filtran al modelo | Todo va dentro de un `Pipeline` que se ajusta solo con train, también en cada fold de la validación cruzada |
| 7 | **Retardos que cruzan huecos** | `shift(k)` deja de significar "hace k × 5 min" | Reindexado a una rejilla regular de 5 min |

Los tests de `tests/test_forecast.py` lo comprueban de forma automática. Por ejemplo:
si se modifican los datos posteriores a un instante *t*, las variables de las filas
≤ *t* **no cambian**.

Fuera del modelo quedan dos puntos a vigilar:

- La limpieza (paso 0) imputa huecos aislados con la media del valor anterior y el
  siguiente, lo que usa un dato futuro. Hoy no afecta (0 valores imputados), pero en
  tiempo real habría que imputar solo con el pasado.
- El consumo de la isla (`consumption.py`) es **sintético** y se calibra con la media de
  cada mes completo. Si se entrena un modelo de demanda con esa serie, el modelo solo
  aprenderá el generador sintético.

## 4. Resultados (test: 12 al 30 de junio de 2026)

MAE de generación en kW (menor es mejor). *Skill* = mejora sobre la persistencia.
Capacidad del parque: 27 000 kW.

| Modelo | 5 min | 1 h | 6 h | 24 h |
|---|---:|---:|---:|---:|
| Persistencia | 47 | 355 | 1 735 | 585 |
| Naive estacional (ayer misma hora) | 582 | 582 | 583 | 585 |
| Media por hora | 416 | 416 | 418 | **422** |
| Regresión lineal | **40** | 175 | 375 | 533 |
| **Gradient Boosting** | 47 | **162** | **358** | **422** |
| *Skill* de GB vs. persistencia | 0 % | 54 % | 79 % | 28 % |

Lectura honesta:

- **1 h y 6 h** (los horizontes útiles para el despacho intradiario): Gradient Boosting es
  el mejor modelo y supera a la regresión lineal y a todos los baselines.
- **5 min**: casi no hay nada que aprender. La persistencia y la regresión lineal bastan;
  un modelo complejo no aporta.
- **24 h**: Gradient Boosting empata con la media por hora. Con solo datos de estaciones,
  lo único predecible a un día vista es el ciclo diario. **Para mejorar a 24 h hacen
  falta pronósticos meteorológicos numéricos (NWP) como variables de entrada**; es lo que
  más valor añadiría al proyecto.
- **Hora punta de error**: todos los modelos con lags fallan más alrededor de las 18 h
  locales, cuando el viento cambia de tendencia. La media por hora acierta ahí. Conviene
  revisarlo con `reports/wind_model_mae_by_hour.csv`.

### Ajuste que hizo falta: ráfagas y sobreajuste

La primera versión (`loss: squared_error`, `max_depth: 6`) **perdía frente a la media por
hora a 6 h y 24 h**, con un sesgo de +250 kW. La pérdida cuadrática penaliza mucho los
errores grandes, así que los árboles profundos perseguían las ráfagas aisladas y
sobreajustaban. Con `loss: absolute_error` (predice la mediana, robusta a ráfagas),
`max_depth: 3` y `min_samples_leaf: 200`, el MAE a 6 h bajó de 545 a 358 kW y el sesgo
casi desapareció.

## 5. Siguientes pasos

1. **NWP como variable de entrada** para el horizonte de 24 h.
2. **Pérdida cuantílica** (`loss: quantile`, `quantile: 0.3`) para el despacho: se
   prefiere quedarse corto en viento que pasarse.
3. **Ajuste de hiperparámetros** con `TimeSeriesSplit` (ya integrado en el script como
   validación cruzada temporal con `gap = horizonte`).
4. **Un modelo por horizonte** o un único modelo con el horizonte como variable, para
   cubrir toda la ventana de 24 h del despacho.
5. Recalcular cuando haya **más meses de datos**: con un año completo cambia el balance
   entre modelos (estacionalidad anual).
