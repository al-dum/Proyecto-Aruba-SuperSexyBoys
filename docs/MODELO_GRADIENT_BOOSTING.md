# Por qué Gradient Boosting para el pronóstico eólico

Este documento justifica la elección de **Gradient Boosting**
(`HistGradientBoostingRegressor` de scikit-learn) como modelo principal para pronosticar la
generación del parque eólico Vader Piet a 15 minutos. También explica cómo se eliminó la fuga
de datos (*data leakage*) y qué resultados da frente a los modelos de referencia.

- Código actual (t+15 min): `src/energy_dispatch/features/t15.py` (variables),
  `src/energy_dispatch/models/gradient_boosting.py` (el modelo),
  `src/energy_dispatch/models/baselines.py` (referencias) y `src/energy_dispatch/models/t15.py`
  (catálogo `build_models`, métricas y validación)
- Entrenamiento: `python scripts/train_t15.py --problems wind`
- Tests de no-fuga: `tests/test_t15.py`

> **Nota:** las cifras de este documento (§4) proceden de `reports/t15/`, la última ejecución de
> `scripts/train_t15.py`. Las versiones anteriores de este documento, que usaban horizontes de
> 6 h y 24 h y un pipeline ya eliminado, han quedado obsoletas. El detalle completo de
> resultados está en `docs/RESULTADOS_T9_T10.md`.

## 1. El problema

| | |
|---|---|
| **Qué se predice** | Velocidad del viento en el parque en *t + 15 min*; la generación (kW) se obtiene con la curva de potencia V90-3.0 MW |
| **Granularidad** | 5 minutos |
| **Horizontes evaluados** | 5 min, 15 min, 30 min y 1 h |
| **Datos** | 3 estaciones cercanas al parque, del 1-sep-2025 al 17-sep-2026 (109 988 instantes) |
| **Uso** | Pronóstico a 15 min para `predict.py` y para la interfaz (T13) |

Rasgos de la serie que condicionan el modelo:

- **Relación viento → potencia muy no lineal**: cero por debajo de 4 m/s (*cut-in*), crecimiento
  casi cúbico y saturación a 3 MW por turbina.
- **Un año de datos**: hay un solo ciclo anual, así que no se puede validar la estacionalidad
  anual con los datos disponibles.
- **Huecos de red**: 36 huecos en las lecturas. El pipeline no los rellena.

## 2. Por qué Gradient Boosting

| Modelo | A favor | En contra para este proyecto |
|---|---|---|
| Persistencia / naive estacional | Imbatibles como referencia, coste cero | No aprenden nada; fallan cuando el viento cambia de tendencia |
| Media por hora (climatología) | Captura el ciclo diario | Ignora el estado actual del viento |
| Regresión lineal (Ridge) | Simple, interpretable, buena a muy corto plazo | No capta no linealidades ni interacciones (hora × nivel de viento) |
| Random Forest | No lineal, robusto | Más lento, no gestiona NaN tan bien y no tiene pérdida cuantílica nativa en scikit-learn |
| ARIMA / SARIMA | Clásico en series temporales | Una sola serie, mal con variables exógenas y con estacionalidad de 288 pasos |
| LSTM / redes neuronales | Potentes con mucha historia | Un año y tres estaciones son pocos datos; difícil de validar y mantener |
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
4. **Pérdidas robustas**: `absolute_error` es robusta a ráfagas aisladas; el ajuste de §4
   compara las dos pérdidas.
5. **Rápido y sin dependencias nuevas**: cada configuración se evalúa en unos 4 s con 5 folds
   sobre ~85 000 filas de desarrollo, y viene en scikit-learn. LightGBM o XGBoost darían
   resultados similares.

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
| 3 | **Split aleatorio** (`test_size: 0.2` sin orden) | Entrena con el futuro y evalúa con el pasado | Split temporal: prueba = el último 20 % de la serie |
| 4 | **Sin embargo entre train y test** | Las últimas *h* filas de train tienen como objetivo valores que están en test | Train solo incluye filas cuyo objetivo (*t + h*) es anterior al inicio de test |
| 5 | **Ventanas que miran al futuro** (medias centradas, interpolaciones) | Usan valores posteriores a *t* | Solo `shift(k ≥ 0)` y `rolling` hacia atrás |
| 6 | **Preprocesado ajustado con todos los datos** (imputación, escalado, media por hora) | Las estadísticas de test se filtran al modelo | Todo va dentro de un `Pipeline` o del baseline, que se ajustan solo con train, también en cada fold de la validación cruzada |
| 7 | **Retardos que cruzan huecos** | `shift(k)` deja de significar "hace k × 5 min" | Reindexado a una rejilla regular de 5 min |

Los tests de `tests/test_t15.py` cubren parte de estas garantías: el objetivo está
exactamente a 15 min, los huecos no se puentean, la partición purga los objetivos que
caen en test y la media por hora solo aprende de train.

Fuera del modelo quedan dos puntos a vigilar:

- La limpieza (paso 0) imputa huecos aislados con la media del valor anterior y el
  siguiente, lo que usa un dato futuro. En la última ejecución no se imputó ningún valor
  (`data_cleaning_report.json`: `isolated_values_imputed: 0`), pero en tiempo real habría que
  imputar solo con el pasado.
- El consumo de la isla (`consumption.py`) es **sintético** y se calibra con la media de
  cada mes completo. Si se entrena un modelo de demanda con esa serie, el modelo solo
  aprenderá el generador sintético.

## 4. Resultados (prueba: 6-jul a 17-sep de 2026)

Cifras de `reports/t15/`. Detalle completo en `docs/RESULTADOS_T9_T10.md`.

MAE de generación en kW a 15 min (menor es mejor). *Mejora* = frente a la persistencia.

| Modelo | MAE kW | RMSE kW | R² | Mejora |
|---|---:|---:|---:|---:|
| Persistencia | 106.4 | 181.6 | 0.986 | — |
| Ridge | 75.1 | 144.2 | 0.991 | 29 % |
| **Gradient Boosting** | **64.8** | **101.7** | **0.996** | **39 %** |

Por horizonte, el MAE de Gradient Boosting es 38, 65, 93 y 136 kW a 5, 15, 30 y 60 minutos.
La persistencia da 47, 106, 194 y 365 kW. La mejora sobre la persistencia crece con el
horizonte, de 18 % a 5 min hasta 63 % a 1 h.

Lectura honesta:

- **A 5 min** hay poco que aprender: la persistencia ya acierta casi todo.
- **A 15 min y 1 h** (los horizontes útiles para el despacho intradiario), Gradient Boosting
  supera a Ridge y a todos los baselines.
- **Hora de peor error:** Ridge y la persistencia fallan más a las 18 h (216 y 226 kW).
  Gradient Boosting tiene su peor hora a las 3 h (92 kW).
- **Viento cerca de los umbrales de alerta:** Gradient Boosting da 136 falsas alarmas críticas
  frente a 267 de Ridge, pero deja sin avisar 114 críticas frente a 102 de Ridge (sobre 6 152
  críticas reales). Ver `docs/RESULTADOS_T9_T10.md`.

### Ajuste de hiperparámetros: ráfagas y sobreajuste

La búsqueda compara 72 configuraciones con validación temporal: `learning_rate`,
`max_iter`, `max_leaf_nodes` (7, 15, 31), `min_samples_leaf` (50, 200, 500) y `loss`. La
configuración elegida usa árboles de hasta 31 hojas y hojas con al menos 50 instantes.

El motivo es el mismo que en la primera versión del modelo: a 15 min el cambio del viento es
casi todo ruido, y un modelo flexible lo sobreajusta. Las hojas grandes y el número moderado
de iteraciones lo evitan. En viento, la pérdida cuadrática y la absoluta quedan empatadas
(MAE de validación 65.6 kW en ambos casos); en demanda gana la absoluta, que es robusta a los
picos aislados.

## 5. Siguientes pasos

1. **NWP como variable de entrada** para horizontes de más de 1 h: las estaciones no anticipan
   los cambios de tiempo.
2. **Pérdida cuantílica** (`loss: quantile`, `quantile: 0.3`) para el despacho: se
   prefiere quedarse corto en viento que pasarse. Reduciría las críticas no avisadas.
3. **Un modelo por horizonte** o un único modelo con el horizonte como variable, para
   cubrir más de una hora del despacho. Ahora solo hay modelo a 15 min.
4. Validar la **estacionalidad anual** cuando haya más de un año de datos: con el año actual
   solo hay un ciclo.
