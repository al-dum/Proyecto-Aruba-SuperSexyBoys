# Metodología de trabajo

Guía de cómo se aborda este proyecto desde cero: de datos crudos a modelo confiable
conectado al despacho. Cada fase produce algo concreto antes de pasar a la siguiente.

## 0. Define el problema antes de tocar datos

Antes de abrir un notebook, deja escrito (aunque sea en 3 líneas):

- **Qué predices exactamente**: ¿demanda en MW a 1h vista? ¿a 24h vista?
- **Granularidad**: ¿horaria, cada 15 min?
- **Qué harás con la predicción**: aquí, alimentar el despacho — eso condiciona qué
  error importa más (subestimar demanda es peor que sobrestimarla, por ejemplo).

Esto decide la métrica de éxito antes de que los datos sesguen la opinión.

## 1. Ingesta — datos crudos intocables

Los CSV originales van a `data/raw/` **y no se tocan nunca**. Ni un `.strip()`. Si
hace falta limpiarlos, el resultado limpio va a otro sitio (`data/processed/`). Esto
permite rehacer todo el pipeline si algo sale mal en la limpieza, sin perder el
original.

## 2. Auditoría de calidad (antes de limpiar nada)

Con el CSV crudo, antes de decidir cómo limpiarlo, responde:

| Pregunta | Cómo se mira |
|---|---|
| ¿Hay huecos temporales? | `pd.date_range` completo vs. timestamps que realmente existen |
| ¿Hay duplicados de timestamp? | `df.duplicated(subset='timestamp')` |
| ¿Hay valores imposibles? | demanda negativa, % de renovable >100%, capacidad excedida |
| ¿Hay outliers reales o errores de sensor? | boxplot por hora del día, z-score, o simplemente ojo |
| ¿Cambian las unidades a mitad de serie? | kW vs MW, o cambios de medidor |
| ¿Coincide la zona horaria / cambio de hora (DST)? | muy típico en series eléctricas y se cuela sin avisar |
| ¿Los NaN son aleatorios o hay tramos enteros perdidos? | `df.isna().sum()` + gráfico de huecos en el tiempo |

Se hace en `notebooks/01_auditoria.ipynb`. El resultado no es código, es una lista
de decisiones a tomar en el paso 3.

## 3. Limpieza (con reglas explícitas, no a ojo)

Aquí ya se escriben funciones, porque cada regla de limpieza debe repetirse igual
cada vez que llegue un CSV nuevo:

- **Huecos temporales**: ¿interpolar (si son horas sueltas) o dejar el hueco (si son
  días)? Interpolar 3 días de demanda es inventar datos.
- **Outliers**: no se borran a ciegas — un pico de demanda real (ola de calor) no es
  un error, un valor negativo de demanda sí lo es.
- **Duplicados**: quedarse con el último, el primero o la media, según cómo se
  generó el dato.
- **Dejar rastro**: cada fila modificada o borrada se anota (columna `is_imputed` o
  log aparte). Sin esto no se puede saber después si un buen resultado se debe al
  modelo o a que se "arregló" el dato de más.

Va a `src/energy_dispatch/data/cleaning.py`; el resultado limpio se guarda en
`data/processed/` (nunca sobrescribe `data/raw/`).

## 4. Validación de coherencia (test, no vista)

No basta con mirarlo una vez — se automatizan chequeos que corren cada vez que entra
un CSV nuevo:

```python
# tests/test_data_quality.py
def test_no_negative_demand(df): ...
def test_no_duplicate_timestamps(df): ...
def test_expected_frequency(df): ...       # ej. exactamente 1 fila/hora
def test_within_physical_bounds(df): ...   # generación <= capacidad instalada
```

Esto separa un proyecto serio de un notebook de un solo uso: cuando lleguen datos
nuevos, `pytest` dice en segundos si algo se rompió.

## 5. Análisis exploratorio (EDA) — ya sobre datos limpios

- Estacionalidad diaria/semanal/anual (heatmap hora × día de semana suele ser muy
  revelador en demanda eléctrica).
- Autocorrelación (¿cuánto pesa el valor de hace 1h, de hace 24h, de hace 1 semana?).
- Relación con variables externas (temperatura, festivos, día de la semana).
- Estacionariedad si se van a usar modelos de series temporales clásicos.

Sigue en notebook (`notebooks/02_eda.ipynb`); aquí se decide qué features tienen
sentido probar.

## 6. Features — de la hipótesis al código

Cada variable que el EDA sugiere que importa se prueba en notebook y, si aporta, se
congela en `features/build.py`. Típicas en demanda energética:

- Lags (`demanda_t-1`, `demanda_t-24`, `demanda_t-168`)
- Medias móviles
- Festivos / fin de semana
- Temperatura (si se tiene) — suele ser la variable más potente
- Codificación cíclica de hora/mes (`sin`/`cos`), mejor que hora como entero plano

## 7. Split temporal — no aleatorio

**Nunca** usar `train_test_split` aleatorio en series temporales — filtra
información del futuro al pasado. Se hace por corte de fecha: entrenar con todo
antes de X, validar con lo de después. Si hay estacionalidad anual, el test debería
cubrir al menos un ciclo completo (ej. un año) para no evaluar solo en una estación.

## 8. Modelado — empieza simple

1. **Baseline tonto primero**: "la demanda de mañana a las 14h = la demanda de hoy a
   las 14h" o la media histórica de esa hora. Si el modelo de ML no le gana a esto,
   algo va mal.
2. Luego modelo simple (regresión lineal / árbol simple) como segundo baseline.
3. Luego XGBoost/LightGBM, ajustando con validación cruzada **temporal**
   (`TimeSeriesSplit`, no k-fold normal).

## 9. Evaluación — con las métricas ya definidas

MAE/RMSE/MAPE (`evaluation/metrics.py`), pero también **por franja horaria** — un
modelo puede tener buen MAE global y fallar sistemáticamente en horas punta, que es
justo cuando más importa para el despacho.

## 10. Conectar con el despacho

Solo cuando el modelo predictivo es fiable, se pasa la predicción al optimizador
(`dispatch/optimizer.py`). Meter predicciones malas en el despacho amplifica el
error — por eso este paso va el último, no en paralelo.

## Resumen

| Fase | Dónde vive | Se congela en `.py` cuando... |
|---|---|---|
| Auditoría | notebook | nunca — de un solo uso, pero el resultado (reglas) pasa a limpieza |
| Limpieza | notebook → `src/data/cleaning.py` | la regla de limpieza es estable |
| Validación calidad | `tests/` | desde el principio — corre en cada CSV nuevo |
| EDA | notebook | nunca — es exploración |
| Features | notebook → `src/features/build.py` | se decide que esa variable se queda |
| Modelo | notebook → `src/models/train.py` | se fijan hiperparámetros finales |
| Pipeline completo | `main.py` | todo lo anterior es estable |

## Regla general de notebook → código

El notebook es para **pensar**, `src/` es para **lo que ya se decidió**. En cuanto
una celda se va a ejecutar una tercera vez, o empieza a tener funciones con `def` de
más de unas pocas líneas, esa lógica se mueve a un módulo en `src/`. El notebook pasa
entonces a *llamar* a esas funciones en vez de tener la lógica pegada.

`app/` (Streamlit) sigue la misma regla: solo importa funciones ya congeladas en
`src/energy_dispatch/`, nunca tiene lógica de negocio propia.
