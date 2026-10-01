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

## Metodología

Ver [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md): flujo de trabajo desde datos
crudos (auditoría, limpieza, validación) hasta modelo en producción conectado al
despacho.
