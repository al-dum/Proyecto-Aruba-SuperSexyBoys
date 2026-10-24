import sys
from pathlib import Path

# Configurar la ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.energy_dispatch.data.ingestion import ejecutar_pipeline_ingesta
if __name__ == "__main__":
    print("Iniciando pipeline de ingesta...")
    ejecutar_pipeline_ingesta(PROJECT_ROOT)