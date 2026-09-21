"""Punto de entrada: datos -> features -> modelo -> predicción -> despacho."""
import argparse

from energy_dispatch.utils.config import load_config


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    print(f"Configuración cargada. Modelo: {cfg['model']['name']}")
    # TODO: cargar datos, construir features, entrenar, predecir y despachar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)
