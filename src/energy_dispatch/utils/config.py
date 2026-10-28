from os import PathLike

import yaml


def load_config(path: str | PathLike[str] = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
