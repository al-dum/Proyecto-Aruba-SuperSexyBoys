import numpy as np


def dispatch(demand: np.ndarray, capacities: dict[str, float], costs: dict[str, float]) -> dict:
    """Despacho por orden de mérito (placeholder; sustituir por LP/MILP con pulp)."""
    remaining = demand.copy().astype(float)
    result = {}
    for name in sorted(costs, key=costs.get):
        gen = np.minimum(remaining, capacities[name])
        result[name] = gen
        remaining -= gen
    return result
