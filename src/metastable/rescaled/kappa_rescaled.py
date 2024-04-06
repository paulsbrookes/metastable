import numpy as np


def calculate_kappa_rescaled(kappa: float, delta: float) -> float:
    """Calculates the rescaled decay rate from the original parameters."""
    kappa_rescaled = kappa / (2 * np.abs(delta))
    return kappa_rescaled
