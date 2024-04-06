import numpy as np


def map_beta_to_epsilon(beta: float, delta: float, chi: float) -> float:
    lambda_ = abs(chi / delta)
    epsilon = delta * np.sqrt(beta / (2 * lambda_))
    return epsilon
