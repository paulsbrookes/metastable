from typing import Tuple


def calculate_beta_limits(kappa_rescaled: float) -> Tuple[float, float]:
    """Calculates the rescaled power parameter at the bifurcation points from the rescaled decay rate."""
    beta_1 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 - (1 - 3 * kappa_rescaled**2) ** (3 / 2)
    )
    beta_2 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 + (1 - 3 * kappa_rescaled**2) ** (3 / 2)
    )
    return beta_1, beta_2
