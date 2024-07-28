import numpy as np
import numpy.typing as npt
from typing import Union


def calculate_beta_limits(
    kappa_rescaled: Union[float, npt.NDArray[np.floating]]
) -> npt.NDArray[np.floating]:
    """Calculates the rescaled power parameter at the bifurcation points from the rescaled decay rate."""
    beta_1 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 - np.power(1 - 3 * kappa_rescaled**2, 3 / 2)
    )
    beta_2 = (2 / 27) * (
        1 + 9 * kappa_rescaled**2 + np.power(1 - 3 * kappa_rescaled**2, 3 / 2)
    )
    return np.array([beta_1, beta_2])
