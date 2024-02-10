import numpy as np
import numpy.typing as npt
from typing import Union


def calculate_kappa_rescaled(
    kappa: Union[float, npt.NDArray[np.floating]],
    delta: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    """Calculates the rescaled decay rate from the original parameters."""
    kappa_rescaled = kappa / np.abs(delta)
    return kappa_rescaled
