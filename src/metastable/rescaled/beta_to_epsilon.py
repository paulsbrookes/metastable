import numpy as np
import numpy.typing as npt
from typing import Union


def map_beta_to_epsilon(
    beta: Union[float, npt.NDArray[np.floating]],
    delta: Union[float, npt.NDArray[np.floating]],
    chi: Union[float, npt.NDArray[np.floating]],
) -> Union[float, npt.NDArray[np.floating]]:
    lambda_ = abs(chi / delta)
    epsilon = delta * np.sqrt(beta / (2 * lambda_))
    return epsilon
