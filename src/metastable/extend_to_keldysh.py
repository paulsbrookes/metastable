import numpy as np
from numpy.typing import NDArray


def extend_to_keldysh_state(classical_state: NDArray[float]):
    return np.hstack([classical_state, [0.0, 0.0]])
