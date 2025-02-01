from typing import Tuple


import numpy as np
from numpy.typing import NDArray


from metastable.eom import EOM, Params


def calculate_jacobian_eigenvectors(
    keldysh_fixed_point: NDArray[float], params: Params
) -> Tuple[NDArray[float], NDArray[float]]:
    """Find the inverse of the manifold spanned by the unstable eigenvectors of the jacobian at the saddle point. These
    vectors are orthogonal to the stable manifold, so they can be used to define the boundary condition at the saddle
    point."""
    eom = EOM(params=params)
    jacobian = eom.jacobian_func(keldysh_fixed_point)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    return eigenvalues, eigenvectors
