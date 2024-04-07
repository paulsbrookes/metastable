from typing import Tuple


import numpy as np
from numpy.typing import NDArray


from metastable.eom import EOM, Params


def calculate_manifold_inverses(
    keldysh_fixed_point: NDArray[float], params: Params
) -> Tuple[NDArray[float], NDArray[float]]:
    """Find the inverse of the manifold spanned by the unstable eigenvectors of the jacobian at the saddle point. These
    vectors are orthogonal to the stable manifold, so they can be used to define the boundary condition at the saddle
    point."""
    eom = EOM(params=params)
    jacobian = eom.jacobian_func(keldysh_fixed_point)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    stable_manifold_mask = eigenvalues <= 0
    unstable_manifold_mask = eigenvalues > 0
    stable_manifold = eigenvectors[:, stable_manifold_mask]
    unstable_manifold = eigenvectors[:, unstable_manifold_mask]
    inverse_eigenvectors = np.linalg.inv(eigenvectors)

    inverse_unstable_manifold = inverse_eigenvectors[unstable_manifold_mask, :]
    inverse_stable_manifold = inverse_eigenvectors[stable_manifold_mask, :]



    assert np.allclose(
        np.dot(inverse_unstable_manifold, unstable_manifold),
        np.eye(inverse_unstable_manifold.shape[0], dtype=float),
    ), "The inverse unstable manifold should be orthonormal to the unstable manifold."

    assert np.allclose(
        np.dot(inverse_stable_manifold, stable_manifold),
        np.eye(inverse_stable_manifold.shape[0], dtype=float),
    ), "The inverse stable manifold should be orthonormal to the stable manifold."

    assert np.allclose(
        np.dot(inverse_unstable_manifold, stable_manifold),
        np.zeros(
            [inverse_unstable_manifold.shape[0], inverse_unstable_manifold.shape[0]],
            dtype=float,
        ),
    ), "The inverse unstable manifold should be orthogonal to the stable manifold."

    assert np.allclose(
        np.dot(inverse_stable_manifold, unstable_manifold),
        np.zeros(
            [inverse_stable_manifold.shape[0], inverse_stable_manifold.shape[0]],
            dtype=float,
        ),
    ), "The inverse unstable manifold should be orthogonal to the stable manifold."

    return inverse_stable_manifold, inverse_unstable_manifold
