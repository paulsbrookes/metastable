import numpy as np
from numpy.typing import NDArray


from metastable.eom import EOM, Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state


def calculate_saddle_point_unstable_manifold_inverse(
    classical_saddle_point: NDArray[float], params: Params
) -> NDArray[float]:
    """Find the inverse of the manifold spanned by the unstable eigenvectors of the jacobian at the saddle point. These
    vectors are orthogonal to the stable manifold, so they can be used to define the boundary condition at the saddle
    point."""

    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    eom = EOM(params=params)
    jacobian = eom.jacobian_func(keldysh_saddle_point)
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    assert np.allclose(eigenvalues.imag, 0, atol=1e-8), "All eigenvalues must be real."
    stable_manifold_mask = eigenvalues <= 0
    unstable_manifold_mask = eigenvalues > 0
    stable_manifold = eigenvectors[:, stable_manifold_mask]
    unstable_manifold = eigenvectors[:, unstable_manifold_mask]
    inverse_eigenvectors = np.linalg.inv(eigenvectors)
    inverse_unstable_manifold = inverse_eigenvectors[unstable_manifold_mask, :]

    assert np.allclose(
        np.dot(inverse_unstable_manifold, unstable_manifold),
        np.eye(inverse_unstable_manifold.shape[0], dtype=float),
    ), "The inverse unstable manifold should be orthonormal to the unstable manifold."

    assert np.allclose(
        np.dot(inverse_unstable_manifold, stable_manifold),
        np.zeros(
            [inverse_unstable_manifold.shape[0], inverse_unstable_manifold.shape[0]],
            dtype=float,
        ),
    ), "The inverse unstable manifold should be orthogonal to the stable manifold."

    return inverse_unstable_manifold
