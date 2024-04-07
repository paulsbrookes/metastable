import numpy as np
from numpy.typing import NDArray


from metastable.eom import EOM, Params
from metastable.extend_to_keldysh import extend_to_keldysh_state


def calculate_incoming_quantum_vector(
    classical_saddle_point: NDArray[float], params: Params
) -> NDArray[float]:
    """At the saddle point, calculate the incoming vector which has non-zero quantum components."""

    eom = EOM(params=params)

    classical_jacobian = eom.jacobian_classical_func(classical_saddle_point)
    classical_eigenvalues, classical_eigenvectors = np.linalg.eig(classical_jacobian)
    assert (
        np.prod(classical_eigenvalues) < 0
    ), f"{classical_saddle_point} is not a saddle point."

    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_jacobian = eom.jacobian_func(keldysh_saddle_point)
    keldysh_eigenvalues, keldysh_eigenvectors = np.linalg.eig(keldysh_jacobian)

    incoming_vector_mask = keldysh_eigenvalues < 0
    quantum_vector_mask = ~np.isclose(
        np.linalg.norm(keldysh_eigenvectors[2:, :], axis=0), 0.0
    )
    incoming_quantum_vector_mask = incoming_vector_mask & quantum_vector_mask
    assert (
        np.sum(incoming_quantum_vector_mask) == 1
    ), f"We expect only a single incoming quantum vector. Found: {np.sum(incoming_quantum_vector_mask)}."

    incoming_quantum_vector = keldysh_eigenvectors[
        np.where(incoming_quantum_vector_mask)[0][0]
    ]

    return incoming_quantum_vector
