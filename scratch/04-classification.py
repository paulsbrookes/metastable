import numpy as np
import scipy


from metastable.state import FixedPointMap
from metastable.eom import EOM, Params
from metastable.manifold_inverses import calculate_manifold_inverses
from metastable.incoming_quantum_vector import extend_to_keldysh_state


fixed_point_map = FixedPointMap.load("map.npz")
epsilon_idx = -1
kappa_idx = 20
params = Params(
    epsilon=fixed_point_map.epsilon_linspace[epsilon_idx],
    kappa=fixed_point_map.kappa_linspace[kappa_idx],
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
)
eom = EOM(params=params)
classical_saddle_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 0]
classical_focus_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 1]
print(params)
print(classical_saddle_point, classical_focus_point)

keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)

eom = EOM(params=params)
jacobian = eom.jacobian_func(keldysh_focus_point)
eigenvalues, eigenvectors = np.linalg.eig(jacobian)
print(np.linalg.det(jacobian))

# Identify indices where real part of eigenvalues are <= 0 and > 0
stable_eigenvector_indexes = [i for i, val in enumerate(eigenvalues) if val.real <= 0]
unstable_eigenvector_indexes = [i for i, val in enumerate(eigenvalues) if val.real > 0]

# Assert conditions
assert (
    len(stable_eigenvector_indexes) == 2
), "There should be exactly two stable eigenvectors."
assert (
    len(unstable_eigenvector_indexes) == 2
), "There should be exactly two unstable eigenvectors."

# Check for complex conjugates in each pair
assert np.all(
    np.isclose(
        eigenvectors[:, stable_eigenvector_indexes[0]],
        np.conj(eigenvectors[:, stable_eigenvector_indexes[1]]),
    )
), "The unstable eigenvectors should be conjugate at a focus."
assert np.all(
    np.isclose(
        eigenvectors[:, unstable_eigenvector_indexes[0]],
        np.conj(eigenvectors[:, unstable_eigenvector_indexes[1]]),
    )
), "The stable eigenvectors should be conjugate at a focus."

cov = np.dot(
    np.conjugate(np.transpose(eigenvectors[:, stable_eigenvector_indexes])),
    eigenvectors[:, stable_eigenvector_indexes],
)
...
