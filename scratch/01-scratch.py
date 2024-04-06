import numpy as np
from numpy.typing import NDArray


from metastable.state import FixedPointMap
from metastable.eom import EOM, Params


def calculate_saddle_point_incoming_quantum_vector(
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

    # quantum_vector_indices = np.where(
    #     ~np.isclose(np.linalg.norm(keldysh_eigenvectors[2:, :], axis=0), 0)
    # )[0]
    # incoming_quantum_vector_indices = quantum_vector_indices[
    #     np.where(keldysh_eigenvalues[quantum_vector_indices] < 0)[0]
    # ]
    #
    # if incoming_quantum_vector_indices.shape[0] != 1:
    #     raise Exception("Number of incoming quantum vectors != 1.")
    #
    # incoming_quantum_vector_idx = incoming_quantum_vector_indices[0]
    # incoming_quantum_vector = keldysh_eigenvectors[:, incoming_quantum_vector_idx]
    #
    # return incoming_quantum_vector


def extend_to_keldysh_state(classical_state: NDArray[float]):
    return np.hstack([classical_state, [0.0, 0.0]])


fixed_point_map = FixedPointMap.load("map.npz")
epsilon_idx = 30
kappa_idx = 10
params = Params(
    epsilon=fixed_point_map.epsilon_linspace[epsilon_idx],
    kappa=fixed_point_map.kappa_linspace[kappa_idx],
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
)
eom = EOM(params=params)
classical_saddle_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 0]
incoming_quantum_vector = calculate_saddle_point_incoming_quantum_vector(
    classical_saddle_point, params
)
print(incoming_quantum_vector)


# print(classical_saddle_point)
# classical_jacobian = eom.jacobian_classical_func(classical_saddle_point)
#
# # Calculate the eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(classical_jacobian)
#
#
# keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
# keldysh_jacobian = eom.jacobian_func(keldysh_saddle_point)
#
#
#
# # Calculate the eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(keldysh_jacobian)
#
#
#
#
#
#
# quantum_vector_indices = np.where(
#     ~np.isclose(np.linalg.norm(eigenvectors[2:, :], axis=0), 0)
# )[0]
# incoming_quantum_vector_indices = quantum_vector_indices[
#     np.where(eigenvalues[quantum_vector_indices] < 0)[0]
# ]
#
# if incoming_quantum_vector_indices.shape[0] != 1:
#     raise Exception("Number of incoming quantum vectors != 1.")
#
# incoming_quantum_vector_idx = incoming_quantum_vector_indices[0]
# incoming_quantum_vector = eigenvectors[:, incoming_quantum_vector_idx]
#
# print(incoming_quantum_vector)


# model = EscapeModel(epsilon, delta, chi)
# model.find_fixed_points()
# model.classify_fixed_points()
# model.obtain_incoming_quantum_vector()
#
# unstable_point = np.hstack([model.fixed_points[1], [0, 0]])
# unstable_jac = model.jacobian_func(unstable_point)
# unstable_eigenvalues, unstable_eigenvectors = np.linalg.eig(unstable_jac)
# unstable_incoming_vectors = unstable_eigenvectors[:, [1, 2]]
# unstable_outgoing_vectors = unstable_eigenvectors[:, [0, 3]]
# plane_vectors = unstable_incoming_vectors
# out_of_plane_vectors = unstable_outgoing_vectors.T
# unstable_out_of_plane_vectors = np.array([remove_plane_component(v, plane_vectors) for v in out_of_plane_vectors])
#
# stable_point = np.hstack([model.fixed_points[2], [0, 0]])
# stable_jac = model.jacobian_func(stable_point)
# stable_eigenvalues, stable_eigenvectors = np.linalg.eig(stable_jac)
# stable_incoming_vectors = np.array([(stable_eigenvectors[:, 0] + stable_eigenvectors[:, 1]).real,
#                                     (1j * stable_eigenvectors[:, 0] - 1j * stable_eigenvectors[:, 1]).real]).T
# stable_outgoing_vectors = np.array([(stable_eigenvectors[:, 2] + stable_eigenvectors[:, 3]).real,
#                                     (1j * stable_eigenvectors[:, 2] - 1j * stable_eigenvectors[:, 3]).real]).T
# plane_vectors = stable_outgoing_vectors
# out_of_plane_vectors = stable_incoming_vectors.T
# stable_out_of_plane_vectors = np.array([remove_plane_component(v, plane_vectors) for v in out_of_plane_vectors])
#
# def bc(ya, yb):
#     return np.hstack([np.dot(ya - stable_point, stable_out_of_plane_vectors.T),
#                       np.dot(yb - unstable_point, unstable_out_of_plane_vectors.T)])
#
# t_guess = np.linspace(0, t_end, 10001)
# y_guess = stable_point[:, np.newaxis] + t_guess[np.newaxis, :] * (unstable_point - stable_point)[:,
#                                                                      np.newaxis] / t_end
# wrapper = lambda x, y: model.y_dot_func(y)
# res = scipy.integrate.solve_bvp(wrapper, bc, t_guess, y_guess, tol=1e-14, max_nodes=100000)
