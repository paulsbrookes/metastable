import numpy as np
from typing import Callable
from numpy.typing import NDArray


from metastable.manifold_inverses import calculate_manifold_inverses
from metastable.eom import Params


def generate_boundary_condition_func(
    keldysh_saddle_point: NDArray, keldysh_stable_point: NDArray, params: Params
) -> Callable[[NDArray, NDArray], NDArray]:
    _, saddle_point_unstable_manifold_inverse = calculate_manifold_inverses(
        keldysh_saddle_point, params
    )
    stable_point_stable_manifold_inverse, _ = calculate_manifold_inverses(
        keldysh_stable_point, params
    )
    stable_point_bc_vectors = stable_point_stable_manifold_inverse + np.dot(
        np.array([[0, 1], [-1, 0]]), stable_point_stable_manifold_inverse
    )

    def boundary_condition_func(ya: NDArray, yb: NDArray):
        return np.hstack(
            [
                np.abs(np.dot(stable_point_bc_vectors, ya - keldysh_stable_point)),
                np.abs(
                    np.dot(
                        saddle_point_unstable_manifold_inverse,
                        yb - keldysh_saddle_point,
                    )
                ),
            ]
        )

    return boundary_condition_func
