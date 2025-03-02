import numpy as np
from typing import Tuple
from numpy.typing import NDArray


from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from metastable.paths.data_structures import IndexPair


def generate_guess_from_sol(bvp_result, t_end: float):
    t_guess = np.linspace(0.0, t_end, 10001)
    y_guess = bvp_result.sol(t_guess)
    return t_guess, y_guess


def generate_linear_guess(y0: NDArray, y1: NDArray, t_end: float):
    t_guess = np.linspace(0.0, t_end, 10001)
    y_guess = (
        y0[:, np.newaxis] + t_guess[np.newaxis, :] * (y1 - y0)[:, np.newaxis] / t_end
    )
    return t_guess, y_guess


def generate_linear_guess_from_map(
    fixed_point_map: FixedPointMap,
    index_pair: IndexPair,
    t_end: float = 8.0,
    fixed_point_type: FixedPointType = FixedPointType.BRIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, FixedPointType.SADDLE.value
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, fixed_point_type.value
    ]
    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)
    t_guess, y_guess = generate_linear_guess(
        keldysh_focus_point,
        keldysh_saddle_point,
        t_end,
    )
    return t_guess, y_guess 