import numpy as np
from typing import Tuple, Optional

from metastable.map.map import FixedPointMap, FixedPointType
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from metastable.paths.data_structures import IndexPair


def generate_guess_from_sol(bvp_result, t_end: float):
    t_guess = np.linspace(0.0, t_end, 10001)
    y_guess = bvp_result.sol(t_guess)
    return t_guess, y_guess


def generate_linear_guess(
    start_point: np.ndarray,
    end_point: np.ndarray,
    t_end: float,
    num_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a linear guess path between two points in phase space.
    
    Args:
        start_point: Starting point in phase space
        end_point: Ending point in phase space
        t_end: End time for the path
        num_points: Number of points to use in the discretization
        
    Returns:
        Tuple of (time array, state array)
    """
    t_guess = np.linspace(0, t_end, num_points)
    y_guess = np.zeros((num_points, len(start_point)))
    
    for i in range(num_points):
        t_frac = t_guess[i] / t_end
        y_guess[i, :] = (1 - t_frac) * start_point + t_frac * end_point
        
    return t_guess, y_guess


def generate_linear_guess_from_map(
    fixed_point_map: FixedPointMap,
    index_pair: IndexPair,
    t_end: float = 8.0,
    fixed_point_type: FixedPointType = FixedPointType.BRIGHT,
    num_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a linear guess path between fixed points from a map.
    
    Args:
        fixed_point_map: Map containing fixed points
        index_pair: Indices for epsilon and kappa in the map
        t_end: End time for the path
        fixed_point_type: Type of fixed point to use as the focus point
        num_points: Number of points to use in the discretization
        
    Returns:
        Tuple of (time array, state array)
    """
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, FixedPointType.SADDLE.value
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, fixed_point_type.value
    ]
    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)
    
    kwargs = {}
    if num_points is not None:
        kwargs["num_points"] = num_points
        
    t_guess, y_guess = generate_linear_guess(
        keldysh_focus_point,
        keldysh_saddle_point,
        t_end,
        **kwargs
    )
    return t_guess, y_guess 