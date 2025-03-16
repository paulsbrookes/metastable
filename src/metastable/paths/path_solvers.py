import numpy as np
import contextlib
import io
import logging
from typing import Tuple, Optional
from scipy.integrate import solve_bvp
from scipy.integrate._bvp import BVPResult

from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM
from metastable.paths.boundary_conditions.boundary_conditions_lock import generate_boundary_condition_func, BoundaryLockParams
from metastable.paths.data_structures import IndexPair
from metastable.paths.parameter_utils import extract_params, prepare_saddle_and_stable_points

def solve_path(
    eom, boundary_condition_func, t_guess: np.ndarray, y_guess: np.ndarray
) -> BVPResult:
    wrapper = lambda x, y: eom.y_dot_func(y)

    # Create a StringIO object to capture stdout
    stdout_buffer = io.StringIO()

    # Capture the output of solve_bvp
    with contextlib.redirect_stdout(stdout_buffer):
        result = solve_bvp(
            wrapper,
            boundary_condition_func,
            t_guess,
            y_guess,
            tol=1e-3,
            max_nodes=1000000,
            verbose=2,
        )

    # Log the captured stdout
    logging.info(stdout_buffer.getvalue())

    return result

def process_index(
    fixed_point_map: FixedPointMap,
    index_pair: IndexPair,
    t_guess: np.ndarray,
    y_guess: np.ndarray,
    endpoint_type: FixedPointType,
    lock_params: Optional[BoundaryLockParams] = None,
) -> BVPResult:
    params = extract_params(fixed_point_map, index_pair)
    print(params)
    logging.info(params)

    eom = EOM(params=params)

    keldysh_saddle_point, keldysh_focus_point = prepare_saddle_and_stable_points(
        fixed_point_map, index_pair, endpoint_type
    )

    # Use default BoundaryLockParams if none provided
    if lock_params is None:
        lock_params = BoundaryLockParams()

    boundary_condition_func = generate_boundary_condition_func(
        keldysh_saddle_point, keldysh_focus_point, params, lock_params=lock_params
    )

    path_result = solve_path(eom, boundary_condition_func, t_guess, y_guess)

    return path_result 