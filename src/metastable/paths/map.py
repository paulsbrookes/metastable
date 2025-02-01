import numpy as np
import contextlib
import io
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path
from scipy.integrate import solve_bvp
from scipy.integrate._bvp import BVPResult


from metastable.map.map import PhaseSpaceMap, FixedPointType, PathType
from metastable.eom import EOM, Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from metastable.generate_guess import generate_linear_guess, generate_guess_from_sol
from metastable.generate_boundary_conditions import generate_boundary_condition_func


@dataclass
class IndexPair:
    epsilon_idx: int
    kappa_idx: int


def configure_logging(file_name):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a new file handler to store logs
    file_handler = logging.FileHandler(file_name, mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Define the format of logs
    formatter = logging.Formatter(
        80 * "-"
        + "\n"
        + "%(asctime)s - %(levelname)s:\n"
        + "%(message)s\n"
        + 80 * "-"
        + "\n"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def extract_params(fixed_point_map: PhaseSpaceMap, index_pair: IndexPair) -> Params:
    return Params(
        epsilon=fixed_point_map.epsilon_linspace[index_pair.epsilon_idx],
        kappa=fixed_point_map.kappa_linspace[index_pair.kappa_idx],
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )


def prepare_saddle_and_focus_points(
    fixed_point_map: PhaseSpaceMap,
    index_pair: IndexPair,
    fixed_point_type: FixedPointType,
) -> Tuple[np.ndarray, np.ndarray]:
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, FixedPointType.SADDLE.value
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, fixed_point_type.value
    ]

    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)

    return keldysh_saddle_point, keldysh_focus_point


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
    fixed_point_map: PhaseSpaceMap,
    index_pair: IndexPair,
    t_guess: np.ndarray,
    y_guess: np.ndarray,
    endpoint_type: FixedPointType,
) -> BVPResult:
    params = extract_params(fixed_point_map, index_pair)
    print(params)
    logging.info(params)

    eom = EOM(params=params)

    keldysh_saddle_point, keldysh_focus_point = prepare_saddle_and_focus_points(
        fixed_point_map, index_pair, endpoint_type
    )

    boundary_condition_func = generate_boundary_condition_func(
        keldysh_saddle_point, keldysh_focus_point, params
    )

    path_result = solve_path(eom, boundary_condition_func, t_guess, y_guess)

    return path_result


def generate_linear_guess_from_map(
    fixed_point_map: PhaseSpaceMap,
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


def plot_solution(res: BVPResult, t_guess: np.ndarray):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    t_plot = np.linspace(0, t_guess[-1], 1001)
    y0_plot = res.sol(t_plot)[0]
    y1_plot = res.sol(t_plot)[1]
    axes.plot(y0_plot, y1_plot)
    return fig, axes


def map_switching_paths(
    fixed_point_map: PhaseSpaceMap,
    index_list: List[IndexPair],
    output_path: Path,
    t_end: float = 8.0,
    endpoint_type: FixedPointType = FixedPointType.BRIGHT,
) -> List[BVPResult]:
    t_guess, y_guess = generate_linear_guess_from_map(
        fixed_point_map, index_list[0], t_end, endpoint_type
    )

    if endpoint_type == FixedPointType.BRIGHT:
        path_idx = PathType.BRIGHT_TO_SADDLE.value
    elif endpoint_type == FixedPointType.DIM:
        path_idx = PathType.DIM_TO_SADDLE.value
    else:
        raise ValueError(f"Unsupported endpoint_type: {endpoint_type}.")

    output_map_path = output_path / "output_map.npz"
    log_path = output_path / "logs"
    log_path.mkdir(parents=True)

    results = []

    for index_pair in tqdm(index_list):
        plot_file_path = (
            log_path
            / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).png"
        )
        log_file_path = (
            log_path
            / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).log"
        )
        configure_logging(log_file_path)

        path_result = process_index(
            fixed_point_map, index_pair, t_guess, y_guess, endpoint_type
        )

        logging.info(f"Message: {path_result.message} Success: {path_result.success}")

        fig, _ = plot_solution(path_result, t_guess)
        fig.savefig(plot_file_path, bbox_inches="tight")

        fixed_point_map.path_results[
            index_pair.epsilon_idx, index_pair.kappa_idx, path_idx
        ] = path_result
        fixed_point_map.save_state(output_map_path)

        t_guess, y_guess = generate_guess_from_sol(bvp_result=path_result, t_end=t_end)

        results.append(path_result)

    return results
