import numpy as np
import contextlib
import io
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
from tqdm import tqdm
from pathlib import Path
from scipy.integrate import solve_bvp
from scipy.integrate._bvp import BVPResult


from metastable.map.map import FixedPointMap, FixedPointType, PathType
from metastable.eom import EOM, Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from metastable.generate_guess import generate_linear_guess, generate_guess_from_sol
from metastable.generate_boundary_conditions import generate_boundary_condition_func


@dataclass
class IndexPair:
    epsilon_idx: int
    kappa_idx: int

@dataclass
class BistableBoundaries:
    dim_saddle: Optional[IndexPair]
    bright_saddle: Optional[IndexPair]

@dataclass
class Sweeps:
    """Sweep traversing the bistable region from each boundary point."""
    dim_saddle: List[IndexPair]  # Sweep starting at the dim-saddle bifurcation
    bright_saddle: List[IndexPair]  # Sweep starting at the bright-saddle bifurcation


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


def extract_params(fixed_point_map: FixedPointMap, index_pair: IndexPair) -> Params:
    return Params(
        epsilon=fixed_point_map.epsilon_linspace[index_pair.epsilon_idx],
        kappa=fixed_point_map.kappa_linspace[index_pair.kappa_idx],
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )


def prepare_saddle_and_focus_points(
    fixed_point_map: FixedPointMap,
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
    fixed_point_map: FixedPointMap,
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


def plot_solution(res: BVPResult, t_guess: np.ndarray):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    t_plot = np.linspace(0, t_guess[-1], 1001)
    y0_plot = res.sol(t_plot)[0]
    y1_plot = res.sol(t_plot)[1]
    axes.plot(y0_plot, y1_plot)
    return fig, axes


def map_switching_paths(
    fixed_point_map: FixedPointMap,
    index_list: List[IndexPair],
    output_path: Path,
    t_end: float = 8.0,
    endpoint_type: FixedPointType = FixedPointType.BRIGHT,
    enable_logging: bool = False,
    overwrite_existing: bool = True,
) -> List[BVPResult]:
    """
    Calculate switching paths for a list of parameter points and store results in the fixed_point_map.
    
    Note: This function mutates the input fixed_point_map by storing path results in it.
    
    Args:
        fixed_point_map: FixedPointMap object containing fixed points. Will be modified with path results.
        index_list: List of IndexPair objects specifying parameter points to calculate paths for
        output_path: Directory to save output files
        t_end: End time for the path integration
        endpoint_type: Type of fixed point to use as endpoint (BRIGHT or DIM)
        enable_logging: Whether to enable detailed logging
        overwrite_existing: Whether to overwrite existing results
        
    Returns:
        List of BVPResult objects for each calculated path
    """
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    
    # Set up logging directory once if needed
    if enable_logging:
        log_path.mkdir(parents=True, exist_ok=True)

    results = []

    for index_pair in tqdm(index_list):
        # Handle all logging setup at once
        if enable_logging:
            log_file_path = (
                log_path
                / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).log"
            )
            configure_logging(log_file_path)

        # Core processing - always happens
        path_result = process_index(
            fixed_point_map, index_pair, t_guess, y_guess, endpoint_type
        )
        
        # Store results and update guess - always happens
        fixed_point_map.path_results[
            index_pair.epsilon_idx, index_pair.kappa_idx, path_idx
        ] = path_result
        fixed_point_map.save(output_map_path)
        t_guess, y_guess = generate_guess_from_sol(bvp_result=path_result, t_end=t_end)
        results.append(path_result)

        # Handle all logging output at once
        if enable_logging:
            logging.info(f"Message: {path_result.message} Success: {path_result.success}")
            plot_file_path = (
                log_path
                / f"(epsilon_idx,kappa_idx)-({index_pair.epsilon_idx},{index_pair.kappa_idx}).png"
            )
            fig, _ = plot_solution(path_result, t_guess)
            fig.savefig(plot_file_path, bbox_inches="tight")

    return results


def get_bistable_epsilon_range(
    bistable_region: np.ndarray, kappa_idx: int
) -> BistableBoundaries:
    """
    Get the epsilon indices at the bistable boundaries for a given kappa index.
    
    This function identifies the lower and upper epsilon indices that bound the bistable
    region at a specific kappa value.
    
    Args:
        bistable_region: 2D boolean array indicating where the system is bistable
        kappa_idx: Index in the kappa dimension
        
    Returns:
        BistableBoundary: A dataclass containing:
            - bright_saddle: IndexPair at the bright-saddle bifurcation (lower epsilon boundary)
            - dim_saddle: IndexPair at the dim-saddle bifurcation (upper epsilon boundary)
            Both values are None if no bistable region exists at the specified kappa.
    """
    bistable_epsilon = np.where(bistable_region[:, kappa_idx])[0]
    
    if len(bistable_epsilon) == 0:
        return BistableBoundaries(bright_saddle=None, dim_saddle=None)
        
    bright_saddle_idx = bistable_epsilon[0]  # Lower boundary is bright-saddle bifurcation
    dim_saddle_idx = bistable_epsilon[-1]    # Upper boundary is dim-saddle bifurcation
    
    bright_saddle_pair = IndexPair(epsilon_idx=bright_saddle_idx, kappa_idx=kappa_idx)
    dim_saddle_pair = IndexPair(epsilon_idx=dim_saddle_idx, kappa_idx=kappa_idx)
    
    return BistableBoundaries(bright_saddle=bright_saddle_pair, dim_saddle=dim_saddle_pair)


def get_bistable_kappa_range(
    bistable_region: np.ndarray, epsilon_idx: int
) -> BistableBoundaries:
    """
    Get the kappa indices at the bistable boundaries for a given epsilon index.
    
    This function identifies the lower and upper kappa indices that bound the bistable
    region at a specific epsilon value.
    
    Args:
        bistable_region: 2D boolean array indicating where the system is bistable
        epsilon_idx: Index in the epsilon dimension
        
    Returns:
        BistableBoundary: A dataclass containing:
            - dim_saddle: IndexPair at the dim-saddle bifurcation (lower kappa boundary)
            - bright_saddle: IndexPair at the bright-saddle bifurcation (upper kappa boundary)
            Both values are None if no bistable region exists at the specified epsilon.
    """
    bistable_kappa = np.where(bistable_region[epsilon_idx, :])[0]
    
    if len(bistable_kappa) == 0:
        return BistableBoundaries(dim_saddle=None, bright_saddle=None)
        
    dim_saddle_idx = bistable_kappa[0]       # Lower boundary is dim-saddle bifurcation
    bright_saddle_idx = bistable_kappa[-1]   # Upper boundary is bright-saddle bifurcation
    
    dim_saddle_pair = IndexPair(epsilon_idx=epsilon_idx, kappa_idx=dim_saddle_idx)
    bright_saddle_pair = IndexPair(epsilon_idx=epsilon_idx, kappa_idx=bright_saddle_idx)
    
    return BistableBoundaries(dim_saddle=dim_saddle_pair, bright_saddle=bright_saddle_pair)


def generate_sweep_index_pairs(
    boundary: BistableBoundaries, 
    max_points: Optional[int] = None,
    sweep_fraction: float = 1.0
) -> Sweeps:
    """
    Generate paths of IndexPair objects that traverse the bistable region.
    
    Args:
        boundary: BistableBoundary object containing the dim_saddle and bright_saddle points
        max_points: Optional maximum number of points to include in each path. If None, uses the maximum
                 distance between start and target indices (either epsilon or kappa) as the number of points.
        sweep_fraction: Float between 0 and 1 indicating what fraction of the path to traverse.
                 1.0 means go all the way from one endpoint to the other.
                 0.5 means go halfway from each endpoint toward the other.
        
    Returns:
        Sweeps object containing:
            - dim_saddle: List of IndexPair objects from dim_saddle toward bright_saddle
            - bright_saddle: List of IndexPair objects from bright_saddle toward dim_saddle
            
    Raises:
        ValueError: If boundary points are None or invalid or if sweep_fraction is outside [0,1]
    """
    if boundary.dim_saddle is None or boundary.bright_saddle is None:
        raise ValueError("Both boundary points must be defined")
    
    if not 0 <= sweep_fraction <= 1:
        raise ValueError("sweep_fraction must be between 0 and 1")
    
    # Extract coordinates
    dim_epsilon = boundary.dim_saddle.epsilon_idx
    dim_kappa = boundary.dim_saddle.kappa_idx
    bright_epsilon = boundary.bright_saddle.epsilon_idx
    bright_kappa = boundary.bright_saddle.kappa_idx
    
    # Calculate target points based on sweep_fraction
    target_dim_epsilon = dim_epsilon + sweep_fraction * (bright_epsilon - dim_epsilon)
    target_dim_kappa = dim_kappa + sweep_fraction * (bright_kappa - dim_kappa)
    target_bright_epsilon = bright_epsilon + sweep_fraction * (dim_epsilon - bright_epsilon)
    target_bright_kappa = bright_kappa + sweep_fraction * (dim_kappa - bright_kappa)
    
    # Determine number of points if not specified, based on distance to target
    if max_points is None:
        dim_epsilon_distance = abs(dim_epsilon - target_dim_epsilon)
        dim_kappa_distance = abs(dim_kappa - target_dim_kappa)
        bright_epsilon_distance = abs(bright_epsilon - target_bright_epsilon)
        bright_kappa_distance = abs(bright_kappa - target_bright_kappa)
        
        dim_max_distance = max(dim_epsilon_distance, dim_kappa_distance)
        bright_max_distance = max(bright_epsilon_distance, bright_kappa_distance)
        
        dim_points = int(dim_max_distance) + 1  # +1 to include both endpoints
        bright_points = int(bright_max_distance) + 1  # +1 to include both endpoints
    else:
        dim_points = max_points
        bright_points = max_points
    
    # Ensure we have at least 2 points (start and end)
    dim_points = max(2, dim_points)
    bright_points = max(2, bright_points)
    
    # Create evenly spaced points for dim to bright path
    dim_to_bright = []
    for i in range(dim_points):
        # Linear interpolation between endpoints
        t = i / (dim_points - 1)  # t goes from 0 to 1
        epsilon_idx = round(dim_epsilon + t * (target_dim_epsilon - dim_epsilon))
        kappa_idx = round(dim_kappa + t * (target_dim_kappa - dim_kappa))
        dim_to_bright.append(IndexPair(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx))
    
    # Create evenly spaced points for bright to dim path
    bright_to_dim = []
    for i in range(bright_points):
        # Linear interpolation between endpoints
        t = i / (bright_points - 1)  # t goes from 0 to 1
        epsilon_idx = round(bright_epsilon + t * (target_bright_epsilon - bright_epsilon))
        kappa_idx = round(bright_kappa + t * (target_bright_kappa - bright_kappa))
        bright_to_dim.append(IndexPair(epsilon_idx=epsilon_idx, kappa_idx=kappa_idx))
    
    return Sweeps(dim_saddle=dim_to_bright, bright_saddle=bright_to_dim)
