import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List


from metastable.map.map import FixedPointMap, FixedPointType, PathType
from metastable.paths.data_structures import IndexPair
from metastable.paths.logging_utils import configure_logging
from metastable.paths.guess_generation import generate_linear_guess_from_map, generate_guess_from_sol
from metastable.paths.path_solvers import process_index
from metastable.paths.visualization import plot_solution


def map_switching_paths(
    fixed_point_map: FixedPointMap,
    index_list: List[IndexPair],
    output_path: Path,
    t_end: float = 8.0,
    endpoint_type: FixedPointType = FixedPointType.BRIGHT,
    enable_logging: bool = False,
    overwrite_existing: bool = True,
) -> List:
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
    # Check if output directory exists and raise exception if overwrite_existing is False
    if output_path.exists() and not overwrite_existing:
        raise FileExistsError(f"Output path '{output_path}' already exists and overwrite_existing is False")
    
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