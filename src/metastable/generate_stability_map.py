import os
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
from metastable.extend_to_keldysh import extend_to_keldysh_state

# Set up logging for production diagnostics.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold shared map data for each worker process.
global_map_data = None

def init_worker(map_data: Dict[str, Any]) -> None:
    """Initializer for each worker process; loads shared map data into a global variable."""
    global global_map_data
    global_map_data = map_data

def compute_stability(classical_fixed_point: np.ndarray, params: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors for a given classical fixed point.
    
    Returns arrays of shape (4,) for eigenvalues and (4,4) for eigenvectors.
    In case of an error or if the input contains NaNs, returns arrays filled with NaNs.
    """
    try:
        if np.any(np.isnan(classical_fixed_point)):
            return np.full(4, np.nan, dtype=complex), np.full((4, 4), np.nan, dtype=complex)
        full_fixed_point = extend_to_keldysh_state(classical_fixed_point)
        eom = EOM(params)
        jacobian = eom.jacobian_func(full_fixed_point)
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        
        # Filter out imaginary parts where real parts are nan
        real_parts = eigenvalues.real
        imag_parts = eigenvalues.imag
        imag_parts[np.isnan(real_parts)] = np.nan
        eigenvalues = real_parts + 1j * imag_parts
        
        return eigenvalues, eigenvectors
    except Exception:
        logger.exception("Error in compute_stability for fixed point: %s", classical_fixed_point)
        return np.full(4, np.nan, dtype=complex), np.full((4, 4), np.nan, dtype=complex)

def process_single(epsilon_idx: int, kappa_idx: int) -> Tuple[int, int, Dict]:
    """
    Process a single (epsilon, kappa) pair using the shared map data.
    Computes stability for each fixed point type and returns a tuple of indices and a results dict.
    """
    results = {
        'eigenvalues': {},
        'eigenvectors': {}
    }
    try:
        for fp_type in FixedPointType:
            classical_fixed_point = global_map_data['fixed_points'][epsilon_idx, kappa_idx, fp_type.value, :]
            # Skip if the fixed point data is not available.
            if np.all(np.isnan(classical_fixed_point)):
                continue

            params = Params(
                epsilon=global_map_data['epsilon_linspace'][epsilon_idx],
                kappa=global_map_data['kappa_linspace'][kappa_idx],
                chi=global_map_data['chi'],
                delta=global_map_data['delta'],
            )
            eigenvalues, eigenvectors = compute_stability(classical_fixed_point, params)
            results['eigenvalues'][fp_type.value] = eigenvalues
            results['eigenvectors'][fp_type.value] = eigenvectors
    except Exception:
        logger.exception("Error processing indices (epsilon_idx=%s, kappa_idx=%s)", epsilon_idx, kappa_idx)
    return epsilon_idx, kappa_idx, results

def generate_stability_map(fixed_point_map: FixedPointMap, n_workers: Optional[int] = None) -> FixedPointMap:
    """
    Generate stability analysis for a fixed point map.
    
    Args:
        fixed_point_map: The FixedPointMap object containing fixed points to analyze.
        n_workers: Number of worker processes to use. Defaults to os.cpu_count() if None.
    
    Returns:
        A new FixedPointMap object with computed eigenvalues and eigenvectors.
    """
    # Create a copy of the input map to avoid mutating the original.
    map_copy = fixed_point_map.copy()

    # Prepare the shared map data for workers.
    map_data = {
        'fixed_points': map_copy.fixed_points,
        'epsilon_linspace': map_copy.epsilon_linspace,
        'kappa_linspace': map_copy.kappa_linspace,
        'chi': map_copy.chi,
        'delta': map_copy.delta,
    }

    # Generate list of all (epsilon, kappa) index pairs.
    epsilon_size = map_copy.epsilon_linspace.size
    kappa_size = map_copy.kappa_linspace.size
    tasks = list(product(range(epsilon_size), range(kappa_size)))

    # Determine number of workers.
    workers = n_workers if n_workers is not None else (os.cpu_count() or 1)

    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(map_data,)
    ) as executor:
        # Submit all tasks and use as_completed for robust error handling.
        future_to_indices = {
            executor.submit(process_single, epsilon_idx, kappa_idx): (epsilon_idx, kappa_idx)
            for epsilon_idx, kappa_idx in tasks
        }

        for future in tqdm(as_completed(future_to_indices), total=len(future_to_indices), desc="Computing stability"):
            try:
                epsilon_idx, kappa_idx, result = future.result()
                results.append((epsilon_idx, kappa_idx, result))
            except Exception as exc:
                idx_pair = future_to_indices[future]
                logger.exception("Task for indices %s failed: %s", idx_pair, exc)

    # Update the map copy with computed eigenvalues and eigenvectors.
    for epsilon_idx, kappa_idx, result in results:
        for fp_type_value, eigenvalues in result['eigenvalues'].items():
            map_copy.eigenvalues[epsilon_idx, kappa_idx, fp_type_value, :] = eigenvalues
        for fp_type_value, eigenvectors in result['eigenvectors'].items():
            map_copy.eigenvectors[epsilon_idx, kappa_idx, fp_type_value, :, :] = eigenvectors

    return map_copy
