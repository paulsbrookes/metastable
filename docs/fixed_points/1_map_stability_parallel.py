import numpy as np
import matplotlib.pyplot as plt
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
from metastable.extend_to_keldysh import extend_to_keldysh_state
from tqdm import tqdm
from itertools import product
from concurrent.futures import ProcessPoolExecutor

# Global variable to hold shared map data for each worker
global_map_data = None

def init_worker(map_data):
    """
    Initializer for each worker process.
    Loads the shared map_data into a global variable.
    """
    global global_map_data
    global_map_data = map_data

def compute_stability(classical_fixed_point: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors for a given classical fixed point.
    """
    if np.any(np.isnan(classical_fixed_point)):
        return np.full(4, np.nan, dtype=complex), np.full((4, 4), np.nan, dtype=complex)
    
    full_fixed_point = extend_to_keldysh_state(classical_fixed_point)
    eom = EOM(params)
    jacobian = eom.jacobian_func(full_fixed_point)
    return np.linalg.eig(jacobian)

def process_single(epsilon_idx, kappa_idx):
    """
    Process a single (epsilon, kappa) pair using the globally loaded map data.
    """
    results = {
        'eigenvalues': {},
        'eigenvectors': {}
    }
    
    # Loop over all fixed point types
    for fp_type in FixedPointType:
        classical_fixed_point = global_map_data['fixed_points'][epsilon_idx, kappa_idx, fp_type.value, :]
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
    
    return epsilon_idx, kappa_idx, results

def worker_chunk(chunk):
    """
    Process a chunk of (epsilon, kappa) pairs.
    """
    chunk_results = []
    for epsilon_idx, kappa_idx in chunk:
        result = process_single(epsilon_idx, kappa_idx)
        chunk_results.append(result)
    return chunk_results

def chunkify(lst, chunk_size):
    """
    Yield successive chunks of size chunk_size from lst.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

if __name__ == '__main__':
    # Load the fixed point map
    map = FixedPointMap.load(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map.npz"
    )
    
    # Prepare the shared map data for workers
    map_data = {
        'fixed_points': map.fixed_points,
        'epsilon_linspace': map.epsilon_linspace,
        'kappa_linspace': map.kappa_linspace,
        'chi': map.chi,
        'delta': map.delta,
    }
    
    # Create list of all (epsilon, kappa) index pairs
    tasks = [(i, j) for i, j in product(
        range(map.epsilon_linspace.size),
        range(map.kappa_linspace.size)
    )]
    total_tasks = len(tasks)
    
    # Define number of workers and chunk size (aiming for ~4 chunks per worker)
    workers = 20
    chunk_size = max(1, total_tasks // (workers * 4))
    
    # Create chunks of tasks
    chunks = list(chunkify(tasks, chunk_size))
    
    results = []
    # Use ProcessPoolExecutor with an initializer to load map_data once per worker
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(map_data,)) as executor:
        # Map each chunk to the worker_chunk function with a progress bar
        for chunk_result in tqdm(executor.map(worker_chunk, chunks), total=len(chunks), desc="Computing stability"):
            results.extend(chunk_result)
    
    # Update the map object with the computed eigenvalues and eigenvectors
    for epsilon_idx, kappa_idx, result in results:
        for fp_type_value, eigenvalues in result['eigenvalues'].items():
            map.eigenvalues[epsilon_idx, kappa_idx, fp_type_value, :] = eigenvalues
        for fp_type_value, eigenvectors in result['eigenvectors'].items():
            map.eigenvectors[epsilon_idx, kappa_idx, fp_type_value, :, :] = eigenvectors
    
    # Save the updated map
    map.save("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")
