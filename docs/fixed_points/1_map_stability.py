import numpy as np
import matplotlib.pyplot as plt
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
from metastable.extend_to_keldysh import extend_to_keldysh_state
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots

map = FixedPointMap.load(
    "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-backup.npz"
)

def compute_stability(classical_fixed_point: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors for a given classical fixed point.
    
    Args:
        classical_fixed_point: Array of shape (2,) containing the classical fixed point
        params: Params object containing system parameters
        
    Returns:
        tuple containing:
            eigenvalues: Complex array of shape (4,) containing eigenvalues
            eigenvectors: Complex array of shape (4, 4) containing eigenvectors as columns
    """
    if np.any(np.isnan(classical_fixed_point)):
        return np.full(4, np.nan, dtype=complex), np.full((4, 4), np.nan, dtype=complex)
    
    full_fixed_point = extend_to_keldysh_state(classical_fixed_point)
    eom = EOM(params)
    jacobian = eom.jacobian_func(full_fixed_point)
    return np.linalg.eig(jacobian)

# Main loop with single progress bar
total_points = map.epsilon_linspace.size * map.kappa_linspace.size
with tqdm(total=total_points, desc="Computing stability") as pbar:
    for epsilon_idx in range(map.epsilon_linspace.size):
        for kappa_idx in range(map.kappa_linspace.size):
            # Loop over all fixed point types
            for fp_type in FixedPointType:
                classical_fixed_point = map.fixed_points[epsilon_idx, kappa_idx, fp_type.value, :]
                
                # Skip if fixed point is all nan
                if np.all(np.isnan(classical_fixed_point)):
                    continue
                    
                params = Params(
                    epsilon=map.epsilon_linspace[epsilon_idx],
                    kappa=map.kappa_linspace[kappa_idx],
                    chi=map.chi,
                    delta=map.delta,
                )
                
                eigenvalues, eigenvectors = compute_stability(classical_fixed_point, params)
                map.eigenvalues[epsilon_idx, kappa_idx, fp_type.value, :] = eigenvalues
                map.eigenvectors[epsilon_idx, kappa_idx, fp_type.value, :, :] = eigenvectors
            
            pbar.update(1)

# Save the updated map with eigenvalues and eigenvectors
map.save("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")