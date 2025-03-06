from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
# Import the plotting function from epsilon_sweep
from metastable.paths.visualization import plot_parameter_sweeps
from metastable.action.calculate import integrand_func
import numpy as np
import plotly.graph_objects as go
from metastable.paths.parameter_utils import prepare_saddle_and_stable_points, IndexPair, extract_params
from metastable.manifold_inverses import calculate_manifold_inverses
from metastable.paths.boundary_conditions.boundary_conditions_alt import calculate_jacobian_properties

if __name__ == "__main__":

    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/archive-source-of-disagreement/0/output_map_with_actions.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    epsilon_idx = 380
    kappa_idx = 114
    index_pair = IndexPair(
        epsilon_idx=epsilon_idx,
        kappa_idx=kappa_idx,
    )
    params = extract_params(fixed_point_map, index_pair)
    keldysh_saddle_point, keldysh_stable_point = prepare_saddle_and_stable_points(
        fixed_point_map, index_pair, FixedPointType.BRIGHT
    )
    # print(keldysh_saddle_point)
    # print(keldysh_stable_point)

    _, saddle_point_unstable_manifold_inverse = calculate_manifold_inverses(
        keldysh_saddle_point, params
    )
    stable_point_stable_manifold_inverse, _ = calculate_manifold_inverses(
        keldysh_stable_point, params
    )


    # Calculate Jacobian properties at the saddle point
    saddle_eigenvalues, saddle_eigenvectors, saddle_recip_eigenvectors = calculate_jacobian_properties(
        keldysh_saddle_point, params
    )
    
    # Calculate Jacobian properties at the stable point
    stable_eigenvalues, stable_eigenvectors, stable_recip_eigenvectors = calculate_jacobian_properties(
        keldysh_stable_point, params
    )

    # Extract unstable manifold inverse from saddle point (positive real eigenvalues)
    unstable_indices = np.where(np.real(saddle_eigenvalues) > 0)[0]
    saddle_point_unstable_manifold_inverse_alt = saddle_recip_eigenvectors[:, unstable_indices]
    
    # Extract stable manifold inverse from stable point (negative real eigenvalues)
    stable_indices = np.where(np.real(stable_eigenvalues) < 0)[0]
    stable_point_stable_manifold_inverse_alt = stable_recip_eigenvectors[:, stable_indices]

    ...


