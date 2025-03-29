from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
from metastable.action.map import map_actions
# Import the plotting function from epsilon_sweep
from metastable.paths.visualization import plot_parameter_sweeps
from metastable.paths.boundary_conditions.boundary_conditions_lock import BoundaryLockParams
import numpy as np


if __name__ == "__main__":

    # Load the fixed point map
    map_path = Path(__file__).parent.parent.parent.parent / "fixed_points/examples/map-with-stability.npz"
    fixed_point_map = FixedPointMap.load(map_path)

    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    # Choose an epsilon index for the kappa cut
    epsilon_idx = 380
    
    # Get the bistable kappa range for this epsilon
    kappa_boundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)
    
    # Generate kappa cuts
    kappa_cuts = generate_sweep_index_pairs(kappa_boundaries, bright_sweep_fraction=0.4, dim_sweep_fraction=0.95)
    
    # Get the actual epsilon value from index
    epsilon_value = fixed_point_map.epsilon_linspace[epsilon_idx]
    
    # If kappa_boundary contains valid index pairs, plot them
    if kappa_boundaries.dim_saddle is not None:
        kappa_start = fixed_point_map.kappa_linspace[kappa_boundaries.dim_saddle.kappa_idx]
        kappa_end = fixed_point_map.kappa_linspace[kappa_boundaries.bright_saddle.kappa_idx]
        fig.add_scatter(
            x=[kappa_start, kappa_end], 
            y=[epsilon_value, epsilon_value], 
            mode='markers', 
            marker=dict(size=10, color='red'), 
            name='Bistable Range (κ)'
        )
    
    # Use the plot_parameter_sweeps function instead of manually plotting
    fig = plot_parameter_sweeps(fixed_point_map, kappa_cuts, fig)
    
    # Display the plot
    fig.show()
    
    # Configure boundary lock parameters for bright fixed point
    bright_lock_params = BoundaryLockParams(
        stable_threshold=1e-2,
        stable_linear_coefficient=0.0,
        saddle_threshold=1e-2,
        saddle_linear_coefficient=0.0
    )
    
    # Configure boundary lock parameters for dim fixed point
    dim_lock_params = BoundaryLockParams(
        stable_threshold=1e-2,
        stable_linear_coefficient=0.0,
        saddle_threshold=1e-2,
        saddle_linear_coefficient=0.0
    )
    
    output_path = Path(__file__).parent / "sweep"

    # Map switching paths for bright fixed point
    path_results_bright = map_switching_paths(
        fixed_point_map, 
        kappa_cuts.bright_saddle, 
        output_path,
        t_end=10.0,
        endpoint_type=FixedPointType.BRIGHT,
        lock_params=bright_lock_params,
        max_nodes=1000000,
        tol=1e-3
    )
    
    # Map switching paths for dim fixed point
    path_results_dim = map_switching_paths(
        fixed_point_map, 
        kappa_cuts.dim_saddle, 
        output_path, 
        t_end=10.0,
        endpoint_type=FixedPointType.DIM,
        lock_params=dim_lock_params
    )
    
    # Calculate actions for all switching paths
    fixed_point_map = FixedPointMap.load(output_path / "map.npz")
    fixed_point_map_with_actions = map_actions(fixed_point_map)
    fixed_point_map_with_actions.save(output_path / "map.npz")

