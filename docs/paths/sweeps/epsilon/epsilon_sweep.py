from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_epsilon_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
from metastable.action.map import map_actions
# Import the plotting function from visualization module
from metastable.paths.visualization import plot_parameter_sweeps
from metastable.paths.boundary_conditions.boundary_conditions_lock import BoundaryLockParams
import numpy as np


if __name__ == "__main__":
    # Load the fixed point map
    map_path = Path(__file__).parent.parent.parent.parent.parent / "fixed_points/examples/map-with-stability.npz"
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    # Choose a kappa index for the epsilon cut
    kappa_idx = 150
    
    # Get the bistable epsilon range for this kappa
    epsilon_boundaries = get_bistable_epsilon_range(fixed_point_map.bistable_region, kappa_idx)
    
    # Generate epsilon sweeps
    epsilon_sweeps = generate_sweep_index_pairs(epsilon_boundaries, sweep_fraction=0.4)
    
    # Get the actual kappa value from index
    kappa_value = fixed_point_map.kappa_linspace[kappa_idx]
    
    # If epsilon_boundary contains valid index pairs, plot them
    if epsilon_boundaries.bright_saddle is not None:
        epsilon_start = fixed_point_map.epsilon_linspace[epsilon_boundaries.bright_saddle.epsilon_idx]
        epsilon_end = fixed_point_map.epsilon_linspace[epsilon_boundaries.dim_saddle.epsilon_idx]
        fig.add_scatter(
            x=[kappa_value, kappa_value], 
            y=[epsilon_start, epsilon_end], 
            mode='markers', 
            marker=dict(size=10, color='blue'), 
            name='Bistable Range (ε)'
        )
    
    # Use the plot_parameter_sweeps function instead of manually plotting
    fig = plot_parameter_sweeps(fixed_point_map, epsilon_sweeps, fig)
    
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
        stable_linear_coefficient=1.0,
        saddle_threshold=1e-2,
        saddle_linear_coefficient=1.0
    )

    output_path = Path(__file__).parent / "sweep"
    
    # Map switching paths for bright fixed point
    path_results_bright = map_switching_paths(
        fixed_point_map,
        epsilon_sweeps.bright_saddle, 
        output_path,
        t_end=11.0,
        endpoint_type=FixedPointType.BRIGHT,
        lock_params=bright_lock_params,
        tol=1e-3,
        max_nodes=1000000
    )
    
    # Map switching paths for dim fixed point
    path_results_dim = map_switching_paths(
        fixed_point_map, 
        epsilon_sweeps.dim_saddle, 
        output_path,
        t_end=11.0,
        endpoint_type=FixedPointType.DIM,
        lock_params=dim_lock_params,
        tol=1e-2,
        max_nodes=1000000
    )

    # Calculate actions for all switching paths
    fixed_point_map = FixedPointMap.load(output_path / "map.npz")
    fixed_point_map_with_actions = map_actions(fixed_point_map)
    fixed_point_map_with_actions.save(output_path / "map.npz")