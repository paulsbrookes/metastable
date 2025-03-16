from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_epsilon_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
# Import the plotting function from visualization module
from metastable.paths.visualization import plot_parameter_sweeps
import numpy as np


if __name__ == "__main__":
    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)

    output_path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/19")
    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    # Choose a kappa index for the epsilon cut
    kappa_idx = 150
    
    # Get the bistable epsilon range for this kappa
    epsilon_boundaries = get_bistable_epsilon_range(fixed_point_map.bistable_region, kappa_idx)
    
    # Generate epsilon sweeps
    epsilon_sweeps = generate_sweep_index_pairs(epsilon_boundaries, sweep_fraction=1.0)
    
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
    
    # Map switching paths for bright fixed point
    path_results_bright = map_switching_paths(
        fixed_point_map,
        epsilon_sweeps.bright_saddle, 
        output_path,
        t_end=11.0,
        endpoint_type=FixedPointType.BRIGHT
    )
    
    # Map switching paths for dim fixed point
    path_results_dim = map_switching_paths(
        fixed_point_map, 
        epsilon_sweeps.dim_saddle, 
        output_path,
        t_end=11.0,
        endpoint_type=FixedPointType.DIM
    )
