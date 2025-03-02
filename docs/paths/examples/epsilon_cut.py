from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_epsilon_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
import numpy as np


def plot_epsilon_sweeps(fig, fixed_point_map, sweeps):
    """
    Plot epsilon cuts on a bifurcation diagram.
    
    Args:
        fig: The plotly figure to add the cuts to
        fixed_point_map: The FixedPointMap containing parameter data
        sweeps: The epsilon sweep index pairs
        
    Returns:
        The updated figure
    """
    if sweeps.bright_saddle and sweeps.dim_saddle:
        # Extract kappa and epsilon values for the bright_saddle cut
        bright_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in sweeps.bright_saddle]
        bright_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in sweeps.bright_saddle]
        
        # Extract kappa and epsilon values for the dim_saddle cut
        dim_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in sweeps.dim_saddle]
        dim_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in sweeps.dim_saddle]
        
        # Add the epsilon cuts to the plot
        fig.add_scatter(
            x=bright_cut_kappas, 
            y=bright_cut_epsilons, 
            mode='lines+markers', 
            marker=dict(size=5), 
            line=dict(color='green', dash='dot'), 
            name='Bright-Saddle Cut (ε)'
        )
        
        fig.add_scatter(
            x=dim_cut_kappas, 
            y=dim_cut_epsilons, 
            mode='lines+markers', 
            marker=dict(size=5), 
            line=dict(color='purple', dash='dot'), 
            name='Dim-Saddle Cut (ε)'
        )
    
    return fig


if __name__ == "__main__":
    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)

    output_path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/13")
    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    # Choose a kappa index for the epsilon cut
    kappa_idx = 60
    
    # Get the bistable epsilon range for this kappa
    epsilon_boundaries = get_bistable_epsilon_range(fixed_point_map.bistable_region, kappa_idx)
    
    # Generate epsilon sweeps
    epsilon_sweeps = generate_sweep_index_pairs(epsilon_boundaries, sweep_fraction=0.15)
    
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
    
    # Plot the epsilon cuts
    fig = plot_epsilon_sweeps(fig, fixed_point_map, epsilon_sweeps)
    
    # Display the plot
    fig.show()

    # Set up output path for switching paths
    
    # Map switching paths for bright fixed point
    path_results_bright = map_switching_paths(
        fixed_point_map,
        epsilon_sweeps.bright_saddle, 
        output_path,
        t_end=7.0,
        endpoint_type=FixedPointType.BRIGHT
    )
    
    # Map switching paths for dim fixed point
    path_results_dim = map_switching_paths(
        fixed_point_map, 
        epsilon_sweeps.dim_saddle, 
        output_path,
        t_end=7.0,
        endpoint_type=FixedPointType.DIM
    )
