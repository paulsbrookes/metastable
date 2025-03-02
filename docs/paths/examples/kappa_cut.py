from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths.map import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
import numpy as np


if __name__ == "__main__":
    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    # Choose an epsilon index for the kappa cut
    epsilon_idx = 380
    
    # Get the bistable kappa range for this epsilon
    kappa_boundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)
    
    # Generate kappa cuts
    kappa_cuts = generate_sweep_index_pairs(kappa_boundaries)
    
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
    
    # Plot the kappa cuts
    if kappa_cuts.bright_saddle and kappa_cuts.dim_saddle:
        # Extract kappa and epsilon values for the bright_saddle cut
        bright_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in kappa_cuts.bright_saddle]
        bright_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in kappa_cuts.bright_saddle]
        
        # Extract kappa and epsilon values for the dim_saddle cut
        dim_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in kappa_cuts.dim_saddle]
        dim_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in kappa_cuts.dim_saddle]
        
        # Add the kappa cuts to the plot
        fig.add_scatter(
            x=bright_cut_kappas, 
            y=bright_cut_epsilons, 
            mode='lines+markers', 
            marker=dict(size=5), 
            line=dict(color='orange', dash='dot'), 
            name='Bright-Saddle Cut (κ)'
        )
        
        fig.add_scatter(
            x=dim_cut_kappas, 
            y=dim_cut_epsilons, 
            mode='lines+markers', 
            marker=dict(size=5), 
            line=dict(color='cyan', dash='dot'), 
            name='Dim-Saddle Cut (κ)'
        )
    
    # Display the plot
    fig.show()

    # Set up output path for switching paths
    output_path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/4")
    
    # Map switching paths for bright fixed point
    bright_results = map_switching_paths(
        fixed_point_map, 
        kappa_cuts.bright_saddle, 
        output_path/"bright_kappa_cut", 
        endpoint_type=FixedPointType.BRIGHT
    )
    
    # Map switching paths for dim fixed point
    dim_results = map_switching_paths(
        fixed_point_map, 
        kappa_cuts.dim_saddle, 
        output_path/"dim_kappa_cut", 
        endpoint_type=FixedPointType.DIM
    )
