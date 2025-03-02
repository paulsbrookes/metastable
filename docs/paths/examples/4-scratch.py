from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths.map import get_bistable_kappa_range, get_bistable_epsilon_range, IndexPair, BistableBoundaries, Sweeps, generate_sweep_index_pairs
import numpy as np
from typing import Optional, Tuple
from metastable.paths.map import map_switching_paths



# Example usage:
if __name__ == "__main__":
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)

    epsilon_idx = 450
    kappa_boundaries: BistableBoundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)
    
    # Add points to the plot
    # Get the actual epsilon and kappa values from indices
    epsilon_value = fixed_point_map.epsilon_linspace[epsilon_idx]
    
    # If kappa_boundary contains valid index pairs, plot them
    if kappa_boundaries.dim_saddle is not None:
        kappa_start = fixed_point_map.kappa_linspace[kappa_boundaries.dim_saddle.kappa_idx]
        kappa_end = fixed_point_map.kappa_linspace[kappa_boundaries.bright_saddle.kappa_idx]
        fig.add_scatter(x=[kappa_start, kappa_end], y=[epsilon_value, epsilon_value], 
                        mode='markers', marker=dict(size=10, color='red'), name='Bistable Range (κ)')
    
    kappa_idx = 300
    epsilon_boundaries: BistableBoundaries = get_bistable_epsilon_range(fixed_point_map.bistable_region, kappa_idx)
    epsilon_cuts: Sweeps = generate_sweep_index_pairs(epsilon_boundaries)
    
    # Generate kappa cuts
    kappa_cuts: Sweeps = generate_sweep_index_pairs(kappa_boundaries)
    
    # Add points to the plot
    # Get the actual epsilon and kappa values from indices
    kappa_value = fixed_point_map.kappa_linspace[kappa_idx]
    
    # If epsilon_boundary contains valid index pairs, plot them
    if epsilon_boundaries.bright_saddle is not None:
        epsilon_start = fixed_point_map.epsilon_linspace[epsilon_boundaries.bright_saddle.epsilon_idx]
        epsilon_end = fixed_point_map.epsilon_linspace[epsilon_boundaries.dim_saddle.epsilon_idx]
        fig.add_scatter(x=[kappa_value, kappa_value], y=[epsilon_start, epsilon_end], 
                        mode='markers', marker=dict(size=10, color='blue'), name='Bistable Range (ε)')
    
    # Plot the epsilon cuts
    if epsilon_cuts.bright_saddle and epsilon_cuts.dim_saddle:
        # Extract kappa and epsilon values for the bright_saddle cut
        bright_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in epsilon_cuts.bright_saddle]
        bright_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in epsilon_cuts.bright_saddle]
        
        # Extract kappa and epsilon values for the dim_saddle cut
        dim_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in epsilon_cuts.dim_saddle]
        dim_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in epsilon_cuts.dim_saddle]
        
        # Add the cuts to the plot
        fig.add_scatter(x=bright_cut_kappas, y=bright_cut_epsilons, 
                        mode='lines+markers', marker=dict(size=5), 
                        line=dict(color='green', dash='dot'), name='Bright-Saddle Cut (ε)')
        
        fig.add_scatter(x=dim_cut_kappas, y=dim_cut_epsilons, 
                        mode='lines+markers', marker=dict(size=5), 
                        line=dict(color='purple', dash='dot'), name='Dim-Saddle Cut (ε)')
    
    # Plot the kappa cuts
    if kappa_cuts.bright_saddle and kappa_cuts.dim_saddle:
        # Extract kappa and epsilon values for the bright_saddle cut
        bright_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in kappa_cuts.bright_saddle]
        bright_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in kappa_cuts.bright_saddle]
        
        # Extract kappa and epsilon values for the dim_saddle cut
        dim_cut_kappas = [fixed_point_map.kappa_linspace[pair.kappa_idx] for pair in kappa_cuts.dim_saddle]
        dim_cut_epsilons = [fixed_point_map.epsilon_linspace[pair.epsilon_idx] for pair in kappa_cuts.dim_saddle]
        
        # Add the kappa cuts to the plot
        fig.add_scatter(x=bright_cut_kappas, y=bright_cut_epsilons, 
                        mode='lines+markers', marker=dict(size=5), 
                        line=dict(color='orange', dash='dot'), name='Bright-Saddle Cut (κ)')
        
        fig.add_scatter(x=dim_cut_kappas, y=dim_cut_epsilons, 
                        mode='lines+markers', marker=dict(size=5), 
                        line=dict(color='cyan', dash='dot'), name='Dim-Saddle Cut (κ)')
    
    fig.show()

    output_path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output")

    # # Begin mapping switching paths
    # results = map_switching_paths(
    #     fixed_point_map, epsilon_cuts.bright_saddle, output_path, endpoint_type=FixedPointType.BRIGHT
    # )

        # Begin mapping switching paths
    results = map_switching_paths(
        fixed_point_map, epsilon_cuts.dim_saddle, output_path/"2", endpoint_type=FixedPointType.DIM
    )




