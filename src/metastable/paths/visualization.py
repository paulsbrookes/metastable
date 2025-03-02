import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate._bvp import BVPResult
from typing import Tuple


from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram


def plot_solution(res: BVPResult, t_guess: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    t_plot = np.linspace(0, t_guess[-1], 1001)
    y0_plot = res.sol(t_plot)[0]
    y1_plot = res.sol(t_plot)[1]
    axes.plot(y0_plot, y1_plot)
    return fig, axes


def plot_parameter_sweeps(fixed_point_map, sweeps, fig=None):
    """
    Plot parameter sweeps on a bifurcation diagram.
    
    Args:
        fixed_point_map: The FixedPointMap containing parameter data
        sweeps: The parameter sweep index pairs
        fig: Optional existing plotly figure. If None, a new figure will be created.
        
    Returns:
        The updated figure with parameter sweeps
    """
    # Create bifurcation diagram if no figure is provided
    if fig is None:
        fig = plot_bifurcation_diagram(fixed_point_map)
    
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