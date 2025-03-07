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


if __name__ == "__main__":

    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/1/output_map_with_actions.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    epsilon_idx = 380
    kappa_idx = 150
    path_result = fixed_point_map.path_results[epsilon_idx,kappa_idx,1]
    print(path_result)
    
    # Plot the integrand over time
    # Create time points for plotting
    t_max = path_result.x[-1]
    t_points = np.linspace(0, t_max, 500)
    
    # Calculate integrand values at each time point
    integrand_values = [integrand_func(t, path_result) for t in t_points]
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_points,
        y=integrand_values,
        mode='lines',
        name='Action Integrand'
    ))
    
    # Update layout
    fig.update_layout(
        title='Action Integrand vs Time',
        xaxis_title='Time',
        yaxis_title='Integrand Value',
        template='plotly_white'
    )
    
    # Show the plot
    fig.show()