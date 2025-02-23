import numpy as np
import plotly.graph_objects as go
from metastable.map.map import FixedPointMap

def plot_bistable_region(map: FixedPointMap, filename: str):
    """
    Create an interactive plot of the bistable region.
    
    Args:
        map: FixedPointMap instance containing the fixed point data
        filename: Path to save the HTML plot
    """
    # Create the figure
    fig = go.Figure()
    
    # Add heatmap of bistable region
    fig.add_trace(
        go.Heatmap(
            z=map.bistable_region.astype(int),
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorscale=[[0, 'white'], [1, 'rgb(0,0,255)']],  # White for monostable, blue for bistable
            showscale=False,
            name='Bistable region'
        )
    )
    
    # Add bifurcation lines
    lower_line, upper_line = map.bifurcation_lines
    if lower_line.size > 0:  # Only add if bifurcation lines exist
        fig.add_trace(
            go.Scatter(
                x=lower_line[1],  # kappa values
                y=lower_line[0],  # epsilon values
                mode='lines',
                line=dict(color='red', width=2),
                name='Lower bifurcation'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=upper_line[1],  # kappa values
                y=upper_line[0],  # epsilon values
                mode='lines',
                line=dict(color='red', width=2),
                name='Upper bifurcation'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Bistable Region Map',
        xaxis_title='κ',
        yaxis_title='ε',
        height=600,
        width=800,
    )
    
    # Save the plot
    fig.write_html(filename)

if __name__ == "__main__":
    # Load the map with fixed points
    map = FixedPointMap.load("map-with-stability-no-imag.npz")
    
    # Create the plot
    plot_bistable_region(map, "bistable_region.html")