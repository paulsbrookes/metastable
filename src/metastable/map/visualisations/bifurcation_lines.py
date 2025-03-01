import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_bifurcation_diagram(fixed_point_map, title='Bifurcation Diagram', height=600, width=800):
    """
    Create a bifurcation diagram from a FixedPointMap object.
    
    Parameters
    ----------
    fixed_point_map : FixedPointMap
        The fixed point map containing bifurcation data
    title : str, optional
        Title for the plot, by default 'Bifurcation Diagram'
    height : int, optional
        Height of the plot in pixels, by default 600
    width : int, optional
        Width of the plot in pixels, by default 800
        
    Returns
    -------
    plotly.graph_objects.Figure
        The bifurcation diagram figure
    """    
    # Get the bifurcation lines
    lower_line, upper_line = fixed_point_map.bifurcation_lines
    
    # Get parameter values
    epsilon_values = fixed_point_map.epsilon_linspace
    kappa_values = fixed_point_map.kappa_linspace
    
    # Create a figure with secondary y-axis (secondary x-axis will be added manually)
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    # Plot the bifurcation diagram with parameter values
    if lower_line.size > 0 and upper_line.size > 0:
        # Lower bifurcation line
        fig.add_trace(
            go.Scatter(
                x=lower_line[1],  # kappa values
                y=lower_line[0],  # epsilon values
                mode='lines',
                name='Lower Bifurcation Line',
                line=dict(color='blue', width=2),
                hovertemplate='κ: %{x:.4f}<br>ε: %{y:.4f}<br>κ index: %{customdata[0]}<br>ε index: %{customdata[1]}<extra></extra>',
                customdata=[[np.argmin(np.abs(fixed_point_map.kappa_linspace - kap)), 
                             np.argmin(np.abs(fixed_point_map.epsilon_linspace - eps))] 
                            for kap, eps in zip(lower_line[1], lower_line[0])]
            )
        )
        
        # Upper bifurcation line
        fig.add_trace(
            go.Scatter(
                x=upper_line[1],  # kappa values
                y=upper_line[0],  # epsilon values
                mode='lines',
                name='Upper Bifurcation Line',
                line=dict(color='red', width=2),
                hovertemplate='κ: %{x:.4f}<br>ε: %{y:.4f}<br>κ index: %{customdata[0]}<br>ε index: %{customdata[1]}<extra></extra>',
                customdata=[[np.argmin(np.abs(fixed_point_map.kappa_linspace - kap)), 
                             np.argmin(np.abs(fixed_point_map.epsilon_linspace - eps))] 
                            for kap, eps in zip(upper_line[1], upper_line[0])]
            )
        )
        
        # Fill the bistable region
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lower_line[1], upper_line[1][::-1]]),
                y=np.concatenate([lower_line[0], upper_line[0][::-1]]),
                fill='toself',
                fillcolor='rgba(173, 216, 230, 0.3)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Bistable Region',
                showlegend=True,
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white',
        xaxis=dict(
            title="Damping Rate (κ)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Drive Amplitude (ε)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickmode='array',
        )
    )
    
    return fig
