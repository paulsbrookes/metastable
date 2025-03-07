from pathlib import Path
from metastable.map.map import FixedPointMap, FixedPointType, PathType
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
from metastable.paths import (
    get_bistable_kappa_range, 
    generate_sweep_index_pairs,
    map_switching_paths
)
# Import the plotting function from epsilon_sweep
from metastable.paths.visualization import plot_parameter_sweeps
from metastable.action.calculate import integrand_func
from metastable.rescaled import (
    calculate_kappa_rescaled,
    calculate_beta_limits,
    map_beta_to_epsilon,
)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Create a Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive Path Analysis"),
    html.Div([
        html.Label("Select Path Type:"),
        html.Div([  # Container for better alignment
            dcc.RadioItems(
                id='path-type-toggle',
                options=[
                    {'label': 'Bright to Saddle', 'value': 0},
                    {'label': 'Dim to Saddle', 'value': 1}
                ],
                value=0,
                inline=False,
                labelStyle={'display': 'block', 'margin-bottom': '5px', 'text-align': 'left'},
                style={'display': 'inline-block', 'margin': '10px auto'}
            )
        ], style={'text-align': 'center'}),
        dcc.Graph(id='interactive-plot', style={'height': '600px'}),
    ]),
    html.Div([
        html.P("Click on any dot to see the corresponding action integrand plot.", 
               style={'font-style': 'italic', 'margin-top': '10px'})
    ])
])

# Define callback to update the figure
@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('interactive-plot', 'clickData'),
     Input('path-type-toggle', 'value')]
)
def update_figure(click_data, path_type_slider):
    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/8/output_map_with_actions.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create a visualization of where path_results exist for both path types
    epsilon_indices_bright, kappa_indices_bright = np.where(fixed_point_map.path_results[:, :, PathType.BRIGHT_TO_SADDLE.value] != None)
    epsilon_indices_dim, kappa_indices_dim = np.where(fixed_point_map.path_results[:, :, PathType.DIM_TO_SADDLE.value] != None)
    
    # Get the actual epsilon and kappa values for both path types
    epsilon_values_bright = fixed_point_map.epsilon_linspace[epsilon_indices_bright]
    kappa_values_bright = fixed_point_map.kappa_linspace[kappa_indices_bright]
    
    epsilon_values_dim = fixed_point_map.epsilon_linspace[epsilon_indices_dim]
    kappa_values_dim = fixed_point_map.kappa_linspace[kappa_indices_dim]
    
    # Create a figure with 2 subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Map of Available Paths', 'Action Integrand vs Time'),
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Calculate bifurcation lines
    delta = fixed_point_map.delta
    kappa_rescaled_linspace = calculate_kappa_rescaled(
        fixed_point_map.kappa_linspace, delta=delta
    )
    beta_limits_array = calculate_beta_limits(kappa_rescaled_linspace)
    epsilon_limits_array = map_beta_to_epsilon(
        beta_limits_array, chi=fixed_point_map.chi, delta=delta
    )
    
    # Create arrays for the bifurcation curves
    kappa_values_for_bifurcation = fixed_point_map.kappa_linspace
    epsilon_unstable_bright = epsilon_limits_array[0]
    epsilon_unstable_dim = epsilon_limits_array[1]
    
    # Add bifurcation lines
    fig.add_trace(
        go.Scatter(
            x=kappa_values_for_bifurcation,
            y=epsilon_unstable_bright,
            mode="lines",
            name="Unstable-Bright",
            line=dict(color="red", width=3),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=kappa_values_for_bifurcation,
            y=epsilon_unstable_dim,
            mode="lines",
            name="Unstable-Dim",
            line=dict(color="blue", width=3),
        ),
        row=1, col=1
    )
    
    # Determine which path type to display based on slider value
    if path_type_slider == 0:  # Bright to Saddle
        # Add scatter plot for points where BRIGHT_TO_SADDLE paths exist
        fig.add_trace(
            go.Scatter(
                x=kappa_values_bright,
                y=epsilon_values_bright,
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.7
                ),
                name='Bright to Saddle Paths',
                hovertemplate='κ: %{x:.4f}<br>ε: %{y:.4f}<extra></extra>',
                customdata=np.column_stack((epsilon_indices_bright, kappa_indices_bright, 
                                            np.full(len(epsilon_indices_bright), PathType.BRIGHT_TO_SADDLE.value)))
            ),
            row=1, col=1
        )
    else:  # Dim to Saddle
        # Add scatter plot for points where DIM_TO_SADDLE paths exist
        fig.add_trace(
            go.Scatter(
                x=kappa_values_dim,
                y=epsilon_values_dim,
                mode='markers',
                marker=dict(
                    size=8,
                    color='green',
                    opacity=0.7
                ),
                name='Dim to Saddle Paths',
                hovertemplate='κ: %{x:.4f}<br>ε: %{y:.4f}<extra></extra>',
                customdata=np.column_stack((epsilon_indices_dim, kappa_indices_dim, 
                                            np.full(len(epsilon_indices_dim), PathType.DIM_TO_SADDLE.value)))
            ),
            row=1, col=1
        )
    
    # Initialize empty trace for the integrand plot
    t_points = []
    integrand_values = []
    
    # If a point was clicked, update the integrand plot
    if click_data:
        # Print the click_data to debug
        print("Click data:", click_data)
        
        # Access the point index and curve number
        point_idx = click_data['points'][0]['pointIndex']
        curve_number = click_data['points'][0]['curveNumber']
        
        # Determine which dataset to use based on the curve number and slider value
        valid_point = False
        
        if curve_number == 2:  # This is now the scatter plot (after the two bifurcation lines)
            if path_type_slider == 0:  # Bright to Saddle is displayed
                # Get the epsilon and kappa indices directly from the arrays
                epsilon_idx = epsilon_indices_bright[point_idx]
                kappa_idx = kappa_indices_bright[point_idx]
                path_type = PathType.BRIGHT_TO_SADDLE.value
                valid_point = True
            elif path_type_slider == 1:  # Dim to Saddle is displayed
                # Get the epsilon and kappa indices directly from the arrays
                epsilon_idx = epsilon_indices_dim[point_idx]
                kappa_idx = kappa_indices_dim[point_idx]
                path_type = PathType.DIM_TO_SADDLE.value
                valid_point = True
        else:
            # If a bifurcation line was clicked, don't update the integrand plot
            valid_point = False
        
        # Get the path result for these indices if a valid path type was selected
        if valid_point:
            path_result = fixed_point_map.path_results[epsilon_idx, kappa_idx, path_type]
            
            if path_result is not None:
                # Calculate the integrand values
                t_max = path_result.x[-1]
                t_points = np.linspace(0, t_max, 500)
                integrand_values = [integrand_func(t, path_result) for t in t_points]
                
                # Get path type name for the title
                path_type_name = "Bright to Saddle" if path_type == PathType.BRIGHT_TO_SADDLE.value else "Dim to Saddle"
                
                # Update subplot title
                fig.update_layout(
                    annotations=[
                        dict(
                            text='Map of Available Paths',
                            x=0.25, y=1.0,
                            xref='paper', yref='paper',
                            showarrow=False
                        ),
                        dict(
                            text=f"{path_type_name} Action Integrand (ε={fixed_point_map.epsilon_linspace[epsilon_idx]:.4f}, κ={fixed_point_map.kappa_linspace[kappa_idx]:.4f})",
                            x=0.75, y=1.0,
                            xref='paper', yref='paper',
                            showarrow=False
                        )
                    ]
                )
                
                # Print debug info
                print(f"Selected point: epsilon_idx={epsilon_idx}, kappa_idx={kappa_idx}, path_type={path_type}")
    
    # Add the integrand trace
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=integrand_values, 
            mode='lines',
            name='Action Integrand'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=600,
        showlegend=True,
        uirevision='constant'  # Keep zoom level when switching between path types
    )
    
    # Update subplot title based on slider value
    path_type_title = "Bright to Saddle" if path_type_slider == 0 else "Dim to Saddle"
    fig.update_layout(
        annotations=[
            dict(
                text=f'Map of Available {path_type_title} Paths',
                x=0.25, y=1.0,
                xref='paper', yref='paper',
                showarrow=False
            ),
            dict(
                text='Action Integrand vs Time',
                x=0.75, y=1.0,
                xref='paper', yref='paper',
                showarrow=False
            )
        ]
    )
    
    fig.update_xaxes(title_text='κ', row=1, col=1)
    fig.update_yaxes(title_text='ε', row=1, col=1)
    fig.update_xaxes(title_text='Time', row=1, col=2)
    fig.update_yaxes(title_text='Integrand Value', row=1, col=2)
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)