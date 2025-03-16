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
        html.Div([  # New container for action calculation toggle
            html.Label("Show Action Calculation:"),
            dcc.RadioItems(
                id='action-toggle',
                options=[
                    {'label': 'Yes', 'value': True},
                    {'label': 'No', 'value': False}
                ],
                value=True,
                inline=True,
                labelStyle={'margin-right': '15px'},
                style={'display': 'inline-block', 'margin': '10px auto'}
            )
        ], style={'text-align': 'center'}),
        dcc.Graph(id='interactive-plot', style={'height': '600px'}),
    ]),
    html.Div([
        html.P("Click on any dot to see the corresponding path visualization.", 
               style={'font-style': 'italic', 'margin-top': '10px'})
    ])
])

# Define callback to update the figure
@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('interactive-plot', 'clickData'),
     Input('path-type-toggle', 'value'),
     Input('action-toggle', 'value')]  # Add the new input
)
def update_figure(click_data, path_type_slider, show_action):
    # Load the fixed point map
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/paths/sweeps/kappa/sweep_1/output_map.npz"
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
    
    # Determine number of rows based on whether action is shown
    if show_action:
        num_rows = 3
        row_heights = [0.4, 0.3, 0.3]
        subplot_titles = ('Map of Available Paths', 'Action Integrand vs Time', 'Path First Coordinate vs Time')
    else:
        num_rows = 2
        row_heights = [0.6, 0.4]
        subplot_titles = ('Map of Available Paths', 'Path First Coordinate vs Time')
    
    # Create a figure with appropriate subplots
    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        specs=[[{"type": "scatter"}]] * num_rows
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
    
    # Initialize empty traces for the integrand plot and path coordinate plot
    t_points = []
    integrand_values = []
    path_x_values = []
    fixed_point_markers_x = []
    fixed_point_markers_y = []
    fixed_point_labels = []
    
    # If a point was clicked, update the plots
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
            # If a bifurcation line was clicked, don't update the plots
            valid_point = False
        
        # Get the path result for these indices if a valid path type was selected
        if valid_point:
            path_result = fixed_point_map.path_results[epsilon_idx, kappa_idx, path_type]
            
            if path_result is not None:
                # Calculate the time points
                t_max = path_result.x[-1]
                t_points = np.linspace(0, t_max, 500)
                
                # Calculate the integrand values if action calculation is enabled
                if show_action:
                    integrand_values = [integrand_func(t, path_result) for t in t_points]
                
                # Extract the first coordinate of the path
                # Interpolate the path at the same time points
                path_x_values = [path_result.sol(t)[0] for t in t_points]
                
                # Get the fixed point values from the map
                if path_type == PathType.BRIGHT_TO_SADDLE.value:
                    start_point_type = FixedPointType.BRIGHT.value
                    end_point_type = FixedPointType.SADDLE.value
                else:  # DIM_TO_SADDLE
                    start_point_type = FixedPointType.DIM.value
                    end_point_type = FixedPointType.SADDLE.value
                
                # Get the fixed point coordinates
                start_x = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, start_point_type, 0]
                end_x = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, end_point_type, 0]
                
                # Create markers for the fixed points
                fixed_point_markers_x = [0, t_max]  # Time points for start and end
                fixed_point_markers_y = [start_x, end_x]  # x-coordinate values at fixed points
                
                # Create labels for the markers
                start_label = "Bright" if start_point_type == FixedPointType.BRIGHT.value else "Dim"
                fixed_point_labels = [f"{start_label} State", "Saddle"]
                
                # Get path type name for the title
                path_type_name = "Bright to Saddle" if path_type == PathType.BRIGHT_TO_SADDLE.value else "Dim to Saddle"
                
                # Update subplot titles based on whether action is shown
                annotations = [
                    dict(
                        text=f'Map of Available {path_type_name} Paths',
                        x=0.5, y=1.0,
                        xref='paper', yref='paper',
                        showarrow=False
                    )
                ]
                
                if show_action:
                    annotations.append(
                        dict(
                            text=f"{path_type_name} Action Integrand (ε={fixed_point_map.epsilon_linspace[epsilon_idx]:.4f}, κ={fixed_point_map.kappa_linspace[kappa_idx]:.4f})",
                            x=0.5, y=0.63,
                            xref='paper', yref='paper',
                            showarrow=False
                        )
                    )
                    annotations.append(
                        dict(
                            text=f"{path_type_name} Path First Coordinate (ε={fixed_point_map.epsilon_linspace[epsilon_idx]:.4f}, κ={fixed_point_map.kappa_linspace[kappa_idx]:.4f})",
                            x=0.5, y=0.3,
                            xref='paper', yref='paper',
                            showarrow=False
                        )
                    )
                else:
                    annotations.append(
                        dict(
                            text=f"{path_type_name} Path First Coordinate (ε={fixed_point_map.epsilon_linspace[epsilon_idx]:.4f}, κ={fixed_point_map.kappa_linspace[kappa_idx]:.4f})",
                            x=0.5, y=0.4,
                            xref='paper', yref='paper',
                            showarrow=False
                        )
                    )
                
                fig.update_layout(annotations=annotations)
                
                # Print debug info
                print(f"Selected point: epsilon_idx={epsilon_idx}, kappa_idx={kappa_idx}, path_type={path_type}")
    
    # Add the integrand trace if action calculation is enabled
    if show_action:
        fig.add_trace(
            go.Scatter(
                x=t_points, 
                y=integrand_values, 
                mode='lines',
                name='Action Integrand'
            ),
            row=2, col=1
        )
        
        # Add the path first coordinate trace in the third row
        path_row = 3
    else:
        # Add the path first coordinate trace in the second row
        path_row = 2
    
    # Add the path first coordinate trace
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=path_x_values, 
            mode='lines',
            name='Path First Coordinate'
        ),
        row=path_row, col=1
    )
    
    # Add markers for fixed points
    fig.add_trace(
        go.Scatter(
            x=fixed_point_markers_x,
            y=fixed_point_markers_y,
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                symbol='star'
            ),
            text=fixed_point_labels,
            textposition="top center",
            name='Fixed Points'
        ),
        row=path_row, col=1
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        hovermode='closest',
        height=900 if show_action else 700,  # Adjust height based on number of plots
        showlegend=True,
        uirevision='constant'  # Keep zoom level when switching between path types
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='κ', row=1, col=1)
    fig.update_yaxes(title_text='ε', row=1, col=1)
    
    if show_action:
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Integrand Value', row=2, col=1)
        fig.update_xaxes(title_text='Time', row=3, col=1)
        fig.update_yaxes(title_text='x₁', row=3, col=1)
    else:
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='x₁', row=2, col=1)
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)