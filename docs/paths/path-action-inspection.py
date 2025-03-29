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
    # Header section
    html.Div([
        html.H1("Interactive Path Analysis", 
                style={
                    'textAlign': 'center',
                    'color': '#2c3e50',
                    'marginBottom': '20px',
                    'fontFamily': 'Arial, sans-serif'
                }),
        
        # Documentation section
        html.Div([
            html.H2("About this Tool", 
                    style={
                        'color': '#2c3e50',
                        'fontSize': '1.3em',
                        'marginBottom': '15px'
                    }),
            html.P([
                "This interactive visualization tool helps analyze paths and actions calculated in parameter sweeps. It provides:"
            ], style={'marginBottom': '10px'}),
            html.Ul([
                html.Li("Visual inspection of switching paths between fixed points (bright/dim to saddle)"),
                html.Li("Action calculations and their convergence verification"),
                html.Li("Interactive bifurcation diagram with path availability overlay"),
                html.Li("Detailed time evolution of paths and their action integrands")
            ], style={
                'listStyleType': 'disc',
                'paddingLeft': '30px',
                'marginBottom': '15px'
            }),
            html.P([
                "To use: Enter the path to a sweep output map file (.npz format), select the path type of interest, ",
                "and click on any point in the parameter space to examine the corresponding path details."
            ], style={'marginBottom': '15px'}),
            html.P([
                "The visualization includes bifurcation lines (red/blue) and available paths (dots). ",
                "The action calculation can be toggled to focus on specific aspects of the analysis."
            ], style={'marginBottom': '20px'})
        ], style={
            'backgroundColor': '#fff',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
            'marginBottom': '30px',
            'fontSize': '14px',
            'lineHeight': '1.6',
            'color': '#34495e'
        })
    ], style={'padding': '20px 0'}),

    # Main content container
    html.Div([
        # File input section
        html.Div([
            html.Label("Map File Path:", 
                      style={
                          'fontWeight': 'bold',
                          'color': '#2c3e50',
                          'marginBottom': '8px'
                      }),
            html.Div([
                dcc.Input(
                    id='map-path-input',
                    type='text',
                    placeholder='Enter path to map.npz file',
                    style={
                        'width': '80%',
                        'padding': '8px',
                        'borderRadius': '4px',
                        'border': '1px solid #bdc3c7',
                        'marginRight': '10px'
                    }
                ),
                html.Button(
                    'Load Map',
                    id='load-map-button',
                    n_clicks=0,
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'border': 'none',
                        'padding': '8px 15px',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontWeight': 'bold',
                        'width': '15%'
                    }
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
        ], style={'marginBottom': '30px'}),

        # Controls section
        html.Div([
            # Path type selection
            html.Div([
                html.Label("Select Path Type:", 
                          style={
                              'fontWeight': 'bold',
                              'color': '#2c3e50',
                              'marginBottom': '10px'
                          }),
                dcc.RadioItems(
                    id='path-type-toggle',
                    options=[
                        {'label': 'Bright to Saddle', 'value': 0},
                        {'label': 'Dim to Saddle', 'value': 1}
                    ],
                    value=0,
                    style={'marginBottom': '20px'},
                    className='radio-items',
                    labelStyle={
                        'display': 'block',
                        'marginBottom': '10px',
                        'fontSize': '14px',
                        'color': '#34495e'
                    }
                )
            ], style={'flex': 1}),

            # Action calculation toggle
            html.Div([
                html.Label("Show Action Calculation:", 
                          style={
                              'fontWeight': 'bold',
                              'color': '#2c3e50',
                              'marginBottom': '10px'
                          }),
                dcc.RadioItems(
                    id='action-toggle',
                    options=[
                        {'label': 'Yes', 'value': True},
                        {'label': 'No', 'value': False}
                    ],
                    value=True,
                    inline=True,
                    className='radio-items',
                    labelStyle={
                        'marginRight': '20px',
                        'fontSize': '14px',
                        'color': '#34495e'
                    }
                )
            ], style={'flex': 1})
        ], style={'display': 'flex', 'marginBottom': '20px'}),

        # Graph
        dcc.Graph(
            id='interactive-plot',
            style={
                'height': '2000px',  # Increased from 1500px
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }
        ),

        # Instructions
        html.Div([
            html.P(
                "Click on any dot to see the corresponding path visualization.",
                style={
                    'fontStyle': 'italic',
                    'color': '#7f8c8d',
                    'textAlign': 'center',
                    'marginTop': '20px',
                    'fontSize': '14px'
                }
            )
        ])
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'borderRadius': '8px',
        'boxShadow': '0 0 10px rgba(0,0,0,0.1)'
    })
])

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Interactive Path Analysis</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #ecf0f1;
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .radio-items input[type="radio"] {
                margin-right: 8px;
            }
            button:hover {
                background-color: #2980b9 !important;
                transition: background-color 0.3s ease;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define callback to update the figure
@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('interactive-plot', 'clickData'),
     Input('path-type-toggle', 'value'),
     Input('action-toggle', 'value'),
     Input('load-map-button', 'n_clicks')],  # Change input to button clicks
    [dash.dependencies.State('map-path-input', 'value')]  # Add state for the path input
)
def update_figure(click_data, path_type_slider, show_action, n_clicks, map_path_str):
    # Return empty figure if no path is provided or button hasn't been clicked
    if not map_path_str or n_clicks == 0:
        return go.Figure().add_annotation(
            text="Please enter a map file path and click 'Load Map'",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    try:
        # Load the fixed point map using the provided path
        map_path = Path(map_path_str)
        fixed_point_map = FixedPointMap.load(map_path)
    except Exception as e:
        # Return an error figure if the file can't be loaded
        return go.Figure().add_annotation(
            text=f"Error loading map file: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
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
        num_rows = 6  # For all coordinates
        row_heights = [0.3, 0.14, 0.14, 0.14, 0.14, 0.14]  # More space for bifurcation diagram
        vertical_spacing = 0.05  # Add more space between plots
        subplot_titles = ('Map of Available Paths', 
                        'Action Integrand vs Time',
                        'x_c vs Time', 'p_c vs Time',
                        'x_q vs Time', 'p_q vs Time')
    else:
        num_rows = 5  # Without action
        row_heights = [0.3, 0.175, 0.175, 0.175, 0.175]  # More space for bifurcation diagram
        vertical_spacing = 0.05  # Add more space between plots
        subplot_titles = ('Map of Available Paths',
                        'x_c vs Time', 'p_c vs Time',
                        'x_q vs Time', 'p_q vs Time')
    
    # Create a figure with appropriate subplots
    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=None,
        row_heights=row_heights,
        vertical_spacing=0.02,  # Reduced from 0.08 to 0.02
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
    path_p_values = []
    path_xq_values = []
    path_pq_values = []
    fixed_point_markers_t = []
    fixed_point_markers_x = []
    fixed_point_markers_p = []
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
                
                # Extract both coordinates
                path_x_values = [path_result.sol(t)[0] for t in t_points]  # x_c
                path_p_values = [path_result.sol(t)[1] for t in t_points]  # p_c
                path_xq_values = [path_result.sol(t)[2] for t in t_points]  # x_q
                path_pq_values = [path_result.sol(t)[3] for t in t_points]  # p_q
                
                # Get the fixed point values from the map
                if path_type == PathType.BRIGHT_TO_SADDLE.value:
                    start_point_type = FixedPointType.BRIGHT.value
                    end_point_type = FixedPointType.SADDLE.value
                else:  # DIM_TO_SADDLE
                    start_point_type = FixedPointType.DIM.value
                    end_point_type = FixedPointType.SADDLE.value
                
                # Get the fixed point coordinates for both x_c and p_c
                start_x = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, start_point_type, 0]
                end_x = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, end_point_type, 0]
                start_p = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, start_point_type, 1]
                end_p = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, end_point_type, 1]
                
                # Create markers for both coordinates
                fixed_point_markers_t = [0, t_max]  # Time points for both
                fixed_point_markers_x = [start_x, end_x]  # x_c values
                fixed_point_markers_p = [start_p, end_p]  # p_c values
                
                # Create labels for the markers
                start_label = "Bright" if start_point_type == FixedPointType.BRIGHT.value else "Dim"
                fixed_point_labels = [f"{start_label} State", "Saddle"]
                
                # Get path type name for the title
                path_type_name = "Bright to Saddle" if path_type == PathType.BRIGHT_TO_SADDLE.value else "Dim to Saddle"
                
                # Update subplot titles based on whether action is shown
                if show_action:
                    title_positions = [
                        (0.95, 'Map of Available Paths'),
                        (0.78, 'Action Integrand vs Time'),
                        (0.62, 'x_c vs Time'),
                        (0.46, 'p_c vs Time'),
                        (0.30, 'x_q vs Time'),
                        (0.14, 'p_q vs Time')
                    ]
                else:
                    title_positions = [
                        (0.95, 'Map of Available Paths'),
                        (0.72, 'x_c vs Time'),
                        (0.52, 'p_c vs Time'),
                        (0.32, 'x_q vs Time'),
                        (0.12, 'p_q vs Time')
                    ]
                
                # Create base annotations for titles
                annotations = []
                
                # Add parameter info to titles if a point is selected
                param_text = f" (ε={fixed_point_map.epsilon_linspace[epsilon_idx]:.4f}, κ={fixed_point_map.kappa_linspace[kappa_idx]:.4f})"
                
                fig.update_layout(
                    template='plotly_white',
                    hovermode='closest',
                    height=2000 if show_action else 1800,
                    showlegend=True,
                    uirevision='constant',
                    margin=dict(t=100, b=30, l=50, r=50),  # Reduced bottom margin from 50 to 30
                    bargap=0.2,
                    annotations=[
                        dict(
                            text=f'Map of Available {path_type_name} Paths',
                            x=0.5,
                            y=1.0,
                            xref='paper',
                            yref='paper',
                            showarrow=False,
                            font=dict(size=16)
                        )
                    ]
                )
                
                # Update x and y axis labels for each subplot
                # First plot (Map)
                fig.update_xaxes(title_text="κ", row=1, col=1)
                fig.update_yaxes(title_text="ε", row=1, col=1)

                # Make all time-based plots share x-axis
                if show_action:
                    # Action Integrand plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=2, col=1)
                    fig.update_yaxes(title_text="Action Integrand", row=2, col=1)
                    
                    # x_c plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=3, col=1)
                    fig.update_yaxes(title_text="x_c", row=3, col=1)
                    
                    # p_c plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=4, col=1)
                    fig.update_yaxes(title_text="p_c", row=4, col=1)
                    
                    # x_q plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=5, col=1)
                    fig.update_yaxes(title_text="x_q", row=5, col=1)
                    
                    # p_q plot (bottom plot gets the shared x-axis label)
                    fig.update_xaxes(title_text="Time", row=6, col=1)
                    fig.update_yaxes(title_text="p_q", row=6, col=1)
                else:
                    # x_c plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=2, col=1)
                    fig.update_yaxes(title_text="x_c", row=2, col=1)
                    
                    # p_c plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=3, col=1)
                    fig.update_yaxes(title_text="p_c", row=3, col=1)
                    
                    # x_q plot
                    fig.update_xaxes(title_text=None, showticklabels=False, row=4, col=1)
                    fig.update_yaxes(title_text="x_q", row=4, col=1)
                    
                    # p_q plot (bottom plot gets the shared x-axis label)
                    fig.update_xaxes(title_text="Time", row=5, col=1)
                    fig.update_yaxes(title_text="p_q", row=5, col=1)
                
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
        
        # Add x_c trace
        path_row_x = 3
    else:
        # Add x_c trace
        path_row_x = 2
    
    # Add x_c trace
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=path_x_values, 
            mode='lines',
            name='x_c'
        ),
        row=path_row_x, col=1
    )
    
    # Add p_c trace
    path_row_p = 4 if show_action else 3
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=path_p_values, 
            mode='lines',
            name='p_c'
        ),
        row=path_row_p, col=1
    )
    
    # Add x_q trace
    path_row_xq = 5 if show_action else 4
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=path_xq_values, 
            mode='lines',
            name='x_q'
        ),
        row=path_row_xq, col=1
    )

    # Add p_q trace
    path_row_pq = 6 if show_action else 5
    fig.add_trace(
        go.Scatter(
            x=t_points, 
            y=path_pq_values, 
            mode='lines',
            name='p_q'
        ),
        row=path_row_pq, col=1
    )
    
    # Add markers for x_c fixed points
    fig.add_trace(
        go.Scatter(
            x=fixed_point_markers_t,
            y=fixed_point_markers_x,
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                symbol='star'
            ),
            text=fixed_point_labels,
            textposition="top center",
            name='x_c Fixed Points'
        ),
        row=path_row_x, col=1
    )

    # Add markers for p_c fixed points
    fig.add_trace(
        go.Scatter(
            x=fixed_point_markers_t,
            y=fixed_point_markers_p,
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                symbol='star'
            ),
            text=fixed_point_labels,
            textposition="top center",
            name='p_c Fixed Points'
        ),
        row=path_row_p, col=1
    )
    
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)