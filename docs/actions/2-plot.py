from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, PathType

# Load the fixed point map
path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/actions/output_map_with_actions.npz")
fixed_point_map = FixedPointMap.load(path)

# Get the parameter grids
epsilon = fixed_point_map.epsilon_linspace
kappa = fixed_point_map.kappa_linspace

# Extract action values for each path type
bright_to_saddle_actions = fixed_point_map.path_actions[:, :, PathType.BRIGHT_TO_SADDLE.value, 0]
dim_to_saddle_actions = fixed_point_map.path_actions[:, :, PathType.DIM_TO_SADDLE.value, 0]

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Bright to Saddle Action", "Dim to Saddle Action"),
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.05
)

# Add heatmaps for the action values
fig.add_trace(
    go.Heatmap(
        z=bright_to_saddle_actions,  # No transpose needed with swapped axes
        y=epsilon,
        x=kappa,
        colorscale='Viridis',
        colorbar=dict(title="Action", x=0.46),
    ),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(
        z=dim_to_saddle_actions,  # No transpose needed with swapped axes
        y=epsilon,
        x=kappa,
        colorscale='Viridis',
        colorbar=dict(title="Action", x=1.0),
    ),
    row=1, col=2
)

# Add bifurcation lines if available
lower_line, upper_line = fixed_point_map.bifurcation_lines
if lower_line.size > 0 and upper_line.size > 0:
    # Lower bifurcation line
    fig.add_trace(
        go.Scatter(
            y=lower_line[0], x=lower_line[1],  # Swapped x and y
            mode='lines',
            line=dict(color='red', width=2),
            name='Lower Bifurcation'
        ),
        row=1, col=1
    )
    
    # Upper bifurcation line
    fig.add_trace(
        go.Scatter(
            y=upper_line[0], x=upper_line[1],  # Swapped x and y
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Upper Bifurcation'
        ),
        row=1, col=1
    )
    
    # Add the same lines to the second subplot
    fig.add_trace(
        go.Scatter(
            y=lower_line[0], x=lower_line[1],  # Swapped x and y
            mode='lines',
            line=dict(color='red', width=2),
            name='Lower Bifurcation',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            y=upper_line[0], x=upper_line[1],  # Swapped x and y
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Upper Bifurcation',
            showlegend=False
        ),
        row=1, col=2
    )

# Update layout
fig.update_layout(
    title="Action Values in Parameter Space",
    xaxis_title="κ (Damping Rate)",  # Swapped axis titles
    yaxis_title="ε (Drive Amplitude)",
    xaxis2_title="κ (Damping Rate)",  # Swapped axis title
    height=600,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Save as HTML for interactive viewing
fig.write_html("action_plots.html")

# Show the plot
fig.show()

