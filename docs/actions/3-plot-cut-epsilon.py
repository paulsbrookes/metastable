from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, PathType

# Load the fixed point map
path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/8/output_map_with_actions.npz")
fixed_point_map = FixedPointMap.load(path)

# Get the parameter grids
epsilon = fixed_point_map.epsilon_linspace
kappa = fixed_point_map.kappa_linspace

# Extract action values and errors for each path type
bright_to_saddle_actions = fixed_point_map.path_actions[:, :, PathType.BRIGHT_TO_SADDLE.value]
dim_to_saddle_actions = fixed_point_map.path_actions[:, :, PathType.DIM_TO_SADDLE.value]

# Create a single plot instead of subplots
fig = go.Figure()

# Function to check if a cut has valid data
def has_valid_data(data_slice):
    return np.sum(~np.isnan(data_slice)) >= 2 and np.any(data_slice != 0)

# Plot cuts along epsilon for each kappa value
colors = [f'rgb({int(255*i/len(kappa))}, {int(100+155*(1-i/len(kappa)))}, {int(255*(1-i/len(kappa)))})'
          for i in range(len(kappa))]

# For bright to saddle
for i, k in enumerate(kappa):
    # Get the cut along epsilon at this kappa
    bright_cut_values = bright_to_saddle_actions[:, i, 0]  # Action values
    bright_cut_errors = bright_to_saddle_actions[:, i, 1]  # Error values
    
    # Only plot if there's valid data
    if has_valid_data(bright_cut_values):
        fig.add_trace(
            go.Scatter(
                x=epsilon,
                y=bright_cut_values,
                mode='lines',
                line=dict(color=colors[i], width=2),
                name=f'Bright to Saddle, κ = {k:.4f}',
                legendgroup=f'kappa_{i}',
                error_y=dict(
                    type='data',
                    array=bright_cut_errors,
                    visible=True,
                    color=colors[i],
                    thickness=1,
                    width=3
                )
            )
        )

# For dim to saddle
for i, k in enumerate(kappa):
    # Get the cut along epsilon at this kappa
    dim_cut_values = dim_to_saddle_actions[:, i, 0]  # Action values
    dim_cut_errors = dim_to_saddle_actions[:, i, 1]  # Error values
    
    # Only plot if there's valid data
    if has_valid_data(dim_cut_values):
        fig.add_trace(
            go.Scatter(
                x=epsilon,
                y=dim_cut_values,
                mode='lines',
                line=dict(color=colors[i], width=2, dash='dash'),  # Use dashed lines for dim to saddle
                name=f'Dim to Saddle, κ = {k:.4f}',
                legendgroup=f'kappa_{i}',
                error_y=dict(
                    type='data',
                    array=dim_cut_errors,
                    visible=True,
                    color=colors[i],
                    thickness=1,
                    width=3
                )
            )
        )

# Add bifurcation lines if available
lower_line, upper_line = fixed_point_map.bifurcation_lines
if lower_line.size > 0 and upper_line.size > 0:
    # Add vertical lines at bifurcation points for each subplot
    # This is a simplified approach - for proper visualization you might need to interpolate
    # the bifurcation lines to find where they cross each kappa value
    pass  # Removed bifurcation lines as they don't make sense for epsilon cuts

# Update layout
fig.update_layout(
    title="Action Values vs Drive Amplitude (ε) for Different Damping Rates (κ)",
    xaxis_title="ε (Drive Amplitude)",
    yaxis_title="Action",
    height=600,
    width=1000,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# Save as HTML for interactive viewing
fig.write_html("action_cuts_plots.html")

# Show the plot
fig.show()

