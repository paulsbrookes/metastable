import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, PathType
from metastable.rescaled import (
    calculate_kappa_rescaled,
    calculate_beta_limits,
    map_beta_to_epsilon,
)
from metastable.rescaled.barriers import dykman_actions_calc
from metastable.paths import get_bistable_epsilon_range

# Set global parameters
delta = 7.8  # Scaling factor
y_max = 0.6  # Upper limit for y-axis

# Create figure with two rows and one column, NOT sharing x-axis
fig = make_subplots(
    rows=2, 
    cols=1, 
    shared_xaxes=False,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.5]
)

# Load data
map_path = "/home/paul/Projects/misc/keldysh/metastable/docs/paths/sweeps/epsilon/sweep_16/output_map.npz"
# map_path = "/home/paul/Projects/misc/keldysh/metastable/docs/paths/sweeps/epsilon/archive/18/output_map_with_actions.npz"
fixed_point_map = FixedPointMap.load(map_path)

kappa_rescaled_linspace = calculate_kappa_rescaled(
    fixed_point_map.kappa_linspace, delta=fixed_point_map.delta
)
beta_limits_array = calculate_beta_limits(kappa_rescaled_linspace)
epsilon_limits_array = map_beta_to_epsilon(
    beta_limits_array, chi=fixed_point_map.chi, delta=fixed_point_map.delta
)

# Left column: Fixed kappa, sweep epsilon
kappa_idx = 150
# Filter actions to only include successful paths
successful_bright_to_saddle_mask = np.array([
    fixed_point_map.path_results[eps_idx, kappa_idx, PathType.BRIGHT_TO_SADDLE.value] is not None
    and fixed_point_map.path_results[eps_idx, kappa_idx, PathType.BRIGHT_TO_SADDLE.value].success 
    for eps_idx in range(len(fixed_point_map.epsilon_linspace))
])
successful_dim_to_saddle_mask = np.array([
    fixed_point_map.path_results[eps_idx, kappa_idx, PathType.DIM_TO_SADDLE.value] is not None
    and fixed_point_map.path_results[eps_idx, kappa_idx, PathType.DIM_TO_SADDLE.value].success 
    for eps_idx in range(len(fixed_point_map.epsilon_linspace))
])

# Create DataFrames with only successful paths
actions_bright_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[successful_bright_to_saddle_mask, kappa_idx, PathType.BRIGHT_TO_SADDLE.value, 0],
    index=fixed_point_map.epsilon_linspace[successful_bright_to_saddle_mask],
    columns=[r"$R_{b \to u}$"]
)

actions_dim_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[successful_dim_to_saddle_mask, kappa_idx, PathType.DIM_TO_SADDLE.value, 0],
    index=fixed_point_map.epsilon_linspace[successful_dim_to_saddle_mask],
    columns=[r"$R_{d \to u}$"]
)

# Get kappa value from the fixed point map
kappa = fixed_point_map.kappa_linspace[kappa_idx]

epsilon_boundaries = get_bistable_epsilon_range(fixed_point_map.bistable_region, kappa_idx)
epsilon_min = epsilon_boundaries.bright_saddle.epsilon_idx
epsilon_max = epsilon_boundaries.dim_saddle.epsilon_idx

# Calculate Dykman actions - function returns both bright_to_saddle and dim_to_saddle values
dykman_bright_to_saddle_values, dykman_dim_to_saddle_values = dykman_actions_calc(
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
    eps=fixed_point_map.epsilon_linspace[epsilon_min:epsilon_max],
    kappa=kappa,
)

# Create DataFrames with calculated values
dykman_actions_bright_to_saddle = pd.DataFrame(
    dykman_bright_to_saddle_values,
    index=fixed_point_map.epsilon_linspace[epsilon_min:epsilon_max],
    columns=[r"$R_{b \to u}$"]
)

dykman_actions_dim_to_saddle = pd.DataFrame(
    dykman_dim_to_saddle_values,
    index=fixed_point_map.epsilon_linspace[epsilon_min:epsilon_max],
    columns=[r"$R_{d \to u}$"]
)

# Rescale the index (x-axis) for all DataFrames
for df in [
    actions_bright_to_saddle,
    actions_dim_to_saddle,
    dykman_actions_bright_to_saddle,
    dykman_actions_dim_to_saddle,
]:
    df.index = df.index / delta

# Get the upper limit from dykman_actions_bright_to_saddle and lower limit from dykman_actions_dim_to_saddle
lower_kappa = dykman_actions_bright_to_saddle.index.min()
upper_kappa = dykman_actions_bright_to_saddle.index.max()

# Plot left panel data - bifurcation diagram
# Create arrays for the bifurcation curves
kappa_values = fixed_point_map.kappa_linspace / delta
epsilon_unstable_bright = epsilon_limits_array[0] / delta
epsilon_unstable_dim = epsilon_limits_array[1] / delta

# Add light grey shading to upper panel
# fig.add_shape(
#     type="rect",
#     x0=lower_kappa,
#     x1=upper_kappa,
#     y0=0,
#     y1=3.5,
#     fillcolor="lightgrey",
#     opacity=0.5,
#     layer="below",
#     line_width=0,
#     row=1, col=1
# )

# Plot upper panel data - bifurcation diagram
fig.add_trace(
    go.Scatter(
        x=kappa_values,
        y=epsilon_unstable_bright,
        mode="lines",
        name="Unstable-Bright",
        line=dict(color="red", width=3),
        legendgroup="bifurcation",
        legendgrouptitle_text="Bifurcation Diagram"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=kappa_values,
        y=epsilon_unstable_dim,
        mode="lines",
        name="Unstable-Dim",
        line=dict(color="blue", width=3),
        legendgroup="bifurcation"
    ),
    row=1, col=1
)

# Add dashed black line at kappa = kappa / delta
fig.add_trace(
    go.Scatter(
        x=[kappa / delta, kappa / delta],
        y=[0, 3.5],
        mode="lines",
        line=dict(color="black", width=3, dash="dash"),
        showlegend=False
    ),
    row=1, col=1
)

# Add light grey shading to lower panel
# fig.add_shape(
#     type="rect",
#     x0=lower_kappa,
#     x1=upper_kappa,
#     y0=0,
#     y1=8.0,
#     fillcolor="lightgrey",
#     opacity=0.5,
#     layer="below",
#     line_width=0,
#     row=2, col=1
# )

# Plot lower panel data - actions
fig.add_trace(
    go.Scatter(
        x=actions_bright_to_saddle.index,
        y=-actions_bright_to_saddle.values.flatten(),
        mode="lines",
        name="Keldysh R<sub>b→u</sub>",
        line=dict(color="red", width=3),
        legendgroup="actions",
        legendgrouptitle_text="Action Values",
        showlegend=True
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=actions_dim_to_saddle.index,
        y=-actions_dim_to_saddle.values.flatten(),
        mode="lines",
        name="Keldysh R<sub>d→u</sub>",
        line=dict(color="blue", width=3),
        legendgroup="actions",
        showlegend=True
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=dykman_actions_bright_to_saddle.index,
        y=-dykman_actions_bright_to_saddle.values.flatten(),
        mode="lines",
        name="Kramers R<sub>b→u</sub>",
        line=dict(color="purple", width=3, dash="dashdot"),
        legendgroup="actions",
        showlegend=True
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=dykman_actions_dim_to_saddle.index,
        y=-dykman_actions_dim_to_saddle.values.flatten(),
        mode="lines",
        name="Kramers R<sub>d→u</sub>",
        line=dict(color="green", width=3, dash="dashdot"),
        legendgroup="actions",
        showlegend=True
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    font=dict(size=20),
    legend=dict(
        font=dict(size=18),
        groupclick="toggleitem",
        tracegroupgap=10
    ),
    xaxis=dict(
        title=dict(text="κ/δ", font=dict(size=20)),
        tickformat=".1f"
    ),
    xaxis2=dict(
        title=dict(text="ε/δ", font=dict(size=20)),
        range=[0, 3.5],
        tickformat=".1f"
    ),
    yaxis=dict(
        title=dict(text="ε/δ", font=dict(size=20)),
        range=[0, 3.5],
        tickformat=".1f"
    ),
    yaxis2=dict(
        title=dict(text="R<sub>j→u</sub>/λ", font=dict(size=20)),
        range=[0, 8.0],
        tickformat=".1f"
    ),
    width=800,
    height=800,
    margin=dict(l=80, r=50, t=50, b=80),
    template="plotly_white",
    title="Epsilon Sweep with Action Values"
)

# Create separate legends for each subplot
for trace in fig.data:
    if trace.legendgroup == "bifurcation":
        trace.update(legendgrouptitle_font=dict(size=16))
    elif trace.legendgroup == "actions":
        trace.update(legendgrouptitle_font=dict(size=16))

# Save the plot to an HTML file
output_filename = "epsilon_sweep_with_actions.html"
fig.write_html(output_filename)
print(f"Saved plot to {output_filename}")

fig.show()
