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
from metastable.paths import get_bistable_kappa_range

# Set global parameters
delta = 7.8  # Scaling factor
x_max = 0.6  # Upper limit for x-axis

# Create figure with two rows and one column, sharing x-axis
fig = make_subplots(
    rows=2, 
    cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.5]
)

# Load data
map_path = "/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/4/output_map_with_actions.npz"
fixed_point_map = FixedPointMap.load(map_path)

kappa_rescaled_linspace = calculate_kappa_rescaled(
    fixed_point_map.kappa_linspace, delta=fixed_point_map.delta
)
beta_limits_array = calculate_beta_limits(kappa_rescaled_linspace)
epsilon_limits_array = map_beta_to_epsilon(
    beta_limits_array, chi=fixed_point_map.chi, delta=fixed_point_map.delta
)

# Lower row: Custom graph data
epsilon_idx = 380
# this is a dataframe with an index containing kappa
actions_bright_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[epsilon_idx, :, PathType.BRIGHT_TO_SADDLE.value, 0],
    index=fixed_point_map.kappa_linspace,
    columns=[r"$R_{b \to u}$"]
)

actions_dim_to_saddle = pd.DataFrame(
    fixed_point_map.path_actions[epsilon_idx, :, PathType.DIM_TO_SADDLE.value, 0],
    index=fixed_point_map.kappa_linspace,
    columns=[r"$R_{d \to u}$"]
)

# Get epsilon value from the fixed point map
epsilon = fixed_point_map.epsilon_linspace[epsilon_idx]

kappa_boundaries = get_bistable_kappa_range(fixed_point_map.bistable_region, epsilon_idx)
kappa_min = kappa_boundaries.dim_saddle.kappa_idx
kappa_max = kappa_boundaries.bright_saddle.kappa_idx

# Calculate Dykman actions - function returns both bright_to_saddle and dim_to_saddle values
dykman_bright_to_saddle_values, dykman_dim_to_saddle_values = dykman_actions_calc(
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
    eps=epsilon,
    kappa=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
)

# Create DataFrames with calculated values
dykman_actions_bright_to_saddle = pd.DataFrame(
    dykman_bright_to_saddle_values,
    index=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
    columns=[r"$R_{b \to u}$"]
)

dykman_actions_dim_to_saddle = pd.DataFrame(
    dykman_dim_to_saddle_values,
    index=fixed_point_map.kappa_linspace[kappa_min:kappa_max],
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
lower_epsilon = dykman_actions_bright_to_saddle.index.min()
upper_epsilon = dykman_actions_bright_to_saddle.index.max()

# Add light grey shading to upper panel
fig.add_shape(
    type="rect",
    x0=lower_epsilon,
    x1=upper_epsilon,
    y0=0,
    y1=3.5,
    fillcolor="lightgrey",
    opacity=0.5,
    layer="below",
    line_width=0,
    row=1, col=1
)

# Plot upper panel data
fig.add_trace(
    go.Scatter(
        x=fixed_point_map.kappa_linspace / delta,
        y=epsilon_limits_array[0] / delta,
        mode="lines",
        name="Unstable-Bright",
        line=dict(color="red", width=3),
        legendgroup="bifurcation"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=fixed_point_map.kappa_linspace / delta,
        y=epsilon_limits_array[1] / delta,
        mode="lines",
        name="Unstable-Dim",
        line=dict(color="blue", width=3),
        legendgroup="bifurcation"
    ),
    row=1, col=1
)

# Add dashed black line at epsilon = 19.0 / delta
fig.add_trace(
    go.Scatter(
        x=[0, x_max],
        y=[19.0 / delta, 19.0 / delta],
        mode="lines",
        line=dict(color="black", width=3, dash="dash"),
        showlegend=False
    ),
    row=1, col=1
)

# Add light grey shading to lower panel
fig.add_shape(
    type="rect",
    x0=lower_epsilon,
    x1=upper_epsilon,
    y0=0,
    y1=8.0,
    fillcolor="lightgrey",
    opacity=0.5,
    layer="below",
    line_width=0,
    row=2, col=1
)

# Plot lower panel data
fig.add_trace(
    go.Scatter(
        x=actions_bright_to_saddle.index,
        y=-actions_bright_to_saddle.values.flatten(),
        mode="lines",
        name="Keldysh R<sub>b→u</sub>",
        line=dict(color="red", width=3),
        legendgroup="keldysh"
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
        legendgroup="keldysh"
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
        legendgroup="kramers"
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
        legendgroup="kramers"
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    font=dict(size=20),
    legend=dict(
        font=dict(size=18),
        groupclick="toggleitem"
    ),
    xaxis2=dict(
        title=dict(text="κ/δ", font=dict(size=20)),
        range=[0, x_max],
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
    width=1000,
    height=800,
    margin=dict(l=80, r=50, t=50, b=80),
    template="plotly_white"
)

fig.show()
