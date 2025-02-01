import plotly.graph_objects as go


from metastable.map.map import PhaseSpaceMap
from metastable.map.heatmap import plot_fixed_point_map
from metastable.rescaled import (
    calculate_kappa_rescaled,
    calculate_beta_limits,
    map_beta_to_epsilon,
)


map_path = "/home/paul/Projects/keldysh/metastable/experiments/backup-map/map-601x401-bright-to-saddle.npz"
fixed_point_map = PhaseSpaceMap.load(map_path)
fig = plot_fixed_point_map(fixed_point_map)


kappa_rescaled_linspace = calculate_kappa_rescaled(
    fixed_point_map.kappa_linspace, delta=fixed_point_map.delta
)  # 401
beta_limits_array = calculate_beta_limits(kappa_rescaled_linspace)
epsilon_limits_array = map_beta_to_epsilon(
    beta_limits_array, chi=fixed_point_map.chi, delta=fixed_point_map.delta
)  # 2 x 401


# Add the epsilon_limits vs kappa_linspace plot to each subplot
for idx in range(3):  # Since there are 3 subplots
    fig.add_trace(
        go.Scatter(
            x=fixed_point_map.kappa_linspace,
            y=epsilon_limits_array[0],  # Lower epsilon limit
            mode="lines",
            line=dict(color="red"),
            name="Lower Epsilon Limit",
        ),
        row=idx + 1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=fixed_point_map.kappa_linspace,
            y=epsilon_limits_array[1],  # Upper epsilon limit
            mode="lines",
            line=dict(color="blue"),
            name="Upper Epsilon Limit",
        ),
        row=idx + 1,
        col=1,
    )

# Update the layout to include a legend
fig.update_layout(showlegend=True)

fig.show()
