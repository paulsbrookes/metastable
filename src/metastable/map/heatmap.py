import numpy as np
import plotly.graph_objects as go
from typing import Dict, List
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType


def plot_fixed_point_map(
    fixed_point_map: FixedPointMap, fig_size=(800, 1800), cmap="Viridis"
):

    types_to_plot: List[FixedPointType] = [
        FixedPointType.SADDLE,
        FixedPointType.DIM,
        FixedPointType.BRIGHT,
    ]
    title_map: Dict[FixedPointType, str] = {
        FixedPointType.SADDLE: "Saddle Point",
        FixedPointType.DIM: "Dim State",
        FixedPointType.BRIGHT: "Bright State",
    }
    names: List[str] = [title_map[x] for x in types_to_plot]
    num_subplots: int = len(names)

    # Create a new figure with subplots
    fig = make_subplots(
        rows=num_subplots, cols=1, shared_xaxes=True, subplot_titles=names
    )

    for idx, fixed_point_type in enumerate(types_to_plot):
        norm_fixed_points = np.linalg.norm(
            fixed_point_map.fixed_points[:, :, fixed_point_type.value], axis=2
        )

        heatmap = go.Heatmap(
            z=norm_fixed_points,
            x=fixed_point_map.kappa_linspace,
            y=fixed_point_map.epsilon_linspace,
            colorscale=cmap,
            colorbar=dict(title="Norm of Fixed Points"),
        )

        fig.add_trace(heatmap, row=idx + 1, col=1)

    # Add y-axis labels
    for idx in range(num_subplots):
        fig.update_yaxes(title_text=r"$\epsilon/\delta$", row=idx + 1, col=1)

    # Add x-axis label
    fig.update_xaxes(title_text=r"$\kappa/\delta$", row=num_subplots, col=1)

    # Set the size of the figure
    fig.update_layout(
        height=fig_size[1], width=fig_size[0], title="Fixed Point Map", showlegend=False
    )

    return fig
