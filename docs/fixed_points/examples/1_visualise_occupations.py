from pathlib import Path
from metastable.map.map import FixedPointMap
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the map
map = FixedPointMap.load(Path("map.npz"))

# Calculate occupations for each fixed point
occupations = 0.5 * np.linalg.norm(map.fixed_points, axis=3) ** 2

# Create figure with subplots
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Saddle Point", "Bright State", "Dim State"],
    shared_yaxes=True,
    horizontal_spacing=0.01,  # Reduced from default (0.2)
)

# Set minimum to 0 and use true maximum for colorbar
vmin = 0
vmax = np.nanmax(occupations)

# Create heatmaps for each fixed point type
for idx in range(3):
    fig.add_trace(
        go.Heatmap(
            z=occupations[:, :, idx],
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorscale="Viridis",
            zmin=vmin,
            zmax=vmax,
            showscale=True if idx == 1 else False,  # Only show colorbar for middle plot
            colorbar=dict(
                title=dict(text="Occupation (n)", font=dict(family="Latex")),
                orientation="h",
                y=-0.2,
                len=0.8,
                yanchor="top",
                thickness=20,
            ),
        ),
        row=1,
        col=idx + 1,
    )

# Update layout
fig.update_layout(
    height=500,
    width=800,
    title_text="Fixed Points Occupations (n)",
    showlegend=False,
)

# Update axes labels
for i in range(3):
    fig.update_xaxes(title_text="κ", row=1, col=i + 1)
    if i == 0:  # Only add y-axis label to the first subplot
        fig.update_yaxes(title_text="ε", row=1, col=1)

# Save as HTML file (interactive)
fig.write_html("occupations.html")
