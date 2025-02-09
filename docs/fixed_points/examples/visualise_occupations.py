"""
Visualize the occupations of fixed points in the quantum optical system.

This script creates an interactive visualization of the occupation numbers (n)
for three types of fixed points (saddle point, bright state, and dim state)
as a function of two parameters: κ (kappa) and ε (epsilon).

The visualization consists of three heatmaps showing how the occupation numbers
vary across the parameter space for each fixed point type.

Output:
    - occupations.html: Interactive plotly visualization saved as an HTML file
"""

from pathlib import Path
from metastable.map.map import FixedPointMap
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the previously calculated fixed points map
map = FixedPointMap.load(Path("map.npz"))

# Calculate occupations (n) for each fixed point
# The occupation is defined as |α|²/2 where α is the complex amplitude
occupations = 0.5 * np.linalg.norm(map.fixed_points, axis=3) ** 2

# Create figure with three subplots arranged horizontally
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=["Dim State", "Bright State", "Saddle Point"],
    shared_yaxes=True,  # Share y-axis scale across subplots
    horizontal_spacing=0.01,  # Minimize spacing between subplots
)

# Set colorscale limits: minimum at 0 and maximum at the highest occupation value
vmin = 0
vmax = np.nanmax(occupations)

# Create heatmaps for each fixed point type
for idx in range(3):
    fig.add_trace(
        go.Heatmap(
            z=occupations[:, :, idx],
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorscale="Plasma",
            zmin=vmin,
            zmax=vmax,
            showscale=True if idx == 1 else False,  # Only show colorbar for middle plot
            colorbar=dict(
                title=dict(text="Occupation (n)", font=dict(family="Latex")),
                orientation="h",  # Horizontal colorbar
                y=-0.2,  # Position below the plots
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
    height=450,
    width=800,
    title_text="Occupation at Fixed Points (n)",
    showlegend=False,
)

# Update axes labels
for i in range(3):
    fig.update_xaxes(title_text="κ", row=1, col=i + 1)
    if i == 0:  # Only add y-axis label to the first subplot
        fig.update_yaxes(title_text="ε", row=1, col=1)

# Save as HTML file (interactive)
fig.write_html("occupations.html")
