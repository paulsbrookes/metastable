#!/usr/bin/env python3
"""
Plot Eigenvalues of a Fixed Point Map with Stability Analysis

This script loads a fixed point map from an .npz file, generates heatmaps for the
eigenvalues corresponding to different fixed point types, overlays bifurcation lines,
and saves the plots as interactive HTML files.

Dependencies:
    - numpy
    - plotly
    - metastable
"""

import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType


def plot_eigenvalues(map_obj: FixedPointMap, fixed_point_type: FixedPointType, filename: str) -> None:
    """
    Generate and save an interactive HTML plot of the eigenvalue spectrum for a given fixed point type.

    The function creates a 2x2 grid of subplots for the absolute values of the real and imaginary parts
    of two eigenvalues (λ₀ and λ₁). It also overlays bifurcation lines on each subplot.

    Args:
        map_obj: An instance of FixedPointMap loaded from the data file.
        fixed_point_type: The fixed point type (e.g., DIM, BRIGHT, SADDLE) to plot.
        filename: The filename for saving the generated HTML plot.
    """
    title_map = {
        FixedPointType.DIM: 'Dim',
        FixedPointType.BRIGHT: 'Bright',
        FixedPointType.SADDLE: 'Saddle'
    }

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            '|Re(λ₀)|', '|Im(λ₀)|',
            '|Re(λ₁)|', '|Im(λ₁)|'
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.1,
    )

    # Extract eigenvalues for the specified fixed point type
    eigenvalues = map_obj.eigenvalues[:, :, fixed_point_type.value, :]
    real_parts = [np.abs(eigenvalues[:, :, i].real) for i in range(2)]
    imag_parts = [np.abs(eigenvalues[:, :, i].imag) for i in range(2)]

    # Determine global maximum for consistent color scaling
    global_max = max(
        max(np.nanmax(r) for r in real_parts),
        max(np.nanmax(i) for i in imag_parts)
    )

    # Plot heatmaps for both eigenvalues
    for i, (real, imag) in enumerate(zip(real_parts, imag_parts)):
        row = i + 1

        # Plot absolute real part heatmap in the left column
        fig.add_trace(
            go.Heatmap(
                z=real,
                x=map_obj.kappa_linspace,
                y=map_obj.epsilon_linspace,
                colorbar=dict(
                    title=dict(text='Magnitude', font=dict(family="Latex")),
                    orientation="h",
                    y=-0.1,
                    len=0.8,
                    yanchor="top",
                    thickness=20,
                ) if i == 1 else None,  # Only show colorbar for the bottom left plot
                zmin=0,
                zmax=global_max,
                name=f'|Re(λ{i})|',
                showscale=i == 1
            ),
            row=row, col=1
        )

        # Plot absolute imaginary part heatmap in the right column
        fig.add_trace(
            go.Heatmap(
                z=imag,
                x=map_obj.kappa_linspace,
                y=map_obj.epsilon_linspace,
                showscale=False,  # Hide colorbar for right column
                zmin=0,
                zmax=global_max,
                name=f'|Im(λ{i})|'
            ),
            row=row, col=2
        )

    # Retrieve bifurcation lines and calculate axis limits
    lower_line, upper_line = map_obj.bifurcation_lines
    epsilon_min = min(np.min(lower_line[0]), np.min(upper_line[0]))
    epsilon_max = max(np.max(lower_line[0]), np.max(upper_line[0]))
    kappa_min = min(np.min(lower_line[1]), np.min(upper_line[1]))
    kappa_max = max(np.max(lower_line[1]), np.max(upper_line[1]))

    # Overlay bifurcation lines on all subplots
    for row in [1, 2]:
        for col in [1, 2]:
            # Lower bifurcation line
            fig.add_trace(
                go.Scatter(
                    x=lower_line[1],
                    y=lower_line[0],
                    mode='lines',
                    line=dict(color='red', width=2.0),
                    name='Bright-Saddle Bifurcation',
                    showlegend=(row == 1 and col == 1),
                    hoverinfo='skip'  # Disable hover tooltip
                ),
                row=row, col=col
            )

            # Upper bifurcation line
            fig.add_trace(
                go.Scatter(
                    x=upper_line[1],
                    y=upper_line[0],
                    mode='lines',
                    line=dict(color='cyan', width=2.0),
                    name='Dim-Saddle Bifurcation',
                    showlegend=(row == 1 and col == 1),
                    hoverinfo='skip'  # Disable hover tooltip
                ),
                row=row, col=col
            )

            # Set consistent axis ranges
            fig.update_xaxes(range=[kappa_min, kappa_max], row=row, col=col)
            fig.update_yaxes(range=[epsilon_min, epsilon_max], row=row, col=col)

    # Update overall layout and titles
    fig.update_layout(
        title=f'Jacobian Spectrum at the {title_map[fixed_point_type]} Fixed Point',
        height=800,
        width=800,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Set axis labels for appropriate subplots
    for i in range(2):
        for j in range(2):
            if i == 1:  # Bottom row gets x-axis labels
                fig.update_xaxes(title_text='κ', row=i+1, col=j+1)
            if j == 0:  # Leftmost column gets y-axis labels
                fig.update_yaxes(title_text='ε', row=i+1, col=j+1)

    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Save the plot to an HTML file
    fig.write_html(filename)
    logging.info("Saved plot to %s", filename)


def main() -> None:
    """Main function to load the map and generate eigenvalue plots for all fixed point types."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    map_obj = FixedPointMap.load("map-with-stability.npz")

    # Generate and save plots for each fixed point type
    plot_eigenvalues(map_obj, FixedPointType.DIM, "jacobian_spectrum_dim_fixed_point.html")
    plot_eigenvalues(map_obj, FixedPointType.BRIGHT, "jacobian_spectrum_bright_fixed_point.html")
    plot_eigenvalues(map_obj, FixedPointType.SADDLE, "jacobian_spectrum_saddle_fixed_point.html")


if __name__ == "__main__":
    main()
