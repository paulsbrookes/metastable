"""
Trajectory and Bifurcation Visualization Module

This module calculates trajectories between fixed points using a provided fixed point map,
and visualizes the results with Plotly. It integrates the equations of motion from a specified
initial state, plots the trajectory along with the fixed points, displays an eigenvalue table,
and overlays bifurcation lines. The final interactive plot is saved as an HTML file and displayed.
"""

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params


def calculate_and_plot_trajectory(
    fixed_point_map: FixedPointMap,
    target_kappa: float,
    target_epsilon: float,
    initial_state_fraction: float = 0.01,
    t_final_multiplier: float = 1.0,
    n_points: int = 1000
) -> go.Figure:
    """
    Calculate and plot the trajectory between fixed points.

    Args:
        fixed_point_map: FixedPointMap object containing the fixed points.
        target_kappa: Target kappa value.
        target_epsilon: Target epsilon value.
        initial_state_fraction: Fraction of the distance from the saddle to the dim state used for the initial point.
        t_final_multiplier: Multiplier to determine final integration time (t_final = multiplier / |dim eigenvalue|).
        n_points: Number of points for smooth plotting.

    Returns:
        A Plotly Figure object containing the trajectory and fixed points.
    """
    # Find the indices corresponding to the target parameters
    kappa_idx = np.abs(fixed_point_map.kappa_linspace - target_kappa).argmin()
    epsilon_idx = np.abs(fixed_point_map.epsilon_linspace - target_epsilon).argmin()

    # Retrieve the actual kappa and epsilon values
    actual_kappa = fixed_point_map.kappa_linspace[kappa_idx]
    actual_epsilon = fixed_point_map.epsilon_linspace[epsilon_idx]

    # Extract fixed points: SADDLE (index 2) and DIM (index 0)
    saddle_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 2]
    dim_state = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 0]

    # Retrieve eigenvalues for the fixed points
    dim_eigenvalues = fixed_point_map.eigenvalues[
        epsilon_idx, kappa_idx, FixedPointType.DIM.value
    ]
    saddle_eigenvalues = fixed_point_map.eigenvalues[
        epsilon_idx, kappa_idx, FixedPointType.SADDLE.value
    ]

    # Initialize the equation of motion (EOM)
    params = Params(
        epsilon=actual_epsilon,
        kappa=actual_kappa,
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )
    eom = EOM(params)

    # Compute the initial state via linear interpolation from the saddle to the dim state
    initial_state = saddle_point + initial_state_fraction * (dim_state - saddle_point)

    # Determine the final integration time based on the magnitude of the dim eigenvalue
    t_final = t_final_multiplier / np.abs(np.real(dim_eigenvalues[0]))

    # Integrate the system's equations of motion
    solution = solve_ivp(
        lambda t, y: eom.y_dot_classical_func(y),
        (0, t_final),
        initial_state,
        method='RK45',
        rtol=1e-5,
        atol=1e-5,
        dense_output=True
    )

    # Generate a smooth time grid and evaluate the solution
    t_smooth = np.linspace(solution.t[0], solution.t[-1], n_points)
    y_smooth = solution.sol(t_smooth)

    # Create the trajectory plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_smooth[0],
        y=y_smooth[1],
        mode='lines',
        name='Trajectory',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[saddle_point[0]],
        y=[saddle_point[1]],
        mode='markers',
        name='Saddle Point',
        marker=dict(color='red', size=10, symbol='x')
    ))
    fig.add_trace(go.Scatter(
        x=[dim_state[0]],
        y=[dim_state[1]],
        mode='markers',
        name='Dim State',
        marker=dict(color='green', size=10, symbol='circle')
    ))
    fig.add_trace(go.Scatter(
        x=[initial_state[0]],
        y=[initial_state[1]],
        mode='markers',
        name='Initial State',
        marker=dict(color='purple', size=10, symbol='star')
    ))

    # Update layout for consistent axis scaling and styling
    fig.update_layout(
        xaxis_title='x',
        yaxis_title='p',
        template='plotly_white',
        showlegend=True,
        yaxis=dict(scaleanchor='x', scaleratio=1),
        font=dict(size=14)
    )

    return fig


def main() -> None:
    """
    Main function to generate and display trajectory and bifurcation plots.
    """
    # Load the fixed point map from file
    map_path = 'map-with-stability.npz'
    fixed_point_map = FixedPointMap.load(map_path)

    # Create subplots: two trajectory plots, an eigenvalue table, and a bifurcation diagram
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Low Decay Rate', 'Near Cusp', '', '', 'Bifurcation Lines', ''),
        specs=[[{}, {}],
               [{"colspan": 2, "type": "table"}, None],
               [{"colspan": 2}, None]],
        vertical_spacing=0.12,
        row_heights=[0.4, 0.2, 0.35]
    )

    # Generate trajectory for the Low Decay Rate operating point
    kappa1, epsilon1 = 0.1, 10.0
    fig1 = calculate_and_plot_trajectory(
        fixed_point_map,
        target_kappa=kappa1,
        target_epsilon=epsilon1,
        t_final_multiplier=20.0,
        n_points=10_000,
    )

    # Generate trajectory for the Near Cusp operating point
    kappa2, epsilon2 = 4.3, 25.6
    fig2 = calculate_and_plot_trajectory(
        fixed_point_map,
        target_kappa=kappa2,
        target_epsilon=epsilon2,
        t_final_multiplier=20.0,
        n_points=10_000,
    )

    # Add trajectory traces to the respective subplots
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        trace.showlegend = False  # Avoid duplicate legends
        fig.add_trace(trace, row=1, col=2)

    # Extract eigenvalues for both operating points
    kappa1_idx = np.abs(fixed_point_map.kappa_linspace - kappa1).argmin()
    epsilon1_idx = np.abs(fixed_point_map.epsilon_linspace - epsilon1).argmin()
    kappa2_idx = np.abs(fixed_point_map.kappa_linspace - kappa2).argmin()
    epsilon2_idx = np.abs(fixed_point_map.epsilon_linspace - epsilon2).argmin()

    dim1_eigs = fixed_point_map.eigenvalues[
        epsilon1_idx, kappa1_idx, FixedPointType.DIM.value
    ]
    saddle1_eigs = fixed_point_map.eigenvalues[
        epsilon1_idx, kappa1_idx, FixedPointType.SADDLE.value
    ]
    dim2_eigs = fixed_point_map.eigenvalues[
        epsilon2_idx, kappa2_idx, FixedPointType.DIM.value
    ]
    saddle2_eigs = fixed_point_map.eigenvalues[
        epsilon2_idx, kappa2_idx, FixedPointType.SADDLE.value
    ]

    # Create an eigenvalue table for display
    table = go.Table(
        header=dict(
            values=['Operating Point', 'State', 'λ₀', 'λ₁'],
            font=dict(size=16),
            align='center',
            height=30
        ),
        cells=dict(
            values=[
                [f'Low Decay Rate<br>κ={kappa1:.1f}, ε={epsilon1:.1f}',
                 f'Near Cusp<br>κ={kappa2:.1f}, ε={epsilon2:.1f}'],
                ['Dim<br>Saddle', 'Dim<br>Saddle'],
                [f'{dim1_eigs[0]:.3f}<br>{saddle1_eigs[0]:.3f}',
                 f'{dim2_eigs[0]:.3f}<br>{saddle2_eigs[0]:.3f}'],
                [f'{dim1_eigs[1]:.3f}<br>{saddle1_eigs[1]:.3f}',
                 f'{dim2_eigs[1]:.3f}<br>{saddle2_eigs[1]:.3f}']
            ],
            font=dict(size=14),
            align='center',
            height=30
        )
    )
    fig.add_trace(table, row=2, col=1)

    # Retrieve bifurcation lines
    lower_line, upper_line = fixed_point_map.bifurcation_lines

    # Plot bifurcation lines
    bifurcation_traces = [
        go.Scatter(
            x=lower_line[1], y=lower_line[0],
            mode='lines',
            name='Lower Bifurcation',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        go.Scatter(
            x=upper_line[1], y=upper_line[0],
            mode='lines',
            name='Upper Bifurcation',
            line=dict(color='black', width=2),
            showlegend=False
        )
    ]
    for trace in bifurcation_traces:
        fig.add_trace(trace, row=3, col=1)

    # Add operating point markers on the bifurcation diagram
    operating_point_traces = [
        go.Scatter(
            x=[kappa1], y=[epsilon1],
            mode='markers+text',
            name='Low Decay Rate',
            text=['Low Decay Rate'],
            textposition='top right',
            marker=dict(color='blue', size=10),
            showlegend=False
        ),
        go.Scatter(
            x=[kappa2], y=[epsilon2],
            mode='markers+text',
            name='Near Cusp',
            text=['Near Cusp'],
            textposition='top right',
            marker=dict(color='red', size=10),
            showlegend=False
        )
    ]
    for trace in operating_point_traces:
        fig.add_trace(trace, row=3, col=1)

    # Update overall layout and axis labels
    fig.update_layout(
        title='Saddle to Dim State Trajectories',
        template='plotly_white',
        showlegend=True,
        height=1000,
        width=800,
        margin=dict(t=100, b=50, l=50, r=50, pad=20),
        font=dict(size=14)
    )
    fig.update_xaxes(title='x', scaleanchor='y', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title='x', scaleanchor='y', scaleratio=1, row=1, col=2)
    fig.update_yaxes(title='p', row=1, col=1)
    fig.update_yaxes(title='p', row=1, col=2)
    fig.update_xaxes(title='κ', range=[0, 5.1], row=3, col=1)
    fig.update_yaxes(title='ε', range=[0, 30.0], row=3, col=1)

    # Save and display the final interactive plot
    fig.write_html("trajectories_and_bifurcation.html")
    fig.show()


if __name__ == "__main__":
    main()
