from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        fixed_point_map: FixedPointMap object containing the fixed points
        target_kappa: Target kappa value
        target_epsilon: Target epsilon value
        initial_state_fraction: Fraction of distance from saddle to dim state for initial point
        t_final_multiplier: Multiplier for t_final (t_final = multiplier/kappa)
        n_points: Number of points for smooth plotting
    
    Returns:
        plotly.graph_objects.Figure object
    """
    # Find the closest indices to our target point
    kappa_idx = np.abs(fixed_point_map.kappa_linspace - target_kappa).argmin()
    epsilon_idx = np.abs(fixed_point_map.epsilon_linspace - target_epsilon).argmin()

    # Get the actual values
    actual_kappa = fixed_point_map.kappa_linspace[kappa_idx]
    actual_epsilon = fixed_point_map.epsilon_linspace[epsilon_idx]

    # Get the fixed points
    saddle_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 2]  # 2 is SADDLE
    dim_state = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 0]     # 0 is DIM

    # Get the eigenvalues for both fixed points
    dim_eigenvalues = fixed_point_map.eigenvalues[epsilon_idx, kappa_idx, FixedPointType.DIM.value]
    saddle_eigenvalues = fixed_point_map.eigenvalues[epsilon_idx, kappa_idx, FixedPointType.SADDLE.value]

    # Format eigenvalues for title
    dim_eig_str = f"Dim \\lambda_0={dim_eigenvalues[0]:.3f}, \\lambda_1={dim_eigenvalues[1]:.3f}"
    saddle_eig_str = f"Saddle \\lambda_0={saddle_eigenvalues[0]:.3f}, \\lambda_1={saddle_eigenvalues[1]:.3f}"

    # Initialize EOM
    params = Params(
        epsilon=actual_epsilon,
        kappa=actual_kappa,
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )
    eom = EOM(params)

    # Calculate initial state
    initial_state = saddle_point + initial_state_fraction * (dim_state - saddle_point)

    # Integrate using magnitude of real part of dim eigenvalue
    t_final = t_final_multiplier/np.abs(np.real(dim_eigenvalues[0]))
    solution = solve_ivp(
        lambda t, y: eom.y_dot_classical_func(y),
        (0, t_final),
        initial_state,
        method='RK45',
        rtol=1e-5,
        atol=1e-5,
        dense_output=True
    )

    # Create smooth interpolation
    t_smooth = np.linspace(solution.t[0], solution.t[-1], n_points)
    y_smooth = solution.sol(t_smooth)

    # Create figure
    fig = go.Figure()

    # Plot trajectory
    fig.add_trace(go.Scatter(
        x=y_smooth[0], y=y_smooth[1],
        mode='lines', name='Trajectory',
        line=dict(color='blue', width=2)
    ))

    # Add fixed points
    fig.add_trace(go.Scatter(
        x=[saddle_point[0]], y=[saddle_point[1]],
        mode='markers', name='Saddle Point',
        marker=dict(color='red', size=10, symbol='x')
    ))

    fig.add_trace(go.Scatter(
        x=[dim_state[0]], y=[dim_state[1]],
        mode='markers', name='Dim State',
        marker=dict(color='green', size=10, symbol='circle')
    ))

    # Add initial point
    fig.add_trace(go.Scatter(
        x=[initial_state[0]], y=[initial_state[1]],
        mode='markers', name='Initial State',
        marker=dict(color='purple', size=10, symbol='star')
    ))

    # Update layout with eigenvalue information
    fig.update_layout(
        xaxis_title=r'$x_\mathrm{c}$',
        yaxis_title=r'$p_\mathrm{c}$',
        template='plotly_white',
        showlegend=True,
        yaxis=dict(scaleanchor='x', scaleratio=1)
    )

    return fig

# Example usage:
if __name__ == "__main__":
    # Load the map
    map_path = '/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz'
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create subplots - now 3 rows
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Low Decay Rate', 'Near Cusp', '', '', 'Bifurcation Lines', ''),
        specs=[[{}, {}], 
               [{"colspan": 2, "type": "table"}, None],
               [{"colspan": 2}, None]],
        vertical_spacing=0.12,  # Increased from 0.04
        row_heights=[0.4, 0.175, 0.35]
    )

    # First trajectory (Low Decay Rate)
    kappa1, epsilon1 = 0.1, 10.0
    fig1 = calculate_and_plot_trajectory(
        fixed_point_map,
        target_kappa=kappa1,
        target_epsilon=epsilon1,
        t_final_multiplier=2.0,
        n_points=10_000,
    )

    # Second trajectory (Near Cusp)
    kappa2, epsilon2 = 4.3, 25.6
    fig2 = calculate_and_plot_trajectory(
        fixed_point_map,
        target_kappa=kappa2,
        target_epsilon=epsilon2,
        t_final_multiplier=2.0,
        n_points=10_000,
    )

    # Add traces from both figures to the subplots
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Get eigenvalues for both points
    kappa1_idx = np.abs(fixed_point_map.kappa_linspace - kappa1).argmin()
    epsilon1_idx = np.abs(fixed_point_map.epsilon_linspace - epsilon1).argmin()
    kappa2_idx = np.abs(fixed_point_map.kappa_linspace - kappa2).argmin()
    epsilon2_idx = np.abs(fixed_point_map.epsilon_linspace - epsilon2).argmin()

    dim1_eigs = fixed_point_map.eigenvalues[epsilon1_idx, kappa1_idx, FixedPointType.DIM.value]
    saddle1_eigs = fixed_point_map.eigenvalues[epsilon1_idx, kappa1_idx, FixedPointType.SADDLE.value]
    dim2_eigs = fixed_point_map.eigenvalues[epsilon2_idx, kappa2_idx, FixedPointType.DIM.value]
    saddle2_eigs = fixed_point_map.eigenvalues[epsilon2_idx, kappa2_idx, FixedPointType.SADDLE.value]

    # Add eigenvalue table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Operating Point', 'State', 'λ₀', 'λ₁'],
                font=dict(size=12),
                align='center'
            ),
            cells=dict(
                values=[
                    ['Low Decay Rate<br>' + f'κ={kappa1:.1f}, ε={epsilon1:.1f}',
                     'Near Cusp<br>' + f'κ={kappa2:.1f}, ε={epsilon2:.1f}'],
                    ['Dim<br>Saddle', 'Dim<br>Saddle'],
                    [f'{dim1_eigs[0]:.3f}<br>{saddle1_eigs[0]:.3f}',
                     f'{dim2_eigs[0]:.3f}<br>{saddle2_eigs[0]:.3f}'],
                    [f'{dim1_eigs[1]:.3f}<br>{saddle1_eigs[1]:.3f}',
                     f'{dim2_eigs[1]:.3f}<br>{saddle2_eigs[1]:.3f}']
                ],
                font=dict(size=11),
                align='center'
            )
        ),
        row=2, col=1
    )

    # Get bifurcation lines
    lower_line, upper_line = fixed_point_map.bifurcation_lines

    # Plot bifurcation lines
    fig.add_trace(
        go.Scatter(
            x=lower_line[1], y=lower_line[0],
            mode='lines', name='Lower Bifurcation',
            line=dict(color='black', width=2),
            showlegend=False  # Hide legend for bifurcation lines
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=upper_line[1], y=upper_line[0],
            mode='lines', name='Upper Bifurcation',
            line=dict(color='black', width=2),
            showlegend=False  # Hide legend for bifurcation lines
        ),
        row=3, col=1
    )

    # Add operating points
    fig.add_trace(
        go.Scatter(
            x=[kappa1], y=[epsilon1],
            mode='markers+text',
            name='Low Decay Rate',
            text=['Low Decay Rate'],
            textposition='top right',
            marker=dict(color='blue', size=10),
            showlegend=False
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[kappa2], y=[epsilon2],
            mode='markers+text',
            name='Near Cusp',
            text=['Near Cusp'],
            textposition='top right',
            marker=dict(color='red', size=10),
            showlegend=False
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title='Saddle to Dim State Trajectories',
        template='plotly_white',
        showlegend=True,
        height=1000,
        width=800,
        margin=dict(t=100, b=50, l=50, r=50, pad=20)  # Updated margins
    )

    # Update axes labels for phase spaces
    fig.update_xaxes(title=r'$x_c$', scaleanchor='y', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title=r'$x_c$', scaleanchor='y', scaleratio=1, row=1, col=2)
    fig.update_yaxes(title=r'$p_c$', row=1, col=1)
    fig.update_yaxes(title=r'$p_c$', row=1, col=2)
    
    # Update bifurcation diagram axes
    fig.update_xaxes(title=r'$\kappa$', range=[0, 5.1], row=3, col=1)
    fig.update_yaxes(title=r'$\epsilon$', range=[0,30.0], row=3, col=1)

    # Update subplot titles with eigenvalues
    fig.update_annotations(selector=dict(text='Low Decay Rate'), text='Low Decay Rate')
    fig.update_annotations(selector=dict(text='Near Cusp'), text='Near Cusp')

    # Save as HTML file (interactive)
    fig.write_html("trajectories_and_bifurcation.html")
    
    fig.show()