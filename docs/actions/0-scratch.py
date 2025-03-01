from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import scipy.integrate
from metastable.map.map import FixedPointMap, PathType


path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/output_map.npz")
fixed_point_map = FixedPointMap.load(path)
fixed_point_map.path_results

# Define the action calculation function
def calculate_action(bvp_result):
    def integrand_func(t):
        integrand = -bvp_result.sol(t,nu=1)[0]*bvp_result.sol(t)[2]
        integrand -= bvp_result.sol(t,nu=1)[1]*bvp_result.sol(t)[3]
        return integrand

    action, action_error = scipy.integrate.quad(integrand_func, 0, bvp_result.x[-1], limit=2000, epsabs=1e-2)
    
    return action, action_error

# Create arrays to store points and their corresponding actions
bright_to_saddle_points = []
dim_to_saddle_points = []
bright_to_saddle_actions = []
dim_to_saddle_actions = []

# Iterate through the path_results array to find non-empty results and calculate actions
for eps_idx in range(len(fixed_point_map.epsilon_linspace)):
    for kap_idx in range(len(fixed_point_map.kappa_linspace)):
        # Check BRIGHT_TO_SADDLE paths (index 0)
        path_result = fixed_point_map.path_results[eps_idx, kap_idx, PathType.BRIGHT_TO_SADDLE.value]
        if path_result is not None:
            bright_to_saddle_points.append((
                fixed_point_map.kappa_linspace[kap_idx],
                fixed_point_map.epsilon_linspace[eps_idx]
            ))
            action, _ = calculate_action(path_result)
            bright_to_saddle_actions.append(action)
        
        # Check DIM_TO_SADDLE paths (index 1)
        path_result = fixed_point_map.path_results[eps_idx, kap_idx, PathType.DIM_TO_SADDLE.value]
        if path_result is not None:
            dim_to_saddle_points.append((
                fixed_point_map.kappa_linspace[kap_idx],
                fixed_point_map.epsilon_linspace[eps_idx]
            ))
            action, _ = calculate_action(path_result)
            dim_to_saddle_actions.append(action)

# Convert to numpy arrays for easier manipulation
bright_to_saddle_points = np.array(bright_to_saddle_points) if bright_to_saddle_points else np.empty((0, 2))
dim_to_saddle_points = np.array(dim_to_saddle_points) if dim_to_saddle_points else np.empty((0, 2))
bright_to_saddle_actions = np.array(bright_to_saddle_actions) if bright_to_saddle_actions else np.empty(0)
dim_to_saddle_actions = np.array(dim_to_saddle_actions) if dim_to_saddle_actions else np.empty(0)

# Create the figure
fig = go.Figure()

# Add the points to the plot with color based on action values
if len(bright_to_saddle_points) > 0:
    fig.add_trace(go.Scatter(
        x=bright_to_saddle_points[:, 0], 
        y=bright_to_saddle_points[:, 1],
        mode='markers',
        marker=dict(
            size=10, 
            symbol='circle', 
            color=bright_to_saddle_actions,
            colorscale='Viridis',
            colorbar=dict(
                title="Bright-to-Saddle Action",
                x=0.45
            ),
            showscale=True
        ),
        name='Bright-to-Saddle Paths'
    ))

if len(dim_to_saddle_points) > 0:
    fig.add_trace(go.Scatter(
        x=dim_to_saddle_points[:, 0], 
        y=dim_to_saddle_points[:, 1],
        mode='markers',
        marker=dict(
            size=10, 
            symbol='square', 
            color=dim_to_saddle_actions,
            colorscale='Plasma',
            colorbar=dict(
                title="Dim-to-Saddle Action",
                x=1.0
            ),
            showscale=True
        ),
        name='Dim-to-Saddle Paths'
    ))

# Add bifurcation lines if they exist
lower_line, upper_line = fixed_point_map.bifurcation_lines
if lower_line.size > 0 and upper_line.size > 0:
    fig.add_trace(go.Scatter(
        x=lower_line[1], 
        y=lower_line[0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Lower Bifurcation Line'
    ))
    
    fig.add_trace(go.Scatter(
        x=upper_line[1], 
        y=upper_line[0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Upper Bifurcation Line'
    ))

# Update layout
fig.update_layout(
    title='Path Results with Action Values in Parameter Space',
    xaxis_title='κ (Damping)',
    yaxis_title='ε (Drive Amplitude)',
    legend_title='Path Types',
    template='plotly_white'
)

# Create a second figure to show action values vs parameters
fig2 = go.Figure()

# Add traces for bright-to-saddle actions
if len(bright_to_saddle_points) > 0:
    fig2.add_trace(go.Scatter3d(
        x=bright_to_saddle_points[:, 0],
        y=bright_to_saddle_points[:, 1],
        z=bright_to_saddle_actions,
        mode='markers',
        marker=dict(
            size=5,
            color=bright_to_saddle_actions,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Bright-to-Saddle Actions'
    ))

# Add traces for dim-to-saddle actions
if len(dim_to_saddle_points) > 0:
    fig2.add_trace(go.Scatter3d(
        x=dim_to_saddle_points[:, 0],
        y=dim_to_saddle_points[:, 1],
        z=dim_to_saddle_actions,
        mode='markers',
        marker=dict(
            size=5,
            color=dim_to_saddle_actions,
            colorscale='Plasma',
            opacity=0.8
        ),
        name='Dim-to-Saddle Actions'
    ))

fig2.update_layout(
    title='Action Values vs Parameters',
    scene=dict(
        xaxis_title='κ (Damping)',
        yaxis_title='ε (Drive Amplitude)',
        zaxis_title='Action'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Display both figures
fig.show()
fig2.show()

# Print summary statistics
if len(bright_to_saddle_actions) > 0:
    print(f"Bright-to-Saddle Actions: min={bright_to_saddle_actions.min():.4f}, max={bright_to_saddle_actions.max():.4f}, mean={bright_to_saddle_actions.mean():.4f}")
if len(dim_to_saddle_actions) > 0:
    print(f"Dim-to-Saddle Actions: min={dim_to_saddle_actions.min():.4f}, max={dim_to_saddle_actions.max():.4f}, mean={dim_to_saddle_actions.mean():.4f}")