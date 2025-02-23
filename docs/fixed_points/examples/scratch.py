from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# Load the map
map_path = '/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz'
fixed_point_map = FixedPointMap.load(map_path)

# Find the closest indices to our target point
# target_kappa = 0.1
# target_epsilon = 10.0

target_kappa = 4.3
target_epsilon = 25.6

kappa_idx = np.abs(fixed_point_map.kappa_linspace - target_kappa).argmin()
epsilon_idx = np.abs(fixed_point_map.epsilon_linspace - target_epsilon).argmin()

# Get the actual values we're looking at
actual_kappa = fixed_point_map.kappa_linspace[kappa_idx]
actual_epsilon = fixed_point_map.epsilon_linspace[epsilon_idx]

# Get the fixed points
saddle_point = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 2]  # 2 is SADDLE
dim_state = fixed_point_map.fixed_points[epsilon_idx, kappa_idx, 0]     # 0 is DIM

# Print the 0th eigenvalue of the dim state
dim_eigenvalue = fixed_point_map.eigenvalues[epsilon_idx, kappa_idx, FixedPointType.DIM.value, 0]
print(f"Dim state 0th eigenvalue: {dim_eigenvalue}")

# Initialize EOM with the same parameters as the map
params = Params(
    epsilon=actual_epsilon,
    kappa=actual_kappa,
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
)
eom = EOM(params)

# Calculate initial state (0.05 of the way from saddle to dim)
initial_state = saddle_point + 0.01 * (dim_state - saddle_point)

# Set up integration time
t_final = 1000/actual_kappa
t_span = (0, t_final)

# Integrate using classical EOM
solution = solve_ivp(
    lambda t, y: eom.y_dot_classical_func(y),
    t_span,
    initial_state,
    method='RK45',
    rtol=1e-5,
    atol=1e-5,
    dense_output=True  # Enable interpolation
)

print(f"Integration completed with {len(solution.t)} timesteps")
print(f"Final state: {solution.y[:, -1]}")

# Create a finer time grid for smoother plotting
t_smooth = np.linspace(solution.t[0], solution.t[-1], 1000)
y_smooth = solution.sol(t_smooth)  # Interpolate the solution

# Create the figure
fig = go.Figure()

# Plot the trajectory using the smoothed data
fig.add_trace(go.Scatter(
    x=y_smooth[0],  # x coordinate
    y=y_smooth[1],  # p coordinate
    mode='lines',
    name='Trajectory',
    line=dict(color='blue', width=2)
))

# Add the fixed points
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

# Add the initial point
fig.add_trace(go.Scatter(
    x=[initial_state[0]],
    y=[initial_state[1]],
    mode='markers',
    name='Initial State',
    marker=dict(color='purple', size=10, symbol='star')
))

# Update layout
fig.update_layout(
    title='Phase Space Trajectory',
    xaxis_title='x',
    yaxis_title='p',
    template='plotly_white',
    showlegend=True
)

# Make the plot aspect ratio equal
fig.update_layout(yaxis=dict(scaleanchor='x', scaleratio=1))

# Show the plot
fig.show()
