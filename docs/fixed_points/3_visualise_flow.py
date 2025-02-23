from metastable.eom import EOM, Params
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from tqdm import tqdm

# Helper function for parallel processing
def calculate_velocity(args):
    x, p = args
    # Create new EOM instance for each process
    eom = EOM(Params(epsilon=0.0, delta=7.8, chi=0.0, kappa=1.0))
    state = [x, p]
    return eom.y_dot_classical_func(state)

# Create a grid of classical coordinates
x_c = np.linspace(-40, 40, 50)
p_c = np.linspace(-40, 40, 50)
X_c, P_c = np.meshgrid(x_c, p_c)

# Prepare inputs for parallel processing
points = list(product(x_c, p_c))

# Calculate velocities in parallel
V_x = np.zeros_like(X_c)
V_p = np.zeros_like(P_c)

total_points = len(points)
with ProcessPoolExecutor() as executor:
    velocities = list(tqdm(
        executor.map(calculate_velocity, points),
        total=total_points,
        desc="Calculating velocities"
    ))

# Reshape results back into grid form
for idx, (i, j) in enumerate(product(range(len(x_c)), range(len(p_c)))):
    v_x, v_p = velocities[idx]
    V_x[i,j] = v_x
    V_p[i,j] = v_p

# Calculate grid spacing and maximum velocity for scaling
grid_spacing = min(x_c[1] - x_c[0], p_c[1] - p_c[0])
velocities_magnitude = np.sqrt(V_x**2 + V_p**2)
max_velocity = np.max(velocities_magnitude)

# Scale to maintain reasonable arrow lengths
scale_factor = 2 * grid_spacing / max_velocity
V_x = V_x * scale_factor
V_p = V_p * scale_factor

# Create quiver plot with scaled vectors
fig = ff.create_quiver(X_c, P_c, V_x, V_p,
                      scale=1.0,  # Use scale=1.0 since we pre-scaled the vectors
                      arrow_scale=.3,
                      name='Phase Space Flow',
                      line_width=1)

# Update layout
fig.update_layout(
    title='Classical Phase Space Flow',  # Removed "(Normalized)" from title
    xaxis_title='x_c',
    yaxis_title='p_c',
    width=800,
    height=800,
    showlegend=False
)

fig.show()
