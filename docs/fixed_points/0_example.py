import numpy as np
import matplotlib.pyplot as plt
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import EOM, Params
from metastable.extend_to_keldysh import extend_to_keldysh_state
import plotly.graph_objects as go
from tqdm import tqdm
from plotly.subplots import make_subplots

map = FixedPointMap.load(
    "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map.npz"
)

def compute_stability(classical_fixed_point: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors for a given classical fixed point.
    
    Args:
        classical_fixed_point: Array of shape (2,) containing the classical fixed point
        params: Params object containing system parameters
        
    Returns:
        tuple containing:
            eigenvalues: Complex array of shape (4,) containing eigenvalues
            eigenvectors: Complex array of shape (4, 4) containing eigenvectors as columns
    """
    if np.any(np.isnan(classical_fixed_point)):
        return np.full(4, np.nan, dtype=complex), np.full((4, 4), np.nan, dtype=complex)
    
    full_fixed_point = extend_to_keldysh_state(classical_fixed_point)
    eom = EOM(params)
    jacobian = eom.jacobian_func(full_fixed_point)
    return np.linalg.eig(jacobian)

n_epsilon = map.epsilon_linspace.size
n_kappa = map.kappa_linspace.size
n_eigenvalues = 4

# Main loop with single progress bar
eigenvalues_array = np.full((n_epsilon, n_kappa, n_eigenvalues), np.nan, dtype=complex)
eigenvectors_array = np.full((n_epsilon, n_kappa, n_eigenvalues, n_eigenvalues), np.nan, dtype=complex)

total_points = n_epsilon * n_kappa  # Adjusted total points
with tqdm(total=total_points, desc="Computing stability") as pbar:
    for epsilon_idx in range(n_epsilon):  # Step by 10
        for kappa_idx in range(n_kappa):  # Step by 10
            classical_fixed_point = map.fixed_points[epsilon_idx, kappa_idx, FixedPointType.SADDLE.value, :]
            
            # Skip if fixed point is all nan
            if np.all(np.isnan(classical_fixed_point)):
                pbar.update(1)
                continue
                
            params = Params(
                epsilon=map.epsilon_linspace[epsilon_idx],
                kappa=map.kappa_linspace[kappa_idx],
                chi=map.chi,
                delta=map.delta,
            )
            
            eigenvalues, eigenvectors = compute_stability(classical_fixed_point, params)
            eigenvalues_array[epsilon_idx, kappa_idx, :] = eigenvalues
            eigenvectors_array[epsilon_idx, kappa_idx, :, :] = eigenvectors
            
            pbar.update(1)

# Extract real and imaginary parts of eigenvalues
real_parts = eigenvalues_array.real
imag_parts = eigenvalues_array.imag

# Find global min/max across both real and imaginary parts
global_min = min(np.nanmin(real_parts), np.nanmin(imag_parts))
global_max = max(np.nanmax(real_parts), np.nanmax(imag_parts))

# Create a 2x4 subplot grid (4 eigenvalues, real and imaginary parts side by side)
fig = make_subplots(
    rows=2, 
    cols=4,
    subplot_titles=[
        'Re(λ1)', 'Re(λ2)', 'Re(λ3)', 'Re(λ4)',
        'Im(λ1)', 'Im(λ2)', 'Im(λ3)', 'Im(λ4)'
    ]
)

for i in range(4):  # Loop over each eigenvalue
    # Real part heatmap
    fig.add_trace(
        go.Heatmap(
            z=real_parts[:, :, i],
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorbar=dict(title='Value'),
            showscale=(i == 3),  # Only show colorbar for last column
            zmin=global_min,  # Set consistent color scale across all plots
            zmax=global_max,
            name=f'Re(λ{i+1})'
        ),
        row=1, col=i+1
    )
    
    # Imaginary part heatmap
    fig.add_trace(
        go.Heatmap(
            z=imag_parts[:, :, i],
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            showscale=False,  # Don't show colorbar
            zmin=global_min,  # Set consistent color scale across all plots
            zmax=global_max,
            name=f'Im(λ{i+1})'
        ),
        row=2, col=i+1
    )

# Update layout
fig.update_layout(
    title='Eigenvalue Components Across Parameter Space',
    height=800,  # Increase height for better visibility
    width=1600,  # Increase width for better visibility
)

# Update axes labels
for i in range(4):
    fig.update_xaxes(title_text='κ', row=1, col=i+1)
    fig.update_xaxes(title_text='κ', row=2, col=i+1)
    fig.update_yaxes(title_text='ε', row=1, col=i+1)
    fig.update_yaxes(title_text='ε', row=2, col=i+1)

# Save the plot as HTML
fig.write_html("eigenvalue_stability.html")

