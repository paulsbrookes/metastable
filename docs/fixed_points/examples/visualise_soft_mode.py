import numpy as np
import plotly.graph_objects as go
from metastable.map.map import FixedPointMap, FixedPointType

def plot_saddle_eigenvalue_ratio(map, filename):
    # Get eigenvalues for saddle point
    eigenvalues = map.eigenvalues[:, :, FixedPointType.SADDLE.value, :]
    
    # Calculate absolute magnitudes of eigenvalues
    magnitudes = [np.sqrt(eigenvalues[:, :, i].real**2 + eigenvalues[:, :, i].imag**2) for i in range(2)]
    
    # Calculate ratio of magnitudes
    magnitude_ratio = np.divide(magnitudes[0], magnitudes[1], where=(magnitudes[1] != 0))
    magnitude_ratio[magnitudes[1] == 0] = np.nan
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=magnitude_ratio,
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorscale='Plasma',
            zmin=0,  # Changed from 1 since ratio can now be < 1
            zmax=1,  # Adjusted since we expect |λ₀| < |λ₁| for saddles
            colorbar=dict(
                title='|λ₀|/|λ₁|',  # Updated ratio label
                thickness=20,
                tickformat='.2f'
            )
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Saddle Point Eigenvalue Ratio',
        xaxis_title='κ',
        yaxis_title='ε',
        height=550,  # Changed from 600 to 550
        width=800,
        showlegend=False,
        margin=dict(b=150)  # Changed margin to match other plot
    )
    
    # Save the plot
    fig.write_html(filename)

# Load the map and create the plot
map = FixedPointMap.load("map-with-stability.npz")
plot_saddle_eigenvalue_ratio(map, "saddle_eigenvalue_ratio.html")