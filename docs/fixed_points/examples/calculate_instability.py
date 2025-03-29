import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")

def plot_eigenvalue_ratios(map, filename):
    # Create 1x2 subplot grid (one plot per fixed point type)
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=[
            '|Im(λ₀)|/|Re(λ₀)| Dim',
            '|Im(λ₀)|/|Re(λ₀)| Bright'
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
    )
    
    fixed_point_types = {
        'Dim': FixedPointType.DIM,
        'Bright': FixedPointType.BRIGHT
    }
    
    imag_real_ratio_max = 20  # Maximum for |Im(λ)|/|Re(λ)|
    
    # Add plots for each fixed point type
    for idx, (name, fp_type) in enumerate(fixed_point_types.items()):
        eigenvalues = map.eigenvalues[:, :, fp_type.value, :]
        
        # Process 0th eigenvalue only
        real_part = np.abs(eigenvalues[:, :, 0].real)
        imag_part = np.abs(eigenvalues[:, :, 0].imag)
        
        # Calculate ratio
        ratio = np.divide(imag_part, real_part, where=(real_part != 0))
        ratio[real_part == 0] = np.nan
        
        fig.add_trace(
            go.Heatmap(
                z=ratio,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                colorbar=dict(
                    title='|Im(λ₀)|/|Re(λ₀)|',
                    thickness=20,
                    orientation='h',  # Horizontal colorbar
                    y=-0.4,         # Moved down further to -0.3
                    len=0.9,         # Length of colorbar
                    x=0.5,          # Center the colorbar
                ) if idx == 1 else None,  # Only show for last plot
                zmin=0.01,
                zmax=imag_real_ratio_max,
                colorscale='plasma',
                name=f'|Im(λ₀)|/|Re(λ₀)| {name}',
                showscale=(idx == 1)  # Only show colorbar for the last plot
            ),
            row=1, col=idx+1
        )
    
    # Update layout
    fig.update_layout(
        title='Ratio of Imaginary to Real Parts of λ₀',
        height=550,
        width=800,
        showlegend=False,
        margin=dict(b=150)  # Increased bottom margin to 150
    )
    
    # Update axes labels
    for i in range(2):
        fig.update_xaxes(title_text='κ', row=1, col=i+1)
        if i == 0:  # Only leftmost plot gets y-axis label
            fig.update_yaxes(title_text='ε', row=1, col=i+1)
    
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    # Save the plot
    fig.write_html(filename)

# Create plot with dim and bright fixed points
plot_eigenvalue_ratios(map, "instability_ratios.html")
