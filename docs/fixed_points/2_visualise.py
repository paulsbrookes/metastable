import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")

# Function to create and save plot for a specific fixed point type
def plot_eigenvalue(eigenvalues, point_type, filename):
    # Create arrays for both eigenvalues, taking absolute values
    real_parts = [np.abs(eigenvalues[:, :, i].real) for i in range(2)]
    imag_parts = [np.abs(eigenvalues[:, :, i].imag) for i in range(2)]
    
    # Calculate ratios of imaginary to real parts
    ratios = []
    for real, imag in zip(real_parts, imag_parts):
        ratio = np.divide(imag, real, where=(real != 0))  # Avoid division by zero
        ratio[real == 0] = np.nan  # Set ratio to nan where real part is zero
        ratios.append(ratio)
    
    # Set imaginary parts to nan where real parts are nan
    for imag, real in zip(imag_parts, real_parts):
        imag[np.isnan(real)] = np.nan
    
    # Find global max for magnitude color scaling
    global_max_magnitude = max(max(np.nanmax(r) for r in real_parts), 
                             max(np.nanmax(i) for i in imag_parts))
    
    # Set fixed max for ratio color scaling
    ratio_max = 100
    
    # Create 2x3 subplot grid with shared axes
    fig = make_subplots(
        rows=2, 
        cols=3,
        subplot_titles=['|Re(λ₀)|', '|Im(λ₀)|', '|Im(λ₀)|/|Re(λ₀)|',
                       '|Re(λ₁)|', '|Im(λ₁)|', '|Im(λ₁)|/|Re(λ₁)|'],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.1,
    )
    
    # Add heatmaps for both eigenvalues
    for i, (real, imag, ratio) in enumerate(zip(real_parts, imag_parts, ratios)):
        row = i + 1
        # Real part
        fig.add_trace(
            go.Heatmap(
                z=real,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                colorbar=dict(
                    title='Magnitude',
                    x=-0.15,  # Position colorbar to the left of all plots
                    y=-0.15,  # Position below the plots
                    len=0.5,  # Make it span two columns
                    yanchor='top',
                    orientation='h',
                    thickness=20,
                ) if i == 1 else None,  # Only show colorbar for second row
                zmin=0,
                zmax=global_max_magnitude,
                name=f'|Re(λ{i})|'
            ),
            row=row, col=1
        )
        
        # Imaginary part
        fig.add_trace(
            go.Heatmap(
                z=imag,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                showscale=False,  # Hide all imaginary part colorbars
                zmin=0,
                zmax=global_max_magnitude,
                name=f'|Im(λ{i})|'
            ),
            row=row, col=2
        )
        
        # Ratio with logarithmic scale
        fig.add_trace(
            go.Heatmap(
                z=ratio,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                colorbar=dict(
                    title='Ratio (log scale)',
                    x=1.15,  # Position colorbar to the right
                    y=-0.15,  # Position below the plots
                    len=0.25,  # Make it span one column
                    yanchor='top',
                    orientation='h',
                    thickness=20,
                    tickformat='.1e',  # Scientific notation
                ) if i == 1 else None,  # Only show colorbar for second row
                zmin=0.01,  # Minimum value for log scale
                zmax=ratio_max,
                colorscale='Viridis',
                type='heatmap',
                name=f'|Im(λ{i})|/|Re(λ{i})|'
            ),
            row=row, col=3
        )
    
    # Update layout with more bottom margin for colorbars
    fig.update_layout(
        title=f'{point_type} State Eigenvalues Across Parameter Space',
        height=1000,
        width=1800,
        margin=dict(t=100, b=150),  # Increased bottom margin for colorbars
    )
    
    # Update axes labels and sync zooming
    for i in range(2):
        for j in range(3):
            if i == 1:  # Only bottom row gets x-axis labels
                fig.update_xaxes(
                    title_text='κ',
                    row=i+1,
                    col=j+1,
                    scaleanchor=f'x{1}',  # Anchor to first x-axis
                    scaleratio=1
                )
            if j == 0:  # Only leftmost column gets y-axis labels
                fig.update_yaxes(
                    title_text='ε',
                    row=i+1,
                    col=j+1,
                    scaleanchor=f'y{1}',  # Anchor to first y-axis
                    scaleratio=1
                )

    # Synchronize all x-axes and y-axes to a reference axis
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')

    
    # Save the plot as HTML
    fig.write_html(filename)

# Create plots for each fixed point type
fixed_point_types = {
    'Bright': FixedPointType.BRIGHT,
    'Dim': FixedPointType.DIM,
    'Saddle': FixedPointType.SADDLE
}

for name, fp_type in fixed_point_types.items():
    eigenvalues = map.eigenvalues[:, :, fp_type.value, :]
    plot_eigenvalue(eigenvalues, name, f"{name.lower()}_state_eigenvalue0_stability.html")
