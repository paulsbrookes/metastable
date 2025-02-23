import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")

def plot_all_eigenvalues(map, filename):
    # Create 6x3 subplot grid (2 rows per fixed point type)
    fig = make_subplots(
        rows=6, 
        cols=4,
        subplot_titles=[
            '|Re(λ₀)| Dim', '|Im(λ₀)| Dim', '|Im(λ₀)|/|Re(λ₀)| Dim', '|λ₁|/|λ₀| Dim',
            '|Re(λ₁)| Dim', '|Im(λ₁)| Dim', '|Im(λ₁)|/|Re(λ₁)| Dim', '',
            '|Re(λ₀)| Saddle', '|Im(λ₀)| Saddle', '|Im(λ₀)|/|Re(λ₀)| Saddle', '|λ₁|/|λ₀| Saddle',
            '|Re(λ₁)| Saddle', '|Im(λ₁)| Saddle', '|Im(λ₁)|/|Re(λ₁)| Saddle', '',
            '|Re(λ₀)| Bright', '|Im(λ₀)| Bright', '|Im(λ₀)|/|Re(λ₀)| Bright', '|λ₁|/|λ₀| Bright',
            '|Re(λ₁)| Bright', '|Im(λ₁)| Bright', '|Im(λ₁)|/|Re(λ₁)| Bright', ''
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.03,
    )
    
    # Find global max across all fixed point types
    global_max_magnitude = 0
    fixed_point_types = {
        'Dim': FixedPointType.DIM,
        'Saddle': FixedPointType.SADDLE,
        'Bright': FixedPointType.BRIGHT
    }
    
    for fp_type in fixed_point_types.values():
        eigenvalues = map.eigenvalues[:, :, fp_type.value, :]
        real_parts = [np.abs(eigenvalues[:, :, i].real) for i in range(2)]
        imag_parts = [np.abs(eigenvalues[:, :, i].imag) for i in range(2)]
        global_max_magnitude = max(global_max_magnitude,
                                 max(np.nanmax(r) for r in real_parts),
                                 max(np.nanmax(i) for i in imag_parts))
    
    magnitude_ratio_max = 20  # Maximum for |λ₁|/|λ₀|
    imag_real_ratio_max = 20  # Maximum for |Im(λ)|/|Re(λ)|
    
    # Add plots for each fixed point type
    for idx, (name, fp_type) in enumerate(fixed_point_types.items()):
        eigenvalues = map.eigenvalues[:, :, fp_type.value, :]
        base_row = idx * 2  # Starting row for this fixed point type
        
        # Calculate absolute magnitudes of eigenvalues
        magnitudes = [np.sqrt(eigenvalues[:, :, i].real**2 + eigenvalues[:, :, i].imag**2) for i in range(2)]
        # Calculate ratio of magnitudes
        magnitude_ratio = np.divide(magnitudes[1], magnitudes[0], where=(magnitudes[0] != 0))
        magnitude_ratio[magnitudes[0] == 0] = np.nan
        
        # Process eigenvalues similar to before
        real_parts = [np.abs(eigenvalues[:, :, i].real) for i in range(2)]
        imag_parts = [np.abs(eigenvalues[:, :, i].imag) for i in range(2)]
        
        ratios = []
        for real, imag in zip(real_parts, imag_parts):
            ratio = np.divide(imag, real, where=(real != 0))
            ratio[real == 0] = np.nan
            ratios.append(ratio)
        
        for imag, real in zip(imag_parts, real_parts):
            imag[np.isnan(real)] = np.nan
        
        # Add heatmaps for both eigenvalues
        for i, (real, imag, ratio) in enumerate(zip(real_parts, imag_parts, ratios)):
            row = base_row + i + 1
            
            # Real part
            fig.add_trace(
                go.Heatmap(
                    z=real,
                    x=map.kappa_linspace,
                    y=map.epsilon_linspace,
                    colorbar=dict(
                        title='Magnitude',
                        x=-0.15,
                        y=0.5,
                        len=0.9,
                        yanchor='middle',
                        thickness=20,
                    ) if row == 6 else None,  # Only show for last row
                    zmin=0,
                    zmax=global_max_magnitude,
                    name=f'|Re(λ{i})| {name}'
                ),
                row=row, col=1
            )
            
            # Imaginary part
            fig.add_trace(
                go.Heatmap(
                    z=imag,
                    x=map.kappa_linspace,
                    y=map.epsilon_linspace,
                    showscale=False,
                    zmin=0,
                    zmax=global_max_magnitude,
                    name=f'|Im(λ{i})| {name}'
                ),
                row=row, col=2
            )
            
            # Ratio of imaginary to real parts
            fig.add_trace(
                go.Heatmap(
                    z=ratio,
                    x=map.kappa_linspace,
                    y=map.epsilon_linspace,
                    colorbar=dict(
                        title='Ratio (log scale)',
                        x=1.15,
                        y=0.5,
                        len=0.9,
                        yanchor='middle',
                        thickness=20,
                        tickformat='.1e',
                    ) if row == 6 else None,  # Only show for last row
                    zmin=0.01,
                    zmax=imag_real_ratio_max,  # Using 100 for Im/Re ratio
                    colorscale='Viridis',
                    name=f'|Im(λ{i})|/|Re(λ{i})| {name}'
                ),
                row=row, col=3
            )

            # Add magnitude ratio plot (only for first eigenvalue row)
            if i == 0:  # Only add for λ₀ row
                fig.add_trace(
                    go.Heatmap(
                        z=magnitude_ratio,
                        x=map.kappa_linspace,
                        y=map.epsilon_linspace,
                        colorbar=dict(
                            title='Magnitude Ratio',
                            x=1.45,
                            y=0.5,
                            len=0.9,
                            yanchor='middle',
                            thickness=20,
                            tickformat='.2f',
                        ) if row == 5 else None,  # Only show for second-to-last row
                        zmin=1,
                        zmax=magnitude_ratio_max,
                        colorscale='Plasma',
                        name=f'|λ₁|/|λ₀| {name}'
                    ),
                    row=row, col=4
                )
    
    # Update layout
    fig.update_layout(
        title='Fixed Point Eigenvalues Across Parameter Space',
        height=2000,
        width=2200,  # Increased width to accommodate new column
        margin=dict(t=100, b=50, l=150, r=200),  # Adjusted right margin
    )
    
    # Update axes labels
    for i in range(6):
        for j in range(4):  # Updated to include 4th column
            if i == 5:  # Only bottom row gets x-axis labels
                fig.update_xaxes(title_text='κ', row=i+1, col=j+1)
            if j == 0:  # Only leftmost column gets y-axis labels
                fig.update_yaxes(title_text='ε', row=i+1, col=j+1)
    
    # Synchronize all axes
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    
    # Save the plot
    fig.write_html(filename)

# Create single plot with all fixed points
plot_all_eigenvalues(map, "all_fixed_points_eigenvalues.html")
