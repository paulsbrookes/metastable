import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("map-with-stability-no-imag.npz")

def plot_eigenvalues(map, fixed_point_type, filename):
    # Create 2x2 subplot grid for λ₀ and λ₁
    title_map = {
        FixedPointType.DIM: 'Low Amplitude',
        FixedPointType.BRIGHT: 'High Amplitude',
        FixedPointType.SADDLE: 'Saddle'
    }
    
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[
            '|Re(λ₀)|', '|Im(λ₀)|',
            '|Re(λ₁)|', '|Im(λ₁)|'
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.1,
    )
    
    # Get eigenvalues for the specified fixed point type
    eigenvalues = map.eigenvalues[:, :, fixed_point_type.value, :]
    
    # Calculate real and imaginary parts
    real_parts = [np.abs(eigenvalues[:, :, i].real) for i in range(2)]
    imag_parts = [np.abs(eigenvalues[:, :, i].imag) for i in range(2)]
    
    # Find global max for consistent colorscale
    global_max = max(
        max(np.nanmax(r) for r in real_parts),
        max(np.nanmax(i) for i in imag_parts)
    )
    
    # Add heatmaps for both eigenvalues
    for i, (real, imag) in enumerate(zip(real_parts, imag_parts)):
        row = i + 1
        
        # Real part (left column)
        fig.add_trace(
            go.Heatmap(
                z=real,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                colorbar=dict(
                    title=dict(text='Magnitude', font=dict(family="Latex")),
                    orientation="h",  # Horizontal colorbar
                    y=-0.1,  # Changed from -0.2 to bring closer to plots
                    len=0.8,
                    yanchor="top",
                    thickness=20,
                ) if i == 1 else None,  # Only show colorbar for bottom left plot
                zmin=0,
                zmax=global_max,
                name=f'|Re(λ{i})|',
                showscale=i == 1  # Only show colorbar for bottom left plot
            ),
            row=row, col=1
        )
        
        # Imaginary part (right column)
        fig.add_trace(
            go.Heatmap(
                z=imag,
                x=map.kappa_linspace,
                y=map.epsilon_linspace,
                showscale=False,  # Hide colorbar for right column
                zmin=0,
                zmax=global_max,
                name=f'|Im(λ{i})|'
            ),
            row=row, col=2
        )
    
    # Get bifurcation lines
    lower_line, upper_line = map.bifurcation_lines
    
    # Calculate axis limits from bifurcation lines
    epsilon_min = min(np.min(lower_line[0]), np.min(upper_line[0]))
    epsilon_max = max(np.max(lower_line[0]), np.max(upper_line[0]))
    kappa_min = min(np.min(lower_line[1]), np.min(upper_line[1]))
    kappa_max = max(np.max(lower_line[1]), np.max(upper_line[1]))
    
    # Add bifurcation lines to each subplot
    for row in [1, 2]:
        for col in [1, 2]:
            # Lower bifurcation line
            fig.add_trace(
                go.Scatter(
                    x=lower_line[1],
                    y=lower_line[0],
                    mode='lines',
                    line=dict(color='red', width=2.0),
                    name='Lower bifurcation',
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
            
            # Upper bifurcation line
            fig.add_trace(
                go.Scatter(
                    x=upper_line[1],
                    y=upper_line[0],
                    mode='lines',
                    line=dict(color='cyan', width=2.0),
                    name='Upper bifurcation',
                    showlegend=(row == 1 and col == 1)
                ),
                row=row, col=col
            )
            
            # Update axis ranges for this subplot
            fig.update_xaxes(range=[kappa_min, kappa_max], row=row, col=col)
            fig.update_yaxes(range=[epsilon_min, epsilon_max], row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=f'Jacobian Spectrum at the {title_map[fixed_point_type]} Fixed Point',
        height=800,
        width=800,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    for i in range(2):
        for j in range(2):
            if i == 1:  # Only bottom row gets x-axis labels
                fig.update_xaxes(title_text='κ', row=i+1, col=j+1)
            if j == 0:  # Only leftmost column gets y-axis labels
                fig.update_yaxes(title_text='ε', row=i+1, col=j+1)
    
    
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    
    # Save the plot
    fig.write_html(filename)

# Create plots for all fixed point types
plot_eigenvalues(map, FixedPointType.DIM, "dim_fixed_point_jacobian_spectrum.html")
plot_eigenvalues(map, FixedPointType.BRIGHT, "bright_fixed_point_jacobian_spectrum.html")
plot_eigenvalues(map, FixedPointType.SADDLE, "saddle_fixed_point_jacobian_spectrum.html")
