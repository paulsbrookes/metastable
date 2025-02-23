import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("map-with-stability-no-imag.npz")

def plot_dim_eigenvalues(map, filename):
    # Create 2x2 subplot grid for λ₀ and λ₁
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[
            '|Re(λ₀)|', '|Im(λ₀)|',
            '|Re(λ₁)|', '|Im(λ₁)|'
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    
    # Find bistable region boundary
    bistable_region = ~np.isnan(map.eigenvalues[:, :, :, 0]).any(axis=2)
    
    # Find the boundary points
    boundary_points = []
    for i in range(len(map.kappa_linspace)):
        # Find transitions in bistable_region along epsilon axis
        transitions = np.where(np.diff(bistable_region[i, :]))[0]
        for j in transitions:
            boundary_points.append((map.kappa_linspace[i], map.epsilon_linspace[j]))
    
    boundary_points = np.array(boundary_points)
    
    # Get eigenvalues for dim fixed point
    eigenvalues = map.eigenvalues[:, :, FixedPointType.DIM.value, :]
    
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
                    y=-0.2,  # Position below the plots
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
        
        # Add bifurcation line to real part
        fig.add_trace(
            go.Scatter(
                x=boundary_points[:, 0],
                y=boundary_points[:, 1],
                mode='lines',
                line=dict(color='black', width=2),
                name='Bistable boundary'
            ),
            row=row, col=1
        )
        
        # Add bifurcation line to imaginary part
        fig.add_trace(
            go.Scatter(
                x=boundary_points[:, 0],
                y=boundary_points[:, 1],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False  # Only show legend once
            ),
            row=row, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Eigenvalues of Low Amplitude Fixed Point',
        height=800,
        width=1000,
        showlegend=False,
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

def plot_bistable_region(map, filename):
    # Create boolean map of bistable region
    bistable_region = ~np.isnan(map.eigenvalues[:, :, :, 0]).any(axis=2)
    
    # Create the figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=bistable_region.astype(int),
            x=map.kappa_linspace,
            y=map.epsilon_linspace,
            colorscale=[[0, 'white'], [1, 'rgb(0,0,255)']],  # White for monostable, blue for bistable
            showscale=False,
            name='Bistable region'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Bistable Region Map',
        xaxis_title='κ',
        yaxis_title='ε',
        height=600,
        width=800,
    )
    
    # Save the plot
    fig.write_html(filename)

# Create both plots
plot_dim_eigenvalues(map, "dim_fixed_point_eigenvalues.html")
plot_bistable_region(map, "bistable_region.html")
