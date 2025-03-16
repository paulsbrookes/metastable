from pathlib import Path
from metastable.map.map import FixedPointMap
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
import numpy as np

"""
Sweep Positioning Visualization Tool

This script helps visualize the bistable regime in a fixed point map and assists in
choosing optimal locations for parameter sweeps. By displaying the bifurcation diagram,
users can identify regions where the system exhibits bistability (multiple stable fixed points)
and determine the best parameter values to use for experimental sweeps.

Usage:
- Load a pre-computed fixed point map
- Generate a bifurcation diagram visualization
- Use the visualization to identify bistable regions and select sweep parameters

"""

# Example usage:
if __name__ == "__main__":
    # Path to a pre-computed fixed point map with stability information
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create and display the bifurcation diagram
    # This visualization shows where the system transitions between different stability regimes
    # and helps identify optimal positions for parameter sweeps
    fig = plot_bifurcation_diagram(fixed_point_map)
    fig.show()
