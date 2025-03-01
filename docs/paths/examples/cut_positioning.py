from pathlib import Path
from metastable.map.map import FixedPointMap
from metastable.map.visualisations.bifurcation_lines import plot_bifurcation_diagram
import numpy as np

# Example usage:
if __name__ == "__main__":

    
    map_path = Path(
        "/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz"
    )
    fixed_point_map = FixedPointMap.load(map_path)
    
    # Create and display the bifurcation diagram
    fig = plot_bifurcation_diagram(fixed_point_map)
    fig.show()
