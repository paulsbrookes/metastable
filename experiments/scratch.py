from metastable.map.map import PhaseSpaceMap
from metastable.map.heatmap import plot_fixed_point_map


map_path = "/home/paul/Projects/keldysh/metastable/experiments/backup-map/map-601x401-bright-to-saddle.npz"
fixed_point_map = PhaseSpaceMap.load(map_path)
fig = plot_fixed_point_map(fixed_point_map)
fig.show()
