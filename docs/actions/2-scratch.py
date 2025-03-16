from pathlib import Path


from metastable.action.map import map_actions
from metastable.map.map import FixedPointMap


path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/3/output_map.npz")
fixed_point_map = FixedPointMap.load(path)
...