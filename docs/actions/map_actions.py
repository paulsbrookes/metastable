from pathlib import Path


from metastable.action.map import map_actions
from metastable.map.map import FixedPointMap


path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/12/output_map.npz")
fixed_point_map = FixedPointMap.load(path)

fixed_point_map_with_actions = map_actions(fixed_point_map)

# Derive output path from input path by modifying the filename
output_path = path.with_name(path.stem + "_with_actions" + path.suffix)
fixed_point_map_with_actions.save(output_path)


