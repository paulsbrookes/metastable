from pathlib import Path


from metastable.action.map import map_actions
from metastable.map.map import FixedPointMap


path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/paths/examples/output/output_map.npz")
fixed_point_map = FixedPointMap.load(path)

fixed_point_map_with_actions = map_actions(fixed_point_map)

output_path = Path("/home/paul/Projects/misc/keldysh/metastable/docs/actions/output_map_with_actions.npz")
fixed_point_map_with_actions.save(output_path)


