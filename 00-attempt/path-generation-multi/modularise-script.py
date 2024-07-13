from pathlib import Path


from metastable.state import FixedPointMap, FixedPointType
from metastable.paths.map import map_switching_paths, IndexPair


map_path = "/home/paul/Projects/keldysh/metastable/00-attempt/map.npz"

output_path = Path("/home/paul/Projects/keldysh/metastable/00-attempt/output")
# Check if the directory exists
if output_path.exists():
    raise FileExistsError(f"The directory {output_path} already exists.")
else:
    # Create the directory
    output_path.mkdir(parents=True)


fixed_point_map = FixedPointMap.load(map_path)


# Example list of IndexPair for epsilon and kappa indexes
epsilon_idx = 360
seed_kappa_idx = 230
index_list = [
    IndexPair(epsilon_idx, seed_kappa_idx),
    IndexPair(epsilon_idx, seed_kappa_idx - 1),
]

# Call the function with the list of IndexPair
results = map_switching_paths(
    fixed_point_map, index_list, output_path, endpoint_type=FixedPointType.BRIGHT
)
