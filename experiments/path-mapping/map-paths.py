from pathlib import Path


from metastable.map.map import FixedPointMap, FixedPointType
from metastable.paths.map import map_switching_paths, IndexPair


map_path = Path("/home/paul/Projects/keldysh/metastable/experiments/map.npz")
output_path = Path("/home/paul/Projects/keldysh/metastable/experiments/output_dim")

# Check if the directory exists
if output_path.exists():
    raise FileExistsError(f"The directory {output_path} already exists.")
else:
    # Create the directory
    output_path.mkdir(parents=True)

# Load the map of fixed points
fixed_point_map = FixedPointMap.load(map_path)


# Build the queue of IndexPairs whose paths we will map
epsilon_idx = 360
seed_kappa_idx = 230
index_list = [
    IndexPair(epsilon_idx, seed_kappa_idx),
    IndexPair(epsilon_idx, seed_kappa_idx - 1),
]

# Begin mapping switching paths
results = map_switching_paths(
    fixed_point_map, index_list, output_path, endpoint_type=FixedPointType.DIM
)
