from pathlib import Path
import numpy as np

from metastable.map.map import PhaseSpaceMap, FixedPointType
from metastable.paths.map import map_switching_paths, IndexPair


map_path = Path(
    "/home/paul/Projects/keldysh/metastable/experiments/output/output_map.npz"
)
output_path = Path("/home/paul/Projects/keldysh/metastable/experiments/output2")

# Check if the directory exists
if output_path.exists():
    raise FileExistsError(f"The directory {output_path} already exists.")
else:
    # Create the directory
    output_path.mkdir(parents=True)

# Load the map of fixed points
fixed_point_map = PhaseSpaceMap.load(map_path)


# Build the queue of IndexPairs whose paths we will map
epsilon_idx = 380
# kappa_indexes = list(range(230, 400))
kappa_indexes = list(range(229, 200, -1))
index_list = []
for kappa_idx in kappa_indexes:
    index_list.append(IndexPair(epsilon_idx, kappa_idx))

# Begin mapping switching paths
results = map_switching_paths(
    fixed_point_map, index_list, output_path, endpoint_type=FixedPointType.BRIGHT
)
