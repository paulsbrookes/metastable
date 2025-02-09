from __future__ import annotations
import numpy as np


from metastable.zero_damping import solve_zero_damping
from metastable.map.map import FixedPointMap
from metastable.extend_map import fill_map


seed_map = FixedPointMap(
    epsilon_linspace=np.linspace(start=0.0, stop=30.0, num=601),
    kappa_linspace=np.linspace(start=0.0, stop=5.0, num=401),
    delta=7.8,
    chi=-0.1,
)

# Choose a point in parameter space for the seed solution
epsilon_idx = 0
kappa_idx = 0

# Double check that we are at zero damping
assert seed_map.kappa_linspace[kappa_idx] == 0.0

# Generate the seed solution analytically
seed_points = solve_zero_damping(
    epsilon=seed_map.epsilon_linspace[epsilon_idx],
    delta=seed_map.delta,
    chi=seed_map.chi,
)

# We need to start with seeds for all three types of fixed point
assert len([point for point in seed_points if point is not None]) == 3

# Update the state of the arrays
seed_map.update_map(
    epsilon_idx=epsilon_idx, kappa_idx=kappa_idx, new_fixed_points=seed_points
)


fixed_points_map = fill_map(seed_map)

fixed_points_map.save(file_path="map-601x401.npz")
