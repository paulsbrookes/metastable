from pathlib import Path
import numpy as np

from metastable.map.map import FixedPointMap, FixedPointType
from metastable.paths.map import (
    map_switching_paths,
    IndexPair,
    prepare_saddle_and_focus_points,
)
from metastable.eom import EOM, Params


map_path = Path(
    "/home/paul/Projects/misc/keldysh/metastable/experiments/output/output_map.npz"
)

# Load the map of fixed points
fixed_point_map = FixedPointMap.load(map_path)

# epsilon_idx = 428  # dim saddle bifurcation
epsilon_idx = 340  # dim saddle bifurcation


kappa_idx = 220
index_pair = IndexPair(
    epsilon_idx=epsilon_idx,
    kappa_idx=kappa_idx,
)
fixed_points = fixed_point_map.fixed_points[epsilon_idx, kappa_idx]
print("Saddle:", fixed_points[0])
print("Bright:", fixed_points[1])
print("Dim:", fixed_points[2])
params = Params(
    epsilon=fixed_point_map.epsilon_linspace[epsilon_idx],
    kappa=fixed_point_map.kappa_linspace[kappa_idx],
    delta=fixed_point_map.delta,
    chi=fixed_point_map.chi,
)
print("epsilon:", params.epsilon / params.delta)
print("kappa:", params.kappa / params.delta)
eom = EOM(params=params)


keldysh_saddle_point, keldysh_focus_point = prepare_saddle_and_focus_points(
    fixed_point_map=fixed_point_map,
    index_pair=index_pair,
    fixed_point_type=FixedPointType.DIM,
)
jacobian = eom.jacobian_func(keldysh_focus_point)
eigenvalues, eigenvectors = np.linalg.eig(jacobian)
print("Dim Eigenavalues:", eigenvalues)


keldysh_saddle_point, keldysh_focus_point = prepare_saddle_and_focus_points(
    fixed_point_map=fixed_point_map,
    index_pair=index_pair,
    fixed_point_type=FixedPointType.BRIGHT,
)
jacobian = eom.jacobian_func(keldysh_focus_point)
eigenvalues, eigenvectors = np.linalg.eig(jacobian)
print("Bright Eigenavalues:", eigenvalues)
