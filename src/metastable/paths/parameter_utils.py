import numpy as np
from metastable.map.map import FixedPointMap, FixedPointType
from metastable.eom import Params
from metastable.incoming_quantum_vector import extend_to_keldysh_state
from typing import Tuple

from metastable.paths.data_structures import IndexPair


def extract_params(fixed_point_map: FixedPointMap, index_pair: IndexPair) -> Params:
    return Params(
        epsilon=fixed_point_map.epsilon_linspace[index_pair.epsilon_idx],
        kappa=fixed_point_map.kappa_linspace[index_pair.kappa_idx],
        delta=fixed_point_map.delta,
        chi=fixed_point_map.chi,
    )

def prepare_saddle_and_focus_points(
    fixed_point_map: FixedPointMap,
    index_pair: IndexPair,
    fixed_point_type: FixedPointType,
) -> Tuple[np.ndarray, np.ndarray]:
    classical_saddle_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, FixedPointType.SADDLE.value
    ]
    classical_focus_point = fixed_point_map.fixed_points[
        index_pair.epsilon_idx, index_pair.kappa_idx, fixed_point_type.value
    ]

    keldysh_saddle_point = extend_to_keldysh_state(classical_saddle_point)
    keldysh_focus_point = extend_to_keldysh_state(classical_focus_point)

    return keldysh_saddle_point, keldysh_focus_point 