from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm


from metastable.eom import Params
from metastable.map.map import FixedPointMap
from metastable.neighbour_average import neighbour_average
from metastable.find_boundary import find_boundary
from metastable.calculate_fixed_points import calculate_fixed_points


def extend_map(seeded_map: FixedPointMap) -> FixedPointMap:
    fixed_points_map = seeded_map.copy()
    with ProcessPoolExecutor(max_workers=20) as executor:
        for round_idx in tqdm(
            range(
                max(
                    fixed_points_map.kappa_linspace.shape[0],
                    fixed_points_map.epsilon_linspace.shape[0],
                )
            )
        ):
            futures = []
            boundary_indexes_queue: List[Tuple[int, int]] = find_boundary(
                fixed_points_map.checked_points
            )
            boundary_guesses_queue: List[NDArray] = [
                neighbour_average(fixed_points_map.fixed_points, *bi)
                for bi in boundary_indexes_queue
            ]
            for (epsilon_idx, kappa_idx), guesses in zip(
                boundary_indexes_queue, boundary_guesses_queue
            ):
                params = Params(
                    epsilon=fixed_points_map.epsilon_linspace[epsilon_idx],
                    kappa=fixed_points_map.kappa_linspace[kappa_idx],
                    delta=fixed_points_map.delta,
                    chi=fixed_points_map.chi,
                )
                future = executor.submit(
                    calculate_fixed_points,
                    params,
                    guesses,
                )
                futures.append(future)
            for (epsilon_idx, kappa_idx), future in zip(
                boundary_indexes_queue, futures
            ):
                new_fixed_points = future.result()
                fixed_points_map.update_map(
                    epsilon_idx=epsilon_idx,
                    kappa_idx=kappa_idx,
                    new_fixed_points=new_fixed_points,
                )
    return fixed_points_map
