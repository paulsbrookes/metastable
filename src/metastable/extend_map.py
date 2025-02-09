from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
from numpy.typing import NDArray
from tqdm import tqdm


from metastable.eom import Params
from metastable.map.map import PhaseSpaceMap
from metastable.neighbour_average import neighbour_average
from metastable.find_boundary import find_boundary
from metastable.calculate_fixed_points import calculate_fixed_points


def _extend_map(
    fixed_points_map: PhaseSpaceMap,
    executor: ProcessPoolExecutor
) -> None:
    """Process boundary points to find and update new fixed points.
    
    Args:
        fixed_points_map: The map to update with new fixed points
        executor: ProcessPoolExecutor for parallel processing
    """
    futures = []
    boundary_indexes_queue: List[Tuple[int, int]] = find_boundary(
        fixed_points_map.checked_points
    )
    boundary_guesses_queue: List[NDArray] = [
        neighbour_average(fixed_points_map.fixed_points, *bi)
        for bi in boundary_indexes_queue
    ]
    
    # Submit all calculations to the executor
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
    
    # Process results and update the map
    for (epsilon_idx, kappa_idx), future in zip(
        boundary_indexes_queue, futures
    ):
        new_fixed_points = future.result()
        fixed_points_map.update_map(
            epsilon_idx=epsilon_idx,
            kappa_idx=kappa_idx,
            new_fixed_points=new_fixed_points,
        )

def fill_map(seeded_map: PhaseSpaceMap, max_workers: int = 20) -> PhaseSpaceMap:
    """Completes a map of fixed points using numeric continuation from an initial seed solution. The seed solution should be inside the 
    bistable regime and should contain all three fixed points in order for numeric 
    continuation to find all fixed points throughout the parameter space.
    
    Extends a phase space map by finding fixed points around its boundaries.

    This function implements a numeric continuation approach to extend a seeded phase
    space map. It iteratively:
    1. Finding the boundary points of the currently mapped region
    2. Making initial guesses for fixed points using neighboring solutions
    3. Calculating new fixed points in parallel using Powell's hybrid method
    4. Updating the map with newly found fixed points

    The numeric continuation method works by using known solutions to make educated
    guesses about nearby solutions, allowing the map to be gradually extended outward
    from the initial seeded points.

    Args:
        seeded_map: A PhaseSpaceMap containing initial fixed points. The map is defined
            over a parameter space of epsilon (ε) and kappa (κ).
        max_workers: Maximum number of parallel processes to use for calculations.
            Defaults to 20.

    Returns:
        PhaseSpaceMap: A new map containing the original fixed points plus newly
        calculated ones around the boundaries. The map's parameters (delta, chi)
        remain unchanged from the input map.

    Note:
        The function uses parallel processing to speed up calculations. Each worker
        process calculates fixed points for a different boundary point using initial
        guesses derived from neighboring solutions.
    """
    fixed_points_map = seeded_map.copy()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for round_idx in tqdm(
            range(
                max(
                    fixed_points_map.kappa_linspace.shape[0],
                    fixed_points_map.epsilon_linspace.shape[0],
                )
            )
        ):
            _extend_map(fixed_points_map, executor)
    return fixed_points_map
