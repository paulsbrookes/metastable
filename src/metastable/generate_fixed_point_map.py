import numpy as np
from typing import Optional

from metastable.zero_damping import solve_zero_damping
from metastable.map.map import FixedPointMap
from metastable.extend_map import fill_map
from metastable.generate_seed_points import generate_seed_points
from metastable.eom import EOM, Params


def generate_fixed_point_map(
    epsilon_max: float,
    kappa_max: float,
    epsilon_points: int,
    kappa_points: int,
    delta: float,
    chi: float,
    max_workers: Optional[int] = None,
) -> FixedPointMap:
    """
    Generate a complete fixed point map starting from zero damping.

    The initial seed points are generated at kappa = 0 (zero damping) and finite epsilon,
    as this is where analytical solutions exist. These seed points are then used as 
    starting points to numerically extend the map across the full parameter space.

    Args:
        epsilon_max: Maximum epsilon value (range starts from 0)
        kappa_max: Maximum kappa value (range starts from 0)
        epsilon_points: Number of points along the epsilon axis
        kappa_points: Number of points along the kappa axis
        delta: Detuning parameter
        chi: Nonlinearity parameter
        max_workers: Maximum number of worker processes for parallel computation.
                     If None, uses all available CPU cores.

    Returns:
        FixedPointMap: The completed fixed point map
    """
    # Initialize the map
    seed_map = FixedPointMap(
        epsilon_linspace=np.linspace(start=0.0, stop=epsilon_max, num=epsilon_points),
        kappa_linspace=np.linspace(start=0.0, stop=kappa_max, num=kappa_points),
        delta=delta,
        chi=chi,
    )

    # Start at zero damping (kappa = 0) where analytical solutions exist
    kappa_idx = 0

    # To avoid bifurcation points and allow determination of stability, start at finite epsilon
    epsilon_idx = 100

    seed_points = generate_seed_points(
        epsilon=seed_map.epsilon_linspace[epsilon_idx],
        delta=seed_map.delta,
        chi=seed_map.chi,
    )

    if seed_map.kappa_linspace[kappa_idx] != 0.0:
        raise ValueError("Seed solutions are generated at kappa = 0, so must be inserted at kappa = 0")

    # Initialize the map with seed points
    seed_map.add_seed(
        epsilon_idx=epsilon_idx,
        kappa_idx=kappa_idx,
        dim_fixed_point=seed_points.dim,
        bright_fixed_point=seed_points.bright,
        saddle_fixed_point=seed_points.saddle,
    )

    # Extend the map across full parameter space
    return fill_map(seed_map, max_workers=max_workers)
