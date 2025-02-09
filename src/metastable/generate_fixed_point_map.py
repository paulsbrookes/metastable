import numpy as np
from typing import Optional, Tuple


from metastable.map.map import FixedPointMap
from metastable.extend_map import fill_map
from metastable.generate_seed_points import generate_seed_points


def calculate_max_epsilon_bistable(delta: float, chi: float) -> float:
    """
    Calculate the maximum epsilon value in the bistable regime at kappa = 0.

    Args:
        delta: Detuning parameter
        chi: Nonlinearity parameter

    Returns:
        float: Maximum epsilon value in the bistable regime
    """
    return np.sqrt(-2 * delta**3 / (27 * chi))


def choose_seed_point_location(
    epsilon_linspace: np.ndarray, kappa_linspace: np.ndarray, delta: float, chi: float
) -> Tuple[int, int]:
    """
    Choose the location of the seed point in the parameter space. Aims to find a point 
    with zero damping and with a drive strength which places the system far from bifurcations.

    Args:
        epsilon_linspace: Array of epsilon values
        kappa_linspace: Array of kappa values
        delta: Detuning parameter
        chi: Nonlinearity parameter

    Returns:
        Tuple[int, int]: The (epsilon_idx, kappa_idx) location for the seed point

    Raises:
        ValueError: If kappa = 0 point cannot be found in kappa_linspace
    """
    # Find kappa_idx for kappa = 0
    kappa_idx = np.where(kappa_linspace == 0.0)[0]
    if len(kappa_idx) == 0:
        raise ValueError("Could not find kappa = 0 in kappa_linspace")
    kappa_idx = kappa_idx[0]

    # Calculate maximum epsilon in bistable regime at kappa = 0
    max_epsilon_bistable = calculate_max_epsilon_bistable(delta, chi)

    # Find epsilon_idx closest to half the maximum bistable epsilon
    target_epsilon = max_epsilon_bistable / 2
    epsilon_idx = np.abs(epsilon_linspace - target_epsilon).argmin()

    return epsilon_idx, kappa_idx


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

    epsilon_idx, kappa_idx = choose_seed_point_location(
        epsilon_linspace=seed_map.epsilon_linspace,
        kappa_linspace=seed_map.kappa_linspace,
        delta=delta,
        chi=chi,
    )

    seed_points = generate_seed_points(
        epsilon=seed_map.epsilon_linspace[epsilon_idx],
        delta=seed_map.delta,
        chi=seed_map.chi,
    )

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
