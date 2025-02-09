from __future__ import annotations
import numpy as np
from typing import Optional

from metastable.zero_damping import solve_zero_damping
from metastable.map.map import PhaseSpaceMap
from metastable.extend_map import extend_map


def generate_fixed_point_map(
    epsilon_max: float,
    kappa_max: float,
    epsilon_points: int,
    kappa_points: int,
    delta: float,
    chi: float,
    max_workers: Optional[int] = None,
) -> PhaseSpaceMap:
    """
    Generate a complete fixed point map starting from zero damping.
    
    Args:
        epsilon_max: Maximum epsilon value (range starts from 0)
        kappa_max: Maximum kappa value (range starts from 0)
        epsilon_points: Number of points in epsilon dimension
        kappa_points: Number of points in kappa dimension
        delta: Detuning parameter
        chi: Nonlinearity parameter
        max_workers: Maximum number of worker processes for parallel computation.
                    If None, uses all available CPU cores.
        
    Returns:
        PhaseSpaceMap: The completed fixed point map
    """
    # Initialize the map
    seed_map = PhaseSpaceMap(
        epsilon_linspace=np.linspace(start=0.0, stop=epsilon_max, num=epsilon_points),
        kappa_linspace=np.linspace(start=0.0, stop=kappa_max, num=kappa_points),
        delta=delta,
        chi=chi,
    )

    # Start at zero damping
    epsilon_idx = 0
    kappa_idx = 0
    assert seed_map.kappa_linspace[kappa_idx] == 0.0

    # Generate seed solutions
    seed_points = solve_zero_damping(
        epsilon=seed_map.epsilon_linspace[epsilon_idx],
        delta=seed_map.delta,
        chi=seed_map.chi,
    )

    # Verify we have all three types of fixed points
    assert len([point for point in seed_points if point is not None]) == 3

    # Initialize the map with seed points
    seed_map.update_map(
        epsilon_idx=epsilon_idx, kappa_idx=kappa_idx, new_fixed_points=seed_points
    )

    # Extend the map to full parameter space
    return extend_map(seed_map, max_workers=max_workers)


# Example usage:
if __name__ == "__main__":
    map = generate_fixed_point_map(
        epsilon_max=30.0,
        kappa_max=5.0,
        epsilon_points=601,
        kappa_points=401,
        delta=7.8,
        chi=-0.1,
        max_workers=4,  # Use 4 worker processes
    )