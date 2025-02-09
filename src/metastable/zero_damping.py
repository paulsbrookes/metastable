import numpy as np
from typing import List, Tuple


def solve_zero_damping(
    epsilon: float, delta: float, chi: float
) -> List[Tuple[float, float]]:
    """Solves for the steady-state solutions of a system with zero damping.
    
    This function finds the equilibrium points (xₒ, pₒ) of a system by solving a cubic equation
    for pₒ. The function verifies the number of real roots matches theoretical expectations
    based on the discriminant of the cubic equation.
    
    Args:
        epsilon (float): Parameter affecting the constant term in the cubic equation
        delta (float): Parameter affecting the linear term in the cubic equation
        chi (float): Parameter affecting the cubic term coefficient
        
    Returns:
        List[Tuple[float, float]]: A list of (xₒ, pₒ) pairs representing steady-state solutions.
            xₒ is always 0.0, and pₒ values are the real roots of the cubic equation.
            Complex solutions are filtered out (represented as None in the output list).
            
    Raises:
        ValueError: If the number of real roots found doesn't match the theoretical expectation
            based on the discriminant.
            
    Notes:
        The function solves the cubic equation: (chi/2)pₒ³ + δpₒ - 2ε = 0
        The discriminant determines whether there are 1 or 3 real roots:
        - If discriminant > 0: three real roots
        - If discriminant ≤ 0: one real root
    """

    # Calculate the roots of steady state eom for p_c
    p_c_candidates = np.roots([chi / 2.0, 0.0, delta, 2.0 * epsilon])

    # Check the number of real roots is as expected based on the discriminant
    discriminant = (
        -4 * (chi / 2.0) * delta**3 - 27 * ((chi / 2.0) ** 2) * (2.0 * epsilon) ** 2
    )
    real_roots = [root.real for root in p_c_candidates if np.isreal(root)]
    expected_real_roots_count = 3 if discriminant > 0 else 1
    actual_real_roots_count = len(real_roots)
    if actual_real_roots_count != expected_real_roots_count:
        raise ValueError(
            f"The number of real roots was found to be {actual_real_roots_count}, but we expected "
            f"{expected_real_roots_count}."
        )

    # Filter out the complex solutions
    x_c_p_c_pairs = [
        (0.0, p_c.real) if np.isreal(p_c) else None for p_c in p_c_candidates
    ]

    return x_c_p_c_pairs
